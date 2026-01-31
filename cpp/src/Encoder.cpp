#include "Encoder.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "bgr_to_nv12.h"

Encoder::Encoder(const std::string& filename, int width, int height, int fps, std::string codec, int bitrate)
    : filename(filename), width(width), height(height), fps(fps), bitrate(bitrate) {
    init_ffmpeg(codec);
}

Encoder::~Encoder() {
    finish();
    cleanup();
}

void Encoder::init_ffmpeg(std::string codec) {
    av_log_set_level(AV_LOG_ERROR);

    const char* format_name = nullptr;
    if (filename.find("rtsp://") == 0) {
        format_name = "rtsp";
    } else if (filename.find("rtmp://") == 0) {
        format_name = "flv";
    }

    int ret = avformat_alloc_output_context2(&format_ctx, nullptr, format_name, filename.c_str());
    if (!format_ctx) {
        ret = avformat_alloc_output_context2(&format_ctx, nullptr, "flv", filename.c_str());
    }
    if (!format_ctx) {
        throw std::runtime_error("Could not create output context for " + filename);
    }

    const AVCodec* encoder_codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!encoder_codec) {
        throw std::runtime_error("h264_nvenc codec not found");
    }

    video_stream = avformat_new_stream(format_ctx, encoder_codec);
    if (!video_stream) {
        throw std::runtime_error("Could not allocate stream");
    }

    codec_ctx = avcodec_alloc_context3(encoder_codec);
    if (!codec_ctx) {
        throw std::runtime_error("Could not allocate codec context");
    }

    codec_ctx->width        = width;
    codec_ctx->height       = height;
    codec_ctx->time_base    = {1, fps};
    codec_ctx->framerate    = {fps, 1};
    codec_ctx->pix_fmt      = AV_PIX_FMT_CUDA;
    codec_ctx->bit_rate     = bitrate;
    codec_ctx->gop_size     = fps;
    codec_ctx->max_b_frames = 0;

    video_stream->time_base = codec_ctx->time_base;

    if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        throw std::runtime_error("Failed to create CUDA HW device");
    }

    {
        AVHWDeviceContext*   device_ctx      = (AVHWDeviceContext*)hw_device_ctx->data;
        AVCUDADeviceContext* cuda_device_ctx = device_ctx ? (AVCUDADeviceContext*)device_ctx->hwctx : nullptr;
        ffmpeg_cuda_stream                  = cuda_device_ctx ? (void*)cuda_device_ctx->stream : nullptr;
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ref) {
        throw std::runtime_error("Failed to allocate HW frames context");
    }

    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;
    frames_ctx->format            = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format         = AV_PIX_FMT_NV12;
    frames_ctx->width             = width;
    frames_ctx->height            = height;
    frames_ctx->initial_pool_size = 20;

    ret = av_hwframe_ctx_init(hw_frames_ref);
    if (ret < 0) {
        av_buffer_unref(&hw_frames_ref);
        throw std::runtime_error("Failed to initialize HW frames context");
    }

    codec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
    av_buffer_unref(&hw_frames_ref);

    AVDictionary* opts = NULL;
    av_dict_set(&opts, "preset", "p1", 0);
    av_dict_set(&opts, "tune", "ull", 0);
    av_dict_set(&opts, "zerolatency", "1", 0);

    ret = avcodec_open2(codec_ctx, encoder_codec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        throw std::runtime_error("Could not open codec");
    }

    ret = avcodec_parameters_from_context(video_stream->codecpar, codec_ctx);
    if (ret < 0) {
        throw std::runtime_error("Could not copy stream parameters");
    }

    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&format_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            throw std::runtime_error("Could not open output file: " + filename);
        }
    }

    ret = avformat_write_header(format_ctx, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Error occurred when opening output file");
    }

    frame  = av_frame_alloc();
    packet = av_packet_alloc();
}

void Encoder::cleanup() {
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) {
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx->pb);
        }
        avformat_free_context(format_ctx);
    }
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
}

void Encoder::encode(torch::Tensor tensor, double pts) {
    if (!tensor.is_cuda() || tensor.dtype() != torch::kUInt8) {
        throw std::runtime_error("Input tensor must be CUDA uint8");
    }

    c10::cuda::CUDAGuard device_guard(tensor.device());

    cudaStream_t torch_stream  = c10::cuda::getCurrentCUDAStream().stream();
    cudaStream_t encode_stream = ffmpeg_cuda_stream ? (cudaStream_t)ffmpeg_cuda_stream : torch_stream;

    int ret = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, frame, 0);
    if (ret < 0) {
        throw std::runtime_error("Failed to allocate frame from HW pool");
    }

    ret = av_frame_make_writable(frame);
    if (ret < 0) {
        throw std::runtime_error("Frame not writable");
    }

    const uint8_t* pSrc     = tensor.data_ptr<uint8_t>();
    int            nSrcStep = width * 3;

    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
        pSrc   = tensor.data_ptr<uint8_t>();
    }

    uint8_t* pDstY      = (uint8_t*)frame->data[0];
    int      nDstYStep  = frame->linesize[0];
    uint8_t* pDstUV     = (uint8_t*)frame->data[1];
    int      nDstUVStep = frame->linesize[1];

    if (encode_stream != torch_stream) {
        cudaEvent_t ev;
        cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaEventCreateWithFlags failed: ") + cudaGetErrorString(err));
        }
        err = cudaEventRecord(ev, torch_stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(ev);
            throw std::runtime_error(std::string("cudaEventRecord failed: ") + cudaGetErrorString(err));
        }
        err = cudaStreamWaitEvent(encode_stream, ev, 0);
        if (err != cudaSuccess) {
            cudaEventDestroy(ev);
            throw std::runtime_error(std::string("cudaStreamWaitEvent failed: ") + cudaGetErrorString(err));
        }
        cudaEventDestroy(ev);
    }

    bgr_to_nv12(pSrc, nSrcStep, pDstY, nDstYStep, pDstUV, nDstUVStep, width, height, encode_stream);

    if (pts >= 0) {
        // pts is in seconds
        frame->pts = (int64_t)(pts / av_q2d(codec_ctx->time_base));
    } else {
        frame->pts = frame_index++;
    }

    ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0) {
        throw std::runtime_error("Error sending frame to encoder");
    }
    av_frame_unref(frame);

    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            throw std::runtime_error("Error encoding frame");
        }

        av_packet_rescale_ts(packet, codec_ctx->time_base, video_stream->time_base);
        packet->stream_index = video_stream->index;

        ret = av_interleaved_write_frame(format_ctx, packet);
        av_packet_unref(packet);
        if (ret < 0) {
            char err_buf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, err_buf, AV_ERROR_MAX_STRING_SIZE);
            std::cerr << "Error writing packet: " << err_buf << std::endl;
            // throw std::runtime_error("Error writing packet");
        }
    }
}

void Encoder::finish() {
    if (is_finished) return;
    is_finished = true;

    if (!codec_ctx) return;

    int ret = avcodec_send_frame(codec_ctx, nullptr);

    while (true) {
        ret = avcodec_receive_packet(codec_ctx, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            std::cerr << "Error flushing encoder" << std::endl;
            break;
        }

        av_packet_rescale_ts(packet, codec_ctx->time_base, video_stream->time_base);
        packet->stream_index = video_stream->index;

        av_interleaved_write_frame(format_ctx, packet);
        av_packet_unref(packet);
    }

    av_write_trailer(format_ctx);
}
