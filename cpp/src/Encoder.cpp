#include "Encoder.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bgr_to_nv12.h"

extern "C" {
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

namespace {

std::string ff_err(int ret) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, buf, sizeof(buf));
    return std::string(buf);
}

}  // namespace

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

    cleanup();
    is_finished        = false;
    frame_index        = 0;
    ffmpeg_cuda_stream = nullptr;

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

    const AVCodec* encoder_codec = avcodec_find_encoder_by_name("libx264");
    if (!encoder_codec) {
        throw std::runtime_error("libx264 codec not found");
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
    codec_ctx->bit_rate     = bitrate;
    codec_ctx->gop_size     = fps;
    codec_ctx->max_b_frames = 0;
    codec_ctx->pix_fmt      = AV_PIX_FMT_YUV420P;

    video_stream->time_base = codec_ctx->time_base;

    if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    AVDictionary* opts = NULL;
    av_dict_set(&opts, "preset", "veryfast", 0);
    av_dict_set(&opts, "tune", "zerolatency", 0);
    av_dict_set(&opts, "profile", "baseline", 0);

    ret = avcodec_open2(codec_ctx, encoder_codec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        throw std::runtime_error("Could not open codec libx264: " + ff_err(ret));
    }

    ret = avcodec_parameters_from_context(video_stream->codecpar, codec_ctx);
    if (ret < 0) {
        throw std::runtime_error("Could not copy stream parameters: " + ff_err(ret));
    }

    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&format_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            throw std::runtime_error("Could not open output: " + filename + ": " + ff_err(ret));
        }
    }

    ret = avformat_write_header(format_ctx, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Error occurred when opening output: " + ff_err(ret));
    }

    frame  = av_frame_alloc();
    packet = av_packet_alloc();
    if (!frame || !packet) {
        throw std::runtime_error("Failed to allocate frame/packet");
    }

    frame->format = codec_ctx->pix_fmt;
    frame->width  = width;
    frame->height = height;
    ret           = av_frame_get_buffer(frame, 32);
    if (ret < 0) {
        throw std::runtime_error("Failed to allocate SW frame buffer: " + ff_err(ret));
    }

    sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_BGR24, width, height, codec_ctx->pix_fmt, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx) {
        throw std::runtime_error("Failed to create sws context");
    }

    std::cerr << "[Encoder] Using libx264 for " << filename << std::endl;
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
    if (sws_ctx) sws_freeContext(sws_ctx);

    format_ctx        = nullptr;
    codec_ctx         = nullptr;
    hw_device_ctx     = nullptr;
    video_stream      = nullptr;
    packet            = nullptr;
    frame             = nullptr;
    sws_ctx           = nullptr;
    ffmpeg_cuda_stream = nullptr;
}

void Encoder::encode(torch::Tensor tensor, double pts) {
    if (!codec_ctx || !frame) {
        throw std::runtime_error("Encoder is not initialized");
    }

    int ret = 0;

    if (tensor.dtype() != torch::kUInt8) {
        throw std::runtime_error("Input tensor must be uint8 for libx264");
    }
    if (tensor.is_cuda()) {
        tensor = tensor.cpu();
    }
    if (tensor.dim() != 3 || tensor.size(0) != height || tensor.size(1) != width || tensor.size(2) != 3) {
        throw std::runtime_error("Input tensor must be HWC uint8 BGR with encoder resolution");
    }
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    ret = av_frame_make_writable(frame);
    if (ret < 0) {
        throw std::runtime_error("Frame not writable: " + ff_err(ret));
    }

    uint8_t* src_slices[1]  = {tensor.data_ptr<uint8_t>()};
    int      src_strides[1] = {width * 3};

    int scaled = sws_scale(sws_ctx, src_slices, src_strides, 0, height, frame->data, frame->linesize);
    if (scaled != height) {
        throw std::runtime_error("sws_scale failed");
    }

    if (pts >= 0) {
        // pts is in seconds
        frame->pts = (int64_t)(pts / av_q2d(codec_ctx->time_base));
    } else {
        frame->pts = frame_index++;
    }

    ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0) {
        throw std::runtime_error("Error sending frame to encoder: " + ff_err(ret));
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            throw std::runtime_error("Error encoding frame: " + ff_err(ret));
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
