#include "Decoder.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <libavutil/rational.h>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>

#include <stdexcept>
#include <thread>

static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA)
            return *p;
    }
    throw std::runtime_error("[Decoder] Failed to get HW surface format.");
}

Decoder::Decoder(const std::string& filename,
                 bool               enable_frame_skip_,
                 int                output_width,
                 int                output_height,
                 bool               enable_auto_reconnect,
                 int                reconnect_delay_ms_,
                 int                max_reconnects_,
                 int                open_timeout_ms_,
                 int                read_timeout_ms_,
                 int                buffer_size_,
                 int                max_delay_ms_,
                 int                reorder_queue_size_,
                 int                decoder_threads_,
                 int                surfaces_,
                 std::string        hwaccel_)
    : source_url(filename),
      requested_width(output_width),
      requested_height(output_height),
      enable_frame_skip(enable_frame_skip_),
      output_this_frame(true),
      enable_reconnect(enable_auto_reconnect),
      reconnect_delay_ms(reconnect_delay_ms_),
      max_reconnects(max_reconnects_),
      open_timeout_ms(open_timeout_ms_),
      read_timeout_ms(read_timeout_ms_),
      buffer_size(buffer_size_),
      max_delay_ms(max_delay_ms_),
      reorder_queue_size(reorder_queue_size_),
      decoder_threads(decoder_threads_),
      surfaces(surfaces_),
      hwaccel(std::move(hwaccel_)) {
    init_ffmpeg(filename);
}

Decoder::~Decoder() {
    cleanup();
}

void Decoder::init_ffmpeg(const std::string& filename) {
    av_log_set_level(AV_LOG_INFO);
    AVDictionary* opts  = nullptr;
    is_streaming_source = false;
    if (filename.rfind("rtsp://", 0) == 0) {
        is_streaming_source = true;
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    }
    if (filename.rfind("rtp://", 0) == 0) {
        is_streaming_source = true;
    }

    if (open_timeout_ms > 0) {
        av_dict_set(&opts, "stimeout", std::to_string(open_timeout_ms * 1000).c_str(), 0);
    }
    if (read_timeout_ms > 0) {
        av_dict_set(&opts, "rw_timeout", std::to_string(read_timeout_ms * 1000).c_str(), 0);
    }
    if (buffer_size > 0) {
        av_dict_set(&opts, "buffer_size", std::to_string(buffer_size).c_str(), 0);
        av_dict_set(&opts, "rtbufsize", std::to_string(buffer_size).c_str(), 0);
    }
    if (max_delay_ms > 0) {
        av_dict_set(&opts, "max_delay", std::to_string(max_delay_ms * 1000).c_str(), 0);
    }
    if (reorder_queue_size > 0) {
        av_dict_set(&opts, "reorder_queue_size", std::to_string(reorder_queue_size).c_str(), 0);
    }

    if (avformat_open_input(&format_ctx, filename.c_str(), nullptr, &opts) != 0) {
        if (opts) {
            av_dict_free(&opts);
        }
        throw std::runtime_error("[Decoder] Could not open input file: " + filename);
    }
    if (opts) {
        av_dict_free(&opts);
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        throw std::runtime_error("[Decoder] Could not find stream info");
    }
    av_dump_format(format_ctx, 0, filename.c_str(), 0);

    video_stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        throw std::runtime_error("[Decoder] Could not find video stream");
    }

    AVStream*          stream   = format_ctx->streams[video_stream_idx];
    AVCodecParameters* codecpar = stream->codecpar;

    AVRational frame_rate = stream->avg_frame_rate;
    if (frame_rate.num == 0 || frame_rate.den == 0) {
        frame_rate = stream->r_frame_rate;
    }
    if (frame_rate.num != 0 && frame_rate.den != 0) {
        fps = av_q2d(frame_rate);
    }
    if (fps > 0.0) {
        nominal_frame_delta = 1.0 / fps;
    }

    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        throw std::runtime_error("[Decoder] Codec not found");
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("[Decoder] Could not allocate codec context");
    }

    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
        throw std::runtime_error("[Decoder] Could not copy codec params");
    }

    if (decoder_threads <= 0) {
        decoder_threads = 2;
    }
    codec_ctx->thread_count = decoder_threads;
    codec_ctx->thread_type  = FF_THREAD_FRAME;

    if (surfaces < 2) surfaces = 2;
    if (surfaces > 5) surfaces = 5;

    if (hwaccel == "cuda") {
        if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
            throw std::runtime_error("[Decoder] Failed to create CUDA HW device");
        }
        codec_ctx->hw_device_ctx   = av_buffer_ref(hw_device_ctx);
        codec_ctx->get_format      = get_hw_format;
        codec_ctx->extra_hw_frames = surfaces;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("[Decoder] Could not open codec");
    }

    decode_width  = codec_ctx->width;
    decode_height = codec_ctx->height;

    if (requested_width > 0 && requested_height > 0) {
        width  = requested_width;
        height = requested_height;
    } else {
        width  = decode_width;
        height = decode_height;
    }

    frame              = av_frame_alloc();
    packet             = av_packet_alloc();
    reconnect_attempts = 0;
}

void Decoder::cleanup() {
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
}

bool Decoder::try_reconnect() {
    if (!enable_reconnect || !is_streaming_source) return false;
    while (max_reconnects == 0 || reconnect_attempts < max_reconnects) {
        reconnect_attempts += 1;
        cleanup();
        flushing          = false;
        finished          = false;
        output_this_frame = true;
        last_input_pts    = -1.0;
        if (nominal_frame_delta <= 0.0 && fps > 0.0) {
            nominal_frame_delta = 1.0 / fps;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        try {
            init_ffmpeg(source_url);
            return true;
        } catch (const std::exception&) {
            continue;
        }
    }
    return false;
}

double Decoder::align_pts(double pts) {
    double expected = nominal_frame_delta > 0.0 ? nominal_frame_delta : (fps > 0.0 ? 1.0 / fps : 0.04);
    if (pts < 0.0) {
        if (last_output_pts < 0.0) {
            last_output_pts = 0.0;
        } else {
            last_output_pts += expected;
        }
        return last_output_pts;
    }
    if (last_input_pts < 0.0) {
        last_input_pts = pts;
        if (last_output_pts < 0.0) {
            last_output_pts = 0.0;
        } else {
            last_output_pts += expected;
        }
        pts_offset = last_output_pts - pts;
        return last_output_pts;
    }
    double raw_delta = pts - last_input_pts;
    if (raw_delta <= 0.0) {
        raw_delta = expected;
    }
    double tol       = max_delay_ms > 0 ? (max_delay_ms / 1000.0) : (expected * 2.0);
    double min_delta = expected - tol;
    if (min_delta < 0.0) min_delta = 0.0;
    double max_delta = expected + tol;
    double clamped   = raw_delta;
    if (raw_delta < min_delta || raw_delta > max_delta) {
        clamped = expected;
    }
    nominal_frame_delta  = expected * 0.98 + clamped * 0.02;
    last_input_pts       = pts;
    last_output_pts     += clamped;
    pts_offset           = last_output_pts - pts;
    return last_output_pts;
}

std::pair<torch::Tensor, double> Decoder::next_frame() {
    auto process_frame = [&](AVFrame* f) -> torch::Tensor {
        if (f->format != AV_PIX_FMT_CUDA) {
            std::cerr << "[Decoder] Frame format is not CUDA: " << f->format << std::endl;
            return torch::Tensor();
        }

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        NppStreamContext npp_stream_ctx;
        NppStatus npp_stream_status = nppGetStreamContext(&npp_stream_ctx);
        if (npp_stream_status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] nppGetStreamContext failed: " + std::to_string(npp_stream_status));
        }
        npp_stream_ctx.hStream = stream;

        auto          options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA).layout(torch::kStrided);
        torch::Tensor bgr     = torch::empty({decode_height, decode_width, 3}, options);
        const Npp8u*  pSrc[2];
        pSrc[0]           = (const Npp8u*)f->data[0];
        pSrc[1]           = (const Npp8u*)f->data[1];
        int      nSrcStep = f->linesize[0];
        Npp8u*   pDst     = bgr.data_ptr<uint8_t>();
        int      nDstStep = decode_width * 3;
        NppiSize oSizeROI;
        oSizeROI.width   = decode_width;
        oSizeROI.height  = decode_height;
        NppStatus status = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, npp_stream_ctx);
        if (status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] NPP conversion failed: " + std::to_string(status));
        }

        if (width == decode_width && height == decode_height) {
            return bgr;
        }

        torch::Tensor resized    = torch::empty({height, width, 3}, options);
        const Npp8u*  pResizeSrc = bgr.data_ptr<uint8_t>();
        Npp8u*        pResizeDst = resized.data_ptr<uint8_t>();

        NppiSize srcSize;
        srcSize.width  = decode_width;
        srcSize.height = decode_height;
        int srcStep    = decode_width * 3;
        int dstStep    = width * 3;

        NppiRect srcROI;
        srcROI.x      = 0;
        srcROI.y      = 0;
        srcROI.width  = decode_width;
        srcROI.height = decode_height;

        NppiRect dstROI;
        dstROI.x      = 0;
        dstROI.y      = 0;
        dstROI.width  = width;
        dstROI.height = height;

        double xFactor = static_cast<double>(width) / static_cast<double>(decode_width);
        double yFactor = static_cast<double>(height) / static_cast<double>(decode_height);

        NppStatus resize_status = nppiResizeSqrPixel_8u_C3R_Ctx(
            pResizeSrc,
            srcSize,
            srcStep,
            srcROI,
            pResizeDst,
            dstStep,
            dstROI,
            xFactor,
            yFactor,
            0.0,
            0.0,
            NPPI_INTER_LINEAR,
            npp_stream_ctx);
        if (resize_status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] NPP resize failed: " + std::to_string(resize_status));
        }

        return resized;
    };

    if (finished) return {torch::Tensor(), -1.0};

    while (true) {
        int ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret >= 0) {
            if (enable_frame_skip && !output_this_frame) {
                output_this_frame = true;
                av_frame_unref(frame);
                continue;
            }
            output_this_frame = enable_frame_skip ? false : true;

            double  pts     = -1.0;
            int64_t best_ts = frame->best_effort_timestamp;
            if (best_ts != AV_NOPTS_VALUE) {
                AVRational tb = format_ctx->streams[video_stream_idx]->time_base;
                pts           = best_ts * av_q2d(tb);
            }
            pts = align_pts(pts);

            torch::Tensor out = process_frame(frame);
            av_frame_unref(frame);
            return {out, pts};
        } else if (ret == AVERROR_EOF) {
            if (try_reconnect()) {
                continue;
            }
            finished = true;
            return {torch::Tensor(), -1.0};
        } else if (ret != AVERROR(EAGAIN)) {
            throw std::runtime_error("[Decoder] Error receiving frame: " + std::to_string(ret));
        }

        if (flushing) {
            finished = true;
            return {torch::Tensor(), -1.0};
        }

        ret = av_read_frame(format_ctx, packet);
        if (ret < 0) {
            if (try_reconnect()) {
                continue;
            }
            flushing = true;
            avcodec_send_packet(codec_ctx, nullptr);
            continue;
        }

        if (packet->stream_index == video_stream_idx) {
            ret = avcodec_send_packet(codec_ctx, packet);
            av_packet_unref(packet);
            if (ret < 0) throw std::runtime_error("[Decoder] Error sending packet");
        } else {
            av_packet_unref(packet);
        }
    }
}

