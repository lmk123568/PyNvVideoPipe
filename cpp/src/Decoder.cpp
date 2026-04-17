#include "Decoder.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <libavutil/error.h>
#include <libavutil/rational.h>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>

#include <stdexcept>
#include <thread>

static std::string ff_err(int ret) {
    char buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, buf, sizeof(buf));
    return std::string(buf);
}

static void sync_stream_throw(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[Decoder] CUDA流同步失败: ") + cudaGetErrorString(err));
    }
}

static void sync_stream_noexcept(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA)
            return *p;
    }
    throw std::runtime_error("[Decoder] 无法获取硬件解码表面格式");
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
    try {
        init_ffmpeg(filename);
    } catch (const std::exception& e) {
        bool is_stream = filename.rfind("rtsp://", 0) == 0 || filename.rfind("rtp://", 0) == 0;
        if (!is_stream) {
            throw;
        }
        is_streaming_source = true;
        std::cerr << "[Decoder] 初始化流失败，进入重连模式: " << source_url << ", 原因: " << e.what() << std::endl;
    }
}

Decoder::~Decoder() {
    cleanup();
}

void Decoder::init_ffmpeg(const std::string& filename) {
    // 日志只报错误信息
    av_log_set_level(AV_LOG_ERROR);
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
        throw std::runtime_error("[Decoder] 无法打开输入流: " + filename);
    }
    if (opts) {
        av_dict_free(&opts);
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        throw std::runtime_error("[Decoder] 无法获取媒体流信息");
    }

    // 打印媒体流信息
    // av_dump_format(format_ctx, 0, filename.c_str(), 0);

    video_stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        throw std::runtime_error("[Decoder] 未找到视频流");
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
        throw std::runtime_error("[Decoder] 未找到视频解码器");
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("[Decoder] 无法分配解码器上下文");
    }

    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
        throw std::runtime_error("[Decoder] 无法复制解码器参数");
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
            throw std::runtime_error("[Decoder] 创建 CUDA 硬件解码设备失败");
        }
        codec_ctx->hw_device_ctx   = av_buffer_ref(hw_device_ctx);
        codec_ctx->get_format      = get_hw_format;
        codec_ctx->extra_hw_frames = surfaces;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("[Decoder] 打开解码器失败");
    }

    if (codec_ctx->hw_frames_ctx) {
        AVHWFramesContext* frames_ctx = reinterpret_cast<AVHWFramesContext*>(codec_ctx->hw_frames_ctx->data);
        if (frames_ctx) {
            int pool_size = frames_ctx->initial_pool_size;
            std::cerr << "[Decoder] 当前硬件解码帧池大小: " << source_url << ", 大小: " << pool_size << std::endl;
        }
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
    if (frame) av_frame_unref(frame);
    if (packet) av_packet_unref(packet);
    if (codec_ctx) avcodec_flush_buffers(codec_ctx);
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
        // std::cerr << "[Decoder] 视频流断线中，正在重拉("
        //           << reconnect_attempts
        //           << "/"
        //           << (max_reconnects == 0 ? std::string("无限") : std::to_string(max_reconnects))
        //           << "): "
        //           << source_url
        //           << std::endl;
        cleanup();
        flushing          = false;
        finished          = false;
        output_this_frame = true;
        bad_frame_streak  = 0;
        last_input_pts    = -1.0;
        if (nominal_frame_delta <= 0.0 && fps > 0.0) {
            nominal_frame_delta = 1.0 / fps;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        try {
            init_ffmpeg(source_url);
            // std::cerr << "[Decoder] 重连成功: " << source_url << std::endl;
            return true;
        } catch (const std::exception& e) {
            // std::cerr << "[Decoder] 重连失败: " << source_url << ", 原因: " << e.what() << std::endl;
            continue;
        }
    }
    // std::cerr << "[Decoder] 连续重连失败" << reconnect_attempts << "次，判定无信号: " << source_url << std::endl;
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
    if (!codec_ctx || !format_ctx || !frame || !packet) {
        if (!try_reconnect()) {
            throw std::runtime_error("[Decoder] 无信号，重连3次失败");
        }
    }

    auto process_frame = [&](AVFrame* f) -> torch::Tensor {
        if (f->format != AV_PIX_FMT_CUDA) {
            throw std::runtime_error("[Decoder] 帧像素格式不是 CUDA: " + std::to_string(f->format));
        }
        if (!f->data[0] || !f->data[1] || f->linesize[0] <= 0) {
            throw std::runtime_error("[Decoder] NV12 硬件帧数据无效");
        }

        int frame_w = f->width > 0 ? f->width : decode_width;
        int frame_h = f->height > 0 ? f->height : decode_height;
        if (frame_w <= 0 || frame_h <= 0) {
            throw std::runtime_error("[Decoder] 无效的帧尺寸");
        }
        if (frame_w != decode_width || frame_h != decode_height) {
            decode_width  = frame_w;
            decode_height = frame_h;
            if (requested_width <= 0 || requested_height <= 0) {
                width  = decode_width;
                height = decode_height;
            }
            std::cerr << "[Decoder] 检测到分辨率变化: " << source_url << ", 新分辨率: " << decode_width << "x" << decode_height << std::endl;
        }

        cudaStream_t     stream = c10::cuda::getCurrentCUDAStream().stream();
        NppStreamContext npp_stream_ctx;
        NppStatus        npp_stream_status = nppGetStreamContext(&npp_stream_ctx);
        if (npp_stream_status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] nppGetStreamContext 失败, 错误码: " + std::to_string(npp_stream_status));
        }
        npp_stream_ctx.hStream = stream;

        auto          options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA).layout(torch::kStrided);
        torch::Tensor bgr     = torch::empty({frame_h, frame_w, 3}, options);
        const Npp8u*  pSrc[2];
        pSrc[0]           = (const Npp8u*)f->data[0];
        pSrc[1]           = (const Npp8u*)f->data[1];
        int      nSrcStep = f->linesize[0];
        Npp8u*   pDst     = bgr.data_ptr<uint8_t>();
        int      nDstStep = static_cast<int>(bgr.stride(0) * bgr.element_size());
        NppiSize oSizeROI;
        oSizeROI.width   = frame_w;
        oSizeROI.height  = frame_h;
        NppStatus status = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, npp_stream_ctx);
        if (status != NPP_SUCCESS) {
            throw std::runtime_error("[Decoder] NPP NV12→BGR 颜色转换失败, 错误码: " + std::to_string(status));
        }

        torch::Tensor out = bgr;
        if (width != frame_w || height != frame_h) {
            torch::Tensor resized    = torch::empty({height, width, 3}, options);
            const Npp8u*  pResizeSrc = bgr.data_ptr<uint8_t>();
            Npp8u*        pResizeDst = resized.data_ptr<uint8_t>();

            NppiSize srcSize;
            srcSize.width  = frame_w;
            srcSize.height = frame_h;
            int srcStep    = static_cast<int>(bgr.stride(0) * bgr.element_size());
            int dstStep    = static_cast<int>(resized.stride(0) * resized.element_size());

            NppiRect srcROI;
            srcROI.x      = 0;
            srcROI.y      = 0;
            srcROI.width  = frame_w;
            srcROI.height = frame_h;

            NppiRect dstROI;
            dstROI.x      = 0;
            dstROI.y      = 0;
            dstROI.width  = width;
            dstROI.height = height;

            double xFactor = static_cast<double>(width) / static_cast<double>(frame_w);
            double yFactor = static_cast<double>(height) / static_cast<double>(frame_h);

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
                throw std::runtime_error("[Decoder] NPP 尺寸缩放失败, 错误码: " + std::to_string(resize_status));
            }
            out = resized;
        }
        sync_stream_throw(stream);
        return out;
    };

    if (finished) {
        throw std::runtime_error("[Decoder] 视频流已结束或无可用信号");
    }

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

            torch::Tensor out;
            try {
                out = process_frame(frame);
            } catch (const std::exception& e) {
                sync_stream_noexcept(c10::cuda::getCurrentCUDAStream().stream());
                std::cerr << "[Decoder] 异常帧已跳过: " << source_url << ", 原因: " << e.what() << std::endl;
                bad_frame_streak += 1;
                av_frame_unref(frame);
                if (bad_frame_streak >= 5) {
                    std::cerr << "[Decoder] 连续异常帧过多，执行重拉恢复: " << source_url << std::endl;
                    if (try_reconnect()) {
                        continue;
                    }
                    finished = true;
                    return {torch::Tensor(), -1.0};
                }
                continue;
            } catch (...) {
                sync_stream_noexcept(c10::cuda::getCurrentCUDAStream().stream());
                std::cerr << "[Decoder] 异常帧已跳过: " << source_url << ", 原因: 未知异常" << std::endl;
                bad_frame_streak += 1;
                av_frame_unref(frame);
                if (bad_frame_streak >= 5) {
                    std::cerr << "[Decoder] 连续异常帧过多，执行重拉恢复: " << source_url << std::endl;
                    if (try_reconnect()) {
                        continue;
                    }
                    finished = true;
                    return {torch::Tensor(), -1.0};
                }
                continue;
            }
            bad_frame_streak = 0;
            av_frame_unref(frame);
            return {out, pts};
        } else if (ret == AVERROR_EOF) {
            if (try_reconnect()) {
                continue;
            }
            finished = true;
            throw std::runtime_error("[Decoder] 无信号，重连3次失败");
        } else if (ret != AVERROR(EAGAIN)) {
            if (ret == AVERROR_INVALIDDATA) {
                std::cerr << "[Decoder] 接收解码帧无效，已跳过当前帧: " << source_url << std::endl;
                continue;
            }
            std::cerr << "[Decoder] 接收解码帧失败: " << source_url << ", 原因: " << ff_err(ret) << std::endl;
            if (try_reconnect()) {
                continue;
            }
            finished = true;
            throw std::runtime_error("[Decoder] 无信号，重连3次失败");
        }

        if (flushing) {
            finished = true;
            throw std::runtime_error("[Decoder] 视频流已结束或无可用信号");
        }

        ret = av_read_frame(format_ctx, packet);
        if (ret < 0) {
            std::cerr << "[Decoder] 读取视频包失败: " << source_url << ", 原因: " << ff_err(ret) << std::endl;
            if (try_reconnect()) {
                continue;
            }
            flushing      = true;
            int flush_ret = avcodec_send_packet(codec_ctx, nullptr);
            if (flush_ret < 0) {
                std::cerr << "[Decoder] 刷新解码器失败: " << source_url << ", 原因: " << ff_err(flush_ret) << std::endl;
                finished = true;
                throw std::runtime_error("[Decoder] 无信号，重连3次失败");
            }
            continue;
        }

        if (packet->stream_index == video_stream_idx) {
            ret = avcodec_send_packet(codec_ctx, packet);
            av_packet_unref(packet);
            if (ret < 0) {
                if (ret == AVERROR_INVALIDDATA) {
                    std::cerr << "[Decoder] 无效视频包已丢弃: " << source_url << std::endl;
                    continue;
                }
                if (ret == AVERROR(EAGAIN)) {
                    continue;
                }
                std::cerr << "[Decoder] 发送视频包到解码器失败: " << source_url << ", 原因: " << ff_err(ret) << std::endl;
                if (try_reconnect()) {
                    continue;
                }
                finished = true;
                throw std::runtime_error("[Decoder] 无信号，重连3次失败");
            }
        } else {
            av_packet_unref(packet);
        }
    }
}
