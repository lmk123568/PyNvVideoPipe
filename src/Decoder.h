#pragma once

#include <torch/extension.h>

#include <memory>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

class Decoder {
public:
    Decoder(const std::string &filename,
            bool               enable_frame_skip     = false,
            int                output_width          = 0,
            int                output_height         = 0,
            bool               enable_auto_reconnect = true,
            int                reconnect_delay_ms    = 500,
            int                max_reconnects        = 0,
            int                open_timeout_ms       = 5000,
            int                read_timeout_ms       = 5000,
            int                buffer_size           = 4 * 1024 * 1024,
            int                max_delay_ms          = 200,
            int                reorder_queue_size    = 0);
    ~Decoder();

    std::pair<torch::Tensor, double> next_frame();
    int                              get_width() const { return width; }
    int                              get_height() const { return height; }
    double                           get_fps() const { return fps; }

private:
    void   init_ffmpeg(const std::string &filename);
    void   cleanup();
    bool   try_reconnect();
    double align_pts(double pts);

    AVFormatContext *format_ctx       = nullptr;
    AVCodecContext  *codec_ctx        = nullptr;
    AVBufferRef     *hw_device_ctx    = nullptr;
    int              video_stream_idx = -1;

    AVFrame  *frame  = nullptr;
    AVPacket *packet = nullptr;

    std::string source_url;
    int         width               = 0;
    int         height              = 0;
    int         decode_width        = 0;
    int         decode_height       = 0;
    int         requested_width     = 0;
    int         requested_height    = 0;
    double      fps                 = 0.0;
    bool        flushing            = false;
    bool        finished            = false;
    bool        enable_frame_skip   = false;
    bool        output_this_frame   = true;
    bool        enable_reconnect    = true;
    int         reconnect_delay_ms  = 500;
    int         max_reconnects      = 0;
    int         open_timeout_ms     = 5000;
    int         read_timeout_ms     = 5000;
    int         buffer_size         = 4 * 1024 * 1024;
    int         max_delay_ms        = 200;
    int         reorder_queue_size  = 0;
    int         reconnect_attempts  = 0;
    bool        is_streaming_source = false;
    double      last_input_pts      = -1.0;
    double      last_output_pts     = -1.0;
    double      nominal_frame_delta = 0.0;
    double      pts_offset          = 0.0;
};
