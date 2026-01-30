#include <torch/extension.h>

#include "src/Decoder.h"
#include "src/Encoder.h"

#define MODULE_NAME nv_accel

PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<Decoder>(m, "Decoder")
        .def(py::init([](const std::string& filename,
                         bool               enable_frame_skip,
                         int                output_width,
                         int                output_height,
                         bool               enable_auto_reconnect,
                         int                reconnect_delay_ms,
                         int                max_reconnects,
                         int                open_timeout_ms,
                         int                read_timeout_ms,
                         int                buffer_size,
                         int                max_delay_ms,
                         int                reorder_queue_size,
                         int                decoder_threads,
                         int                surfaces,
                         const std::string& hwaccel) {
                 return std::make_unique<Decoder>(filename,
                                                  enable_frame_skip,
                                                  output_width,
                                                  output_height,
                                                  enable_auto_reconnect,
                                                  reconnect_delay_ms,
                                                  max_reconnects,
                                                  open_timeout_ms,
                                                  read_timeout_ms,
                                                  buffer_size,
                                                  max_delay_ms,
                                                  reorder_queue_size,
                                                  decoder_threads,
                                                  surfaces,
                                                  hwaccel);
             }),
             py::arg("filename"),
             py::arg("enable_frame_skip")     = false,
             py::arg("output_width")          = 0,
             py::arg("output_height")         = 0,
             py::arg("enable_auto_reconnect") = true,
             py::arg("reconnect_delay_ms")    = 10000,
             py::arg("max_reconnects")        = 0,
             py::arg("open_timeout_ms")       = 5000,
             py::arg("read_timeout_ms")       = 5000,
             py::arg("buffer_size")           = 4 * 1024 * 1024,
             py::arg("max_delay_ms")          = 200,
             py::arg("reorder_queue_size")    = 0,
             py::arg("decoder_threads")       = 2,
             py::arg("surfaces")              = 2,
             py::arg("hwaccel")               = "cuda",
             py::call_guard<py::gil_scoped_release>())
        .def("next_frame", &Decoder::next_frame, py::call_guard<py::gil_scoped_release>())
        .def("get_width", &Decoder::get_width)
        .def("get_height", &Decoder::get_height)
        .def("get_fps", &Decoder::get_fps);

    py::class_<Encoder>(m, "Encoder")
        .def(py::init([](const std::string& output_url, int width, int height, float fps, std::string codec, int bitrate) {
                 return std::make_unique<Encoder>(output_url, width, height, (int)fps, codec, bitrate);
             }),
             py::arg("output_url"),
             py::arg("width"),
             py::arg("height"),
             py::arg("fps"),
             py::arg("codec")   = "h264",
             py::arg("bitrate") = 2000000)
        .def("encode", &Encoder::encode, py::arg("frame"), py::arg("pts") = -1.0, py::call_guard<py::gil_scoped_release>())
        .def("finish", &Encoder::finish, py::call_guard<py::gil_scoped_release>());
}
