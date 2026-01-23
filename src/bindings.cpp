#include <torch/extension.h>

#include "Decoder.h"
#include "Encoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Decoder>(m, "Decoder")
        .def(py::init<const std::string&, bool, int, int, bool, int, int, int, int, int, int, int>(),
             py::arg("filename"),
             py::arg("enable_frame_skip")  = false,
             py::arg("output_width")       = 0,
             py::arg("output_height")      = 0,
             py::arg("enable_auto_reconnect") = true,
             py::arg("reconnect_delay_ms") = 10000,
             py::arg("max_reconnects")     = 0,
             py::arg("open_timeout_ms")    = 5000,
             py::arg("read_timeout_ms")    = 5000,
             py::arg("buffer_size")        = 4 * 1024 * 1024,
             py::arg("max_delay_ms")       = 200,
             py::arg("reorder_queue_size") = 0)
        .def("next_frame", &Decoder::next_frame)
        .def("get_width", &Decoder::get_width)
        .def("get_height", &Decoder::get_height)
        .def("get_fps", &Decoder::get_fps);

    py::class_<VideoEncoder>(m, "Encoder")
        .def(py::init([](const std::string& output_url, int width, int height, float fps, std::string codec, int bitrate) {
                 return std::make_unique<VideoEncoder>(output_url, width, height, (int)fps, bitrate);
             }),
             py::arg("output_url"),
             py::arg("width"),
             py::arg("height"),
             py::arg("fps"),
             py::arg("codec")   = "h264",
             py::arg("bitrate") = 2000000)
        .def("encode", &VideoEncoder::encode, py::arg("frame"), py::arg("pts") = -1.0)
        .def("finish", &VideoEncoder::finish);
}
