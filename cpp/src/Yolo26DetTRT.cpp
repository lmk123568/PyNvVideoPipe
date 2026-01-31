#include "Yolo26DetTRT.h"

#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "preprocess.h"

namespace {

torch::ScalarType trt_dtype_to_torch(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT:
            return torch::kFloat32;
        case nvinfer1::DataType::kHALF:
            return torch::kFloat16;
        case nvinfer1::DataType::kINT8:
            return torch::kInt8;
        case nvinfer1::DataType::kINT32:
            return torch::kInt32;
        case nvinfer1::DataType::kBOOL:
            return torch::kBool;
        default:
            throw std::runtime_error("Unsupported TensorRT dtype");
    }
}

std::vector<int64_t> dims_to_sizes(const nvinfer1::Dims& d) {
    std::vector<int64_t> out;
    out.reserve(d.nbDims);
    for (int i = 0; i < d.nbDims; ++i) out.push_back(static_cast<int64_t>(d.d[i]));
    return out;
}

bool looks_like_json(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) return false;
    if (s[i] != '{' && s[i] != '[') return false;
    for (char c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc < 0x09) return false;
        if (uc >= 0x0e && uc < 0x20) return false;
    }
    return true;
}

uint32_t read_u32_le(const char* p) {
    return (static_cast<uint32_t>(static_cast<unsigned char>(p[0])) |
            (static_cast<uint32_t>(static_cast<unsigned char>(p[1])) << 8) |
            (static_cast<uint32_t>(static_cast<unsigned char>(p[2])) << 16) |
            (static_cast<uint32_t>(static_cast<unsigned char>(p[3])) << 24));
}

}  // namespace

void Yolo26DetTRT::TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity == Severity::kVERBOSE) return;
    if (severity == Severity::kINFO) return;
    std::cerr << "[TensorRT] " << msg << std::endl;
}

Yolo26DetTRT::Yolo26DetTRT(std::string engine_path, float conf_thres, int device_id)
    : conf_thres_(conf_thres) {
    set_device(device_id);
    initLibNvInferPlugins(&logger_, "");
    load_engine(engine_path);
    allocate_io();
}

void Yolo26DetTRT::set_device(int device_id) {
    c10::cuda::CUDAGuard guard(c10::Device(c10::kCUDA, device_id));
}

void Yolo26DetTRT::load_engine(const std::string& engine_path) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open engine file: " + engine_path);
    f.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    std::vector<char> buf(size);
    f.read(buf.data(), static_cast<std::streamsize>(size));
    if (!f) throw std::runtime_error("Failed to read engine file: " + engine_path);

    const char* engine_data = buf.data();
    size_t      engine_size = buf.size();

    if (engine_size > 8) {
        uint32_t meta_len = read_u32_le(buf.data());
        if (meta_len > 0 && meta_len < (engine_size - 4) && meta_len < (1u << 20)) {
            std::string meta(buf.data() + 4, buf.data() + 4 + meta_len);
            if (looks_like_json(meta)) {
                engine_data = buf.data() + 4 + meta_len;
                engine_size = buf.size() - 4 - meta_len;
            }
        }
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

    engine_.reset(runtime_->deserializeCudaEngine(engine_data, engine_size));
    if (!engine_) {
        throw std::runtime_error(
            std::string("Failed to deserialize TensorRT engine. ")
            + "This usually means the engine was built with a different TensorRT version. "
            + "Build TensorRT version: " + std::to_string(NV_TENSORRT_MAJOR) + "."
            + std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH));
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create TensorRT execution context");
}

void Yolo26DetTRT::allocate_io() {
    int nb = engine_->getNbIOTensors();
    input_name_.clear();
    output_names_.clear();
    output_tensors_.clear();
    input_tensor_ = torch::Tensor();

    int device_id = 0;
    cudaGetDevice(&device_id);
    auto device = torch::Device(torch::kCUDA, device_id);

    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto        mode = engine_->getTensorIOMode(name);
        auto        dt   = engine_->getTensorDataType(name);
        auto        st   = trt_dtype_to_torch(dt);

        nvinfer1::Dims dims        = engine_->getTensorShape(name);
        bool           has_dynamic = false;
        for (int k = 0; k < dims.nbDims; ++k) {
            if (dims.d[k] < 0) has_dynamic = true;
        }
        if (has_dynamic && mode == nvinfer1::TensorIOMode::kINPUT) {
            if (dims.nbDims == 4) {
                dims.d[0] = 1;
                dims.d[1] = 3;
                dims.d[2] = height_;
                dims.d[3] = width_;
                if (!context_->setInputShape(name, dims)) throw std::runtime_error("Failed to set input shape");
            }
        }

        nvinfer1::Dims       runtime_dims = context_->getTensorShape(name);
        std::vector<int64_t> sizes        = dims_to_sizes(runtime_dims);
        for (auto& v : sizes) {
            if (v < 0) throw std::runtime_error("Tensor has unresolved dynamic shape: " + std::string(name));
        }

        auto options = torch::TensorOptions().device(device).dtype(st).layout(torch::kStrided);
        auto t       = torch::empty(sizes, options);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (!input_name_.empty()) throw std::runtime_error("Only single-input engines are supported");
            input_name_   = name;
            input_tensor_ = t;
            if (t.dim() == 4) {
                height_ = static_cast<int>(t.size(2));
                width_  = static_cast<int>(t.size(3));
            }
        } else {
            output_names_.push_back(name);
            output_tensors_.push_back(t);
        }
    }

    if (input_name_.empty()) throw std::runtime_error("Engine has no inputs");
    if (!input_tensor_.defined()) throw std::runtime_error("Failed to allocate input tensor");

    if (output_tensors_.empty()) throw std::runtime_error("Engine has no outputs");
}

torch::Tensor Yolo26DetTRT::infer(torch::Tensor image_hwc_u8) {
    TORCH_CHECK(image_hwc_u8.is_cuda(), "image must be CUDA tensor");
    TORCH_CHECK(image_hwc_u8.dtype() == torch::kUInt8, "image must be uint8");
    TORCH_CHECK(image_hwc_u8.dim() == 3, "image must be HWC");
    TORCH_CHECK(image_hwc_u8.size(0) == height_ && image_hwc_u8.size(1) == width_ && image_hwc_u8.size(2) == 3,
                "image shape must be (", height_, ", ", width_, ", 3)");
    TORCH_CHECK(image_hwc_u8.is_contiguous(), "image must be contiguous");

    c10::cuda::CUDAGuard guard(image_hwc_u8.device());

    preprocess_bgr_u8_hwc_to_rgb_nchw(image_hwc_u8, input_tensor_);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    context_->setTensorAddress(input_name_.c_str(), input_tensor_.data_ptr());
    for (size_t i = 0; i < output_names_.size(); ++i) {
        context_->setTensorAddress(output_names_[i].c_str(), output_tensors_[i].data_ptr());
    }

    if (!context_->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueueV3 failed");

    torch::Tensor pred = output_tensors_[0];
    if (pred.dim() == 3 && pred.size(0) == 1) pred = pred.squeeze(0);
    TORCH_CHECK(pred.dim() == 2, "pred must be 2D after squeeze");
    TORCH_CHECK(pred.size(1) >= 6, "pred last dim must be >= 6");

    using namespace at::indexing;

    auto mask     = pred.index({Slice(), 4}) > conf_thres_;
    auto filtered = pred.index({mask});

    if (filtered.numel() == 0) return filtered;

    filtered.index({Slice(), 0}).clamp_(0, static_cast<double>(width_));
    filtered.index({Slice(), 2}).clamp_(0, static_cast<double>(width_));
    filtered.index({Slice(), 1}).clamp_(0, static_cast<double>(height_));
    filtered.index({Slice(), 3}).clamp_(0, static_cast<double>(height_));

    return filtered;
}

std::vector<int64_t> Yolo26DetTRT::input_shape() const {
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(input_tensor_.dim()));
    for (int i = 0; i < input_tensor_.dim(); ++i) out.push_back(input_tensor_.size(i));
    return out;
}

std::vector<std::vector<int64_t>> Yolo26DetTRT::output_shapes() const {
    std::vector<std::vector<int64_t>> out;
    out.reserve(output_tensors_.size());
    for (const auto& t : output_tensors_) {
        std::vector<int64_t> s;
        s.reserve(static_cast<size_t>(t.dim()));
        for (int i = 0; i < t.dim(); ++i) s.push_back(t.size(i));
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<std::string> Yolo26DetTRT::input_names() const {
    if (input_name_.empty()) return {};
    return {input_name_};
}

std::vector<std::string> Yolo26DetTRT::output_names() const {
    return output_names_;
}
