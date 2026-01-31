#pragma once

#include <NvInfer.h>
#include <torch/extension.h>

#include <memory>
#include <string>
#include <vector>

class Yolo26DetTRT {
public:
    Yolo26DetTRT(std::string engine_path, float conf_thres = 0.25f, int device_id = 0);

    torch::Tensor infer(torch::Tensor image_hwc_u8);

    std::vector<int64_t> input_shape() const;
    std::vector<std::vector<int64_t>> output_shapes() const;
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;

private:
    void load_engine(const std::string& engine_path);
    void allocate_io();
    void set_device(int device_id);

    float conf_thres_ = 0.25f;
    int width_        = 1024;
    int height_       = 576;

    struct TrtLogger final : nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    struct TrtDeleter {
        template <typename T>
        void operator()(T* p) const {
            delete p;
        }
    };

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<torch::Tensor> output_tensors_;
    torch::Tensor input_tensor_;
};
