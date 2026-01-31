#include "preprocess.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

template <typename OutT>
__global__ void bgr_u8_hwc_to_rgb_nchw_norm_kernel(
    const uint8_t* __restrict__ input,
    int                      height,
    int                      width,
    int64_t                  in_row_stride,
    OutT* __restrict__       output,
    int64_t                  out_c_stride,
    int64_t                  out_row_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int64_t in_base = y * in_row_stride + x * 3;
    uint8_t b       = input[in_base + 0];
    uint8_t g       = input[in_base + 1];
    uint8_t r       = input[in_base + 2];

    float rf = static_cast<float>(r) * (1.0f / 255.0f);
    float gf = static_cast<float>(g) * (1.0f / 255.0f);
    float bf = static_cast<float>(b) * (1.0f / 255.0f);

    int64_t out_base = y * out_row_stride + x;
    output[out_base + 0 * out_c_stride] = static_cast<OutT>(rf);
    output[out_base + 1 * out_c_stride] = static_cast<OutT>(gf);
    output[out_base + 2 * out_c_stride] = static_cast<OutT>(bf);
}

template <>
__global__ void bgr_u8_hwc_to_rgb_nchw_norm_kernel<at::Half>(
    const uint8_t* __restrict__ input,
    int                      height,
    int                      width,
    int64_t                  in_row_stride,
    at::Half* __restrict__   output,
    int64_t                  out_c_stride,
    int64_t                  out_row_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int64_t in_base = y * in_row_stride + x * 3;
    uint8_t b       = input[in_base + 0];
    uint8_t g       = input[in_base + 1];
    uint8_t r       = input[in_base + 2];

    __half rf = __float2half(static_cast<float>(r) * (1.0f / 255.0f));
    __half gf = __float2half(static_cast<float>(g) * (1.0f / 255.0f));
    __half bf = __float2half(static_cast<float>(b) * (1.0f / 255.0f));

    int64_t out_base = y * out_row_stride + x;
    reinterpret_cast<__half*>(output)[out_base + 0 * out_c_stride] = rf;
    reinterpret_cast<__half*>(output)[out_base + 1 * out_c_stride] = gf;
    reinterpret_cast<__half*>(output)[out_base + 2 * out_c_stride] = bf;
}

}  // namespace

void preprocess_bgr_u8_hwc_to_rgb_nchw(torch::Tensor input_hwc_u8, torch::Tensor output_nchw) {
    TORCH_CHECK(input_hwc_u8.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output_nchw.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input_hwc_u8.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(input_hwc_u8.dim() == 3, "input must be HWC");
    TORCH_CHECK(input_hwc_u8.size(2) == 3, "input last dim must be 3");
    TORCH_CHECK(output_nchw.dim() == 4, "output must be NCHW");
    TORCH_CHECK(output_nchw.size(0) == 1, "output batch must be 1");
    TORCH_CHECK(output_nchw.size(1) == 3, "output channels must be 3");
    TORCH_CHECK(output_nchw.size(2) == input_hwc_u8.size(0) && output_nchw.size(3) == input_hwc_u8.size(1),
                "output spatial size mismatch");
    TORCH_CHECK(input_hwc_u8.is_contiguous(), "input must be contiguous HWC");
    TORCH_CHECK(output_nchw.is_contiguous(), "output must be contiguous NCHW");

    int height = static_cast<int>(input_hwc_u8.size(0));
    int width  = static_cast<int>(input_hwc_u8.size(1));

    const uint8_t* input_ptr = input_hwc_u8.data_ptr<uint8_t>();

    int64_t in_row_stride  = input_hwc_u8.stride(0);
    int64_t out_row_stride = output_nchw.stride(2);
    int64_t out_c_stride   = output_nchw.stride(1);

    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (output_nchw.dtype() == torch::kFloat16) {
        bgr_u8_hwc_to_rgb_nchw_norm_kernel<at::Half><<<grid, block, 0, stream>>>(
            input_ptr,
            height,
            width,
            in_row_stride,
            output_nchw.data_ptr<at::Half>(),
            out_c_stride,
            out_row_stride);
    } else if (output_nchw.dtype() == torch::kFloat32) {
        bgr_u8_hwc_to_rgb_nchw_norm_kernel<float><<<grid, block, 0, stream>>>(
            input_ptr,
            height,
            width,
            in_row_stride,
            output_nchw.data_ptr<float>(),
            out_c_stride,
            out_row_stride);
    } else {
        TORCH_CHECK(false, "output dtype must be float16 or float32");
    }
}
