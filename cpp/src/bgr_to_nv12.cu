#include "bgr_to_nv12.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ int clamp_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

__global__ void bgr_to_nv12_2x2_kernel(
    const uint8_t* __restrict__ src,
    int                       srcStep,
    uint8_t* __restrict__     dstY,
    int                       dstYStep,
    uint8_t* __restrict__     dstUV,
    int                       dstUVStep,
    int                       width,
    int                       height) {
    int uv_x = blockIdx.x * blockDim.x + threadIdx.x;
    int uv_y = blockIdx.y * blockDim.y + threadIdx.y;

    int x0 = uv_x * 2;
    int y0 = uv_y * 2;
    if (x0 >= width || y0 >= height) return;

    auto load_bgr = [&](int x, int y, int& b, int& g, int& r) {
        int idx = y * srcStep + x * 3;
        b       = static_cast<int>(src[idx + 0]);
        g       = static_cast<int>(src[idx + 1]);
        r       = static_cast<int>(src[idx + 2]);
    };

    int b00, g00, r00;
    int b01, g01, r01;
    int b10, g10, r10;
    int b11, g11, r11;

    load_bgr(x0, y0, b00, g00, r00);
    load_bgr(min(x0 + 1, width - 1), y0, b01, g01, r01);
    load_bgr(x0, min(y0 + 1, height - 1), b10, g10, r10);
    load_bgr(min(x0 + 1, width - 1), min(y0 + 1, height - 1), b11, g11, r11);

    auto write_y = [&](int x, int y, int b, int g, int r) {
        int y_val = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        dstY[y * dstYStep + x] = static_cast<uint8_t>(clamp_u8(y_val));
    };

    write_y(x0, y0, b00, g00, r00);
    if (x0 + 1 < width) write_y(x0 + 1, y0, b01, g01, r01);
    if (y0 + 1 < height) write_y(x0, y0 + 1, b10, g10, r10);
    if (x0 + 1 < width && y0 + 1 < height) write_y(x0 + 1, y0 + 1, b11, g11, r11);

    int b_sum = b00 + b01 + b10 + b11;
    int g_sum = g00 + g01 + g10 + g11;
    int r_sum = r00 + r01 + r10 + r11;

    int b_avg = (b_sum + 2) >> 2;
    int g_avg = (g_sum + 2) >> 2;
    int r_avg = (r_sum + 2) >> 2;

    int u_val = ((-38 * r_avg - 74 * g_avg + 112 * b_avg + 128) >> 8) + 128;
    int v_val = ((112 * r_avg - 94 * g_avg - 18 * b_avg + 128) >> 8) + 128;

    int uv_idx        = uv_y * dstUVStep + uv_x * 2;
    dstUV[uv_idx + 0] = static_cast<uint8_t>(clamp_u8(u_val));
    dstUV[uv_idx + 1] = static_cast<uint8_t>(clamp_u8(v_val));
}

void bgr_to_nv12(
    const uint8_t* src,
    int            srcStep,
    uint8_t*       dstY,
    int            dstYStep,
    uint8_t*       dstUV,
    int            dstUVStep,
    int            width,
    int            height,
    void*          stream) {
    dim3 block(32, 8);
    dim3 grid((width / 2 + block.x - 1) / block.x, (height / 2 + block.y - 1) / block.y);

    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    bgr_to_nv12_2x2_kernel<<<grid, block, 0, s>>>(src, srcStep, dstY, dstYStep, dstUV, dstUVStep, width, height);
}

