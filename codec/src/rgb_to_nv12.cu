#include "rgb_to_nv12.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void rgb_to_nv12_kernel(const uint8_t* src, int srcStep, uint8_t* dstY, int dstYStep, uint8_t* dstUV, int dstUVStep, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // RGB is interleaved: R G B R G B ...
    // srcStep is in bytes (width * 3 usually, but could be padded)
    int srcIdx = y * srcStep + x * 3;
    
    uint8_t r = src[srcIdx];
    uint8_t g = src[srcIdx + 1];
    uint8_t b = src[srcIdx + 2];

    // BT.601 limited range
    // Y = ( (  66 * R + 129 * G +  25 * B + 128) >> 8) + 16
    int y_val =  (( 66 * r + 129 * g +  25 * b + 128) >> 8) + 16;
    
    // Y plane
    dstY[y * dstYStep + x] = (uint8_t)y_val;

    // UV plane (2x2 subsampling)
    // We compute UV only for even coordinates
    if (y % 2 == 0 && x % 2 == 0) {
        // Average 2x2 block? Or just subsample. 
        // Simple subsampling (taking top-left pixel) is faster but lower quality.
        // Let's just take the current pixel for simplicity first.
        // Better: average 4 pixels if possible, but that requires reading neighbors.
        // For now, simple subsampling.
        
        int u_val = ((-38 * r -  74 * g + 112 * b + 128) >> 8) + 128;
        int v_val = ((112 * r -  94 * g -  18 * b + 128) >> 8) + 128;

        // UV is interleaved: U V U V ...
        // Destination UV width is width/2 (in pixels), but each pixel is 2 bytes (U+V).
        // So row stride is roughly same as Y stride for NV12 if width is same? 
        // Wait, NV12 UV plane width is width bytes (same as Y) because U+V per 2 pixels.
        // x / 2 is the UV pixel index.
        // x / 2 * 2 is the byte index.
        int uvIdx = (y / 2) * dstUVStep + (x / 2) * 2;
        dstUV[uvIdx]     = (uint8_t)u_val;
        dstUV[uvIdx + 1] = (uint8_t)v_val;
    }
}

void rgb_to_nv12(const uint8_t* src,
                 int            srcStep,
                 uint8_t*       dstY,
                 int            dstYStep,
                 uint8_t*       dstUV,
                 int            dstUVStep,
                 int            width,
                 int            height,
                 void*          stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    rgb_to_nv12_kernel<<<grid, block, 0, s>>>(src, srcStep, dstY, dstYStep, dstUV, dstUVStep, width, height);
    // cudaDeviceSynchronize(); // Optional, but safer to debug
}
