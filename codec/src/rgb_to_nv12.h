#pragma once
#include <cstdint>

void rgb_to_nv12(
    const uint8_t* src,
    int            srcStep,
    uint8_t*       dstY,
    int            dstYStep,
    uint8_t*       dstUV,
    int            dstUVStep,
    int            width,
    int            height,
    void*          stream = nullptr);
