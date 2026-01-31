#pragma once

#include <torch/extension.h>

void preprocess_bgr_u8_hwc_to_rgb_nchw(torch::Tensor input_hwc_u8, torch::Tensor output_nchw);
