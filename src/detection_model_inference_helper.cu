#include "detection_model_inference_helper.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cub/cub.cuh>

#ifdef GENERATE_TXT
#include <fstream>
#endif

__constant__ int d_num_anchors;
__constant__ float d_confidence_threshold;
__constant__ int d_top_k;
__constant__ int d_batch_sizes[3]; // min, opt, max
__constant__ int d_input_size;
__constant__ float d_iou_threshold;

__constant__ float2 centroid_destination;
__constant__ float2 normalized_destination[5];

__constant__ int src_w;
__constant__ int src_h;
__constant__ int device_dst_w;
__constant__ int device_dst_h;

// Converts uint8 BGR HWC → float32 RGB CHW, normalized to [0, 1]
__global__ void preprocessBGRToFloatCHW(const uint8_t *src, float *dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;
    int pixel   = y * width + x;
    int plane   = height * width;

    dst[0 * plane + pixel] = src[src_idx + 2] / 255.0f; // R
    dst[1 * plane + pixel] = src[src_idx + 1] / 255.0f; // G
    dst[2 * plane + pixel] = src[src_idx + 0] / 255.0f; // B
}

void DetectionModelInferenceHelper::launchPreprocessKernel(const uint8_t *src, float *dst, int batch, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((model_input_width_ + block.x - 1) / block.x,
              (model_input_height_ + block.y - 1) / block.y);

    for (int b = 0; b < batch; ++b) {
        const uint8_t *src_b = src + b * model_input_height_ * model_input_width_ * 3;
        float         *dst_b = dst + b * 3 * model_input_height_ * model_input_width_;
        preprocessBGRToFloatCHW<<<grid, block, 0, stream>>>(src_b, dst_b, model_input_width_, model_input_height_);
    }
}

