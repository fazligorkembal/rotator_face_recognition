#include "detection_model_inference_helper.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cub/cub.cuh>

#ifdef GENERATE_TXT
#include <fstream>
#endif

__constant__ int d_slices_coordinates[24]; // 6 slices * 4 (x1,y1,x2,y2)

// src : full camera frame, uint8 BGR HWC  (1920x1080x3), zero-copy pinned
// dst : model input buffer, float32 RGB CHW (batch x 3 x 640 x 640)
// blockIdx.z = slice/batch index → coordinates read from d_slices_coordinates
__global__ void preprocessBGRToFloatCHW(
    const uint8_t *src, float *dst,
    int src_width,
    int dst_width, int dst_height)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int b     = blockIdx.z;

    if (out_x >= dst_width || out_y >= dst_height) return;

    int x1 = d_slices_coordinates[b * 4 + 0];
    int y1 = d_slices_coordinates[b * 4 + 1];

    int src_x = x1 + out_x;
    int src_y = y1 + out_y;

    int src_idx      = (src_y * src_width + src_x) * 3;
    int plane        = dst_height * dst_width;
    int pixel        = out_y * dst_width + out_x;
    int batch_offset = b * 3 * plane;

    dst[batch_offset + 0 * plane + pixel] = src[src_idx + 2] / 255.0f; // R
    dst[batch_offset + 1 * plane + pixel] = src[src_idx + 1] / 255.0f; // G
    dst[batch_offset + 2 * plane + pixel] = src[src_idx + 0] / 255.0f; // B
}

void DetectionModelInferenceHelper::launchPreprocessKernel(const uint8_t *src, float *dst, int batch, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (model_input_width_  + block.x - 1) / block.x,
        (model_input_height_ + block.y - 1) / block.y,
        batch
    );
    preprocessBGRToFloatCHW<<<grid, block, 0, stream>>>(
        src, dst,
        camera_input_width_,
        model_input_width_, model_input_height_
    );
}

bool DetectionModelInferenceHelper::setDeviceSymbols(std::vector<int> slice_coordinates)
{
    if (slice_coordinates.size() > 24) {
        return false;
    }
    cudaError_t err = cudaMemcpyToSymbol(d_slices_coordinates, slice_coordinates.data(),
                                         slice_coordinates.size() * sizeof(int));
    return err == cudaSuccess;
}