#include "detection_model_inference_helper.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cub/cub.cuh>

#ifdef GENERATE_TXT
#include <fstream>
#endif

static constexpr int MAX_SLICES = 16; // max supported 

__constant__ int d_num_anchors;
__constant__ float d_confidence_threshold;
__constant__ int d_top_k;

__constant__ int d_slices_coordinates[MAX_SLICES * 4];
__constant__ int d_source_width;
__constant__ int d_source_height;
__constant__ int d_dest_width;
__constant__ int d_dest_height;



// src : full camera frame, uint8 BGR HWC  (1920x1080x3), zero-copy pinned
// dst : model input buffer, float32 RGB CHW (batch x 3 x 640 x 640)
// blockIdx.z = slice/batch index → coordinates read from d_slices_coordinates
__global__ void preprocessBGRToFloatCHW(const uint8_t *src, float *dst)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int b     = blockIdx.z;

    if (out_x >= d_dest_width || out_y >= d_dest_height) return;

    int x1 = d_slices_coordinates[b * 4 + 0];
    int y1 = d_slices_coordinates[b * 4 + 1];

    int src_idx      = ((y1 + out_y) * d_source_width + (x1 + out_x)) * 3;
    int plane        = d_dest_height * d_dest_width;
    int pixel        = out_y * d_dest_width + out_x;
    int batch_offset = b * 3 * plane;

    dst[batch_offset + 0 * plane + pixel] = (src[src_idx + 2] - 127.5f) / 128.0f; // R
    dst[batch_offset + 1 * plane + pixel] = (src[src_idx + 1] - 127.5f) / 128.0f; // G
    dst[batch_offset + 2 * plane + pixel] = (src[src_idx + 0] - 127.5f) / 128.0f; // B
}


void DetectionModelInferenceHelper::launchPreprocessKernel(const uint8_t *src, float *dst, int batch, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (model_input_width_  + block.x - 1) / block.x,
        (model_input_height_ + block.y - 1) / block.y,
        batch
    );
    preprocessBGRToFloatCHW<<<grid, block, 0, stream>>>(src, dst);
}


bool DetectionModelInferenceHelper::setDeviceSymbols(std::vector<int> slice_coordinates, int camera_width, int camera_height, int model_width, int model_height, int num_anchors, float confidence_threshold, int top_k)
{
    if (slice_coordinates.size() > MAX_SLICES * 4) {
        std::cerr << "[DetectionModelInferenceHelper] slice_coordinates size ("
                  << slice_coordinates.size() << ") exceeds maximum allowed ("
                  << MAX_SLICES * 4 << " = " << MAX_SLICES << " slices * 4 coords). "
                  << "Increase MAX_SLICES if more slices are needed.\n";
        return false;
    }
    slice_coordinates.resize(MAX_SLICES * 4, 0);
    cudaError_t err = cudaMemcpyToSymbol(d_slices_coordinates, slice_coordinates.data(),
                                         MAX_SLICES * 4 * sizeof(int));
    err = cudaMemcpyToSymbol(d_source_width,          &camera_width,         sizeof(int));
    err = cudaMemcpyToSymbol(d_source_height,         &camera_height,        sizeof(int));
    err = cudaMemcpyToSymbol(d_dest_width,            &model_width,          sizeof(int));
    err = cudaMemcpyToSymbol(d_dest_height,           &model_height,         sizeof(int));
    err = cudaMemcpyToSymbol(d_num_anchors,           &num_anchors,          sizeof(int));
    err = cudaMemcpyToSymbol(d_confidence_threshold,  &confidence_threshold, sizeof(float));
    err = cudaMemcpyToSymbol(d_top_k,                 &top_k,                sizeof(int));
    return err == cudaSuccess;
}

