#include "detection_model_inference_helper.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cub/cub.cuh>

#ifdef GENERATE_TXT
#include <fstream>
#endif

static constexpr int MAX_SLICES = 16; // max supported 
static constexpr int kAlignedFaceWidth = DetectionModelInferenceHelper::kAlignedFaceWidth;
static constexpr int kAlignedFaceHeight = DetectionModelInferenceHelper::kAlignedFaceHeight;
static constexpr int kAlignedFaceChannels = DetectionModelInferenceHelper::kAlignedFaceChannels;
static constexpr int kLandmarkCount = DetectionModelInferenceHelper::kLandmarkCount;
static constexpr int kAffineMatrixElements = DetectionModelInferenceHelper::kAffineMatrixElements;

__constant__ int d_num_anchors;
__constant__ float d_confidence_threshold;
__constant__ float d_iou_threshold;
__constant__ int d_top_k;

__constant__ int d_slices_coordinates[MAX_SLICES * 4];
__constant__ int d_source_width;
__constant__ int d_source_height;
__constant__ int d_dest_width;
__constant__ int d_dest_height;
__constant__ float2 d_arcface_centroid_destination;
__constant__ float2 d_arcface_normalized_destination[kLandmarkCount];

__device__ void getCentroid(
    const float2 *landmarks,
    float2 *centroid)
{
    centroid->x = 0.0f;
    centroid->y = 0.0f;
#pragma unroll
    for (int i = 0; i < kLandmarkCount; ++i)
    {
        centroid->x += landmarks[i].x;
        centroid->y += landmarks[i].y;
    }
    centroid->x *= 1.0f / static_cast<float>(kLandmarkCount);
    centroid->y *= 1.0f / static_cast<float>(kLandmarkCount);
}

__device__ void normalizeLandmarks(
    const float2 *landmarks,
    const float2 *centroid,
    float2 *normalized_landmarks)
{
#pragma unroll
    for (int i = 0; i < kLandmarkCount; ++i)
    {
        normalized_landmarks[i].x = landmarks[i].x - centroid->x;
        normalized_landmarks[i].y = landmarks[i].y - centroid->y;
    }
}

__device__ float4 distance_to_bbox(
    float2 anchor_center,
    float4 distance,
    int stride)
{
    float4 bbox;
    bbox.x = fmaxf(anchor_center.x - distance.x * stride, 0.0f);
    bbox.y = fmaxf(anchor_center.y - distance.y * stride, 0.0f);
    bbox.z = fminf(anchor_center.x + distance.z * stride, static_cast<float>(d_dest_width));
    bbox.w = fminf(anchor_center.y + distance.w * stride, static_cast<float>(d_dest_height));
    return bbox;
}

__device__ float2 distance_to_landmark(
    float2 anchor_center,
    float2 distance,
    int stride)
{
    return make_float2(anchor_center.x + distance.x * stride,
                       anchor_center.y + distance.y * stride);
}

__device__ float computeIOU(float4 box1, float4 box2)
{
    const float inter_x1 = fmaxf(box1.x, box2.x);
    const float inter_y1 = fmaxf(box1.y, box2.y);
    const float inter_x2 = fminf(box1.z, box2.z);
    const float inter_y2 = fminf(box1.w, box2.w);

    const float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    const float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float area1 = fmaxf(0.0f, box1.z - box1.x) * fmaxf(0.0f, box1.w - box1.y);
    const float area2 = fmaxf(0.0f, box2.z - box2.x) * fmaxf(0.0f, box2.w - box2.y);
    return inter_area / (area1 + area2 - inter_area + 1e-6f);
}

__device__ float loadBatchMajorScore(
    const float *scores_s8,
    const float *scores_s16,
    const float *scores_s32,
    int batch,
    int global_anchor_idx,
    int scores_s8_count,
    int scores_s16_count,
    int scores_s32_count)
{
    if (global_anchor_idx < scores_s8_count)
    {
        return scores_s8[batch * scores_s8_count + global_anchor_idx];
    }

    int local_idx = global_anchor_idx - scores_s8_count;
    if (local_idx < scores_s16_count)
    {
        return scores_s16[batch * scores_s16_count + local_idx];
    }

    local_idx -= scores_s16_count;
    if (local_idx < scores_s32_count)
    {
        return scores_s32[batch * scores_s32_count + local_idx];
    }

    return 0.0f;
}

__device__ void decodeGlobalAnchorIndex(
    int global_anchor_idx,
    int &local_anchor_idx,
    int &stride,
    int &feat_w)
{
    if (global_anchor_idx < 12800)
    {
        local_anchor_idx = global_anchor_idx;
        stride = 8;
        feat_w = d_dest_width / stride;
        return;
    }

    if (global_anchor_idx < 16000)
    {
        local_anchor_idx = global_anchor_idx - 12800;
        stride = 16;
        feat_w = d_dest_width / stride;
        return;
    }

    local_anchor_idx = global_anchor_idx - 16000;
    stride = 32;
    feat_w = d_dest_width / stride;
}

__device__ float4 loadBatchMajorBbox(
    const float *bboxes_s8,
    const float *bboxes_s16,
    const float *bboxes_s32,
    int batch,
    int global_anchor_idx,
    int scores_s8_count,
    int scores_s16_count,
    int scores_s32_count)
{
    const int bbox_channels = 4;
    const int base_anchor =
        global_anchor_idx < scores_s8_count ? global_anchor_idx :
        (global_anchor_idx < scores_s8_count + scores_s16_count ? global_anchor_idx - scores_s8_count :
         global_anchor_idx - scores_s8_count - scores_s16_count);

    if (global_anchor_idx < scores_s8_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s8_count + base_anchor) * bbox_channels;
        return make_float4(bboxes_s8[base + 0], bboxes_s8[base + 1], bboxes_s8[base + 2], bboxes_s8[base + 3]);
    }

    if (global_anchor_idx < scores_s8_count + scores_s16_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s16_count + base_anchor) * bbox_channels;
        return make_float4(bboxes_s16[base + 0], bboxes_s16[base + 1], bboxes_s16[base + 2], bboxes_s16[base + 3]);
    }

    const size_t base = (static_cast<size_t>(batch) * scores_s32_count + base_anchor) * bbox_channels;
    return make_float4(bboxes_s32[base + 0], bboxes_s32[base + 1], bboxes_s32[base + 2], bboxes_s32[base + 3]);
}

__device__ float2 loadBatchMajorLandmark(
    const float *landmarks_s8,
    const float *landmarks_s16,
    const float *landmarks_s32,
    int batch,
    int global_anchor_idx,
    int landmark_point_idx,
    int scores_s8_count,
    int scores_s16_count,
    int scores_s32_count)
{
    const int landmark_channels = 10;
    const int base_anchor =
        global_anchor_idx < scores_s8_count ? global_anchor_idx :
        (global_anchor_idx < scores_s8_count + scores_s16_count ? global_anchor_idx - scores_s8_count :
         global_anchor_idx - scores_s8_count - scores_s16_count);
    const int channel = landmark_point_idx * 2;

    if (global_anchor_idx < scores_s8_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s8_count + base_anchor) * landmark_channels;
        return make_float2(landmarks_s8[base + channel + 0], landmarks_s8[base + channel + 1]);
    }

    if (global_anchor_idx < scores_s8_count + scores_s16_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s16_count + base_anchor) * landmark_channels;
        return make_float2(landmarks_s16[base + channel + 0], landmarks_s16[base + channel + 1]);
    }

    const size_t base = (static_cast<size_t>(batch) * scores_s32_count + base_anchor) * landmark_channels;
    return make_float2(landmarks_s32[base + channel + 0], landmarks_s32[base + channel + 1]);
}

__global__ void filterScoresFilterKernel(
    const float *scores_s8,
    const float *scores_s16,
    const float *scores_s32,
    int scores_s8_count,
    int scores_s16_count,
    int scores_s32_count,
    int batch_size,
    float *scores_filtered,
    int32_t *selected_indexes,
    int32_t *num_selected)
{
    const int global_anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    const int total_anchor_count = scores_s8_count + scores_s16_count + scores_s32_count;

    if (batch >= batch_size || global_anchor_idx >= total_anchor_count)
        return;

    const float score = loadBatchMajorScore(
        scores_s8,
        scores_s16,
        scores_s32,
        batch,
        global_anchor_idx,
        scores_s8_count,
        scores_s16_count,
        scores_s32_count);

    if (score < d_confidence_threshold)
        return;

    int selected_idx = atomicAdd(&num_selected[batch], 1);
    if (selected_idx >= d_top_k)
        return;

    const int output_idx = batch * d_top_k + selected_idx;
    selected_indexes[output_idx] = global_anchor_idx;
    scores_filtered[output_idx] = score;
}

__global__ void gatherAllKernel(
    const int32_t *device_indexes_sorted,
    const float *bboxes_s8,
    const float *bboxes_s16,
    const float *bboxes_s32,
    const float *landmarks_s8,
    const float *landmarks_s16,
    const float *landmarks_s32,
    int scores_s8_count,
    int scores_s16_count,
    int scores_s32_count,
    int batch_size,
    float4 *device_bboxes_sorted,
    float2 *device_landmarks_sorted,
    const int32_t *device_num_filtered)
{
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (batch >= batch_size)
        return;

    const int valid_count = min(device_num_filtered[batch], d_top_k);
    if (rank >= valid_count)
        return;

    const int out_idx = batch * d_top_k + rank;
    const int global_anchor_idx = device_indexes_sorted[out_idx];

    int local_anchor_idx;
    int stride;
    int feat_w;
    decodeGlobalAnchorIndex(global_anchor_idx, local_anchor_idx, stride, feat_w);

    const int cell = local_anchor_idx / 2;
    const int cell_x = cell % feat_w;
    const int cell_y = cell / feat_w;
    const float2 anchor_center = make_float2(static_cast<float>(cell_x * stride),
                                             static_cast<float>(cell_y * stride));

    const float4 bbox_distance = loadBatchMajorBbox(
        bboxes_s8,
        bboxes_s16,
        bboxes_s32,
        batch,
        global_anchor_idx,
        scores_s8_count,
        scores_s16_count,
        scores_s32_count);

    device_bboxes_sorted[out_idx] = distance_to_bbox(anchor_center, bbox_distance, stride);

    for (int i = 0; i < 5; ++i)
    {
        const float2 landmark_distance = loadBatchMajorLandmark(
            landmarks_s8,
            landmarks_s16,
            landmarks_s32,
            batch,
            global_anchor_idx,
            i,
            scores_s8_count,
            scores_s16_count,
            scores_s32_count);

        device_landmarks_sorted[out_idx * 5 + i] =
            distance_to_landmark(anchor_center, landmark_distance, stride);
    }
}

__global__ void bitmaskNMSKernel(
    const float4 *device_bboxes_sorted,
    uint32_t *device_suppression_mask,
    const int32_t *device_num_filtered,
    int batch_size)
{
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (batch >= batch_size)
        return;

    const int valid_count = min(device_num_filtered[batch], d_top_k);
    if (rank >= valid_count)
        return;

    const int mask_words_per_batch = (d_top_k + 31) / 32;
    const int batch_box_offset = batch * d_top_k;
    const int batch_mask_offset = batch * mask_words_per_batch;
    const float4 box_a = device_bboxes_sorted[batch_box_offset + rank];

    for (int j = rank + 1; j < valid_count; ++j)
    {
        const float4 box_b = device_bboxes_sorted[batch_box_offset + j];
        const float iou = computeIOU(box_a, box_b);
        if (iou > d_iou_threshold)
        {
            atomicOr(&device_suppression_mask[batch_mask_offset + j / 32], 1u << (j % 32));
        }
    }
}

__global__ void gatherFinalResultKernel(
    const int32_t *device_sorted_indexes,
    const float *device_sorted_scores,
    const float4 *device_sorted_bboxes,
    const float2 *device_sorted_landmarks,
    const uint32_t *device_suppression_mask,
    int32_t *device_final_indexes,
    float *device_final_scores,
    float4 *device_final_bboxes,
    float2 *device_final_landmarks,
    int32_t *device_final_count,
    const int32_t *device_num_filtered,
    int batch_size)
{
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (batch >= batch_size)
        return;

    const int valid_count = min(device_num_filtered[batch], d_top_k);
    if (rank >= valid_count)
        return;

    const int sorted_offset = batch * d_top_k + rank;
    const int mask_words_per_batch = (d_top_k + 31) / 32;
    const int batch_mask_offset = batch * mask_words_per_batch;

    if (device_sorted_scores[sorted_offset] < d_confidence_threshold)
        return;

    if (device_suppression_mask[batch_mask_offset + rank / 32] & (1u << (rank % 32)))
        return;

    const int out_rank = atomicAdd(&device_final_count[batch], 1);
    if (out_rank >= d_top_k)
        return;

    const int final_offset = batch * d_top_k + out_rank;
    device_final_indexes[final_offset] = device_sorted_indexes[sorted_offset];
    device_final_scores[final_offset] = device_sorted_scores[sorted_offset];
    device_final_bboxes[final_offset] = device_sorted_bboxes[sorted_offset];

    for (int i = 0; i < 5; ++i)
    {
        device_final_landmarks[final_offset * 5 + i] =
            device_sorted_landmarks[sorted_offset * 5 + i];
    }
}

__global__ void estimateSimilarityKernel(
    const float2 *device_final_landmarks,
    const int32_t *device_final_num_detections,
    float *device_face_affine_matrices,
    int batch_size)
{
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (batch >= batch_size)
        return;

    const int valid_count = min(device_final_num_detections[batch], d_top_k);
    if (rank >= valid_count)
        return;

    const int flat_face_index = batch * d_top_k + rank;
    const int slice_x1 = d_slices_coordinates[batch * 4 + 0];
    const int slice_y1 = d_slices_coordinates[batch * 4 + 1];

    float2 source_landmarks[kLandmarkCount];
    float2 centroid_src;
    float2 normalized_source[kLandmarkCount];

#pragma unroll
    for (int i = 0; i < kLandmarkCount; ++i)
    {
        const float2 landmark = device_final_landmarks[flat_face_index * kLandmarkCount + i];
        source_landmarks[i].x = landmark.x + static_cast<float>(slice_x1);
        source_landmarks[i].y = landmark.y + static_cast<float>(slice_y1);
    }

    getCentroid(source_landmarks, &centroid_src);
    normalizeLandmarks(source_landmarks, &centroid_src, normalized_source);

    float ss = 0.0f;
    float sd = 0.0f;
    float num = 0.0f;
    float den = 0.0f;

#pragma unroll
    for (int i = 0; i < kLandmarkCount; ++i)
    {
        ss += normalized_source[i].x * normalized_source[i].x +
              normalized_source[i].y * normalized_source[i].y;
        sd += d_arcface_normalized_destination[i].x * d_arcface_normalized_destination[i].x +
              d_arcface_normalized_destination[i].y * d_arcface_normalized_destination[i].y;
        num += normalized_source[i].x * d_arcface_normalized_destination[i].x +
               normalized_source[i].y * d_arcface_normalized_destination[i].y;
        den += normalized_source[i].x * d_arcface_normalized_destination[i].y -
               normalized_source[i].y * d_arcface_normalized_destination[i].x;
    }

    ss = fmaxf(ss, 1e-6f);

    const float scale = sqrtf(sd / ss);
    const float theta = atan2f(den, num);
    const float c = cosf(theta);
    const float s = sinf(theta);

    const float a = scale * c;
    const float b = -scale * s;
    const float d = scale * s;
    const float e = scale * c;

    const float tx = d_arcface_centroid_destination.x - (a * centroid_src.x + b * centroid_src.y);
    const float ty = d_arcface_centroid_destination.y - (d * centroid_src.x + e * centroid_src.y);

    float det = a * e - b * d;
    det = fabsf(det) < 1e-8f ? 1e-8f : det;

    const float ia = e / det;
    const float ib = -b / det;
    const float id = -d / det;
    const float ie = a / det;
    const float ic = -(ia * tx + ib * ty);
    const float if_ = -(id * tx + ie * ty);

    float *matrix = device_face_affine_matrices + flat_face_index * kAffineMatrixElements;
    matrix[0] = ia;
    matrix[1] = ib;
    matrix[2] = ic;
    matrix[3] = id;
    matrix[4] = ie;
    matrix[5] = if_;
}

__global__ void warpAffineKernel(
    const uint8_t *src,
    float *dst,
    const float *affine_matrices,
    const int32_t *device_final_num_detections,
    int batch_size)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int flat_face_index = blockIdx.z;

    const int total_capacity = batch_size * d_top_k;
    if (flat_face_index >= total_capacity || out_x >= kAlignedFaceWidth || out_y >= kAlignedFaceHeight)
        return;

    const int batch = flat_face_index / d_top_k;
    const int rank = flat_face_index - batch * d_top_k;
    const int valid_count = min(device_final_num_detections[batch], d_top_k);
    if (rank >= valid_count)
        return;

    const float *matrix = affine_matrices + flat_face_index * kAffineMatrixElements;
    const float src_x = matrix[0] * out_x + matrix[1] * out_y + matrix[2];
    const float src_y = matrix[3] * out_x + matrix[4] * out_y + matrix[5];

    const int face_plane = kAlignedFaceHeight * kAlignedFaceWidth;
    const size_t face_offset = static_cast<size_t>(flat_face_index) * kAlignedFaceChannels * face_plane;
    const size_t pixel_offset = static_cast<size_t>(out_y) * kAlignedFaceWidth + out_x;

    if (src_x < 0.0f || src_x >= static_cast<float>(d_source_width - 1) ||
        src_y < 0.0f || src_y >= static_cast<float>(d_source_height - 1))
    {
        dst[face_offset + 0 * face_plane + pixel_offset] = -0.99609375f;
        dst[face_offset + 1 * face_plane + pixel_offset] = -0.99609375f;
        dst[face_offset + 2 * face_plane + pixel_offset] = -0.99609375f;
        return;
    }

    const int x0 = static_cast<int>(floorf(src_x));
    const int x1 = x0 + 1;
    const int y0 = static_cast<int>(floorf(src_y));
    const int y1 = y0 + 1;

    const float dx = src_x - static_cast<float>(x0);
    const float dy = src_y - static_cast<float>(y0);

    const int s00 = (y0 * d_source_width + x0) * 3;
    const int s01 = (y0 * d_source_width + x1) * 3;
    const int s10 = (y1 * d_source_width + x0) * 3;
    const int s11 = (y1 * d_source_width + x1) * 3;

#pragma unroll
    for (int c = 0; c < kAlignedFaceChannels; ++c)
    {
        const float value =
            src[s00 + c] * (1.0f - dx) * (1.0f - dy) +
            src[s01 + c] * dx * (1.0f - dy) +
            src[s10 + c] * (1.0f - dx) * dy +
            src[s11 + c] * dx * dy;

        const float normalized = (value - 127.5f) / 128.0f;
        const int channel_idx = (c == 0) ? 2 : (c == 2 ? 0 : 1);
        dst[face_offset + channel_idx * face_plane + pixel_offset] = normalized;
    }
}


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

    dst[batch_offset + 0 * plane + pixel] = (src[src_idx + 2] - 127.5f) / 128.0f; // R (src+2)
    dst[batch_offset + 1 * plane + pixel] = (src[src_idx + 1] - 127.5f) / 128.0f; // G
    dst[batch_offset + 2 * plane + pixel] = (src[src_idx + 0] - 127.5f) / 128.0f; // B (src+0)
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

void DetectionModelInferenceHelper::launchFilterScoresKernel(int batch, cudaStream_t stream) {
    const int scores_s8_count = static_cast<int>(output_elements_per_batch_[0]);
    const int scores_s16_count = static_cast<int>(output_elements_per_batch_[1]);
    const int scores_s32_count = static_cast<int>(output_elements_per_batch_[2]);
    const int total_anchor_count = scores_s8_count + scores_s16_count + scores_s32_count;

    dim3 block(256, 1, 1);
    dim3 grid((total_anchor_count + block.x - 1) / block.x, batch, 1);

    filterScoresFilterKernel<<<grid, block, 0, stream>>>(
        device_model_output_buffers_[0],
        device_model_output_buffers_[1],
        device_model_output_buffers_[2],
        scores_s8_count,
        scores_s16_count,
        scores_s32_count,
        batch,
        device_filtered_scores_,
        device_filtered_indexes_,
        device_num_selected_);
}

void DetectionModelInferenceHelper::ensureSortStorageInitialized() {
    if (device_sort_storage_) {
        return;
    }

    cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        sort_storage_bytes_,
        device_filtered_scores_,
        device_sorted_scores_,
        device_filtered_indexes_,
        device_sorted_indexes_,
        top_k_,
        0,
        sizeof(float) * 8,
        stream_);
    cudaMalloc(&device_sort_storage_, sort_storage_bytes_);
}

void DetectionModelInferenceHelper::launchSortFilteredScoresKernel(int batch, cudaStream_t stream) {
    ensureSortStorageInitialized();

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        const size_t offset = static_cast<size_t>(batch_index) * static_cast<size_t>(top_k_);
        cub::DeviceRadixSort::SortPairsDescending(
            device_sort_storage_,
            sort_storage_bytes_,
            device_filtered_scores_ + offset,
            device_sorted_scores_ + offset,
            device_filtered_indexes_ + offset,
            device_sorted_indexes_ + offset,
            top_k_,
            0,
            sizeof(float) * 8,
            stream);
    }
}

void DetectionModelInferenceHelper::launchGatherAllKernel(int batch, cudaStream_t stream) {
    const int scores_s8_count = static_cast<int>(output_elements_per_batch_[0]);
    const int scores_s16_count = static_cast<int>(output_elements_per_batch_[1]);
    const int scores_s32_count = static_cast<int>(output_elements_per_batch_[2]);

    dim3 block(256, 1, 1);
    dim3 grid((top_k_ + block.x - 1) / block.x, batch, 1);

    gatherAllKernel<<<grid, block, 0, stream>>>(
        device_sorted_indexes_,
        device_model_output_buffers_[3],
        device_model_output_buffers_[4],
        device_model_output_buffers_[5],
        device_model_output_buffers_[6],
        device_model_output_buffers_[7],
        device_model_output_buffers_[8],
        scores_s8_count,
        scores_s16_count,
        scores_s32_count,
        batch,
        device_sorted_bboxes_,
        device_sorted_landmarks_,
        device_num_selected_);
}

void DetectionModelInferenceHelper::launchBitmaskNMSKernel(int batch, cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((top_k_ + block.x - 1) / block.x, batch, 1);

    bitmaskNMSKernel<<<grid, block, 0, stream>>>(
        device_sorted_bboxes_,
        device_suppression_mask_,
        device_num_selected_,
        batch);
}

void DetectionModelInferenceHelper::launchGatherFinalResultKernel(int batch, cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((top_k_ + block.x - 1) / block.x, batch, 1);

    gatherFinalResultKernel<<<grid, block, 0, stream>>>(
        device_sorted_indexes_,
        device_sorted_scores_,
        device_sorted_bboxes_,
        device_sorted_landmarks_,
        device_suppression_mask_,
        device_final_indexes_,
        device_final_scores_,
        device_final_bboxes_,
        device_final_landmarks_,
        device_final_num_detections_,
        device_num_selected_,
        batch);
}

void DetectionModelInferenceHelper::launchEstimateSimilarityKernel(int batch, cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((top_k_ + block.x - 1) / block.x, batch, 1);

    estimateSimilarityKernel<<<grid, block, 0, stream>>>(
        device_final_landmarks_,
        device_final_num_detections_,
        device_face_affine_matrices_,
        batch);
}

void DetectionModelInferenceHelper::launchWarpAffineKernel(int batch, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid(
        (kAlignedFaceWidth + block.x - 1) / block.x,
        (kAlignedFaceHeight + block.y - 1) / block.y,
        batch * top_k_);

    warpAffineKernel<<<grid, block, 0, stream>>>(
        device_jetson_ptr_uint8_t_,
        device_face_crops_,
        device_face_affine_matrices_,
        device_final_num_detections_,
        batch);
}


bool DetectionModelInferenceHelper::setDeviceSymbols(std::vector<int> slice_coordinates, int camera_width, int camera_height, int model_width, int model_height, int num_anchors, float confidence_threshold, float iou_threshold, int top_k)
{
    if (slice_coordinates.size() > MAX_SLICES * 4) {
        std::cerr << "[DetectionModelInferenceHelper] slice_coordinates size ("
                  << slice_coordinates.size() << ") exceeds maximum allowed ("
                  << MAX_SLICES * 4 << " = " << MAX_SLICES << " slices * 4 coords). "
                  << "Increase MAX_SLICES if more slices are needed.\n";
        return false;
    }
    const size_t active_coordinate_count = slice_coordinates.size();
    slice_coordinates.resize(MAX_SLICES * 4, 0);
    cudaError_t err = cudaMemcpyToSymbol(d_slices_coordinates, slice_coordinates.data(),
                                         MAX_SLICES * 4 * sizeof(int));
    err = cudaMemcpyToSymbol(d_source_width,          &camera_width,         sizeof(int));
    err = cudaMemcpyToSymbol(d_source_height,         &camera_height,        sizeof(int));
    err = cudaMemcpyToSymbol(d_dest_width,            &model_width,          sizeof(int));
    err = cudaMemcpyToSymbol(d_dest_height,           &model_height,         sizeof(int));
    err = cudaMemcpyToSymbol(d_num_anchors,           &num_anchors,          sizeof(int));
    err = cudaMemcpyToSymbol(d_confidence_threshold,  &confidence_threshold, sizeof(float));
    err = cudaMemcpyToSymbol(d_iou_threshold,         &iou_threshold,        sizeof(float));
    err = cudaMemcpyToSymbol(d_top_k,                 &top_k,                sizeof(int));

    float2 host_arcface_destination[kLandmarkCount] = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f},
    };
    float2 host_arcface_centroid = {0.0f, 0.0f};
    float2 host_arcface_normalized[kLandmarkCount] = {};

    for (int i = 0; i < kLandmarkCount; ++i)
    {
        host_arcface_centroid.x += host_arcface_destination[i].x;
        host_arcface_centroid.y += host_arcface_destination[i].y;
    }
    host_arcface_centroid.x *= 1.0f / static_cast<float>(kLandmarkCount);
    host_arcface_centroid.y *= 1.0f / static_cast<float>(kLandmarkCount);

    for (int i = 0; i < kLandmarkCount; ++i)
    {
        host_arcface_normalized[i].x = host_arcface_destination[i].x - host_arcface_centroid.x;
        host_arcface_normalized[i].y = host_arcface_destination[i].y - host_arcface_centroid.y;
    }

    err = cudaMemcpyToSymbol(d_arcface_centroid_destination, &host_arcface_centroid, sizeof(float2));
    err = cudaMemcpyToSymbol(d_arcface_normalized_destination,
                             host_arcface_normalized,
                             sizeof(host_arcface_normalized));

    // Verify constant memory was written correctly
    int gpu_coords[MAX_SLICES * 4] = {};
    cudaMemcpyFromSymbol(gpu_coords, d_slices_coordinates, sizeof(gpu_coords));
    int n_slices = static_cast<int>(active_coordinate_count) / 4;
    std::cerr << "[setDeviceSymbols] GPU d_slices_coordinates verify (" << n_slices << " slices):\n";
    for (int i = 0; i < n_slices; ++i)
        std::cerr << "  GPU slice " << i << ": ["
                  << gpu_coords[i*4+0] << "," << gpu_coords[i*4+1] << " -> "
                  << gpu_coords[i*4+2] << "," << gpu_coords[i*4+3] << "]\n";

    return err == cudaSuccess;
}
