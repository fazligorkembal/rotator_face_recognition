#include "detection_model_inference_helper.h"
#include <cub/cub.cuh>

static constexpr int MAX_SLICES = 16; // max supported 

__constant__ int d_num_anchors;
__constant__ float d_confidence_threshold;
__constant__ float d_iou_threshold;
__constant__ int d_top_k;
__constant__ int d_min_box_length;
__constant__ int d_slices_coordinates[DetectionModelInferenceHelper::kMaxSlices * 4]; // Each slice has 4 coordinates: x1, y1, x2, y2
__constant__ int d_source_width;
__constant__ int d_source_height;
__constant__ int d_dest_width;
__constant__ int d_dest_height;
__constant__ float2 d_arcface_centroid_destination;
__constant__ float2 d_arcface_normalized_destination[DetectionModelInferenceHelper::kLandmarkCount];

__device__ float4 distanceToBbox(float2 anchor_center, float4 distance, int stride)
{
    return make_float4(
        fmaxf(anchor_center.x - distance.x * stride, 0.0f),
        fmaxf(anchor_center.y - distance.y * stride, 0.0f),
        fminf(anchor_center.x + distance.z * stride, static_cast<float>(d_dest_width)),
        fminf(anchor_center.y + distance.w * stride, static_cast<float>(d_dest_height)));
}

__device__ float2 distanceToLandmark(float2 anchor_center, float2 distance, int stride)
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
    int scores_s8_count,
    int scores_s16_count,
    int &local_anchor_idx,
    int &stride,
    int &feat_w)
{
    if (global_anchor_idx < scores_s8_count)
    {
        local_anchor_idx = global_anchor_idx;
        stride = 8;
        feat_w = d_dest_width / stride;
        return;
    }

    if (global_anchor_idx < scores_s8_count + scores_s16_count)
    {
        local_anchor_idx = global_anchor_idx - scores_s8_count;
        stride = 16;
        feat_w = d_dest_width / stride;
        return;
    }

    local_anchor_idx = global_anchor_idx - scores_s8_count - scores_s16_count;
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
    constexpr int bbox_channels = 4;
    if (global_anchor_idx < scores_s8_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s8_count + global_anchor_idx) * bbox_channels;
        return make_float4(bboxes_s8[base + 0], bboxes_s8[base + 1], bboxes_s8[base + 2], bboxes_s8[base + 3]);
    }

    int local_idx = global_anchor_idx - scores_s8_count;
    if (local_idx < scores_s16_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s16_count + local_idx) * bbox_channels;
        return make_float4(bboxes_s16[base + 0], bboxes_s16[base + 1], bboxes_s16[base + 2], bboxes_s16[base + 3]);
    }

    local_idx -= scores_s16_count;
    const size_t base = (static_cast<size_t>(batch) * scores_s32_count + local_idx) * bbox_channels;
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
    constexpr int landmark_channels = 10;
    const int channel = landmark_point_idx * 2;

    if (global_anchor_idx < scores_s8_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s8_count + global_anchor_idx) * landmark_channels;
        return make_float2(landmarks_s8[base + channel + 0], landmarks_s8[base + channel + 1]);
    }

    int local_idx = global_anchor_idx - scores_s8_count;
    if (local_idx < scores_s16_count)
    {
        const size_t base = (static_cast<size_t>(batch) * scores_s16_count + local_idx) * landmark_channels;
        return make_float2(landmarks_s16[base + channel + 0], landmarks_s16[base + channel + 1]);
    }

    local_idx -= scores_s16_count;
    const size_t base = (static_cast<size_t>(batch) * scores_s32_count + local_idx) * landmark_channels;
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
    {
        return;
    }

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
    {
        return;
    }

    const int selected_idx = atomicAdd(&num_selected[batch], 1);
    if (selected_idx >= d_top_k)
    {
        return;
    }

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
    {
        return;
    }

    const int valid_count = min(device_num_filtered[batch], d_top_k);
    if (rank >= valid_count)
    {
        return;
    }

    const int out_idx = batch * d_top_k + rank;
    const int global_anchor_idx = device_indexes_sorted[out_idx];

    int local_anchor_idx = 0;
    int stride = 0;
    int feat_w = 0;
    decodeGlobalAnchorIndex(global_anchor_idx,
                            scores_s8_count,
                            scores_s16_count,
                            local_anchor_idx,
                            stride,
                            feat_w);

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
    device_bboxes_sorted[out_idx] = distanceToBbox(anchor_center, bbox_distance, stride);

    for (int i = 0; i < DetectionModelInferenceHelper::kLandmarkCount; ++i)
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

        device_landmarks_sorted[out_idx * DetectionModelInferenceHelper::kLandmarkCount + i] =
            distanceToLandmark(anchor_center, landmark_distance, stride);
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
    {
        return;
    }

    const int valid_count = min(device_num_filtered[batch], d_top_k);
    if (rank >= valid_count)
    {
        return;
    }

    const int mask_words_per_batch = (d_top_k + 31) / 32;
    const int batch_box_offset = batch * d_top_k;
    const int batch_mask_offset = batch * mask_words_per_batch;
    const float4 box_a = device_bboxes_sorted[batch_box_offset + rank];
    const float box_a_width = fmaxf(0.0f, box_a.z - box_a.x);
    const float box_a_height = fmaxf(0.0f, box_a.w - box_a.y);

    if (box_a_width < static_cast<float>(d_min_box_length) ||
        box_a_height < static_cast<float>(d_min_box_length))
    {
        atomicOr(&device_suppression_mask[batch_mask_offset + rank / 32], 1u << (rank % 32));
        return;
    }

    for (int j = rank + 1; j < valid_count; ++j)
    {
        const float4 box_b = device_bboxes_sorted[batch_box_offset + j];
        const float box_b_width = fmaxf(0.0f, box_b.z - box_b.x);
        const float box_b_height = fmaxf(0.0f, box_b.w - box_b.y);

        if (box_b_width < static_cast<float>(d_min_box_length) ||
            box_b_height < static_cast<float>(d_min_box_length))
        {
            atomicOr(&device_suppression_mask[batch_mask_offset + j / 32], 1u << (j % 32));
            continue;
        }

        if (computeIOU(box_a, box_b) > d_iou_threshold)
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
    {
        return;
    }

    const int valid_count = min(device_num_filtered[batch], d_top_k);
    if (rank >= valid_count)
    {
        return;
    }

    const int sorted_offset = batch * d_top_k + rank;
    const int mask_words_per_batch = (d_top_k + 31) / 32;
    const int batch_mask_offset = batch * mask_words_per_batch;
    const float4 bbox = device_sorted_bboxes[sorted_offset];
    const float box_width = fmaxf(0.0f, bbox.z - bbox.x);
    const float box_height = fmaxf(0.0f, bbox.w - bbox.y);

    if (device_sorted_scores[sorted_offset] < d_confidence_threshold)
    {
        return;
    }

    if (box_width < static_cast<float>(d_min_box_length) ||
        box_height < static_cast<float>(d_min_box_length))
    {
        return;
    }

    if (device_suppression_mask[batch_mask_offset + rank / 32] & (1u << (rank % 32)))
    {
        return;
    }

    const int out_rank = atomicAdd(&device_final_count[batch], 1);
    if (out_rank >= d_top_k)
    {
        return;
    }

    const int final_offset = batch * d_top_k + out_rank;
    device_final_indexes[final_offset] = device_sorted_indexes[sorted_offset];
    device_final_scores[final_offset] = device_sorted_scores[sorted_offset];
    device_final_bboxes[final_offset] = bbox;

    for (int i = 0; i < DetectionModelInferenceHelper::kLandmarkCount; ++i)
    {
        device_final_landmarks[final_offset * DetectionModelInferenceHelper::kLandmarkCount + i] =
            device_sorted_landmarks[sorted_offset * DetectionModelInferenceHelper::kLandmarkCount + i];
    }
}

__global__ void estimateSimilarityKernel(
    const float2 *device_final_landmarks,
    const int32_t *device_final_count,
    float *device_similarity_transforms,
    int batch_size,
    bool is_search)
{
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (batch >= batch_size)
    {
        return;
    }

    const int count = min(device_final_count[batch], d_top_k);
    if (rank >= count)
    {
        return;
    }

    const int face_index = batch * d_top_k + rank;
    const int landmark_offset = face_index * DetectionModelInferenceHelper::kLandmarkCount;
    const int offset_x = is_search ? d_slices_coordinates[batch * 4 + 0] : 0;
    const int offset_y = is_search ? d_slices_coordinates[batch * 4 + 1] : 0;

    float2 src_centroid = make_float2(0.0f, 0.0f);
    float2 src_points[DetectionModelInferenceHelper::kLandmarkCount];
    for (int i = 0; i < DetectionModelInferenceHelper::kLandmarkCount; ++i)
    {
        const float2 point = device_final_landmarks[landmark_offset + i];
        src_points[i] = make_float2(point.x + static_cast<float>(offset_x),
                                    point.y + static_cast<float>(offset_y));
        src_centroid.x += src_points[i].x;
        src_centroid.y += src_points[i].y;
    }
    src_centroid.x *= 1.0f / static_cast<float>(DetectionModelInferenceHelper::kLandmarkCount);
    src_centroid.y *= 1.0f / static_cast<float>(DetectionModelInferenceHelper::kLandmarkCount);

    float numerator_a = 0.0f;
    float numerator_b = 0.0f;
    float denominator = 0.0f;
    for (int i = 0; i < DetectionModelInferenceHelper::kLandmarkCount; ++i)
    {
        const float sx = src_points[i].x - src_centroid.x;
        const float sy = src_points[i].y - src_centroid.y;
        const float dx = d_arcface_normalized_destination[i].x;
        const float dy = d_arcface_normalized_destination[i].y;

        numerator_a += sx * dx + sy * dy;
        numerator_b += sy * dx - sx * dy;
        denominator += dx * dx + dy * dy;
    }

    const float inv_denominator = denominator > 1e-6f ? 1.0f / denominator : 0.0f;
    const float a = numerator_a * inv_denominator;
    const float b = numerator_b * inv_denominator;
    const float tx =
        src_centroid.x - (a * d_arcface_centroid_destination.x - b * d_arcface_centroid_destination.y);
    const float ty =
        src_centroid.y - (b * d_arcface_centroid_destination.x + a * d_arcface_centroid_destination.y);

    const int transform_offset = face_index * 6;
    device_similarity_transforms[transform_offset + 0] = a;
    device_similarity_transforms[transform_offset + 1] = -b;
    device_similarity_transforms[transform_offset + 2] = tx;
    device_similarity_transforms[transform_offset + 3] = b;
    device_similarity_transforms[transform_offset + 4] = a;
    device_similarity_transforms[transform_offset + 5] = ty;
}

__global__ void warpAffineKernel(
    const uint8_t *source_image,
    int source_width,
    int source_height,
    const int32_t *device_final_count,
    const float *device_similarity_transforms,
    float *device_warped_faces,
    int batch_size)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int face_index = blockIdx.z;

    if (out_x >= DetectionModelInferenceHelper::kWarpedFaceSize ||
        out_y >= DetectionModelInferenceHelper::kWarpedFaceSize)
    {
        return;
    }

    const int batch = face_index / d_top_k;
    const int rank = face_index % d_top_k;
    if (batch >= batch_size)
    {
        return;
    }

    const int count = min(device_final_count[batch], d_top_k);
    if (rank >= count)
    {
        return;
    }

    const int transform_offset = face_index * 6;
    const float m0 = device_similarity_transforms[transform_offset + 0];
    const float m1 = device_similarity_transforms[transform_offset + 1];
    const float m2 = device_similarity_transforms[transform_offset + 2];
    const float m3 = device_similarity_transforms[transform_offset + 3];
    const float m4 = device_similarity_transforms[transform_offset + 4];
    const float m5 = device_similarity_transforms[transform_offset + 5];

    const float src_x = m0 * static_cast<float>(out_x) + m1 * static_cast<float>(out_y) + m2;
    const float src_y = m3 * static_cast<float>(out_x) + m4 * static_cast<float>(out_y) + m5;
    const int out_idx =
        (face_index * DetectionModelInferenceHelper::kWarpedFaceSize *
             DetectionModelInferenceHelper::kWarpedFaceSize +
         out_y * DetectionModelInferenceHelper::kWarpedFaceSize + out_x) *
        DetectionModelInferenceHelper::kWarpedFaceChannels;

    if (src_x < 0.0f || src_y < 0.0f ||
        src_x > static_cast<float>(source_width - 1) ||
        src_y > static_cast<float>(source_height - 1))
    {
        device_warped_faces[out_idx + 0] = 0.0f;
        device_warped_faces[out_idx + 1] = 0.0f;
        device_warped_faces[out_idx + 2] = 0.0f;
        return;
    }

    const int x0 = min(max(static_cast<int>(floorf(src_x)), 0), source_width - 1);
    const int y0 = min(max(static_cast<int>(floorf(src_y)), 0), source_height - 1);
    const int x1 = min(x0 + 1, source_width - 1);
    const int y1 = min(y0 + 1, source_height - 1);
    const float wx = src_x - static_cast<float>(x0);
    const float wy = src_y - static_cast<float>(y0);

    for (int channel = 0; channel < DetectionModelInferenceHelper::kWarpedFaceChannels; ++channel)
    {
        const float p00 = static_cast<float>(source_image[(y0 * source_width + x0) * 3 + channel]);
        const float p01 = static_cast<float>(source_image[(y0 * source_width + x1) * 3 + channel]);
        const float p10 = static_cast<float>(source_image[(y1 * source_width + x0) * 3 + channel]);
        const float p11 = static_cast<float>(source_image[(y1 * source_width + x1) * 3 + channel]);

        const float top = p00 * (1.0f - wx) + p01 * wx;
        const float bottom = p10 * (1.0f - wx) + p11 * wx;
        device_warped_faces[out_idx + channel] = top * (1.0f - wy) + bottom * wy;
    }
}

__global__ void preprocessWarpedFacesForIdentificationKernel(
    const float *device_warped_faces,
    const int32_t *device_final_count,
    float *device_warped_faces_identification,
    int batch_size)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int face_index = blockIdx.z;

    if (out_x >= DetectionModelInferenceHelper::kWarpedFaceSize ||
        out_y >= DetectionModelInferenceHelper::kWarpedFaceSize)
    {
        return;
    }

    const int batch = face_index / d_top_k;
    const int rank = face_index % d_top_k;
    if (batch >= batch_size)
    {
        return;
    }

    const int count = min(device_final_count[batch], d_top_k);
    if (rank >= count)
    {
        return;
    }

    const int plane =
        DetectionModelInferenceHelper::kWarpedFaceSize * DetectionModelInferenceHelper::kWarpedFaceSize;
    const int pixel = out_y * DetectionModelInferenceHelper::kWarpedFaceSize + out_x;
    const int src_offset =
        (face_index * plane + pixel) * DetectionModelInferenceHelper::kWarpedFaceChannels;
    const int dst_offset = face_index * DetectionModelInferenceHelper::kWarpedFaceChannels * plane + pixel;

    device_warped_faces_identification[dst_offset + 0 * plane] =
        (device_warped_faces[src_offset + 2] - 127.5f) / 128.0f;
    device_warped_faces_identification[dst_offset + 1 * plane] =
        (device_warped_faces[src_offset + 1] - 127.5f) / 128.0f;
    device_warped_faces_identification[dst_offset + 2 * plane] =
        (device_warped_faces[src_offset + 0] - 127.5f) / 128.0f;
}

// SEARCH: src is the full camera frame in BGR HWC. Each graph batch item reads
// one fixed slice coordinate from constant memory and writes RGB CHW float input.
__global__ void preprocessSearchBGRToFloatCHW(const uint8_t *src, float *dst, int batch)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (b >= batch || out_x >= d_dest_width || out_y >= d_dest_height)
    {
        return;
    }

    const int x1 = d_slices_coordinates[b * 4 + 0];
    const int y1 = d_slices_coordinates[b * 4 + 1];
    const int src_idx = ((y1 + out_y) * d_source_width + (x1 + out_x)) * 3;
    const int plane = d_dest_height * d_dest_width;
    const int pixel = out_y * d_dest_width + out_x;
    const int batch_offset = b * 3 * plane;

    dst[batch_offset + 0 * plane + pixel] = (static_cast<float>(src[src_idx + 2]) - 127.5f) / 128.0f;
    dst[batch_offset + 1 * plane + pixel] = (static_cast<float>(src[src_idx + 1]) - 127.5f) / 128.0f;
    dst[batch_offset + 2 * plane + pixel] = (static_cast<float>(src[src_idx + 0]) - 127.5f) / 128.0f;
}

// TRACK: src is already a single model-sized crop in BGR HWC.
__global__ void preprocessTrackBGRToFloatCHW(const uint8_t *src, float *dst)
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= d_dest_width || out_y >= d_dest_height)
    {
        return;
    }

    const int src_idx = (out_y * d_dest_width + out_x) * 3;
    const int plane = d_dest_height * d_dest_width;
    const int pixel = out_y * d_dest_width + out_x;

    dst[0 * plane + pixel] = (static_cast<float>(src[src_idx + 2]) - 127.5f) / 128.0f;
    dst[1 * plane + pixel] = (static_cast<float>(src[src_idx + 1]) - 127.5f) / 128.0f;
    dst[2 * plane + pixel] = (static_cast<float>(src[src_idx + 0]) - 127.5f) / 128.0f;
}

void DetectionModelInferenceHelper::launchPreprocessKernel(
    const uint8_t *src,
    float *dst,
    int batch,
    DetectionType detection_type,
    cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (model_input_width_ + block.x - 1) / block.x,
        (model_input_height_ + block.y - 1) / block.y,
        detection_type == DetectionType::SEARCH ? batch : 1);

    if (detection_type == DetectionType::SEARCH)
    {
        preprocessSearchBGRToFloatCHW<<<grid, block, 0, stream>>>(src, dst, batch);
        return;
    }

    preprocessTrackBGRToFloatCHW<<<grid, block, 0, stream>>>(src, dst);
}

void DetectionModelInferenceHelper::launchFilterScoresKernel(int batch, cudaStream_t stream)
{
    const int scores_s8_count = static_cast<int>(output_elements_per_batch_[0]);
    const int scores_s16_count = static_cast<int>(output_elements_per_batch_[1]);
    const int scores_s32_count = static_cast<int>(output_elements_per_batch_[2]);
    const int total_anchor_count = scores_s8_count + scores_s16_count + scores_s32_count;

    const dim3 block(256, 1, 1);
    const dim3 grid((total_anchor_count + block.x - 1) / block.x, batch, 1);

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

void DetectionModelInferenceHelper::launchSortFilteredScoresKernel(int batch, cudaStream_t stream)
{
    if (!device_sort_storage_)
    {
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr,
            sort_storage_bytes_,
            device_filtered_scores_,
            device_sorted_scores_,
            device_filtered_indexes_,
            device_sorted_indexes_,
            detection_top_k_,
            0,
            sizeof(float) * 8,
            stream);
        cudaMalloc(&device_sort_storage_, sort_storage_bytes_);
    }

    for (int batch_index = 0; batch_index < batch; ++batch_index)
    {
        const size_t offset =
            static_cast<size_t>(batch_index) * static_cast<size_t>(detection_top_k_);

        cub::DeviceRadixSort::SortPairsDescending(
            device_sort_storage_,
            sort_storage_bytes_,
            device_filtered_scores_ + offset,
            device_sorted_scores_ + offset,
            device_filtered_indexes_ + offset,
            device_sorted_indexes_ + offset,
            detection_top_k_,
            0,
            sizeof(float) * 8,
            stream);
    }
}

void DetectionModelInferenceHelper::launchGatherAllKernel(int batch, cudaStream_t stream)
{
    const int scores_s8_count = static_cast<int>(output_elements_per_batch_[0]);
    const int scores_s16_count = static_cast<int>(output_elements_per_batch_[1]);
    const int scores_s32_count = static_cast<int>(output_elements_per_batch_[2]);

    const dim3 block(256, 1, 1);
    const dim3 grid((detection_top_k_ + block.x - 1) / block.x, batch, 1);

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

void DetectionModelInferenceHelper::launchBitmaskNMSKernel(int batch, cudaStream_t stream)
{
    const dim3 block(256, 1, 1);
    const dim3 grid((detection_top_k_ + block.x - 1) / block.x, batch, 1);

    bitmaskNMSKernel<<<grid, block, 0, stream>>>(
        device_sorted_bboxes_,
        device_suppression_mask_,
        device_num_selected_,
        batch);
}

void DetectionModelInferenceHelper::launchGatherFinalResultKernel(int batch, cudaStream_t stream)
{
    const dim3 block(256, 1, 1);
    const dim3 grid((detection_top_k_ + block.x - 1) / block.x, batch, 1);

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

void DetectionModelInferenceHelper::launchEstimateSimilarityKernel(
    int batch,
    DetectionType detection_type,
    cudaStream_t stream)
{
    const dim3 block(256, 1, 1);
    const dim3 grid((detection_top_k_ + block.x - 1) / block.x, batch, 1);

    estimateSimilarityKernel<<<grid, block, 0, stream>>>(
        device_final_landmarks_,
        device_final_num_detections_,
        device_similarity_transforms_,
        batch,
        detection_type == DetectionType::SEARCH);
}

void DetectionModelInferenceHelper::launchWarpAffineKernel(
    int batch,
    DetectionType detection_type,
    cudaStream_t stream)
{
    const uint8_t *source_image = deviceInputForDetectionType(detection_type);
    const int source_width =
        detection_type == DetectionType::SEARCH ? camera_input_width_ : model_input_width_;
    const int source_height =
        detection_type == DetectionType::SEARCH ? camera_input_height_ : model_input_height_;
    const int total_faces = batch * detection_top_k_;
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (kWarpedFaceSize + block.x - 1) / block.x,
        (kWarpedFaceSize + block.y - 1) / block.y,
        total_faces);

    warpAffineKernel<<<grid, block, 0, stream>>>(
        source_image,
        source_width,
        source_height,
        device_final_num_detections_,
        device_similarity_transforms_,
        device_warped_faces_,
        batch);
}

void DetectionModelInferenceHelper::launchPreprocessWarpedFacesForIdentificationKernel(
    int batch,
    cudaStream_t stream)
{
    const int total_faces = batch * detection_top_k_;
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (kWarpedFaceSize + block.x - 1) / block.x,
        (kWarpedFaceSize + block.y - 1) / block.y,
        total_faces);

    preprocessWarpedFacesForIdentificationKernel<<<grid, block, 0, stream>>>(
        device_warped_faces_,
        device_final_num_detections_,
        device_warped_faces_identification_,
        batch);
}

bool DetectionModelInferenceHelper::setDeviceSymbols(int camera_width,
                                                     int camera_height,
                                                     int model_width,
                                                     int model_height,
                                                     int num_anchors,
                                                     float confidence_threshold,
                                                     float iou_threshold,
                                                     int top_k,
                                                     int min_box_length)
{
    std::vector<int> coords;
    coords.reserve(slices_.size() * 4);
    for (const auto &s : slices_) {
        coords.push_back(s.x1);
        coords.push_back(s.y1);
        coords.push_back(s.x2);
        coords.push_back(s.y2);
    }

    if (coords.size() > MAX_SLICES * 4) {
        return false;
    }

    std::vector<int> padded_coords(MAX_SLICES * 4, 0);
    for (size_t i = 0; i < coords.size(); ++i)
    {
        padded_coords[i] = coords[i];
    }

    cudaError_t err = cudaSuccess;
    err = cudaMemcpyToSymbol(d_slices_coordinates, padded_coords.data(), padded_coords.size() * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_source_width, &camera_width, sizeof(camera_width));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_source_height, &camera_height, sizeof(camera_height));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_dest_width, &model_width, sizeof(model_width));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_dest_height, &model_height, sizeof(model_height));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_num_anchors, &num_anchors, sizeof(num_anchors));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_confidence_threshold, &confidence_threshold, sizeof(confidence_threshold));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_iou_threshold, &iou_threshold, sizeof(iou_threshold));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_top_k, &top_k, sizeof(top_k));
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_min_box_length, &min_box_length, sizeof(min_box_length));
    if (err != cudaSuccess) return false;

    constexpr int kLandmarkCount = 5;
    const float2 host_arcface_destination[kLandmarkCount] = {
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
    if (err != cudaSuccess) return false;

    err = cudaMemcpyToSymbol(d_arcface_normalized_destination,
                             host_arcface_normalized,
                             sizeof(host_arcface_normalized));
    if (err != cudaSuccess) return false;

    return true;
}
