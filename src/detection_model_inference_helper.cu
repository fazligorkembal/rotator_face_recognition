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
__constant__ int d_input_size;
__constant__ float d_iou_threshold;

__constant__ float2 centroid_destination;
__constant__ float2 normalized_destination[5];

__constant__ int src_w = 1280;
__constant__ int src_h = 720;
__constant__ int device_dst_w = 112;
__constant__ int device_dst_h = 112;

const int host_src_w = 1280;
const int host_src_h = 720;
const int host_dst_w = 112;
const int host_dst_h = 112;

inline cv::Mat generate_mat(int batch_no, float *&data)
{
    cv::Mat response(112, 112, CV_32FC3);
    cv::Mat rgb;
    for (int c = 0; c < 3; c++)
    {
        for (int y = 0; y < 112; y++)
        {
            for (int x = 0; x < 112; x++)
            {
                response.at<cv::Vec3f>(y, x)[c] = data[batch_no * 3 * 112 * 112 + c * 112 * 112 + y * 112 + x];
            }
        }
    }
    response.convertTo(response, CV_8UC3, 128.0, 127.5);
    return response;
}

__device__ void getCentroid(
    const float2 *landmarks,
    float2 *centroid)
{
    centroid->x = 0.0f;
    centroid->y = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; i++)
    {
        centroid->x += landmarks[i].x;
        centroid->y += landmarks[i].y;
    }
    centroid->x *= 0.2f;
    centroid->y *= 0.2f;
}

__device__ void normalizeLandmarks(
    const float2 *landmarks,
    const float2 *centroid,
    float2 *normalized_landmarks)
{
#pragma unroll
    for (int i = 0; i < 5; i++)
    {
        normalized_landmarks[i].x = landmarks[i].x - centroid->x;
        normalized_landmarks[i].y = landmarks[i].y - centroid->y;
    }
}

__global__ void estimateSimilarityKernel(
    const float2 *device_final_landmarks, // [num_detections, 5]
    float *d2s_M_out,                     // [num_detections, 6]
    const int *device_final_num_detections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *device_final_num_detections)
        return;

    // --------------------------------------------------
    // Scale landmarks from model input size to original image size
    // Model input: 640x640, Original image: 1280x720
    // --------------------------------------------------
    float scale_x = (float)src_w / (float)d_input_size; // 1280 / 640 = 2.0
    float scale_y = (float)src_h / (float)d_input_size; // 720 / 640 = 1.125

    float2 scaled_landmarks[5];
    for (int i = 0; i < 5; i++)
    {
        scaled_landmarks[i].x = device_final_landmarks[idx * 5 + i].x * scale_x;
        scaled_landmarks[i].y = device_final_landmarks[idx * 5 + i].y * scale_y;
    }

    // --------------------------------------------------
    // 1. centroid + normalize (source)
    // --------------------------------------------------
    float2 centroid_src;
    float2 norm_src[5];

    getCentroid(scaled_landmarks, &centroid_src);
    normalizeLandmarks(scaled_landmarks, &centroid_src, norm_src);

    // --------------------------------------------------
    // 2. Procrustes sums
    // --------------------------------------------------
    float ss = 0.f;
    float sd = 0.f;
    float num = 0.f;
    float den = 0.f;

#pragma unroll
    for (int i = 0; i < 5; i++)
    {
        // source energy
        ss += norm_src[i].x * norm_src[i].x +
              norm_src[i].y * norm_src[i].y;

        // destination energy
        sd += normalized_destination[i].x * normalized_destination[i].x +
              normalized_destination[i].y * normalized_destination[i].y;

        // dot & cross
        num += norm_src[i].x * normalized_destination[i].x +
               norm_src[i].y * normalized_destination[i].y;

        den += norm_src[i].x * normalized_destination[i].y -
               norm_src[i].y * normalized_destination[i].x;
    }

    ss = fmaxf(ss, 1e-6f);

    float scale = sqrtf(sd / ss);
    float theta = atan2f(den, num);

    float c = cosf(theta);
    float s = sinf(theta);

    float a = scale * c;
    float b = -scale * s;
    float d = scale * s;
    float e = scale * c;

    float tx = centroid_destination.x - (a * centroid_src.x + b * centroid_src.y);
    float ty = centroid_destination.y - (d * centroid_src.x + e * centroid_src.y);

    float det = a * e - b * d;
    det = fabsf(det) < 1e-8f ? 1e-8f : det;

    float ia = e / det;
    float ib = -b / det;
    float id = -d / det;
    float ie = a / det;

    float ic = -(ia * tx + ib * ty);
    float if_ = -(id * tx + ie * ty);

    d2s_M_out[idx * 6 + 0] = ia;
    d2s_M_out[idx * 6 + 1] = ib;
    d2s_M_out[idx * 6 + 2] = ic;
    d2s_M_out[idx * 6 + 3] = id;
    d2s_M_out[idx * 6 + 4] = ie;
    d2s_M_out[idx * 6 + 5] = if_;
}

__global__ void warpAffineKernel(
    const uint8_t *src,   // [H, W, 3] BGR original image
    float *dst,           // [max_faces, 3, 112, 112] RGB normalized output crops
    const float *M,       // [max_faces, 6] affine matrices
    const int *num_faces) // GPU pointer to actual face count
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // dst x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // dst y
    int face_id = blockIdx.z;
    if (face_id >= *num_faces)
        return;
    if (x >= device_dst_w || y >= device_dst_h)
        return;

    const float *M_face = M + face_id * 6;
    float src_x = fminf(fmaxf(M_face[0] * x + M_face[1] * y + M_face[2], 0.f), src_w - 1.001f);
    float src_y = fminf(fmaxf(M_face[3] * x + M_face[4] * y + M_face[5], 0.f), src_h - 1.001f);

    // Output layout: [face_id, C, H, W]
    int base_idx = face_id * 3 * device_dst_h * device_dst_w + y * device_dst_w + x;

    if (src_x < 0 || src_x >= src_w - 1 || src_y < 0 || src_y >= src_h - 1)
    {
        // Normalized zero is (0 - 127.5) / 128.0 = -0.99609375
        dst[base_idx + 0 * device_dst_h * device_dst_w] = -0.99609375f; // R channel
        dst[base_idx + 1 * device_dst_h * device_dst_w] = -0.99609375f; // G channel
        dst[base_idx + 2 * device_dst_h * device_dst_w] = -0.99609375f; // B channel
        return;
    }

    int x0 = static_cast<int>(floorf(src_x));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(floorf(src_y));
    int y1 = y0 + 1;

    float dx = src_x - x0;
    float dy = src_y - y0;

    int s00 = (y0 * src_w + x0) * 3;
    int s01 = (y0 * src_w + x1) * 3;
    int s10 = (y1 * src_w + x0) * 3;
    int s11 = (y1 * src_w + x1) * 3;

    // Bilinear interpolation for BGR, then convert to RGB and normalize
#pragma unroll
    for (int c = 0; c < 3; c++)
    {
        // Bilinear interpolation
        float v =
            src[s00 + c] * (1 - dx) * (1 - dy) +
            src[s01 + c] * dx * (1 - dy) +
            src[s10 + c] * (1 - dx) * dy +
            src[s11 + c] * dx * dy;

        // Normalize: (v - 127.5) / 128.0 (matching blobFromImage preprocessing)
        float normalized = (v - 127.5f) / 128.0f;

        // BGR to RGB conversion (swapRB=true in blobFromImage)
        // Input: c=0 is B, c=1 is G, c=2 is R
        // Output: channel 0 = R, channel 1 = G, channel 2 = B
        int channel_idx;
        if (c == 0)          // B channel in input
            channel_idx = 2; // goes to B channel in output (index 2)
        else if (c == 2)     // R channel in input
            channel_idx = 0; // goes to R channel in output (index 0)
        else                 // G channel
            channel_idx = 1;

        // Write to [face_id, C, H, W] layout (matching blobFromImage output)
        dst[base_idx + channel_idx * device_dst_h * device_dst_w] = normalized;
    }
}

__device__ float4 distance_to_bbox(
    const float2 *anchor_center,
    const float4 *distance,
    const int stride)
{
    float4 bbox;
    // Distance değerlerini stride ile çarp
    bbox.x = fmaxf(anchor_center->x - distance->x * stride, 0.0f);         // x_min
    bbox.y = fmaxf(anchor_center->y - distance->y * stride, 0.0f);         // y_min
    bbox.z = fminf(anchor_center->x + distance->z * stride, d_input_size); // x_max
    bbox.w = fminf(anchor_center->y + distance->w * stride, d_input_size); // y_max
    return bbox;
}

__device__ float2 distance_to_landmark(
    const float2 *anchor_center,
    const float2 *distance,
    const int stride)
{
    float2 landmark;
    landmark.x = anchor_center->x + distance->x * stride;
    landmark.y = anchor_center->y + distance->y * stride;

    return landmark;
}

__global__ void filterScoresFilterKernel(
    const float *scores,
    float *scores_filtered,
    int32_t *selected_indexes,
    int32_t *num_selected)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_num_anchors)
        return;

    float score = scores[idx];
    if (score < d_confidence_threshold)
        return;
    int selected_idx = atomicAdd(num_selected, 1);
    if (selected_idx >= d_top_k)
        return;
    selected_indexes[selected_idx] = idx;
    scores_filtered[selected_idx] = score;
}

__global__ void gatherAllKernel(
    const int32_t *device_indexes_sorted,
    const float2 *device_anchor_centers,
    const float4 *device_model_output_bboxes,
    const float2 *device_model_output_landmarks,
    const float *device_model_output_scores,
    float4 *device_bboxes_sorted,
    float2 *device_sorted_landmarks,
    const int32_t *device_num_filtered)

{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *device_num_filtered)
        return;

    int32_t selected_index = device_indexes_sorted[idx];

    int stride;
    if (selected_index < 12800)
    {
        stride = 8;
    }
    else if (selected_index < 16000)
    {
        stride = 16;
    }
    else
    {
        stride = 32;
    }

    device_bboxes_sorted[idx] = distance_to_bbox(
        &device_anchor_centers[selected_index],
        &device_model_output_bboxes[selected_index],
        stride);

    for (int i = 0; i < 5; i++)
    {
        device_sorted_landmarks[idx * 5 + i] = distance_to_landmark(
            &device_anchor_centers[selected_index],
            &device_model_output_landmarks[selected_index * 5 + i],
            stride);
    }
}

__device__ float computeIOU(float4 box1, float4 box2)
{
    float inter_x1 = fmaxf(box1.x, box2.x);
    float inter_y1 = fmaxf(box1.y, box2.y);
    float inter_x2 = fminf(box1.z, box2.z);
    float inter_y2 = fminf(box1.w, box2.w);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area1 = (box1.z - box1.x) * (box1.w - box1.y);
    float area2 = (box2.z - box2.x) * (box2.w - box2.y);
    return inter_area / (area1 + area2 - inter_area + 1e-6f);
}

__global__ void bitmaskNMSKernel(
    const float4 *device_bboxes,
    const float *device_scores,
    uint32_t *device_suppression_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_top_k)
        return;

    float4 boxA = device_bboxes[idx];

    for (int j = idx + 1; j < d_top_k; ++j)
    {
        float4 boxB = device_bboxes[j];
        float iou = computeIOU(boxA, boxB);
        if (iou > d_iou_threshold)
        {
            atomicOr(&device_suppression_mask[j / 32], 1u << (j % 32));
        }
    }
}

__global__ void gatherFinalResultKernel(
    const float4 *device_bboxes_sorted,
    const float *device_scores_sorted,
    const float2 *device_landmarks_sorted,
    const uint32_t *device_suppression_mask,
    float4 *device_final_bboxes,
    float *device_final_scores,
    float2 *device_final_landmarks,
    int32_t *device_final_count,
    const int32_t *device_num_filtered) // Gerçek detection sayısı
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *device_num_filtered) // topK değil, gerçek sayıyı kullan
        return;

    // Score kontrolü - sıfır olan score'ları atla
    if (device_scores_sorted[idx] < d_confidence_threshold)
        return;

    // Bu detection suppress edilmiş mi kontrol et
    if (device_suppression_mask[idx / 32] & (1u << (idx % 32)))
        return; // Suppress edilmiş, atla

    // Valid detection, output'a ekle
    int out_idx = atomicAdd(device_final_count, 1);
    device_final_bboxes[out_idx] = device_bboxes_sorted[idx];
    device_final_scores[out_idx] = device_scores_sorted[idx];

    // Landmarks'i de kopyala
    for (int i = 0; i < 5; i++)
    {
        device_final_landmarks[out_idx * 5 + i] = device_landmarks_sorted[idx * 5 + i];
    }
}

void DetectionModelInferenceHelper::infer(
    float *input_image,
    SCRFDResults &results)
{

    cudaMemsetAsync(device_num_detections_, 0, batch_sizes_.back() * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_final_num_detections_, 0, batch_sizes_.back() * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_suppression_mask_, 0, batch_sizes_.back() * ((top_k_ + 31) / 32) * sizeof(uint32_t), stream_);
    cudaMemsetAsync(device_filtered_scores_, 0, batch_sizes_.back() * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_filtered_indexes_, 0, batch_sizes_.back() * top_k_ * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_sorted_scores_, 0, batch_sizes_.back() * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_sorted_indexes_, 0, batch_sizes_.back() * top_k_ * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_sorted_bboxes_, 0, batch_sizes_.back() * top_k_ * sizeof(float4), stream_);
    cudaMemsetAsync(device_sorted_landmarks_, 0, batch_sizes_.back() * top_k_ * 5 * sizeof(float2), stream_);
    cudaMemsetAsync(device_final_bboxes_, 0, batch_sizes_.back() * top_k_ * sizeof(float4), stream_);
    cudaMemsetAsync(device_final_landmarks_, 0, batch_sizes_.back() * top_k_ * 5 * sizeof(float2), stream_);
    cudaMemsetAsync(device_final_scores_, 0, batch_sizes_.back() * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_final_num_detections_, 0, batch_sizes_.back() * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_images_face_, 0, batch_sizes_.back() * top_k_ * 3 * 112 * 112 * sizeof(float), stream_);

    cudaMemcpyAsync(
        device_input_,
        input_image,
        batch_sizes_.back() * 3 * input_height_ * input_width_ * sizeof(float),
        cudaMemcpyHostToDevice,
        stream_);

    context_detection_->enqueueV3(stream_);

    filterScoresFilterKernel<<<blocks_anchor_count_, threads_anchor_count_, 0, stream_>>>(
        device_model_output_scores_,
        device_filtered_scores_,
        device_filtered_indexes_,
        device_num_detections_);

    cub::DeviceRadixSort::SortPairsDescending(
        device_sort_storage_,
        sort_storage_bytes_,
        device_filtered_scores_,
        device_sorted_scores_,
        device_filtered_indexes_,
        device_sorted_indexes_,
        top_k_,
        0,
        sizeof(float) * 8,
        stream_);

    gatherAllKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_sorted_indexes_,
        device_anchor_centers_,
        device_model_output_bboxes_,
        device_model_output_landmarks_,
        device_model_output_scores_,
        device_sorted_bboxes_,
        device_sorted_landmarks_,
        device_num_detections_);

    bitmaskNMSKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_sorted_bboxes_,
        device_sorted_scores_,
        device_suppression_mask_);

    gatherFinalResultKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_sorted_bboxes_,
        device_sorted_scores_,
        device_sorted_landmarks_,
        device_suppression_mask_,
        device_final_bboxes_,
        device_final_scores_,
        device_final_landmarks_,
        device_final_num_detections_,
        device_num_detections_);

    cudaMemcpyAsync(
        results.scores_,
        device_final_scores_,
        batch_sizes_.back() * top_k_ * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_);

    cudaMemcpyAsync(
        results.bboxes_,
        device_final_bboxes_,
        batch_sizes_.back() * top_k_ * sizeof(float4),
        cudaMemcpyDeviceToHost,
        stream_);

    cudaMemcpyAsync(
        results.landmarks_,
        device_final_landmarks_,
        batch_sizes_.back() * top_k_ * 5 * sizeof(float2),
        cudaMemcpyDeviceToHost,
        stream_);

    cudaStreamSynchronize(stream_);

    estimateSimilarityKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_final_landmarks_,
        device_d2s_M_,
        device_final_num_detections_);

    dim3 threads(16, 16);
    dim3 blocks((host_dst_w + threads.x - 1) / threads.x,
                (host_dst_h + threads.y - 1) / threads.y,
                batch_sizes_.back() * top_k_);

    cudaMallocAsync(&device_original_image_, batch_sizes_.back() * host_src_h * host_src_w * 3 * sizeof(uint8_t), stream_);

    cudaMemcpyAsync(
        device_original_image_,
        results.img_display_,
        batch_sizes_.back() * host_src_h * host_src_w * 3 * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        stream_);

    warpAffineKernel<<<blocks, threads, 0, stream_>>>(
        device_original_image_,
        device_images_face_,
        device_d2s_M_,
        device_final_num_detections_);

    cudaMemcpyAsync(
        &results.detected_count_,
        device_final_num_detections_,
        batch_sizes_.back() * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream_);

    cudaFreeAsync(device_original_image_, stream_);

    /*
    cudaMemcpyAsync(
        host_images_face,
        device_images_face_,
        batch_sizes_.back() * top_k_ * 3 * 112 * 112 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_);
    */

    // cudaMemcpyAsync(
    //     output_scores,
    //     device_final_scores_,
    //     batch_sizes_.back() * top_k_ * sizeof(float),
    //     cudaMemcpyDeviceToHost,
    //     stream_);

    // cudaMemcpyAsync(
    //     output_bboxes,
    //     device_final_bboxes_,
    //     batch_sizes_.back() * top_k_ * sizeof(float4),
    //     cudaMemcpyDeviceToHost,
    //     stream_);

    // cudaMemcpyAsync(
    //     output_landmarks,
    //     device_final_landmarks_,
    //     batch_sizes_.back() * top_k_ * 5 * sizeof(float2),
    //     cudaMemcpyDeviceToHost,
    //     stream_);

    /*
    #ifdef GENERATE_TXT
        float *host_sorted_scores = new float[top_k_];

        int32_t host_num_detections;
        cudaMemcpyAsync(
            &host_num_detections,
            device_num_detections_,
            batch_sizes_.back() * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream_);
        cudaStreamSynchronize(stream_);
        std::cout << "Number of detections: " << host_num_detections << std::endl;

        cudaMemcpyAsync(
            host_sorted_scores,
            device_sorted_scores_,
            top_k_ * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_);

        cudaStreamSynchronize(stream_);

        std::ofstream output_file("engine_sorted_scores.txt");
        for (int i = 0; i < host_num_detections; ++i)
        {
            output_file << host_sorted_scores[i] << std::endl;
        }
        output_file.close();

        float4 *host_sorted_bboxes = new float4[top_k_];
        cudaMemcpyAsync(
            host_sorted_bboxes,
            device_sorted_bboxes_,
            top_k_ * sizeof(float4),
            cudaMemcpyDeviceToHost,
            stream_);
        cudaStreamSynchronize(stream_);
        std::ofstream bboxes_file("engine_sorted_bboxes.txt");
        for (int i = 0; i < host_num_detections; ++i)
        {
            bboxes_file << host_sorted_bboxes[i].x << "\n"
                        << host_sorted_bboxes[i].y << "\n"
                        << host_sorted_bboxes[i].z << "\n"
                        << host_sorted_bboxes[i].w << std::endl;
        }
        bboxes_file.close();

        int32_t *host_num_detections_array = new int32_t[batch_sizes_.back()];
        cudaMemcpyAsync(
            host_num_detections_array,
            device_final_num_detections_,
            batch_sizes_.back() * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream_);

        cudaStreamSynchronize(stream_);
        for (int i = 0; i < batch_sizes_.back(); ++i)
        {
            std::cout << "Batch " << i << " - Number of detections: " << host_num_detections_array[i] << std::endl;
        }

        delete[] host_sorted_bboxes;
        delete[] host_sorted_scores;

    #endif
    */
}

void DetectionModelInferenceHelper::infer_facecrop(
    float *input_image,
    SCRFDResults &results)
{

    cudaMemsetAsync(device_num_detections_, 0, batch_sizes_.back() * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_final_num_detections_, 0, batch_sizes_.back() * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_suppression_mask_, 0, batch_sizes_.back() * ((top_k_ + 31) / 32) * sizeof(uint32_t), stream_);
    cudaMemsetAsync(device_filtered_scores_, 0, batch_sizes_.back() * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_filtered_indexes_, 0, batch_sizes_.back() * top_k_ * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_sorted_scores_, 0, batch_sizes_.back() * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_sorted_indexes_, 0, batch_sizes_.back() * top_k_ * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_sorted_bboxes_, 0, batch_sizes_.back() * top_k_ * sizeof(float4), stream_);
    cudaMemsetAsync(device_sorted_landmarks_, 0, batch_sizes_.back() * top_k_ * 5 * sizeof(float2), stream_);
    cudaMemsetAsync(device_final_bboxes_, 0, batch_sizes_.back() * top_k_ * sizeof(float4), stream_);
    cudaMemsetAsync(device_final_landmarks_, 0, batch_sizes_.back() * top_k_ * 5 * sizeof(float2), stream_);
    cudaMemsetAsync(device_final_scores_, 0, batch_sizes_.back() * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_final_num_detections_, 0, batch_sizes_.back() * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_images_face_, 0, batch_sizes_.back() * top_k_ * 3 * 112 * 112 * sizeof(float), stream_);

    cudaMemcpyAsync(
        device_input_,
        input_image,
        batch_sizes_.back() * 3 * input_height_ * input_width_ * sizeof(float),
        cudaMemcpyHostToDevice,
        stream_);

    context_detection_->enqueueV3(stream_);

    filterScoresFilterKernel<<<blocks_anchor_count_, threads_anchor_count_, 0, stream_>>>(
        device_model_output_scores_,
        device_filtered_scores_,
        device_filtered_indexes_,
        device_num_detections_);

    cub::DeviceRadixSort::SortPairsDescending(
        device_sort_storage_,
        sort_storage_bytes_,
        device_filtered_scores_,
        device_sorted_scores_,
        device_filtered_indexes_,
        device_sorted_indexes_,
        top_k_,
        0,
        sizeof(float) * 8,
        stream_);

    gatherAllKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_sorted_indexes_,
        device_anchor_centers_,
        device_model_output_bboxes_,
        device_model_output_landmarks_,
        device_model_output_scores_,
        device_sorted_bboxes_,
        device_sorted_landmarks_,
        device_num_detections_);

    bitmaskNMSKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_sorted_bboxes_,
        device_sorted_scores_,
        device_suppression_mask_);

    gatherFinalResultKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_sorted_bboxes_,
        device_sorted_scores_,
        device_sorted_landmarks_,
        device_suppression_mask_,
        device_final_bboxes_,
        device_final_scores_,
        device_final_landmarks_,
        device_final_num_detections_,
        device_num_detections_);

    cudaMemcpyAsync(
        results.scores_,
        device_final_scores_,
        batch_sizes_.back() * top_k_ * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_);

    cudaMemcpyAsync(
        results.bboxes_,
        device_final_bboxes_,
        batch_sizes_.back() * top_k_ * sizeof(float4),
        cudaMemcpyDeviceToHost,
        stream_);

    cudaMemcpyAsync(
        results.landmarks_,
        device_final_landmarks_,
        batch_sizes_.back() * top_k_ * 5 * sizeof(float2),
        cudaMemcpyDeviceToHost,
        stream_);

    cudaStreamSynchronize(stream_);

    estimateSimilarityKernel<<<blocks_top_k_, threads_top_k_, 0, stream_>>>(
        device_final_landmarks_,
        device_d2s_M_,
        device_final_num_detections_);

    dim3 threads(16, 16);
    dim3 blocks((host_dst_w + threads.x - 1) / threads.x,
                (host_dst_h + threads.y - 1) / threads.y,
                batch_sizes_.back() * top_k_);

    cudaMemcpyAsync(
        device_original_image_,
        results.img_display_,
        batch_sizes_.back() * host_src_h * host_src_w * 3 * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        stream_);

    warpAffineKernel<<<blocks, threads, 0, stream_>>>(
        device_original_image_,
        device_images_face_,
        device_d2s_M_,
        device_final_num_detections_);

    /*
    cudaMemcpyAsync(
        results.faces_warped_,
        device_images_face_,
        batch_sizes_.back() * top_k_ * 3 * 112 * 112 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_);
    */

    /*
    for (int i = 0; i < 20; i++)
    {
        cv::Mat face1 = generate_mat(i, host_images_face);
        cv::imshow("face1", face1);
        cv::waitKey(0);
    }
    */

    /*
    #ifdef GENERATE_TXT
        float *host_sorted_scores = new float[top_k_];

        int32_t host_num_detections;
        cudaMemcpyAsync(
            &host_num_detections,
            device_num_detections_,
            batch_sizes_.back() * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream_);
        cudaStreamSynchronize(stream_);
        std::cout << "Number of detections: " << host_num_detections << std::endl;

        cudaMemcpyAsync(
            host_sorted_scores,
            device_sorted_scores_,
            top_k_ * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_);

        cudaStreamSynchronize(stream_);

        std::ofstream output_file("engine_sorted_scores.txt");
        for (int i = 0; i < host_num_detections; ++i)
        {
            output_file << host_sorted_scores[i] << std::endl;
        }
        output_file.close();

        float4 *host_sorted_bboxes = new float4[top_k_];
        cudaMemcpyAsync(
            host_sorted_bboxes,
            device_sorted_bboxes_,
            top_k_ * sizeof(float4),
            cudaMemcpyDeviceToHost,
            stream_);
        cudaStreamSynchronize(stream_);
        std::ofstream bboxes_file("engine_sorted_bboxes.txt");
        for (int i = 0; i < host_num_detections; ++i)
        {
            bboxes_file << host_sorted_bboxes[i].x << "\n"
                        << host_sorted_bboxes[i].y << "\n"
                        << host_sorted_bboxes[i].z << "\n"
                        << host_sorted_bboxes[i].w << std::endl;
        }
        bboxes_file.close();

        int32_t *host_num_detections_array = new int32_t[batch_sizes_.back()];
        cudaMemcpyAsync(
            host_num_detections_array,
            device_final_num_detections_,
            batch_sizes_.back() * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream_);

        cudaStreamSynchronize(stream_);
        for (int i = 0; i < batch_sizes_.back(); ++i)
        {
            std::cout << "Batch " << i << " - Number of detections: " << host_num_detections_array[i] << std::endl;
        }

        delete[] host_sorted_bboxes;
        delete[] host_sorted_scores;

    #endif
    */
}

bool DetectionModelInferenceHelper::setDeviceSymbols(
    int32_t num_anchors, float confidence_threshold, int32_t top_k, float iou_threshold, float2 &host_centroid_destination, float2 *host_normalized_destination)
{
    cudaMemcpyToSymbol(d_num_anchors, &num_anchors, sizeof(int32_t));
    cudaMemcpyToSymbol(d_confidence_threshold, &confidence_threshold, sizeof(float));
    cudaMemcpyToSymbol(d_top_k, &top_k, sizeof(int32_t));
    cudaMemcpyToSymbol(d_input_size, &input_width_, sizeof(int32_t));
    cudaMemcpyToSymbol(d_iou_threshold, &iou_threshold, sizeof(float));
    cudaMemcpyToSymbol(centroid_destination, &host_centroid_destination, sizeof(float2));
    cudaMemcpyToSymbol(normalized_destination, host_normalized_destination, 5 * sizeof(float2));
    std::cout << "centroid_destination: (" << host_centroid_destination.x << ", " << host_centroid_destination.y << ")\n";
    std::cout << "normalized_destination[0]: (" << host_normalized_destination[0].x << ", " << host_normalized_destination[0].y << ")\n";
    std::cout << "normalized_destination[1]: (" << host_normalized_destination[1].x << ", " << host_normalized_destination[1].y << ")\n";
    std::cout << "normalized_destination[2]: (" << host_normalized_destination[2].x << ", " << host_normalized_destination[2].y << ")\n";
    std::cout << "normalized_destination[3]: (" << host_normalized_destination[3].x << ", " << host_normalized_destination[3].y << ")\n";
    std::cout << "normalized_destination[4]: (" << host_normalized_destination[4].x << ", " << host_normalized_destination[4].y << ")\n";
    return true;
}

float DetectionModelInferenceHelper::setCubMemories()
{
    cub::DeviceRadixSort::SortPairsDescending(
        device_sort_storage_,
        sort_storage_bytes_,
        device_filtered_scores_,
        device_sorted_scores_,
        device_filtered_indexes_,
        device_sorted_indexes_,
        top_k_,
        0,
        sizeof(float) * 8,
        stream_);
    cudaMallocAsync(&device_sort_storage_, sort_storage_bytes_, stream_);
    return sort_storage_bytes_ / 1024.0 / 1024.0;
}