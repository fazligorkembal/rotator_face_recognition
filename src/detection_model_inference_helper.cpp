#include "spd_logger_helper.h"
#include "detection_model_inference_helper.h"

#ifdef GENERATE_TXT
#include <fstream>
#endif

DetectionModelInferenceHelper::DetectionModelInferenceHelper(
    ICudaEngine *&engine,
    std::vector<int> batch_sizes,
    int32_t input_height,
    int32_t input_width,
    std::vector<int32_t> strides,
    int32_t top_k,
    float confidence_threshold,
    float iou_threshold)
{
    batch_sizes_ = batch_sizes;
    input_height_ = input_height;
    input_width_ = input_width;
    top_k_ = top_k;
    confidence_threshold_ = confidence_threshold;
    iou_threshold_ = iou_threshold;

    float2 host_destination_arcface[5] = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f}};

    float2 host_centroid_destination;
    host_centroid_destination.x = 0.0f;
    host_centroid_destination.y = 0.0f;
    float2 host_normalized_destination[5];

    for (int i = 0; i < 5; ++i)
    {
        host_centroid_destination.x += host_destination_arcface[i].x;
        host_centroid_destination.y += host_destination_arcface[i].y;
    }

    host_centroid_destination.x *= 0.2f;
    host_centroid_destination.y *= 0.2f;

    for (int i = 0; i < 5; ++i)
    {
        host_normalized_destination[i].x = host_destination_arcface[i].x - host_centroid_destination.x;
        host_normalized_destination[i].y = host_destination_arcface[i].y - host_centroid_destination.y;
    }

    LOG_INFO("Copying ArcFace destination landmarks to device constant memory");
    LOG_DEBUG("\tCentroid Destination: ({}, {})", host_centroid_destination.x, host_centroid_destination.y);
    LOG_DEBUG("\tNormalized Destination Landmarks:");
    for (int i = 0; i < 5; ++i)
    {
        LOG_DEBUG("\t\tLandmark {}: ({}, {})", i, host_normalized_destination[i].x, host_normalized_destination[i].y);
    }

    for (int i = 0; i < 3; ++i)
    {
        strides_[i] = strides[i];
    }

    context_detection_ = engine->createExecutionContext();
    cudaStreamCreate(&stream_);
    allocateBuffers();
    generateAnchors();

    const char *input_tensor_name = engine->getIOTensorName(0);
    const char *output_tensor_name_scores = engine->getIOTensorName(1);
    const char *output_tensor_name_bboxes = engine->getIOTensorName(2);
    const char *output_tensor_name_landmarks = engine->getIOTensorName(3);

    context_detection_->setTensorAddress(input_tensor_name, device_input_);
    context_detection_->setTensorAddress(output_tensor_name_scores, device_model_output_scores_);
    context_detection_->setTensorAddress(output_tensor_name_bboxes, device_model_output_bboxes_);
    context_detection_->setTensorAddress(output_tensor_name_landmarks, device_model_output_landmarks_);

    LOG_INFO("TensorRT tensor addresses binded successfully");

    if (setDeviceSymbols(anchor_count_, confidence_threshold_, top_k_, iou_threshold_, host_centroid_destination, host_normalized_destination))
    {
        LOG_INFO("Device symbols set successfully");
        LOG_DEBUG("\tnum_anchors: {}", anchor_count_);
        LOG_DEBUG("\tconfidence_threshold: {}", confidence_threshold_);
        LOG_DEBUG("\ttop_k: {}", top_k_);
    }
    else
    {
        LOG_ERROR("Failed to set device symbols");
        throw std::runtime_error("Failed to set device symbols");
    }

    threads_anchor_count_ = 256;
    blocks_anchor_count_ = (anchor_count_ + threads_anchor_count_ - 1) / threads_anchor_count_;

    threads_top_k_ = 256;
    blocks_top_k_ = (top_k_ + threads_top_k_ - 1) / threads_top_k_;

    LOG_WARN("Anchor Stack is fixed and equal to {}", anchor_stack_);
    LOG_INFO("Constructor called");
    LOG_INFO("\tinput shape: [B={}..{}, C=3, H={}, W={}]", batch_sizes_.front(), batch_sizes_.back(), input_height_, input_width_);
    LOG_INFO("\tstrides_: [{}, {}, {}]", strides_[0], strides_[1], strides_[2]);
    LOG_INFO("\tanchor_count_: {}", anchor_count_);
    LOG_INFO("\ttop_k: {}", top_k_);
    LOG_INFO("\tconfidence_threshold: {}", confidence_threshold_);
    LOG_DEBUG("\tanchor_stack_: {}", anchor_stack_);
    LOG_DEBUG("\tthreads_anchor_count_: {}", threads_anchor_count_);
    LOG_DEBUG("\tblocks_anchor_count_: {}", blocks_anchor_count_);
}

DetectionModelInferenceHelper::~DetectionModelInferenceHelper()
{
    // CRITICAL: Proper cleanup order to avoid CUDA event errors
    // 1. First synchronize the stream
    cudaStreamSynchronize(stream_);

    // 2. Delete TensorRT context while stream is still valid
    //    This allows TensorRT to properly clean up its internal CUDA events
    if (context_detection_)
    {
        delete context_detection_;
        context_detection_ = nullptr;
    }

    // 3. Now free all CUDA memory resources
    if (device_anchor_centers_)
    {
        cudaFree(device_anchor_centers_);
        device_anchor_centers_ = nullptr;
    }

    if (device_input_)
    {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }

    if (device_model_output_scores_)
    {
        cudaFree(device_model_output_scores_);
        device_model_output_scores_ = nullptr;
    }

    if (device_model_output_bboxes_)
    {
        cudaFree(device_model_output_bboxes_);
        device_model_output_bboxes_ = nullptr;
    }

    if (device_model_output_landmarks_)
    {
        cudaFree(device_model_output_landmarks_);
        device_model_output_landmarks_ = nullptr;
    }

    if (device_filtered_scores_)
    {
        cudaFree(device_filtered_scores_);
        device_filtered_scores_ = nullptr;
    }

    if (device_filtered_indexes_)
    {
        cudaFree(device_filtered_indexes_);
        device_filtered_indexes_ = nullptr;
    }

    if (device_num_detections_)
    {
        cudaFree(device_num_detections_);
        device_num_detections_ = nullptr;
    }

    if (device_sorted_scores_)
    {
        cudaFree(device_sorted_scores_);
        device_sorted_scores_ = nullptr;
    }

    if (device_sorted_indexes_)
    {
        cudaFree(device_sorted_indexes_);
        device_sorted_indexes_ = nullptr;
    }

    if (device_sorted_bboxes_)
    {
        cudaFree(device_sorted_bboxes_);
        device_sorted_bboxes_ = nullptr;
    }

    if (device_sorted_landmarks_)
    {
        cudaFree(device_sorted_landmarks_);
        device_sorted_landmarks_ = nullptr;
    }

    if (device_suppression_mask_)
    {
        cudaFree(device_suppression_mask_);
        device_suppression_mask_ = nullptr;
    }

    if (device_final_scores_)
    {
        cudaFree(device_final_scores_);
        device_final_scores_ = nullptr;
    }

    if (device_final_bboxes_)
    {
        cudaFree(device_final_bboxes_);
        device_final_bboxes_ = nullptr;
    }

    if (device_final_landmarks_)
    {
        cudaFree(device_final_landmarks_);
        device_final_landmarks_ = nullptr;
    }

    if (device_final_num_detections_)
    {
        cudaFree(device_final_num_detections_);
        device_final_num_detections_ = nullptr;
    }

    if (device_sort_storage_)
    {
        cudaFree(device_sort_storage_);
        device_sort_storage_ = nullptr;
    }

    if (device_images_face_)
    {
        cudaFree(device_images_face_);
        device_images_face_ = nullptr;
    }

    // 4. Finally destroy the stream
    cudaStreamDestroy(stream_);

    LOG_INFO("*******************************************************");
    LOG_INFO("Destructor called, resources freed.");
}

void DetectionModelInferenceHelper::allocateBuffers()
{
    for (auto stride : strides_)
    {
        int32_t feature_map_height = input_height_ / stride;
        int32_t feature_map_width = input_width_ / stride;
        int32_t anchors_per_feature_map = feature_map_height * feature_map_width;
        anchor_count_ += anchors_per_feature_map * anchor_stack_;
    }

    float total = 0;

    int32_t size_device_anchor_centers = anchor_count_ * sizeof(float2);
    cudaMallocAsync(&device_anchor_centers_, size_device_anchor_centers, stream_);
    total += size_device_anchor_centers / 1024.0 / 1024.0;

    int32_t size_device_original_image = batch_sizes_.back() * input_height_ * input_width_ * 3 * sizeof(uint8_t);
    cudaMallocAsync(&device_original_image_, size_device_original_image, stream_);
    total += size_device_original_image / 1024.0 / 1024.0;

    int32_t size_device_input = batch_sizes_.back() * 3 * input_height_ * input_width_ * sizeof(float);
    cudaMallocAsync(&device_input_, size_device_input, stream_);
    total += size_device_input / 1024.0 / 1024.0;

    int32_t size_device_model_output_scores = batch_sizes_.back() * anchor_count_ * sizeof(float);
    cudaMallocAsync(&device_model_output_scores_, size_device_model_output_scores, stream_);
    total += size_device_model_output_scores / 1024.0 / 1024.0;

    int32_t size_device_model_output_bboxes = batch_sizes_.back() * anchor_count_ * sizeof(float4);
    cudaMallocAsync(&device_model_output_bboxes_, size_device_model_output_bboxes, stream_);
    total += size_device_model_output_bboxes / 1024.0 / 1024.0;

    int32_t size_device_model_output_landmarks = batch_sizes_.back() * anchor_count_ * 5 * sizeof(float2);
    cudaMallocAsync(&device_model_output_landmarks_, size_device_model_output_landmarks, stream_);
    total += size_device_model_output_landmarks / 1024.0 / 1024.0;

    int32_t size_device_filtered_scores = batch_sizes_.back() * top_k_ * sizeof(float);
    cudaMallocAsync(&device_filtered_scores_, size_device_filtered_scores, stream_);
    total += size_device_filtered_scores / 1024.0 / 1024.0;

    int32_t size_device_filtered_indexes = batch_sizes_.back() * top_k_ * sizeof(int32_t);
    cudaMallocAsync(&device_filtered_indexes_, size_device_filtered_indexes, stream_);
    total += size_device_filtered_indexes / 1024.0 / 1024.0;

    int32_t size_device_num_detections = batch_sizes_.back() * sizeof(int32_t);
    cudaMallocAsync(&device_num_detections_, size_device_num_detections, stream_);
    total += size_device_num_detections / 1024.0 / 1024.0;

    int32_t size_device_sorted_scores = batch_sizes_.back() * top_k_ * sizeof(float);
    cudaMallocAsync(&device_sorted_scores_, size_device_sorted_scores, stream_);
    total += size_device_sorted_scores / 1024.0 / 1024.0;

    int32_t size_device_sorted_indexes = batch_sizes_.back() * top_k_ * sizeof(int32_t);
    cudaMallocAsync(&device_sorted_indexes_, size_device_sorted_indexes, stream_);
    total += size_device_sorted_indexes / 1024.0 / 1024.0;

    int32_t size_device_sorted_bboxes = batch_sizes_.back() * top_k_ * sizeof(float4);
    cudaMallocAsync(&device_sorted_bboxes_, size_device_sorted_bboxes, stream_);
    total += size_device_sorted_bboxes / 1024.0 / 1024.0;

    int32_t size_device_sorted_landmarks = batch_sizes_.back() * top_k_ * 5 * sizeof(float2);
    cudaMallocAsync(&device_sorted_landmarks_, size_device_sorted_landmarks, stream_);
    total += size_device_sorted_landmarks / 1024.0 / 1024.0;

    int32_t size_device_final_scores = batch_sizes_.back() * top_k_ * sizeof(float);
    cudaMallocAsync(&device_final_scores_, size_device_final_scores, stream_);
    total += size_device_final_scores / 1024.0 / 1024.0;

    int32_t size_device_final_bboxes = batch_sizes_.back() * top_k_ * sizeof(float4);
    cudaMallocAsync(&device_final_bboxes_, size_device_final_bboxes, stream_);
    total += size_device_final_bboxes / 1024.0 / 1024.0;

    int32_t size_device_final_landmarks = batch_sizes_.back() * top_k_ * 5 * sizeof(float2);
    cudaMallocAsync(&device_final_landmarks_, size_device_final_landmarks, stream_);
    total += size_device_final_landmarks / 1024.0 / 1024.0;

    int32_t size_device_final_num_detections = batch_sizes_.back() * sizeof(int32_t);
    cudaMallocAsync(&device_final_num_detections_, size_device_final_num_detections, stream_);
    total += size_device_final_num_detections / 1024.0 / 1024.0;

    int32_t size_device_d2s_M = top_k_ * 6 * sizeof(float);
    cudaMallocAsync(&device_d2s_M_, size_device_d2s_M, stream_);
    total += size_device_d2s_M / 1024.0 / 1024.0;

    int32_t size_faces_image = batch_sizes_.back() * top_k_ * 112 * 112 * 3 * sizeof(float);
    cudaMallocAsync(&device_images_face_, size_faces_image, stream_);
    total += size_faces_image / 1024.0 / 1024.0;

    total += setCubMemories();

    cudaStreamSynchronize(stream_);

    int32_t size_device_suppression_mask = batch_sizes_.back() * ((top_k_ + 31) / 32) * sizeof(uint32_t);
    cudaMallocAsync(&device_suppression_mask_, size_device_suppression_mask, stream_);
    total += size_device_suppression_mask / 1024.0 / 1024.0;

    LOG_INFO("*******************************************************");
    LOG_INFO("Allocation Done");
    LOG_INFO("\tanchor_count_: {}", anchor_count_);

    LOG_DEBUG("Allocated Device Memories:");
    LOG_DEBUG("\tdevice_anchor_centers: {} mb", (size_device_anchor_centers) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_input_: {} mb", (size_device_input) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_original_image: {} mb", (size_device_original_image) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_model_output_scores: {} mb", (size_device_model_output_scores) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_model_output_bboxes: {} mb", (size_device_model_output_bboxes) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_model_output_landmarks: {} mb", (size_device_model_output_landmarks) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_filtered_scores: {} mb", (size_device_filtered_scores) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_filtered_indexes: {} mb", (size_device_filtered_indexes) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_num_detections: {} mb", (size_device_num_detections) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_sorted_scores: {} mb", (size_device_sorted_scores) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_sorted_indexes: {} mb", (size_device_sorted_indexes) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_sorted_bboxes: {} mb", (size_device_sorted_bboxes) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_sorted_landmarks: {} mb", (size_device_sorted_landmarks) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_sort_storage: {} mb", (sort_storage_bytes_) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_suppression_mask_: {} mb", (size_device_suppression_mask) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_final_scores_: {} mb", (size_device_final_scores) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_final_bboxes_: {} mb", (size_device_final_bboxes) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_final_landmarks_: {} mb", (size_device_final_landmarks) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_final_num_detections_: {} mb", (size_device_final_num_detections) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_d2s_M_: {} mb", (size_device_d2s_M) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_images_face_: {} mb", (size_faces_image) / (1024.0 * 1024.0));
    LOG_INFO("Total allocated device memory: {} mb", total);
}

void DetectionModelInferenceHelper::generateAnchors()
{
    LOG_INFO("*******************************************************");
    float2 *host_anchor_centers = new float2[anchor_count_];

    int32_t feature_map_height_8 = input_height_ / strides_[0];
    int32_t feature_map_width_8 = input_width_ / strides_[0];
    int32_t feature_map_height_16 = input_height_ / strides_[1];
    int32_t feature_map_width_16 = input_width_ / strides_[1];
    int32_t feature_map_height_32 = input_height_ / strides_[2];
    int32_t feature_map_width_32 = input_width_ / strides_[2];

    float2 *host_anchor_centers_8 = new float2[feature_map_height_8 * feature_map_width_8];

    int32_t idx = 0;
    for (int32_t y = 0; y < feature_map_height_8; ++y)
    {
        for (int32_t x = 0; x < feature_map_width_8; ++x)
        {
            host_anchor_centers_8[idx].x = x * strides_[0];
            host_anchor_centers_8[idx].y = y * strides_[0];
            idx++;
        }
    }

    float2 *host_anchor_centers_16 = new float2[feature_map_height_16 * feature_map_width_16];
    idx = 0;
    for (int32_t y = 0; y < feature_map_height_16; ++y)
    {
        for (int32_t x = 0; x < feature_map_width_16; ++x)
        {
            host_anchor_centers_16[idx].x = x * strides_[1];
            host_anchor_centers_16[idx].y = y * strides_[1];
            idx++;
        }
    }

    float2 *host_anchor_centers_32 = new float2[feature_map_height_32 * feature_map_width_32];
    idx = 0;
    for (int32_t y = 0; y < feature_map_height_32; ++y)
    {
        for (int32_t x = 0; x < feature_map_width_32; ++x)
        {
            host_anchor_centers_32[idx].x = x * strides_[2];
            host_anchor_centers_32[idx].y = y * strides_[2];
            idx++;
        }
    }

    // Merge all anchors into a single array
    idx = 0;
    for (int32_t hw = 0; hw < feature_map_height_8 * feature_map_width_8; ++hw)
    {
        for (int32_t a = 0; a < anchor_stack_; ++a)
        {
            host_anchor_centers[idx].x = host_anchor_centers_8[hw].x;
            host_anchor_centers[idx].y = host_anchor_centers_8[hw].y;
            idx++;
        }
    }
    for (int32_t hw = 0; hw < feature_map_height_16 * feature_map_width_16; ++hw)
    {
        for (int32_t a = 0; a < anchor_stack_; ++a)
        {
            host_anchor_centers[idx].x = host_anchor_centers_16[hw].x;
            host_anchor_centers[idx].y = host_anchor_centers_16[hw].y;
            idx++;
        }
    }
    for (int32_t hw = 0; hw < feature_map_height_32 * feature_map_width_32; ++hw)
    {
        for (int32_t a = 0; a < anchor_stack_; ++a)
        {
            host_anchor_centers[idx].x = host_anchor_centers_32[hw].x;
            host_anchor_centers[idx].y = host_anchor_centers_32[hw].y;
            idx++;
        }
    }
#ifdef GENERATE_TXT
    // Write anchors to a text file for verification
    std::ofstream anchor_file("/home/user/Documents/tensorrt_scrfd/build/anchor_centers_engine.txt");
    for (int32_t i = 0; i < anchor_count_; ++i)
    {
        anchor_file << host_anchor_centers[i].x << "\n"
                    << host_anchor_centers[i].y << "\n";
    }
    anchor_file.close();
    LOG_INFO("Anchor txt file generated");
#endif

    cudaMemcpyAsync(
        device_anchor_centers_,
        host_anchor_centers,
        anchor_count_ * sizeof(float2),
        cudaMemcpyHostToDevice,
        stream_);
    cudaStreamSynchronize(stream_);

    LOG_INFO("Anchors copied to device memory");

    delete[] host_anchor_centers;
    delete[] host_anchor_centers_8;
    delete[] host_anchor_centers_16;
    delete[] host_anchor_centers_32;
}
