#pragma once
#include <string>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include "inference_results.hpp"

using namespace nvinfer1;

class DetectionModelInferenceHelper
{
public:
    DetectionModelInferenceHelper(
        ICudaEngine *&engine,
        std::vector<int> batch_sizes,
        int32_t input_height,
        int32_t input_width,
        std::vector<int32_t> strides,
        int32_t top_k,
        float confidence_threshold,
        float iou_threshold);

    ~DetectionModelInferenceHelper();
    void infer(
        float *input_image,
        SCRFDResults &results);

    void infer_facecrop(
        float *input_image,
        SCRFDResults &results);

    // Getter for device face images pointer (after inference)
    float *getDeviceImagesFace() const
    {
        return device_images_face_;
    }

    cudaStream_t getStream() const
    {
        return stream_;
    }

    float *getDeviceFacesWarped() const
    {
        return device_images_face_;
    }

private:
    // Model and inference parameters
    std::string model_path_;
    std::vector<int> batch_sizes_;
    int32_t input_height_ = 0;
    int32_t input_width_ = 0;
    int32_t strides_[3] = {0, 0, 0};

    int32_t anchor_count_ = 0;
    int32_t anchor_stack_ = 2;
    int32_t top_k_ = 0;
    float iou_threshold_ = 0.0f;
    float confidence_threshold_ = 0.0f;

    // Anchor Buffers
    float2 *device_anchor_centers_ = nullptr;

    // IO Buffers
    float *device_input_ = nullptr;
    uint8_t *device_original_image_;

    float *device_model_output_scores_ = nullptr;
    float4 *device_model_output_bboxes_ = nullptr;
    float2 *device_model_output_landmarks_ = nullptr;

    // NMS Buffers
    float *device_filtered_scores_ = nullptr;
    int32_t *device_filtered_indexes_ = nullptr;
    int32_t *device_num_detections_ = nullptr;

    float *device_sorted_scores_ = nullptr;
    int32_t *device_sorted_indexes_ = nullptr;
    float4 *device_sorted_bboxes_ = nullptr;
    float2 *device_sorted_landmarks_ = nullptr;
    uint32_t *device_suppression_mask_ = nullptr;

    float *device_final_scores_ = nullptr;
    float4 *device_final_bboxes_ = nullptr;
    float2 *device_final_landmarks_ = nullptr;
    int32_t *device_final_num_detections_ = nullptr;
    float *device_d2s_M_ = nullptr;
    float *device_images_face_ = nullptr;

    // TensorRT related members would go here (e.g., engine, context, etc.)
    IExecutionContext *context_detection_ = nullptr;
    cudaStream_t stream_;

    // threads
    int32_t threads_anchor_count_;
    int32_t blocks_anchor_count_;

    int32_t threads_top_k_;
    int32_t blocks_top_k_;

    // CUB Storages
    void *device_sort_storage_ = nullptr;
    size_t sort_storage_bytes_ = 0;

    void generateAnchors();
    void allocateBuffers();
    bool setDeviceSymbols(
        int32_t num_anchors,
        float confidence_threshold,
        int32_t top_k,
        float iou_threshold,
        float2 &host_centroid_destination,
        float2 *host_normalized_destination);
    float setCubMemories();
};