#pragma once

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

struct IdentificationMatch
{
    int face_index = -1;
    std::string label = {};
    float average_similarity = 0.0f;
};

class IdentificationModelInferenceHelper
{
public:
    IdentificationModelInferenceHelper(
        nvinfer1::ICudaEngine *engine,
        int32_t max_batch_size,
        int32_t input_height,
        int32_t input_width,
        const std::string &selected_faces_root,
        cudaStream_t stream = nullptr);

    ~IdentificationModelInferenceHelper();

    int32_t dbCount() const { return db_count_; }
    int32_t featureDim() const { return feature_dim_; }
    const float *deviceDb() const { return device_db_; }
    const std::vector<std::string> &dbLabels() const { return db_labels_; }
    std::vector<IdentificationMatch> matchWarpedFaces(
        const float *device_warped_faces,
        const int32_t *batch_counts,
        int detection_batch_count,
        int top_k,
        float face_threshold);

private:
    int32_t max_batch_size_ = 0;
    int32_t input_height_ = 0;
    int32_t input_width_ = 0;
    int32_t feature_dim_ = 0;
    int32_t db_count_ = 0;
    int32_t current_input_batch_ = -1;

    float *device_input_ = nullptr;
    float *device_output_features_ = nullptr;
    float *device_db_ = nullptr;
    float *device_similarity_scores_ = nullptr;
    int32_t *device_label_group_offsets_ = nullptr;
    int32_t *device_label_group_counts_ = nullptr;
    int32_t *device_label_group_indexes_ = nullptr;
    int32_t *device_best_label_indexes_ = nullptr;
    float *device_best_label_scores_ = nullptr;

    nvinfer1::ICudaEngine *engine_ = nullptr;
    nvinfer1::IExecutionContext *context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;
    cublasHandle_t cublas_handle_ = nullptr;

    std::vector<std::string> db_labels_ = {};
    std::vector<std::string> unique_labels_ = {};
    std::vector<std::vector<int32_t>> label_groups_ = {};
    std::vector<int> host_source_face_indices_ = {};
    std::vector<int32_t> host_best_label_indexes_ = {};
    std::vector<float> host_best_label_scores_ = {};

    void allocateBuffers();
    void buildDatabase(const std::string &selected_faces_root);
    void buildLabelGroups();
};
