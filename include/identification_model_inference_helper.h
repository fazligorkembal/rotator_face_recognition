#pragma once
#include <string>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include "inference_objects.hpp"

using namespace nvinfer1;
class IdentificationModelInferenceHelper
{
public:
    IdentificationModelInferenceHelper(
        ICudaEngine *&engine,
        int32_t batch_size,
        int32_t input_height,
        int32_t input_width,
        int32_t feature_dim,
        const std::string &root_folder,
        cudaStream_t stream);

    ~IdentificationModelInferenceHelper();

    std::vector<int> infer(
        float *device_images_face, SCRFDResults &results);

private:
    std::string model_path_;
    int32_t batch_size_ = 0;
    int32_t input_height_ = 0;
    int32_t input_width_ = 0;
    int32_t feature_dim_ = 0;

    // IO Buffers
    float *device_input_ = nullptr;
    float *device_output_features_ = nullptr;
    float *device_db_ = nullptr;

    // TensorRT objects
    IRuntime *runtime_ = nullptr;
    ICudaEngine *engine_ = nullptr;
    IExecutionContext *context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    cublasHandle_t handle_cublas_;

    void allocateBuffers();
    void generateDB(const std::string &root_folder);
};