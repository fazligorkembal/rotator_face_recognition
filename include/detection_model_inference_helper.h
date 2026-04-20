#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>

struct Slice {
    int x1, y1, x2, y2;
};

using namespace nvinfer1;

class DetectionModelInferenceHelper
{
public:
    DetectionModelInferenceHelper(
        ICudaEngine *&engine,
        std::vector<int> batch_sizes,
        int32_t model_input_height,
        int32_t model_input_width,
        std::vector<int32_t> strides,
        int32_t top_k,
        float confidence_threshold,
        float iou_threshold,
        int32_t camera_input_height,
        int32_t camera_input_width,
        int32_t num_slices_x,
        int32_t num_slices_y,
        int32_t gap_x,
        int32_t gap_y);

    ~DetectionModelInferenceHelper();
    void infer(const uint8_t *host_image, int batch_size);

   

private:

    // TOP Level model parameters
    std::string model_path_;
    std::vector<int> batch_sizes_ = {};
    int32_t model_input_height_ = 0;
    int32_t model_input_width_ = 0;
    int32_t camera_input_height_ = 0;
    int32_t camera_input_width_ = 0;
    std::vector<int32_t> strides_ = {};
    int32_t top_k_ = 0;
    float confidence_threshold_ = 0.0f;
    float iou_threshold_ = 0.0f;

    //Cuda & TensorRT related members
    cudaDeviceProp device_prop_ = {};
    std::string device_name_ = "";
    IExecutionContext *context_detection_ = nullptr;
    cudaStream_t stream_;
    bool is_discreate_gpu_ = false;

    // Buffers for input and output data
    size_t input_buffer_size_uint8_t_ = 0;
    size_t input_buffer_size_float_ = 0;
    uint8_t *host_jetson_input_buffer_uint8_t_ = nullptr;   // mapped pinned memory, CPU writes here
    uint8_t *device_jetson_ptr_uint8_t_ = nullptr;          // GPU-side pointer to the same physical memory
    float   *device_jetson_input_buffer_float_ = nullptr;   // preprocessing output, TensorRT reads from here

    // Slicing
    std::vector<Slice> slices_ = {};
    int num_slices_x_ = 0;
    int num_slices_y_ = 0;
    int32_t gap_x_ = 0;
    int32_t gap_y_ = 0;

    // Functions for managing buffers and CUDA stream
    void allocateBuffers();
    void freeBuffers();
    void computeSlices();
    std::vector<int> makePositions(int dim, int win, int gap, int num);
    bool setDeviceSymbols(std::vector<int> slice_coordinates);
    void launchPreprocessKernel(const uint8_t *src, float *dst, int batch, cudaStream_t stream);

};