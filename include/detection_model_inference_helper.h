#pragma once
#include <array>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>

struct Slice {
    int x1, y1, x2, y2;
};

enum class DetectionBenchmarkMode {
    kInferenceOnly = 1,
    kUploadPreprocessInference = 2,
    kUploadPreprocessInferenceNms = 3,
};

using namespace nvinfer1;

class DetectionModelInferenceHelper
{
public:
    static constexpr int kAlignedFaceWidth = 112;
    static constexpr int kAlignedFaceHeight = 112;
    static constexpr int kAlignedFaceChannels = 3;
    static constexpr int kLandmarkCount = 5;
    static constexpr int kAffineMatrixElements = 6;

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
    void benchmark(const uint8_t *host_image,
                   int batch_size,
                   int iterations,
                   DetectionBenchmarkMode mode);

    const float *getDeviceAlignedFaceBuffer() const
    {
        return device_face_crops_;
    }

    const float *getDeviceFaceAffineMatrices() const
    {
        return device_face_affine_matrices_;
    }

    const int32_t *getDeviceDetectionCounts() const
    {
        return device_final_num_detections_;
    }

private:
    static constexpr int kNumRawOutputs = 9;

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
    std::string input_tensor_name_ = "";
    std::array<std::string, kNumRawOutputs> output_tensor_names_ = {};
    IExecutionContext *context_detection_ = nullptr;
    cudaStream_t stream_;
    bool is_discreate_gpu_ = false;
    std::array<cudaGraph_t, 3> cuda_graphs_ = {};
    std::array<cudaGraphExec_t, 3> cuda_graph_execs_ = {};
    int cuda_graph_batch_size_ = 0;

    // Buffers for input and output data
    int32_t anchor_stack_ = 2;
    size_t anchor_count_ = 0;
    size_t input_buffer_size_uint8_t_ = 0;
    size_t input_buffer_size_float_ = 0;
    
    uint8_t *host_jetson_input_buffer_uint8_t_ = nullptr;   // mapped pinned memory, CPU writes here
    uint8_t *device_jetson_ptr_uint8_t_ = nullptr;          // GPU-side pointer to the same physical memory
    float   *device_jetson_input_buffer_float_ = nullptr;   // preprocessing output, TensorRT reads from here
    std::array<float *, kNumRawOutputs> device_model_output_buffers_ = {};
    std::array<size_t, kNumRawOutputs> output_element_counts_ = {};
    std::array<size_t, kNumRawOutputs> output_elements_per_batch_ = {};
    std::array<int, kNumRawOutputs> output_batch_dims_ = {};
    std::array<bool, kNumRawOutputs> output_has_batch_dim_ = {};
    float *device_filtered_scores_ = nullptr;
    int32_t *device_filtered_indexes_ = nullptr;
    float *device_sorted_scores_ = nullptr;
    int32_t *device_sorted_indexes_ = nullptr;
    float4 *device_sorted_bboxes_ = nullptr;
    float2 *device_sorted_landmarks_ = nullptr;
    float *device_final_scores_ = nullptr;
    int32_t *device_final_indexes_ = nullptr;
    float4 *device_final_bboxes_ = nullptr;
    float2 *device_final_landmarks_ = nullptr;
    float *device_face_affine_matrices_ = nullptr;
    float *device_face_crops_ = nullptr;
    uint32_t *device_suppression_mask_ = nullptr;
    int32_t *device_num_selected_ = nullptr;
    int32_t *device_final_num_detections_ = nullptr;
    void *device_sort_storage_ = nullptr;
    size_t sort_storage_bytes_ = 0;

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
    void stageUploadHostImage(const uint8_t *host_image);
    void stagePreprocessOnly(int batch_size);
    void stageUploadAndPreprocess(const uint8_t *host_image, int batch_size);
    void stageInferenceOnly();
    void resetPostprocessBuffers(int batch_size);
    void stagePostprocess(int batch_size);
    void stageFaceAlignment(int batch_size);
    size_t graphModeIndex(DetectionBenchmarkMode mode) const;
    bool shouldUseCudaGraph(int batch_size) const;
    void ensureSortStorageInitialized();
    void ensureCudaGraphCaptured(DetectionBenchmarkMode mode, int batch_size);
    void launchCapturedGraph(DetectionBenchmarkMode mode);
    void destroyCudaGraphs();
    bool setDeviceSymbols(std::vector<int> slice_coordinates, int camera_width, int camera_height, int model_width, int model_height, int num_anchors, float confidence_threshold, float iou_threshold, int top_k);
    void launchPreprocessKernel(const uint8_t *src, float *dst, int batch, cudaStream_t stream);
    void launchFilterScoresKernel(int batch, cudaStream_t stream);
    void launchSortFilteredScoresKernel(int batch, cudaStream_t stream);
    void launchGatherAllKernel(int batch, cudaStream_t stream);
    void launchBitmaskNMSKernel(int batch, cudaStream_t stream);
    void launchGatherFinalResultKernel(int batch, cudaStream_t stream);
    void launchEstimateSimilarityKernel(int batch, cudaStream_t stream);
    void launchWarpAffineKernel(int batch, cudaStream_t stream);
#ifdef LOGRESULTS
    void dumpLogResults(int batch_size);
#endif

};
