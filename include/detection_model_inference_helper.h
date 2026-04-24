#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>



using namespace nvinfer1;

enum class DetectionType
{
    SEARCH,
    TRACK
};

struct Slice {
    int x1, y1, x2, y2;
};

struct DetectionBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    float score = 0.0f;
};

struct WarpedFaceBatch {
    int count = 0;
    std::vector<float> faces = {};
};

struct DeviceWarpedFaceBatch {
    int count = 0;
    int batch_count = 0;
    const int32_t *batch_counts = nullptr;
    const float *device_faces = nullptr;
};

static constexpr int kNumRawOutputs = 9;

class DetectionModelInferenceHelper
{
public:

    static constexpr int kMaxSlices = 16; // This should be set according to the maximum expected number of slices (num_slices_x * num_slices_y)
    static constexpr int kLandmarkCount = 5; // Assuming 5 landmarks per detection, adjust if necessary
    static constexpr int kWarpedFaceSize = 112;
    static constexpr int kWarpedFaceChannels = 3;

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
        int32_t gap_y,
        int32_t min_box_length);

    ~DetectionModelInferenceHelper();

    void infer(const uint8_t *host_image,
               DetectionType detection_type,
               const uint8_t *log_visual_image = nullptr,
               int log_visual_width = 0,
               int log_visual_height = 0,
               int log_offset_x = 0,
               int log_offset_y = 0);
    std::vector<DetectionBox> getLastDetections(DetectionType detection_type) const;
    WarpedFaceBatch getLastWarpedFaces(DetectionType detection_type);
    DeviceWarpedFaceBatch getDeviceWarpedFacesForIdentification(DetectionType detection_type);
    cudaStream_t stream() const { return stream_; }
    uint8_t *hostInputBuffer(DetectionType detection_type) const;

private:
    static constexpr int kNumDetectionGraphs = 2;

    void allocateBuffers();
    void bindTensorAddresses(IExecutionContext *context, const char *context_name);
    void freeBuffers();
    void computeSlices();
    std::vector<int> makePositions(int dim, int win, int gap, int num);
    bool setDeviceSymbols(int camera_width,
                          int camera_height,
                          int model_width,
                          int model_height,
                          int num_anchors,
                          float confidence_threshold,
                          float iou_threshold,
                          int top_k,
                          int min_box_length);

    //Stages
    void stageUploadHostImage(const uint8_t *host_image, DetectionType detection_type);
    void stagePreprocessOnly(DetectionType detection_type);
    void stageInferenceOnly(DetectionType detection_type);
    void resetPostprocessBuffers(DetectionType detection_type);
    void stagePostprocess(DetectionType detection_type);
    void ensureCudaGraphCaptured(DetectionType detection_type);
    void launchCapturedGraph(DetectionType detection_type);
    void destroyCudaGraphs();
    size_t graphIndex(DetectionType detection_type) const;
    int batchSizeForDetectionType(DetectionType detection_type) const;
    const uint8_t *deviceInputForDetectionType(DetectionType detection_type) const;
    void launchPreprocessKernel(const uint8_t *src, float *dst, int batch, DetectionType detection_type, cudaStream_t stream);
    void launchFilterScoresKernel(int batch, cudaStream_t stream);
    void launchSortFilteredScoresKernel(int batch, cudaStream_t stream);
    void launchGatherAllKernel(int batch, cudaStream_t stream);
    void launchBitmaskNMSKernel(int batch, cudaStream_t stream);
    void launchGatherFinalResultKernel(int batch, cudaStream_t stream);
    void launchEstimateSimilarityKernel(int batch, DetectionType detection_type, cudaStream_t stream);
    void launchWarpAffineKernel(int batch, DetectionType detection_type, cudaStream_t stream);
    void launchPreprocessWarpedFacesForIdentificationKernel(int batch, cudaStream_t stream);

    // TOP Level model parameters
    std::string model_path_;
    std::vector<int> batch_sizes_ = {};
    int32_t model_input_height_ = 0;
    int32_t model_input_width_ = 0;
    int32_t camera_input_height_ = 0;
    int32_t camera_input_width_ = 0;
    std::vector<int32_t> strides_ = {};
    int32_t detection_top_k_ = 0;
    float detection_confidence_threshold_ = 0.0f;
    float detection_iou_threshold_ = 0.0f;
    int32_t min_box_length_ = 0;
    int32_t batch_size_cuda_graph_search_mode_ = 0;
    int32_t batch_size_cuda_graph_track_mode_ = 0;

    // Buffers for input and output data
    uint8_t *host_jetson_input_buffer_search_uint8_t_ = nullptr;   // full-frame mapped pinned memory for search mode
    uint8_t *host_jetson_input_buffer_track_uint8_t_ = nullptr;    // 640x640 mapped pinned memory for track mode
    uint8_t *device_jetson_ptr_search_uint8_t_ = nullptr;          // GPU alias of the full-frame search input buffer
    uint8_t *device_jetson_ptr_track_uint8_t_ = nullptr;           // GPU alias of the track-mode input buffer
    float   *device_jetson_input_buffer_float_ = nullptr;   // preprocessing output, TensorRT reads from here
    int32_t anchor_stack_ = 2;
    size_t anchor_count_ = 0;
    
    //Buffer sizes for input and output data
    size_t input_buffer_size_search_uint8_t_ = 0;
    size_t input_buffer_size_track_uint8_t_ = 0;
    size_t input_buffer_size_float_ = 0;

    //Cuda & TensorRT related members
    std::string device_name_;
    cudaDeviceProp device_prop_;
    ICudaEngine *engine_ = nullptr;
    IExecutionContext *context_detection_search_mode = nullptr;
    IExecutionContext *context_detection_track_mode = nullptr;
    std::string input_tensor_name_ = "";
    std::array<std::string, kNumRawOutputs> output_tensor_names_ = {};
    std::array<float *, kNumRawOutputs> device_model_output_buffers_ = {};
    std::array<size_t, kNumRawOutputs> output_element_counts_ = {};
    std::array<size_t, kNumRawOutputs> output_elements_per_batch_ = {};
    float *device_filtered_scores_ = nullptr;
    int32_t *device_filtered_indexes_ = nullptr;
    float *device_sorted_scores_ = nullptr;
    int32_t *device_sorted_indexes_ = nullptr;
    float4 *device_sorted_bboxes_ = nullptr;
    float2 *device_sorted_landmarks_ = nullptr;
    uint32_t *device_suppression_mask_ = nullptr;
    int32_t *device_num_selected_ = nullptr;
    float *device_final_scores_ = nullptr;
    int32_t *device_final_indexes_ = nullptr;
    float4 *device_final_bboxes_ = nullptr;
    float2 *device_final_landmarks_ = nullptr;
    int32_t *device_final_num_detections_ = nullptr;
    float *device_similarity_transforms_ = nullptr;
    float *device_warped_faces_ = nullptr;
    float *device_warped_faces_identification_ = nullptr;
    std::array<int32_t, kMaxSlices> host_identification_counts_ = {};
    void *device_sort_storage_ = nullptr;
    size_t sort_storage_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
    std::array<cudaGraph_t, kNumDetectionGraphs> cuda_graphs_ = {};
    std::array<cudaGraphExec_t, kNumDetectionGraphs> cuda_graph_execs_ = {};

    // Slicing
    std::vector<Slice> slices_ = {};
    int num_slices_x_ = 0;
    int num_slices_y_ = 0;
    int32_t gap_x_ = 0;
    int32_t gap_y_ = 0;
};
