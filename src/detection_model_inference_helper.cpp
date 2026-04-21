#include "spd_logger_helper.h"
#include "detection_model_inference_helper.h"
#include <cstring>

#ifdef GENERATE_TXT
#include <fstream>
#endif

#ifdef DEBUG_PREPROCESS_VIS
#include <opencv2/opencv.hpp>
#endif

DetectionModelInferenceHelper::DetectionModelInferenceHelper(
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
        int32_t gap_y)
{

    //INFO
    LOG_INFO("***** Constructor DetectionModelInferenceHelper *****");
    LOG_DEBUG("\tBatch sizes: min: {}, medium: {}, max: {}", batch_sizes[0], batch_sizes[1], batch_sizes[2]);
    LOG_DEBUG("\tModel input height: {}", model_input_height);
    LOG_DEBUG("\tModel input width: {}", model_input_width);
    LOG_DEBUG("\tStrides: {} {} {}", strides[0], strides[1], strides[2]);
    LOG_DEBUG("\tTop K: {}", top_k);
    LOG_DEBUG("\tConfidence threshold: {}", confidence_threshold);
    LOG_DEBUG("\tIoU threshold: {}", iou_threshold);
    LOG_DEBUG("\tCamera input height: {}", camera_input_height);
    LOG_DEBUG("\tCamera input width: {}", camera_input_width);

    cudaGetDeviceProperties(&device_prop_, 0);
    device_name_ = device_prop_.name;
    is_discreate_gpu_ = !device_prop_.integrated && device_name_.find("Jetson") == std::string::npos;
    if (is_discreate_gpu_) {
        LOG_INFO("\tRunning on discrete GPU: {}", device_name_);
    } else {
        LOG_INFO("\tRunning on Jetson: {}", device_name_);
    }

    if (batch_sizes.size() != 3) {
        LOG_ERROR("Expected exactly 3 batch sizes, got {}", batch_sizes.size());
        std::exit(EXIT_FAILURE);
    }
    //INFO

    batch_sizes_ = batch_sizes;
    model_input_height_ = model_input_height;
    model_input_width_ = model_input_width;
    camera_input_height_ = camera_input_height;
    camera_input_width_ = camera_input_width;
    num_slices_x_ = num_slices_x;
    num_slices_y_ = num_slices_y;
    gap_x_ = gap_x;
    gap_y_ = gap_y;
    strides_ = strides;
    top_k_ = top_k;
    confidence_threshold_ = confidence_threshold;
    iou_threshold_ = iou_threshold;


    context_detection_ = engine->createExecutionContext();
    if (!context_detection_) {
        LOG_ERROR("Failed to create execution context for detection model");
        std::exit(EXIT_FAILURE);
    }
    cudaStreamCreate(&stream_);
    allocateBuffers();
    
    const char *input_tensor_name = engine->getIOTensorName(0);
    const char *output_tensor_name_scores = engine->getIOTensorName(1);
    const char *output_tensor_name_bboxes = engine->getIOTensorName(2);
    const char *output_tensor_name_landmarks = engine->getIOTensorName(3);

    context_detection_->setTensorAddress(input_tensor_name, device_jetson_input_buffer_float_);
    context_detection_->setTensorAddress(output_tensor_name_scores, device_model_output_scores_);
    context_detection_->setTensorAddress(output_tensor_name_bboxes, device_model_output_bboxes_);
    context_detection_->setTensorAddress(output_tensor_name_landmarks, device_model_output_landmarks_);

}

DetectionModelInferenceHelper::~DetectionModelInferenceHelper(){
    LOG_INFO("Destroying DetectionModelInferenceHelper");
    freeBuffers();
}


void DetectionModelInferenceHelper::infer(const uint8_t *host_image, int batch_size) {
    std::memcpy(host_jetson_input_buffer_uint8_t_, host_image, input_buffer_size_uint8_t_);
    
    launchPreprocessKernel(device_jetson_ptr_uint8_t_, device_jetson_input_buffer_float_, batch_size, stream_);    
    context_detection_->enqueueV3(stream_);
    


    cudaStreamSynchronize(stream_);


#ifdef DEBUG_PREPROCESS_VIS
    {
        const int H = model_input_height_;
        const int W = model_input_width_;
        const int plane = H * W;
        const int vis_batch = std::min(batch_size, 6);

        std::vector<float> host_float(vis_batch * 3 * plane);
        cudaMemcpy(host_float.data(), device_jetson_input_buffer_float_,
                   vis_batch * 3 * plane * sizeof(float), cudaMemcpyDeviceToHost);

        for (int b = 0; b < vis_batch; ++b) {
            const float *R = host_float.data() + b * 3 * plane + 0 * plane;
            const float *G = host_float.data() + b * 3 * plane + 1 * plane;
            const float *B = host_float.data() + b * 3 * plane + 2 * plane;

            cv::Mat img(H, W, CV_8UC3);
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int i = y * W + x;
                    img.at<cv::Vec3b>(y, x) = {
                        static_cast<uint8_t>(B[i] * 128.0f + 127.5f),
                        static_cast<uint8_t>(G[i] * 128.0f + 127.5f),
                        static_cast<uint8_t>(R[i] * 128.0f + 127.5f)
                    };
                }
            }
            std::string win = "slice_" + std::to_string(b);
            cv::imshow(win, img);
        }
        cv::waitKey(0);
    }
#endif
}

std::vector<int> DetectionModelInferenceHelper::makePositions(int dim, int win, int gap, int num) {
    std::vector<int> pos;
    int step = win - gap;
    int span = (num - 1) * step + win;
    int pad  = (dim - span) / 2;
    for (int i = 0; i < num; ++i)
        pos.push_back(pad + i * step);
    return pos;
}

void DetectionModelInferenceHelper::computeSlices() {
    auto xs = makePositions(camera_input_width_,  model_input_width_,  gap_x_, num_slices_x_);
    auto ys = makePositions(camera_input_height_, model_input_height_, gap_y_, num_slices_y_);

    num_slices_x_ = static_cast<int>(xs.size());
    num_slices_y_ = static_cast<int>(ys.size());
    slices_.reserve(num_slices_x_ * num_slices_y_);

    for (int y : ys)
        for (int x : xs)
            slices_.push_back({x, y, x + model_input_width_, y + model_input_height_});

    LOG_INFO("\tComputed {} slices ({}x{})", slices_.size(), num_slices_x_, num_slices_y_);
    for (size_t i = 0; i < slices_.size(); ++i) {
        const auto &s = slices_[i];
        LOG_DEBUG("\t\tSlice {}: [{},{} -> {},{}]", i, s.x1, s.y1, s.x2, s.y2);
    }
}

// Allocates GPU buffers for model inputs and outputs based on engine binding shapes
void DetectionModelInferenceHelper::allocateBuffers() {
    LOG_INFO("\t***** allocateBuffers *****");

    float memory_usage = 0;
    
    // uint8 input: mapped pinned memory → GPU accesses same physical RAM, no PCIe transfer
    input_buffer_size_uint8_t_ = static_cast<size_t>(camera_input_height_) * camera_input_width_ * 3 * sizeof(uint8_t);
    cudaHostAlloc(&host_jetson_input_buffer_uint8_t_, input_buffer_size_uint8_t_, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&device_jetson_ptr_uint8_t_, host_jetson_input_buffer_uint8_t_, 0);
    
    memory_usage += input_buffer_size_uint8_t_ / (1024.0 * 1024.0);

    // float32 CHW buffer: preprocessing kernel writes here, TensorRT reads from here
    input_buffer_size_float_   = static_cast<size_t>(batch_sizes_[2]) * 3 * model_input_height_ * model_input_width_ * sizeof(float);
    cudaMalloc(&device_jetson_input_buffer_float_, input_buffer_size_float_);
    
    memory_usage += input_buffer_size_float_ / (1024.0 * 1024.0);

    for (auto stride : strides_)
    {
        int32_t feature_map_height = model_input_height_ / stride;
        int32_t feature_map_width = model_input_width_ / stride;
        int32_t anchors_per_feature_map = feature_map_height * feature_map_width;
        anchor_count_ += anchors_per_feature_map * anchor_stack_;
        LOG_DEBUG("\t\t\tStride {}: feature map size: {}x{}, anchors per feature map: {}, total anchors so far: {}",
                  stride, feature_map_width, feature_map_height, anchors_per_feature_map * anchor_stack_, anchor_count_);
    }
    

    size_t size_device_model_output_scores = batch_sizes_[2] * anchor_count_ * sizeof(float);
    cudaMalloc(&device_model_output_scores_, size_device_model_output_scores);    
    memory_usage += size_device_model_output_scores / (1024.0 * 1024.0);


    int32_t size_device_model_output_bboxes = batch_sizes_[2] * anchor_count_ * sizeof(float4);
    cudaMallocAsync(&device_model_output_bboxes_, size_device_model_output_bboxes, stream_);
    memory_usage += size_device_model_output_bboxes / 1024.0 / 1024.0;

    int32_t size_device_model_output_landmarks = batch_sizes_[2] * anchor_count_ * 5 * sizeof(float2);
    cudaMallocAsync(&device_model_output_landmarks_, size_device_model_output_landmarks, stream_);
    memory_usage += size_device_model_output_landmarks / 1024.0 / 1024.0;

    int32_t size_device_jetson_input_buffer_float_ = batch_sizes_[2] * 3 * model_input_height_ * model_input_width_ * sizeof(float);
    cudaMallocAsync(&device_jetson_input_buffer_float_, size_device_jetson_input_buffer_float_, stream_);
    memory_usage += size_device_jetson_input_buffer_float_ / 1024.0 / 1024.0;

    
    LOG_DEBUG("\t\tAllocated zero-copy uint8 input buffer: {} mb", input_buffer_size_uint8_t_ / (1024.0 * 1024.0));
    LOG_DEBUG("\t\tAllocated device float input buffer: {} mb", input_buffer_size_float_ / (1024.0 * 1024.0));
    LOG_DEBUG("\t\tTotal anchors across all strides: {}", anchor_count_);
    LOG_DEBUG("\t\tAllocated device buffer for model output scores: {} mb", size_device_model_output_scores / (1024.0 * 1024.0));
    LOG_DEBUG("\t\tAllocated device buffer for model output bboxes: {} mb", size_device_model_output_bboxes / (1024.0 * 1024.0));
    LOG_DEBUG("\t\tAllocated device buffer for model output landmarks: {} mb", size_device_model_output_landmarks / (1024.0 * 1024.0));
    LOG_DEBUG("\tCalculated size for device float input buffer: {} mb", size_device_jetson_input_buffer_float_ / (1024.0 * 1024.0));
    LOG_INFO("\tTotal GPU memory allocated for buffers: {} mb", memory_usage);
    

}

void DetectionModelInferenceHelper::freeBuffers() {
    LOG_INFO("***** freeBuffers *****");

    if (host_jetson_input_buffer_uint8_t_) {
        cudaFreeHost(host_jetson_input_buffer_uint8_t_);
        host_jetson_input_buffer_uint8_t_ = nullptr;
        device_jetson_ptr_uint8_t_ = nullptr; // invalidated when host is freed
    }
    if (device_jetson_input_buffer_float_) {
        cudaFree(device_jetson_input_buffer_float_);
        device_jetson_input_buffer_float_ = nullptr;
    }
}
