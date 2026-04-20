#include "spd_logger_helper.h"
#include "detection_model_inference_helper.h"
#include <cstring>

#ifdef GENERATE_TXT
#include <fstream>
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
    LOG_INFO("\n\n");
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
    cudaStreamCreate(&stream_);
    allocateBuffers();
    computeSlices();

    std::vector<int> coords;
    coords.reserve(slices_.size() * 4);
    for (const auto &s : slices_)
    {
        coords.push_back(s.x1);
        coords.push_back(s.y1);
        coords.push_back(s.x2);
        coords.push_back(s.y2);
    }
    if (!setDeviceSymbols(coords))
    {
        LOG_ERROR("Failed to copy slice coordinates to device constant memory");
        std::exit(EXIT_FAILURE);
    }
    LOG_INFO("Slice coordinates copied to device constant memory");

}

DetectionModelInferenceHelper::~DetectionModelInferenceHelper(){
    LOG_INFO("Destroying DetectionModelInferenceHelper");
    freeBuffers();
}


// Allocates GPU buffers for model inputs and outputs based on engine binding shapes
void DetectionModelInferenceHelper::allocateBuffers() {
    LOG_INFO("***** allocateBuffers *****");

    input_buffer_size_uint8_t_ = static_cast<size_t>(camera_input_height_) * camera_input_width_ * 3;
    input_buffer_size_float_   = static_cast<size_t>(batch_sizes_[2]) * 3 * model_input_height_ * model_input_width_;

    // uint8 input: mapped pinned memory → GPU accesses same physical RAM, no PCIe transfer
    cudaHostAlloc(&host_jetson_input_buffer_uint8_t_, input_buffer_size_uint8_t_, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&device_jetson_ptr_uint8_t_, host_jetson_input_buffer_uint8_t_, 0);
    LOG_DEBUG("\tAllocated zero-copy uint8 input buffer: {} mb", input_buffer_size_uint8_t_ / (1024.0 * 1024.0));

    // float32 CHW buffer: preprocessing kernel writes here, TensorRT reads from here
    cudaMalloc(&device_jetson_input_buffer_float_, input_buffer_size_float_ * sizeof(float));
    LOG_DEBUG("\tAllocated device float input buffer: {} mb", input_buffer_size_float_ * sizeof(float) / (1024.0 * 1024.0));
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

    LOG_INFO("Computed {} slices ({}x{})", slices_.size(), num_slices_x_, num_slices_y_);
    for (size_t i = 0; i < slices_.size(); ++i) {
        const auto &s = slices_[i];
        LOG_DEBUG("\tSlice {}: [{},{} -> {},{}]", i, s.x1, s.y1, s.x2, s.y2);
    }
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

void DetectionModelInferenceHelper::infer(const uint8_t *host_image, int batch_size) {
    std::memcpy(host_jetson_input_buffer_uint8_t_, host_image, input_buffer_size_uint8_t_);

    // preprocessing: uint8 BGR HWC 1920x1080 → float32 RGB CHW 640x640, kernel handles resize
    launchPreprocessKernel(device_jetson_ptr_uint8_t_, device_jetson_input_buffer_float_, batch_size, stream_);

    cudaStreamSynchronize(stream_);

    // TODO: TensorRT enqueueV3
}