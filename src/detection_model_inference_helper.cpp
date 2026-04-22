#include "spd_logger_helper.h"
#include "detection_model_inference_helper.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#if defined(GENERATE_TXT) || defined(LOGRESULTS)
#include <fstream>
#endif

#ifdef LOGRESULTS
#include <filesystem>
#endif

#if defined(DEBUG_PREPROCESS_VIS) || defined(LOGRESULTS)
#include <opencv2/opencv.hpp>
#endif

namespace {

std::string dimsToString(const nvinfer1::Dims &dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

std::string dataTypeToString(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return "kFLOAT";
        case nvinfer1::DataType::kHALF: return "kHALF";
        case nvinfer1::DataType::kINT8: return "kINT8";
        case nvinfer1::DataType::kINT32: return "kINT32";
        case nvinfer1::DataType::kBOOL: return "kBOOL";
#if NV_TENSORRT_MAJOR >= 8
        case nvinfer1::DataType::kUINT8: return "kUINT8";
#endif
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kFP8: return "kFP8";
        case nvinfer1::DataType::kBF16: return "kBF16";
        case nvinfer1::DataType::kINT64: return "kINT64";
#endif
        default: return "UNKNOWN";
    }
}

int logicalSliceToEngineBatchIndex(int logical_slice_index, int num_slices_x, int num_slices_y) {
    if (num_slices_x <= 0 || num_slices_y <= 0) {
        return logical_slice_index;
    }

    const int row = logical_slice_index / num_slices_x;
    const int col = logical_slice_index % num_slices_x;
    return col * num_slices_y + row;
}

template <typename T>
std::vector<T> copyTensorSliceToHost(const T *device_ptr, size_t slice_index, size_t element_count) {
    std::vector<T> host(element_count);
    cudaMemcpy(host.data(),
               device_ptr + slice_index * element_count,
               element_count * sizeof(T),
               cudaMemcpyDeviceToHost);
    return host;
}

template <typename T>
void reconstructTwoRowAnchorInterleave(const std::vector<T> &raw_top,
                                       const std::vector<T> &raw_bottom,
                                       std::vector<T> &dst,
                                       int row,
                                       size_t elems_per_anchor) {
    const size_t anchor_count = dst.size() / elems_per_anchor;
    for (size_t anchor_idx = 0; anchor_idx < anchor_count; anchor_idx += 2) {
        const size_t even_offset = anchor_idx * elems_per_anchor;
        const size_t odd_offset = (anchor_idx + 1) * elems_per_anchor;

        if (row == 0) {
            std::copy_n(raw_top.data() + even_offset, elems_per_anchor, dst.data() + even_offset);
            std::copy_n(raw_bottom.data() + even_offset, elems_per_anchor, dst.data() + odd_offset);
        } else {
            std::copy_n(raw_top.data() + odd_offset, elems_per_anchor, dst.data() + even_offset);
            std::copy_n(raw_bottom.data() + odd_offset, elems_per_anchor, dst.data() + odd_offset);
        }
    }
}

} // namespace

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
    computeSlices();

    std::vector<int> coords;
    coords.reserve(slices_.size() * 4);
    for (const auto &s : slices_) {
        coords.push_back(s.x1);
        coords.push_back(s.y1);
        coords.push_back(s.x2);
        coords.push_back(s.y2);
    }
    if (!setDeviceSymbols(coords,
                          camera_input_width_, camera_input_height_,
                          model_input_width_,  model_input_height_,
                          anchor_count_,
                          confidence_threshold_,
                          top_k_)) {
        LOG_ERROR("Failed to copy slice coordinates to device constant memory");
        std::exit(EXIT_FAILURE);
    }
    LOG_INFO("Slice coordinates copied to device constant memory");

    const char *input_tensor_name           = engine->getIOTensorName(0);
    const char *output_tensor_name_scores   = engine->getIOTensorName(1);
    const char *output_tensor_name_bboxes   = engine->getIOTensorName(2);
    const char *output_tensor_name_landmarks= engine->getIOTensorName(3);

    context_detection_->setTensorAddress(input_tensor_name,            device_jetson_input_buffer_float_);
    context_detection_->setTensorAddress(output_tensor_name_scores,    device_model_output_scores_);
    context_detection_->setTensorAddress(output_tensor_name_bboxes,    device_model_output_bboxes_);
    context_detection_->setTensorAddress(output_tensor_name_landmarks, device_model_output_landmarks_);

    context_detection_->setInputShape(input_tensor_name, nvinfer1::Dims4{batch_sizes_[1], 3, model_input_height_, model_input_width_});

    int nb_tensors = engine->getNbIOTensors();
    LOG_DEBUG("Engine has {} IO tensors:", nb_tensors);
    for (int i = 0; i < nb_tensors; ++i) {
        const char *tensor_name = engine->getIOTensorName(i);
        const auto engine_dims = engine->getTensorShape(tensor_name);
        const auto context_dims = context_detection_->getTensorShape(tensor_name);
        const auto dtype = engine->getTensorDataType(tensor_name);
        const auto bytes_per_component = engine->getTensorBytesPerComponent(tensor_name);
        const auto components_per_element = engine->getTensorComponentsPerElement(tensor_name);
        const auto vectorized_dim = engine->getTensorVectorizedDim(tensor_name);
        const char *format_desc = engine->getTensorFormatDesc(tensor_name);
        LOG_DEBUG("  [{}] {} | engine_dims={} | context_dims={} | dtype={} | bytes/component={} | components/element={} | vectorized_dim={} | format={}",
                  i,
                  tensor_name,
                  dimsToString(engine_dims),
                  dimsToString(context_dims),
                  dataTypeToString(dtype),
                  bytes_per_component,
                  components_per_element,
                  vectorized_dim,
                  format_desc ? format_desc : "<null>");
    }

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

#ifdef LOGRESULTS
    {
        const int H = model_input_height_;
        const int W = model_input_width_;
        const int plane = H * W;
        const int num_elements = 3 * plane;
        const int num_slices = std::min(batch_size, num_slices_x_ * num_slices_y_);
        const bool use_column_major_output_order_fix =
            batch_size == num_slices_x_ * num_slices_y_ && batch_size > 1;
        const bool use_two_row_anchor_reconstruction =
            use_column_major_output_order_fix && num_slices_y_ == anchor_stack_;

        std::filesystem::path build_dir = std::filesystem::current_path();
        if (build_dir.filename() != "build") {
            build_dir /= "build";
        }
        std::filesystem::create_directories(build_dir);
        const cv::Mat full_frame(camera_input_height_, camera_input_width_, CV_8UC3,
                                 const_cast<uint8_t *>(host_image));
        cv::Mat full_vis = full_frame.clone();

        for (int b = 0; b < num_slices; ++b) {
            const int engine_batch_index =
                use_column_major_output_order_fix ? logicalSliceToEngineBatchIndex(b, num_slices_x_, num_slices_y_) : b;
            const auto &slice = slices_[b];
            std::vector<float> host_float(num_elements);
            cudaMemcpy(host_float.data(),
                       device_jetson_input_buffer_float_ + b * num_elements,
                       num_elements * sizeof(float),
                       cudaMemcpyDeviceToHost);

            {
                std::ofstream ofs(build_dir / ("engine_input_slice" + std::to_string(b) + ".txt"));
                for (int i = 0; i < num_elements; ++i) {
                    ofs << host_float[i] << "\n";
                }
            }

            {
                const float *R = host_float.data() + 0 * plane;
                const float *G = host_float.data() + 1 * plane;
                const float *B = host_float.data() + 2 * plane;
                cv::Mat img(H, W, CV_8UC3);
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        const int i = y * W + x;
                        img.at<cv::Vec3b>(y, x) = {
                            static_cast<uint8_t>(std::clamp(B[i] * 128.0f + 127.5f, 0.0f, 255.0f)),
                            static_cast<uint8_t>(std::clamp(G[i] * 128.0f + 127.5f, 0.0f, 255.0f)),
                            static_cast<uint8_t>(std::clamp(R[i] * 128.0f + 127.5f, 0.0f, 255.0f))
                        };
                    }
                }
                cv::imwrite((build_dir / ("engine_input_slice" + std::to_string(b) + ".png")).string(), img);
            }

            std::vector<float> host_scores(anchor_count_);
            std::vector<float4> host_bboxes(anchor_count_);
            std::vector<float2> host_landmarks(anchor_count_ * 5);

            if (use_two_row_anchor_reconstruction) {
                const int row = b / num_slices_x_;
                const int col = b % num_slices_x_;
                const int top_logical_slice = col;
                const int bottom_logical_slice = col + num_slices_x_;
                const int top_engine_batch_index =
                    logicalSliceToEngineBatchIndex(top_logical_slice, num_slices_x_, num_slices_y_);
                const int bottom_engine_batch_index =
                    logicalSliceToEngineBatchIndex(bottom_logical_slice, num_slices_x_, num_slices_y_);

                const auto raw_top_scores =
                    copyTensorSliceToHost(device_model_output_scores_, static_cast<size_t>(top_engine_batch_index), anchor_count_);
                const auto raw_bottom_scores =
                    copyTensorSliceToHost(device_model_output_scores_, static_cast<size_t>(bottom_engine_batch_index), anchor_count_);
                const auto raw_top_bboxes =
                    copyTensorSliceToHost(device_model_output_bboxes_, static_cast<size_t>(top_engine_batch_index), anchor_count_);
                const auto raw_bottom_bboxes =
                    copyTensorSliceToHost(device_model_output_bboxes_, static_cast<size_t>(bottom_engine_batch_index), anchor_count_);
                const auto raw_top_landmarks =
                    copyTensorSliceToHost(device_model_output_landmarks_, static_cast<size_t>(top_engine_batch_index), anchor_count_ * 5);
                const auto raw_bottom_landmarks =
                    copyTensorSliceToHost(device_model_output_landmarks_, static_cast<size_t>(bottom_engine_batch_index), anchor_count_ * 5);

                reconstructTwoRowAnchorInterleave(raw_top_scores, raw_bottom_scores, host_scores, row, 1);
                reconstructTwoRowAnchorInterleave(raw_top_bboxes, raw_bottom_bboxes, host_bboxes, row, 1);
                reconstructTwoRowAnchorInterleave(raw_top_landmarks, raw_bottom_landmarks, host_landmarks, row, 5);
            } else {
                cudaMemcpy(host_scores.data(),
                           device_model_output_scores_ + static_cast<size_t>(engine_batch_index) * anchor_count_,
                           anchor_count_ * sizeof(float),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(host_bboxes.data(),
                           device_model_output_bboxes_ + static_cast<size_t>(engine_batch_index) * anchor_count_,
                           anchor_count_ * sizeof(float4),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(host_landmarks.data(),
                           device_model_output_landmarks_ + static_cast<size_t>(engine_batch_index) * anchor_count_ * 5,
                           anchor_count_ * 5 * sizeof(float2),
                           cudaMemcpyDeviceToHost);
            }

            {
                std::ofstream sofs(build_dir / ("engine_output_slice" + std::to_string(b) + "_scores.txt"));
                for (size_t i = 0; i < anchor_count_; ++i) {
                    sofs << host_scores[i] << "\n";
                }
            }

            {
                std::ofstream bofs(build_dir / ("engine_output_slice" + std::to_string(b) + "_bboxes.txt"));
                std::ofstream lofs(build_dir / ("engine_output_slice" + std::to_string(b) + "_landmarks.txt"));
                cv::Mat slice_vis = full_frame(cv::Rect(slice.x1, slice.y1, W, H)).clone();
                int drawn_boxes = 0;

                size_t anchor_offset = 0;
                for (auto stride : strides_) {
                    const int feature_map_height = model_input_height_ / stride;
                    const int feature_map_width = model_input_width_ / stride;
                    const size_t anchors_this_stride =
                        static_cast<size_t>(feature_map_height) * feature_map_width * anchor_stack_;

                    for (int y = 0; y < feature_map_height; ++y) {
                        for (int x = 0; x < feature_map_width; ++x) {
                            for (int a = 0; a < anchor_stack_; ++a) {
                                const size_t idx =
                                    anchor_offset + (static_cast<size_t>(y) * feature_map_width + x) * anchor_stack_ + a;
                                const float cx = static_cast<float>(x * stride);
                                const float cy = static_cast<float>(y * stride);

                                const float l = host_bboxes[idx].x * stride;
                                const float t = host_bboxes[idx].y * stride;
                                const float r = host_bboxes[idx].z * stride;
                                const float bb = host_bboxes[idx].w * stride;
                                const float score = host_scores[idx];

                                const float x1 = cx - l;
                                const float y1 = cy - t;
                                const float x2 = cx + r;
                                const float y2 = cy + bb;

                                bofs << x1 << "\n";
                                bofs << y1 << "\n";
                                bofs << x2 << "\n";
                                bofs << y2 << "\n";

                                for (int kp = 0; kp < 5; ++kp) {
                                    const float2 &raw_kp = host_landmarks[idx * 5 + kp];
                                    const float kp_x = cx + raw_kp.x * stride;
                                    const float kp_y = cy + raw_kp.y * stride;
                                    lofs << kp_x << "\n";
                                    lofs << kp_y << "\n";

                                    if (score >= confidence_threshold_) {
                                        cv::circle(slice_vis,
                                                   cv::Point(static_cast<int>(std::round(kp_x)),
                                                             static_cast<int>(std::round(kp_y))),
                                                   2, cv::Scalar(255, 0, 0), -1);
                                        cv::circle(full_vis,
                                                   cv::Point(slice.x1 + static_cast<int>(std::round(kp_x)),
                                                             slice.y1 + static_cast<int>(std::round(kp_y))),
                                                   2, cv::Scalar(255, 0, 0), -1);
                                    }
                                }

                                if (score >= confidence_threshold_) {
                                    const cv::Point slice_tl(static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1)));
                                    const cv::Point slice_br(static_cast<int>(std::round(x2)), static_cast<int>(std::round(y2)));
                                    const cv::Point full_tl(slice.x1 + slice_tl.x, slice.y1 + slice_tl.y);
                                    const cv::Point full_br(slice.x1 + slice_br.x, slice.y1 + slice_br.y);

                                    cv::rectangle(slice_vis, slice_tl, slice_br, cv::Scalar(0, 255, 0), 2);
                                    cv::rectangle(full_vis, full_tl, full_br, cv::Scalar(0, 255, 0), 2);
                                    cv::putText(slice_vis, cv::format("%.2f", score),
                                                cv::Point(slice_tl.x, std::max(12, slice_tl.y - 4)),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
                                    cv::putText(full_vis, cv::format("%.2f", score),
                                                cv::Point(full_tl.x, std::max(12, full_tl.y - 4)),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
                                    ++drawn_boxes;
                                }
                            }
                        }
                    }

                    anchor_offset += anchors_this_stride;
                }

                cv::putText(slice_vis, cv::format("slice %d", b), cv::Point(6, 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                cv::imwrite((build_dir / ("engine_slice" + std::to_string(b) + "_detections.png")).string(), slice_vis);
                LOG_INFO("LOGRESULTS saved engine_slice{}_detections.png with {} boxes >= {:.2f}",
                         b, drawn_boxes, confidence_threshold_);
            }
        }

        for (const auto &slice : slices_) {
            cv::rectangle(full_vis, cv::Point(slice.x1, slice.y1), cv::Point(slice.x2, slice.y2),
                          cv::Scalar(255, 255, 0), 1);
        }
        cv::imwrite((build_dir / "engine_detections_full.png").string(), full_vis);

        LOG_INFO("LOGRESULTS wrote per-slice inputs and outputs to {}{}", build_dir.string(),
                 use_two_row_anchor_reconstruction
                     ? " (output batch order remapped and anchor pairs reconstructed across 2 slice rows)"
                     : (use_column_major_output_order_fix ? " (output batch order remapped column-major -> row-major)" : ""));
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
