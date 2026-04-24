#include "spd_logger_helper.h"
#include "detection_model_inference_helper.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

#if defined(LOGRESULTS) || defined(DISPLAYRESULTS)
#include <opencv2/opencv.hpp>
#endif

#ifdef LOGRESULTS
#include <filesystem>
#include <fstream>
#endif

namespace {

void checkCuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

const char *detectionTypeName(DetectionType detection_type)
{
    return detection_type == DetectionType::SEARCH ? "search" : "track";
}

size_t dimsVolume(const nvinfer1::Dims &dims)
{
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw std::runtime_error("Encountered unresolved tensor dimension while computing volume");
        }
        volume *= static_cast<size_t>(dims.d[i]);
    }
    return volume;
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
    int32_t gap_y,
    int32_t min_box_length)
try
{

    // INFO
    LOG_INFO("***** Constructor DetectionModelInferenceHelper *****");
    LOG_INFO("\tBatch sizes: min: {}, opt: {}, max: {}", batch_sizes[0], batch_sizes[1], batch_sizes[2]);
    LOG_INFO("\tModel input height: {}", model_input_height);
    LOG_INFO("\tModel input width: {}", model_input_width);
    LOG_INFO("\tStrides: {} {} {}", strides[0], strides[1], strides[2]);
    LOG_INFO("\tTop K: {}", top_k);
    LOG_INFO("\tConfidence threshold: {}", confidence_threshold);
    LOG_INFO("\tIoU threshold: {}", iou_threshold);
    LOG_INFO("\tCamera input height: {}", camera_input_height);
    LOG_INFO("\tCamera input width: {}", camera_input_width);
    LOG_INFO("\tNum slices X: {}", num_slices_x);
    LOG_INFO("\tNum slices Y: {}", num_slices_y);
    LOG_INFO("\tGap X: {}", gap_x);
    LOG_INFO("\tGap Y: {}", gap_y);
    LOG_INFO("\tMin box length: {}", min_box_length);

    cudaGetDeviceProperties(&device_prop_, 0);
    device_name_ = device_prop_.name;
    if (!device_prop_.integrated && device_name_.find("Jetson") == std::string::npos)
    {
        LOG_INFO("\tRunning on discrete GPU: {}", device_name_);
    }
    else
    {
        LOG_INFO("\tRunning on Jetson: {}", device_name_);
    }

    if (batch_sizes[2] > kMaxSlices)
    {
        LOG_ERROR("Number of slices (batch_sizes[2] = {}) exceeds the maximum supported ({}).", batch_sizes[2], kMaxSlices);
        throw std::runtime_error("Number of slices exceeds maximum supported.");
    }
    if (batch_sizes[0] != 1)
    {
        LOG_ERROR("Track graph currently expects batch_sizes[0] to be 1, got {}", batch_sizes[0]);
        throw std::runtime_error("Track graph currently expects a single crop");
    }

    engine_ = engine;
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
    detection_top_k_ = top_k;
    detection_confidence_threshold_ = confidence_threshold;
    detection_iou_threshold_ = iou_threshold;
    min_box_length_ = min_box_length;
    batch_size_cuda_graph_search_mode_ = batch_sizes_[1];
    batch_size_cuda_graph_track_mode_ = batch_sizes_[0];

    input_tensor_name_ = engine->getIOTensorName(0);
    for (int i = 0; i < kNumRawOutputs; ++i)
    {
        output_tensor_names_[i] = engine->getIOTensorName(i + 1);
    }

    context_detection_search_mode = engine->createExecutionContext();
    context_detection_track_mode = engine->createExecutionContext();

    if (!context_detection_search_mode || !context_detection_track_mode)
    {
        LOG_ERROR("Failed to create execution contexts");
        throw std::runtime_error("Failed to create execution contexts");
    }

    // tracking context
    if (!context_detection_track_mode->setInputShape(
        input_tensor_name_.c_str(),
        nvinfer1::Dims4{batch_sizes_[0], 3, model_input_height_, model_input_width_}))
    {
        LOG_ERROR("Failed to set tracking input shape");
        throw std::runtime_error("Failed to set tracking input shape");
    }

    // search context
    if (!context_detection_search_mode->setInputShape(
        input_tensor_name_.c_str(),
        nvinfer1::Dims4{batch_sizes_[1], 3, model_input_height_, model_input_width_}))
    {
        LOG_ERROR("Failed to set search input shape");
        throw std::runtime_error("Failed to set search input shape");
    }

    if (cudaStreamCreate(&stream_) != cudaSuccess)
    {
        LOG_ERROR("Failed to create CUDA stream");
        throw std::runtime_error("Failed to create CUDA stream");
    }
    allocateBuffers();
    bindTensorAddresses(context_detection_track_mode, "tracking");
    bindTensorAddresses(context_detection_search_mode, "search");
    computeSlices();
    if (slices_.size() != static_cast<size_t>(batch_size_cuda_graph_search_mode_))
    {
        LOG_ERROR("Search graph batch ({}) must match computed slice count ({})",
                  batch_size_cuda_graph_search_mode_,
                  slices_.size());
        throw std::runtime_error("Search graph batch must match computed slice count");
    }

    if (!setDeviceSymbols(camera_input_width_, camera_input_height_,
                          model_input_width_,  model_input_height_,
                          anchor_count_,
                          detection_confidence_threshold_,
                          detection_iou_threshold_,
                          detection_top_k_,
                          min_box_length_))
    {
        LOG_ERROR("Failed to copy slice coordinates to device constant memory");
        throw std::runtime_error("Failed to copy slice coordinates to device constant memory");
    }

    std::memset(host_jetson_input_buffer_search_uint8_t_, 0, input_buffer_size_search_uint8_t_);
    std::memset(host_jetson_input_buffer_track_uint8_t_, 0, input_buffer_size_track_uint8_t_);
    ensureCudaGraphCaptured(DetectionType::SEARCH);
    ensureCudaGraphCaptured(DetectionType::TRACK);
}
catch (...)
{
    freeBuffers();
    throw;
}

DetectionModelInferenceHelper::~DetectionModelInferenceHelper()
{
    LOG_INFO("Destroying DetectionModelInferenceHelper");
    freeBuffers();
}

void DetectionModelInferenceHelper::infer(const uint8_t *host_image,
                                          DetectionType detection_type,
                                          const uint8_t *log_visual_image,
                                          int log_visual_width,
                                          int log_visual_height,
                                          int log_offset_x,
                                          int log_offset_y)
{
    stageUploadHostImage(host_image, detection_type);
    launchCapturedGraph(detection_type);
    checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize detection inference stream");

#if !defined(LOGRESULTS)
    (void)log_visual_image;
    (void)log_visual_width;
    (void)log_visual_height;
    (void)log_offset_x;
    (void)log_offset_y;
#endif

#ifdef DISPLAYRESULTS
    {
        const int batch = batchSizeForDetectionType(detection_type);
        const size_t sorted_count =
            static_cast<size_t>(batch) * static_cast<size_t>(detection_top_k_);
        std::vector<int32_t> host_num_selected(static_cast<size_t>(batch));
        std::vector<float4> host_sorted_bboxes(sorted_count);

        checkCuda(cudaMemcpy(host_num_selected.data(),
                             device_num_selected_,
                             host_num_selected.size() * sizeof(int32_t),
                             cudaMemcpyDeviceToHost),
                  "Failed to copy detection num selected to host for DISPLAYRESULTS");
        checkCuda(cudaMemcpy(host_sorted_bboxes.data(),
                             device_sorted_bboxes_,
                             host_sorted_bboxes.size() * sizeof(float4),
                             cudaMemcpyDeviceToHost),
                  "Failed to copy detection sorted bboxes to host for DISPLAYRESULTS");

        const int visual_width =
            detection_type == DetectionType::SEARCH ? camera_input_width_ : model_input_width_;
        const int visual_height =
            detection_type == DetectionType::SEARCH ? camera_input_height_ : model_input_height_;
        cv::Mat visual(visual_height, visual_width, CV_8UC3, const_cast<uint8_t *>(host_image));
        visual = visual.clone();

        for (int batch_index = 0; batch_index < batch; ++batch_index)
        {
            const int offset_x =
                detection_type == DetectionType::SEARCH ? slices_[static_cast<size_t>(batch_index)].x1 : 0;
            const int offset_y =
                detection_type == DetectionType::SEARCH ? slices_[static_cast<size_t>(batch_index)].y1 : 0;
            const int count = std::min(host_num_selected[static_cast<size_t>(batch_index)], detection_top_k_);

            for (int rank = 0; rank < count; ++rank)
            {
                const size_t flat_index =
                    static_cast<size_t>(batch_index) * static_cast<size_t>(detection_top_k_) +
                    static_cast<size_t>(rank);
                const auto &bbox = host_sorted_bboxes[flat_index];
                const float box_width = std::max(0.0f, bbox.z - bbox.x);
                const float box_height = std::max(0.0f, bbox.w - bbox.y);
                const cv::Scalar color =
                    (box_width >= static_cast<float>(min_box_length_) &&
                     box_height >= static_cast<float>(min_box_length_))
                        ? cv::Scalar(0, 255, 0)
                        : cv::Scalar(0, 0, 255);

                const int x1 = std::clamp(static_cast<int>(bbox.x) + offset_x, 0, visual_width - 1);
                const int y1 = std::clamp(static_cast<int>(bbox.y) + offset_y, 0, visual_height - 1);
                const int x2 = std::clamp(static_cast<int>(bbox.z) + offset_x, 0, visual_width - 1);
                const int y2 = std::clamp(static_cast<int>(bbox.w) + offset_y, 0, visual_height - 1);
                cv::rectangle(visual, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            }
        }

        const std::string window_name =
            std::string("detection_") + detectionTypeName(detection_type) + "_display";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::imshow(window_name, visual);
        cv::waitKey(1);
        LOG_INFO("Displayed frame resolution: {}x{}", visual_width, visual_height);
    }
#endif

#ifdef LOGRESULTS
    auto build_dir = std::filesystem::current_path();
    if (build_dir.filename() != "build")
    {
        build_dir /= "build";
    }

    const auto log_dir = build_dir / "logresults";
    std::filesystem::create_directories(log_dir);

    auto *context = detection_type == DetectionType::SEARCH
        ? context_detection_search_mode
        : context_detection_track_mode;

    for (int i = 0; i < kNumRawOutputs; ++i)
    {
        const auto dims = context->getTensorShape(output_tensor_names_[i].c_str());
        const size_t element_count = dimsVolume(dims);
        std::vector<float> host_output(element_count);

        checkCuda(cudaMemcpy(host_output.data(),
                             device_model_output_buffers_[i],
                             element_count * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Failed to copy detection output to host for LOGRESULTS");

        const auto output_path =
            log_dir / (std::string("detection_") + detectionTypeName(detection_type) +
                       "_raw_output_" + std::to_string(i) + ".txt");
        std::ofstream ofs(output_path);
        for (float value : host_output)
        {
            ofs << value << "\n";
        }
    }

    const int batch = batchSizeForDetectionType(detection_type);
    const size_t sorted_count =
        static_cast<size_t>(batch) * static_cast<size_t>(detection_top_k_);
    std::vector<int32_t> host_num_selected(static_cast<size_t>(batch));
    std::vector<float> host_sorted_scores(sorted_count);
    std::vector<int32_t> host_sorted_indexes(sorted_count);
    std::vector<int32_t> host_final_num_detections(static_cast<size_t>(batch));
    std::vector<float> host_final_scores(sorted_count);
    std::vector<int32_t> host_final_indexes(sorted_count);
    std::vector<float4> host_final_bboxes(sorted_count);
    std::vector<float2> host_final_landmarks(sorted_count * kLandmarkCount);

    checkCuda(cudaMemcpy(host_num_selected.data(),
                         device_num_selected_,
                         host_num_selected.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection num selected to host for LOGRESULTS");
    checkCuda(cudaMemcpy(host_sorted_scores.data(),
                         device_sorted_scores_,
                         host_sorted_scores.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection sorted scores to host for LOGRESULTS");
    checkCuda(cudaMemcpy(host_sorted_indexes.data(),
                         device_sorted_indexes_,
                         host_sorted_indexes.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection sorted indexes to host for LOGRESULTS");
    checkCuda(cudaMemcpy(host_final_num_detections.data(),
                         device_final_num_detections_,
                         host_final_num_detections.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final counts to host for LOGRESULTS");
    checkCuda(cudaMemcpy(host_final_scores.data(),
                         device_final_scores_,
                         host_final_scores.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final scores to host for LOGRESULTS");
    checkCuda(cudaMemcpy(host_final_indexes.data(),
                         device_final_indexes_,
                         host_final_indexes.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final indexes to host for LOGRESULTS");
    checkCuda(cudaMemcpy(host_final_bboxes.data(),
                         device_final_bboxes_,
                         host_final_bboxes.size() * sizeof(float4),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final bboxes to host for LOGRESULTS");
    checkCuda(cudaMemcpy(host_final_landmarks.data(),
                         device_final_landmarks_,
                         host_final_landmarks.size() * sizeof(float2),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final landmarks to host for LOGRESULTS");

    const std::string prefix =
        std::string("detection_") + detectionTypeName(detection_type);

    {
        std::ofstream ofs(log_dir / (prefix + "_num_selected.txt"));
        for (int32_t value : host_num_selected)
        {
            ofs << value << "\n";
        }
    }

    {
        std::ofstream ofs(log_dir / (prefix + "_sorted_scores.txt"));
        for (float value : host_sorted_scores)
        {
            ofs << value << "\n";
        }
    }

    {
        std::ofstream ofs(log_dir / (prefix + "_sorted_indexes.txt"));
        for (int32_t value : host_sorted_indexes)
        {
            ofs << value << "\n";
        }
    }

    {
        std::ofstream ofs(log_dir / (prefix + "_final_num_detections.txt"));
        for (int32_t value : host_final_num_detections)
        {
            ofs << value << "\n";
        }
    }

    {
        std::ofstream ofs(log_dir / (prefix + "_final_scores.txt"));
        for (float value : host_final_scores)
        {
            ofs << value << "\n";
        }
    }

    {
        std::ofstream ofs(log_dir / (prefix + "_final_indexes.txt"));
        for (int32_t value : host_final_indexes)
        {
            ofs << value << "\n";
        }
    }

    {
        std::ofstream ofs(log_dir / (prefix + "_final_bboxes.txt"));
        for (const auto &bbox : host_final_bboxes)
        {
            ofs << bbox.x << " " << bbox.y << " " << bbox.z << " " << bbox.w << "\n";
        }
    }

    {
        std::ofstream ofs(log_dir / (prefix + "_final_landmarks.txt"));
        for (size_t i = 0; i < host_final_landmarks.size(); i += kLandmarkCount)
        {
            for (int j = 0; j < kLandmarkCount; ++j)
            {
                const auto &point = host_final_landmarks[i + j];
                ofs << point.x << " " << point.y;
                if (j + 1 < kLandmarkCount)
                {
                    ofs << " ";
                }
            }
            ofs << "\n";
        }
    }

    const bool use_log_visual = log_visual_image && log_visual_width > 0 && log_visual_height > 0;
    const int visual_width =
        use_log_visual ? log_visual_width :
        (detection_type == DetectionType::SEARCH ? camera_input_width_ : model_input_width_);
    const int visual_height =
        use_log_visual ? log_visual_height :
        (detection_type == DetectionType::SEARCH ? camera_input_height_ : model_input_height_);
    const uint8_t *visual_image = use_log_visual ? log_visual_image : host_image;
    cv::Mat visual(visual_height, visual_width, CV_8UC3, const_cast<uint8_t *>(visual_image));
    visual = visual.clone();

    for (int batch_index = 0; batch_index < batch; ++batch_index)
    {
        const int offset_x =
            (detection_type == DetectionType::SEARCH ? slices_[static_cast<size_t>(batch_index)].x1 : 0) +
            (use_log_visual ? log_offset_x : 0);
        const int offset_y =
            (detection_type == DetectionType::SEARCH ? slices_[static_cast<size_t>(batch_index)].y1 : 0) +
            (use_log_visual ? log_offset_y : 0);
        const int count = std::min(host_final_num_detections[static_cast<size_t>(batch_index)], detection_top_k_);

        for (int rank = 0; rank < count; ++rank)
        {
            const size_t flat_index =
                static_cast<size_t>(batch_index) * static_cast<size_t>(detection_top_k_) +
                static_cast<size_t>(rank);
            const auto &bbox = host_final_bboxes[flat_index];
            const int x1 = std::clamp(static_cast<int>(bbox.x) + offset_x, 0, visual_width - 1);
            const int y1 = std::clamp(static_cast<int>(bbox.y) + offset_y, 0, visual_height - 1);
            const int x2 = std::clamp(static_cast<int>(bbox.z) + offset_x, 0, visual_width - 1);
            const int y2 = std::clamp(static_cast<int>(bbox.w) + offset_y, 0, visual_height - 1);

            cv::rectangle(visual, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            cv::putText(visual,
                        std::to_string(batch_index) + ":" + std::to_string(rank),
                        cv::Point(x1, std::max(0, y1 - 4)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 0),
                        1);

            for (int landmark_index = 0; landmark_index < kLandmarkCount; ++landmark_index)
            {
                const auto &point = host_final_landmarks[flat_index * kLandmarkCount + landmark_index];
                const int px = std::clamp(static_cast<int>(point.x) + offset_x, 0, visual_width - 1);
                const int py = std::clamp(static_cast<int>(point.y) + offset_y, 0, visual_height - 1);
                cv::circle(visual, cv::Point(px, py), 2, cv::Scalar(0, 0, 255), cv::FILLED);
            }
        }
    }

    if (detection_type == DetectionType::SEARCH)
    {
        const int dummy_x = 50;
        const int dummy_y_positions[] = {50, 200, 300};
        const int dummy_sizes[] = {50, 70, 112};

        for (int i = 0; i < 3; ++i)
        {
            const int x1 = std::clamp(dummy_x, 0, visual_width - 1);
            const int y1 = std::clamp(dummy_y_positions[i], 0, visual_height - 1);
            const int x2 = std::clamp(dummy_x + dummy_sizes[i], 0, visual_width - 1);
            const int y2 = std::clamp(dummy_y_positions[i] + dummy_sizes[i], 0, visual_height - 1);

            cv::rectangle(visual, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255), 2);
            cv::putText(visual,
                        std::to_string(dummy_sizes[i]) + "x" + std::to_string(dummy_sizes[i]),
                        cv::Point(x1, std::max(0, y1 - 6)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 255),
                        1);
        }
    }

    cv::imwrite((log_dir / (prefix + "_final_visual.jpg")).string(), visual);
#endif
}

std::vector<DetectionBox> DetectionModelInferenceHelper::getLastDetections(DetectionType detection_type) const
{
    const int batch = batchSizeForDetectionType(detection_type);
    const size_t capacity =
        static_cast<size_t>(batch) * static_cast<size_t>(detection_top_k_);

    std::vector<int32_t> host_counts(static_cast<size_t>(batch));
    std::vector<float> host_scores(capacity);
    std::vector<float4> host_bboxes(capacity);

    checkCuda(cudaMemcpy(host_counts.data(),
                         device_final_num_detections_,
                         host_counts.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final counts to host");
    checkCuda(cudaMemcpy(host_scores.data(),
                         device_final_scores_,
                         host_scores.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final scores to host");
    checkCuda(cudaMemcpy(host_bboxes.data(),
                         device_final_bboxes_,
                         host_bboxes.size() * sizeof(float4),
                         cudaMemcpyDeviceToHost),
              "Failed to copy detection final bboxes to host");

    std::vector<DetectionBox> detections;
    detections.reserve(capacity);

    for (int batch_index = 0; batch_index < batch; ++batch_index)
    {
        const int offset_x =
            detection_type == DetectionType::SEARCH ? slices_[static_cast<size_t>(batch_index)].x1 : 0;
        const int offset_y =
            detection_type == DetectionType::SEARCH ? slices_[static_cast<size_t>(batch_index)].y1 : 0;
        const int count = std::min(host_counts[static_cast<size_t>(batch_index)], detection_top_k_);

        for (int rank = 0; rank < count; ++rank)
        {
            const size_t flat_index =
                static_cast<size_t>(batch_index) * static_cast<size_t>(detection_top_k_) +
                static_cast<size_t>(rank);
            const auto &bbox = host_bboxes[flat_index];
            DetectionBox detection;
            detection.x1 = static_cast<int>(bbox.x) + offset_x;
            detection.y1 = static_cast<int>(bbox.y) + offset_y;
            detection.x2 = static_cast<int>(bbox.z) + offset_x;
            detection.y2 = static_cast<int>(bbox.w) + offset_y;
            detection.score = host_scores[flat_index];
            detections.push_back(detection);
        }
    }

    return detections;
}

void DetectionModelInferenceHelper::bindTensorAddresses(IExecutionContext *context, const char *context_name)
{
    // Both execution contexts reuse the same backing buffers, but each context must
    // still bind its own tensor addresses after shape configuration is finalized.
    if (!context)
    {
        LOG_ERROR("Failed to bind tensor addresses: {} context is null", context_name);
        throw std::runtime_error("Failed to bind tensor addresses for null context");
    }

    if (!context->setTensorAddress(input_tensor_name_.c_str(), device_jetson_input_buffer_float_))
    {
        LOG_ERROR("Failed to bind {} input tensor address for {}", context_name, input_tensor_name_);
        throw std::runtime_error("Failed to bind input tensor address");
    }

    for (int i = 0; i < kNumRawOutputs; ++i)
    {
        if (!context->setTensorAddress(output_tensor_names_[i].c_str(), device_model_output_buffers_[i]))
        {
            LOG_ERROR("Failed to bind {} output tensor address for {}", context_name, output_tensor_names_[i]);
            throw std::runtime_error("Failed to bind output tensor address");
        }
    }
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

void DetectionModelInferenceHelper::allocateBuffers() {
    LOG_INFO("\t***** allocateBuffers *****");

    float memory_usage = 0;

    // Search mode reads the full camera frame directly from mapped pinned memory.
    input_buffer_size_search_uint8_t_ =
        static_cast<size_t>(camera_input_height_) * camera_input_width_ * 3 * sizeof(uint8_t);
    cudaHostAlloc(&host_jetson_input_buffer_search_uint8_t_, input_buffer_size_search_uint8_t_, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&device_jetson_ptr_search_uint8_t_, host_jetson_input_buffer_search_uint8_t_, 0);
    memory_usage += input_buffer_size_search_uint8_t_ / (1024.0 * 1024.0);
    LOG_DEBUG("\t\thost_jetson_input_buffer_search_uint8_t_ allocated with size: {} mb",
              input_buffer_size_search_uint8_t_ / (1024.0 * 1024.0));

    // Track mode keeps a dedicated single-crop buffer so its input meaning stays fixed for cudaGraph capture.
    input_buffer_size_track_uint8_t_ =
        static_cast<size_t>(model_input_height_) * model_input_width_ * 3 * sizeof(uint8_t);
    cudaHostAlloc(&host_jetson_input_buffer_track_uint8_t_, input_buffer_size_track_uint8_t_, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&device_jetson_ptr_track_uint8_t_, host_jetson_input_buffer_track_uint8_t_, 0);
    memory_usage += input_buffer_size_track_uint8_t_ / (1024.0 * 1024.0);
    LOG_DEBUG("\t\thost_jetson_input_buffer_track_uint8_t_ allocated with size: {} mb",
              input_buffer_size_track_uint8_t_ / (1024.0 * 1024.0));

    // float32 CHW buffer: preprocessing kernel writes here, TensorRT reads from here
    input_buffer_size_float_   = static_cast<size_t>(batch_sizes_[2]) * 3 * model_input_height_ * model_input_width_ * sizeof(float);
    cudaMalloc(&device_jetson_input_buffer_float_, input_buffer_size_float_);
    memory_usage += input_buffer_size_float_ / (1024.0 * 1024.0);
    LOG_DEBUG("\t\tdevice_jetson_input_buffer_float_ allocated with size: {} mb", input_buffer_size_float_ / (1024.0 * 1024.0));

    for (int i = 0; i < kNumRawOutputs; ++i)
    {
        const auto dims = context_detection_search_mode->getTensorShape(output_tensor_names_[i].c_str());
        const size_t output_elements = dimsVolume(dims);
        const size_t output_buffer_size = output_elements * sizeof(float);

        if (output_elements % static_cast<size_t>(batch_size_cuda_graph_search_mode_) != 0)
        {
            throw std::runtime_error("Detection output element count is not divisible by search batch size");
        }

        output_element_counts_[i] = output_elements;
        output_elements_per_batch_[i] = output_elements / static_cast<size_t>(batch_size_cuda_graph_search_mode_);

        cudaMalloc(&device_model_output_buffers_[i], output_buffer_size);
        memory_usage += output_buffer_size / (1024.0 * 1024.0);
        LOG_DEBUG("\t\tdevice_model_output_buffers_[{}] allocated with shape volume {} and size: {} mb",
                  i,
                  output_elements,
                  output_buffer_size / (1024.0 * 1024.0));
    }

    const size_t postprocess_capacity =
        static_cast<size_t>(batch_sizes_[2]) * static_cast<size_t>(detection_top_k_);
    const size_t suppression_mask_words =
        static_cast<size_t>(batch_sizes_[2]) * static_cast<size_t>((detection_top_k_ + 31) / 32);
    cudaMalloc(&device_filtered_scores_, postprocess_capacity * sizeof(float));
    cudaMalloc(&device_filtered_indexes_, postprocess_capacity * sizeof(int32_t));
    cudaMalloc(&device_sorted_scores_, postprocess_capacity * sizeof(float));
    cudaMalloc(&device_sorted_indexes_, postprocess_capacity * sizeof(int32_t));
    cudaMalloc(&device_sorted_bboxes_, postprocess_capacity * sizeof(float4));
    cudaMalloc(&device_sorted_landmarks_, postprocess_capacity * kLandmarkCount * sizeof(float2));
    cudaMalloc(&device_suppression_mask_, suppression_mask_words * sizeof(uint32_t));
    cudaMalloc(&device_num_selected_, static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t));
    cudaMalloc(&device_final_scores_, postprocess_capacity * sizeof(float));
    cudaMalloc(&device_final_indexes_, postprocess_capacity * sizeof(int32_t));
    cudaMalloc(&device_final_bboxes_, postprocess_capacity * sizeof(float4));
    cudaMalloc(&device_final_landmarks_, postprocess_capacity * kLandmarkCount * sizeof(float2));
    cudaMalloc(&device_final_num_detections_, static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t));
    memory_usage += (postprocess_capacity * sizeof(float) +
                     postprocess_capacity * sizeof(int32_t) +
                     postprocess_capacity * sizeof(float) +
                     postprocess_capacity * sizeof(int32_t) +
                     postprocess_capacity * sizeof(float4) +
                     postprocess_capacity * kLandmarkCount * sizeof(float2) +
                     suppression_mask_words * sizeof(uint32_t) +
                     postprocess_capacity * sizeof(float) +
                     postprocess_capacity * sizeof(int32_t) +
                     postprocess_capacity * sizeof(float4) +
                     postprocess_capacity * kLandmarkCount * sizeof(float2) +
                     static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t) +
                     static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t)) /
                    (1024.0 * 1024.0);

    for (auto stride : strides_)
    {
        int32_t feature_map_height = model_input_height_ / stride;
        int32_t feature_map_width = model_input_width_ / stride;
        int32_t anchors_per_feature_map = feature_map_height * feature_map_width;
        anchor_count_ += anchors_per_feature_map * anchor_stack_;
        LOG_DEBUG("\t\t\tStride {}: feature map size: {}x{}, anchors per feature map: {}, total anchors so far: {}",
                  stride, feature_map_width, feature_map_height, anchors_per_feature_map * anchor_stack_, anchor_count_);
    }


    LOG_INFO("\tTotal memory allocated in allocateBuffers: {} mb", memory_usage);
}

void DetectionModelInferenceHelper::freeBuffers()
{
    LOG_INFO("Freeing buffers in DetectionModelInferenceHelper");
    destroyCudaGraphs();

    if (host_jetson_input_buffer_search_uint8_t_)
    {
        cudaFreeHost(host_jetson_input_buffer_search_uint8_t_);
        host_jetson_input_buffer_search_uint8_t_ = nullptr;
        device_jetson_ptr_search_uint8_t_ = nullptr;
    }

    if (host_jetson_input_buffer_track_uint8_t_)
    {
        cudaFreeHost(host_jetson_input_buffer_track_uint8_t_);
        host_jetson_input_buffer_track_uint8_t_ = nullptr;
        device_jetson_ptr_track_uint8_t_ = nullptr;
    }

    if (device_jetson_input_buffer_float_)
    {
        cudaFree(device_jetson_input_buffer_float_);
        device_jetson_input_buffer_float_ = nullptr;
    }

    for (auto &buffer : device_model_output_buffers_)
    {
        if (buffer)
        {
            cudaFree(buffer);
            buffer = nullptr;
        }
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

    if (device_num_selected_)
    {
        cudaFree(device_num_selected_);
        device_num_selected_ = nullptr;
    }

    if (device_final_scores_)
    {
        cudaFree(device_final_scores_);
        device_final_scores_ = nullptr;
    }

    if (device_final_indexes_)
    {
        cudaFree(device_final_indexes_);
        device_final_indexes_ = nullptr;
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
        sort_storage_bytes_ = 0;
    }

    if (context_detection_search_mode)
    {
        delete context_detection_search_mode;
        context_detection_search_mode = nullptr;
    }

    if (context_detection_track_mode)
    {
        delete context_detection_track_mode;
        context_detection_track_mode = nullptr;
    }

    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

//STAGES
void DetectionModelInferenceHelper::stageUploadHostImage(const uint8_t *host_image, DetectionType detection_type)
{
    if (!host_image)
    {
        throw std::invalid_argument("host_image cannot be null");
    }

    if (detection_type == DetectionType::SEARCH)
    {
        std::memcpy(host_jetson_input_buffer_search_uint8_t_, host_image, input_buffer_size_search_uint8_t_);
        return;
    }

    std::memcpy(host_jetson_input_buffer_track_uint8_t_, host_image, input_buffer_size_track_uint8_t_);
}

void DetectionModelInferenceHelper::stagePreprocessOnly(DetectionType detection_type)
{
    launchPreprocessKernel(
        deviceInputForDetectionType(detection_type),
        device_jetson_input_buffer_float_,
        batchSizeForDetectionType(detection_type),
        detection_type,
        stream_);
    checkCuda(cudaGetLastError(), "Failed to launch detection preprocess kernel");
}

void DetectionModelInferenceHelper::stageInferenceOnly(DetectionType detection_type)
{
    auto *context = detection_type == DetectionType::SEARCH
        ? context_detection_search_mode
        : context_detection_track_mode;

    if (!context || !context->enqueueV3(stream_))
    {
        throw std::runtime_error(std::string("Failed to enqueue TensorRT detection context for ") +
                                 detectionTypeName(detection_type));
    }
}

void DetectionModelInferenceHelper::resetPostprocessBuffers(DetectionType detection_type)
{
    const int batch = batchSizeForDetectionType(detection_type);
    const size_t capacity =
        static_cast<size_t>(batch) * static_cast<size_t>(detection_top_k_);
    const size_t suppression_mask_words =
        static_cast<size_t>(batch) * static_cast<size_t>((detection_top_k_ + 31) / 32);

    cudaMemsetAsync(device_num_selected_, 0, static_cast<size_t>(batch) * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_filtered_scores_, 0, capacity * sizeof(float), stream_);
    cudaMemsetAsync(device_filtered_indexes_, 0, capacity * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_suppression_mask_, 0, suppression_mask_words * sizeof(uint32_t), stream_);
    cudaMemsetAsync(device_final_num_detections_, 0, static_cast<size_t>(batch) * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_final_scores_, 0, capacity * sizeof(float), stream_);
    cudaMemsetAsync(device_final_indexes_, 0, capacity * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_final_bboxes_, 0, capacity * sizeof(float4), stream_);
    cudaMemsetAsync(device_final_landmarks_, 0, capacity * kLandmarkCount * sizeof(float2), stream_);
}

void DetectionModelInferenceHelper::stagePostprocess(DetectionType detection_type)
{
    launchFilterScoresKernel(batchSizeForDetectionType(detection_type), stream_);
    checkCuda(cudaGetLastError(), "Failed to launch detection filter scores kernel");
    launchSortFilteredScoresKernel(batchSizeForDetectionType(detection_type), stream_);
    checkCuda(cudaGetLastError(), "Failed to launch detection sort filtered scores kernel");
    launchGatherAllKernel(batchSizeForDetectionType(detection_type), stream_);
    checkCuda(cudaGetLastError(), "Failed to launch detection gather all kernel");
    launchBitmaskNMSKernel(batchSizeForDetectionType(detection_type), stream_);
    checkCuda(cudaGetLastError(), "Failed to launch detection bitmask NMS kernel");
    launchGatherFinalResultKernel(batchSizeForDetectionType(detection_type), stream_);
    checkCuda(cudaGetLastError(), "Failed to launch detection gather final result kernel");
}

void DetectionModelInferenceHelper::ensureCudaGraphCaptured(DetectionType detection_type)
{
    const size_t index = graphIndex(detection_type);
    if (cuda_graph_execs_[index])
    {
        return;
    }

    // TensorRT can do context/resource setup on the first enqueue after setting a
    // shape. Run once outside capture so the captured graph contains steady-state work.
    resetPostprocessBuffers(detection_type);
    stagePreprocessOnly(detection_type);
    stageInferenceOnly(detection_type);
    stagePostprocess(detection_type);
    checkCuda(cudaStreamSynchronize(stream_), "Failed to warm up detection graph capture");

    checkCuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
              "Failed to begin detection CUDA graph capture");

    try
    {
        resetPostprocessBuffers(detection_type);
        stagePreprocessOnly(detection_type);
        stageInferenceOnly(detection_type);
        stagePostprocess(detection_type);
    }
    catch (...)
    {
        cudaGraph_t failed_graph = nullptr;
        cudaStreamEndCapture(stream_, &failed_graph);
        if (failed_graph)
        {
            cudaGraphDestroy(failed_graph);
        }
        throw;
    }

    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &cuda_graphs_[index]);
    if (capture_status != cudaSuccess)
    {
        if (cuda_graphs_[index])
        {
            cudaGraphDestroy(cuda_graphs_[index]);
            cuda_graphs_[index] = nullptr;
        }
        checkCuda(capture_status, "Failed to end detection CUDA graph capture");
    }

    checkCuda(cudaGraphInstantiate(&cuda_graph_execs_[index], cuda_graphs_[index], nullptr, nullptr, 0),
              "Failed to instantiate detection CUDA graph");

    LOG_INFO("Captured detection CUDA graph for {} mode with batch={}",
             detectionTypeName(detection_type),
             batchSizeForDetectionType(detection_type));
}

void DetectionModelInferenceHelper::launchCapturedGraph(DetectionType detection_type)
{
    const size_t index = graphIndex(detection_type);
    if (!cuda_graph_execs_[index])
    {
        throw std::runtime_error(std::string("Detection CUDA graph is not captured for ") +
                                 detectionTypeName(detection_type));
    }

    checkCuda(cudaGraphLaunch(cuda_graph_execs_[index], stream_),
              "Failed to launch detection CUDA graph");
}

void DetectionModelInferenceHelper::destroyCudaGraphs()
{
    for (auto &graph_exec : cuda_graph_execs_)
    {
        if (graph_exec)
        {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
    }

    for (auto &graph : cuda_graphs_)
    {
        if (graph)
        {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
    }
}

size_t DetectionModelInferenceHelper::graphIndex(DetectionType detection_type) const
{
    return detection_type == DetectionType::SEARCH ? 0 : 1;
}

int DetectionModelInferenceHelper::batchSizeForDetectionType(DetectionType detection_type) const
{
    return detection_type == DetectionType::SEARCH
        ? batch_size_cuda_graph_search_mode_
        : batch_size_cuda_graph_track_mode_;
}

const uint8_t *DetectionModelInferenceHelper::deviceInputForDetectionType(DetectionType detection_type) const
{
    return detection_type == DetectionType::SEARCH
        ? device_jetson_ptr_search_uint8_t_
        : device_jetson_ptr_track_uint8_t_;
}
