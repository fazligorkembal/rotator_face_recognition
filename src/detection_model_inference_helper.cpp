#include "spd_logger_helper.h"
#include "detection_model_inference_helper.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <sstream>

#if defined(GENERATE_TXT) || defined(LOGRESULTS)
#include <fstream>
#endif

#include <opencv2/opencv.hpp>

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

size_t dimsVolume(const nvinfer1::Dims &dims) {
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Encountered unresolved tensor dimension while computing volume");
        }
        volume *= static_cast<size_t>(dims.d[i]);
    }
    return volume;
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
std::vector<T> copyTensorToHost(const T *device_ptr, size_t element_count) {
    std::vector<T> host(element_count);
    cudaMemcpy(host.data(),
               device_ptr,
               element_count * sizeof(T),
               cudaMemcpyDeviceToHost);
    return host;
}

std::filesystem::path resolveDebugFaceCropDir() {
    std::filesystem::path build_dir = std::filesystem::current_path();
    if (build_dir.filename() != "build") {
        build_dir /= "build";
    }

    const std::filesystem::path crop_dir = build_dir / "debug_aligned_faces";
    std::filesystem::remove_all(crop_dir);
    std::filesystem::create_directories(crop_dir);
    return crop_dir;
}

cv::Mat makeAlignedFaceMat(const float *face_data) {
    cv::Mat face_bgr(DetectionModelInferenceHelper::kAlignedFaceHeight,
                     DetectionModelInferenceHelper::kAlignedFaceWidth,
                     CV_8UC3);

    const int plane =
        DetectionModelInferenceHelper::kAlignedFaceHeight *
        DetectionModelInferenceHelper::kAlignedFaceWidth;

    for (int y = 0; y < DetectionModelInferenceHelper::kAlignedFaceHeight; ++y) {
        for (int x = 0; x < DetectionModelInferenceHelper::kAlignedFaceWidth; ++x) {
            const int pixel_index = y * DetectionModelInferenceHelper::kAlignedFaceWidth + x;
            const float r = face_data[0 * plane + pixel_index] * 128.0f + 127.5f;
            const float g = face_data[1 * plane + pixel_index] * 128.0f + 127.5f;
            const float b = face_data[2 * plane + pixel_index] * 128.0f + 127.5f;

            face_bgr.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f)));
        }
    }

    return face_bgr;
}

void dumpAlignedFaceCrops(const float *device_face_crops,
                          const int32_t *device_detection_counts,
                          const float *device_scores,
                          int batch_size,
                          int top_k) {
    const auto crop_dir = resolveDebugFaceCropDir();
    const auto host_counts =
        copyTensorToHost(device_detection_counts, static_cast<size_t>(batch_size));
    const auto host_scores =
        copyTensorToHost(device_scores, static_cast<size_t>(batch_size) * static_cast<size_t>(top_k));
    const auto host_crops =
        copyTensorToHost(device_face_crops,
                         static_cast<size_t>(batch_size) * static_cast<size_t>(top_k) *
                             DetectionModelInferenceHelper::kAlignedFaceChannels *
                             DetectionModelInferenceHelper::kAlignedFaceHeight *
                             DetectionModelInferenceHelper::kAlignedFaceWidth);

    static const std::array<cv::Point2f, DetectionModelInferenceHelper::kLandmarkCount> kArcFaceTemplate = {{
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f},
    }};

    const size_t face_stride =
        DetectionModelInferenceHelper::kAlignedFaceChannels *
        DetectionModelInferenceHelper::kAlignedFaceHeight *
        DetectionModelInferenceHelper::kAlignedFaceWidth;

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        const int32_t detection_count = std::min<int32_t>(host_counts[batch_index], top_k);
        for (int32_t det_index = 0; det_index < detection_count; ++det_index) {
            const size_t flat_face_index =
                static_cast<size_t>(batch_index) * static_cast<size_t>(top_k) +
                static_cast<size_t>(det_index);
            cv::Mat face = makeAlignedFaceMat(host_crops.data() + flat_face_index * face_stride);

            for (const auto &point : kArcFaceTemplate) {
                cv::circle(face,
                           cv::Point(static_cast<int>(std::lround(point.x)),
                                     static_cast<int>(std::lround(point.y))),
                           2,
                           cv::Scalar(0, 255, 255),
                           cv::FILLED,
                           cv::LINE_AA);
            }

            const float score = host_scores[flat_face_index];
            cv::putText(face,
                        cv::format("b=%d d=%d s=%.3f", batch_index, det_index, score),
                        cv::Point(6, 16),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.4,
                        cv::Scalar(0, 255, 0),
                        1,
                        cv::LINE_AA);

            cv::imwrite((crop_dir / ("aligned_face_batch" + std::to_string(batch_index) +
                                     "_det" + std::to_string(det_index) + ".png")).string(),
                        face);
        }
    }

    LOG_INFO("Saved aligned face crops to {}", crop_dir.string());
}

#ifdef LOGRESULTS
std::filesystem::path resolveBuildDir() {
    std::filesystem::path build_dir = std::filesystem::current_path();
    if (build_dir.filename() != "build") {
        build_dir /= "build";
    }
    std::filesystem::create_directories(build_dir);
    return build_dir;
}

template <typename T>
void writeValueLines(const std::filesystem::path &path, const std::vector<T> &values) {
    std::ofstream ofs(path);
    for (const auto &value : values) {
        ofs << value << "\n";
    }
}

template <typename T>
void writeRawTensorLines(const std::filesystem::path &path, const std::vector<T> &values) {
    static_assert(sizeof(T) % sizeof(float) == 0, "Expected tensor element to be float-packed");
    std::ofstream ofs(path);
    const size_t float_count = values.size() * sizeof(T) / sizeof(float);
    const auto *flat = reinterpret_cast<const float *>(values.data());
    for (size_t i = 0; i < float_count; ++i) {
        ofs << flat[i] << "\n";
    }
}

template <typename T>
void dumpRawTensorBatch(const std::filesystem::path &path,
                        const T *device_ptr,
                        size_t batch_index,
                        size_t element_count) {
    const auto host_values = copyTensorSliceToHost(device_ptr, batch_index, element_count);
    writeRawTensorLines(path, host_values);
}

template <typename T>
void dumpRawTensorAllBatches(const std::filesystem::path &path,
                             const T *device_ptr,
                             size_t element_count) {
    const auto host_values = copyTensorToHost(device_ptr, element_count);
    writeRawTensorLines(path, host_values);
}

void dumpScoreSelectionSummary(const std::filesystem::path &build_dir,
                               const std::string &prefix,
                               const int32_t *device_num_selected,
                               const int32_t *device_indexes,
                               const float *device_scores,
                               int batch_size,
                               int top_k) {
    const auto host_num_selected =
        copyTensorToHost(device_num_selected, static_cast<size_t>(batch_size));
    const auto host_indexes =
        copyTensorToHost(device_indexes, static_cast<size_t>(batch_size) * static_cast<size_t>(top_k));
    const auto host_scores =
        copyTensorToHost(device_scores, static_cast<size_t>(batch_size) * static_cast<size_t>(top_k));

    writeValueLines(build_dir / ("engine_" + prefix + "_num_selected.txt"), host_num_selected);
    writeValueLines(build_dir / ("engine_" + prefix + "_indexes_all_batches.txt"), host_indexes);
    writeValueLines(build_dir / ("engine_" + prefix + "_scores_all_batches.txt"), host_scores);

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        const int32_t selected_count = host_num_selected[batch_index];
        const int32_t clamped_count = std::min<int32_t>(selected_count, top_k);

        std::vector<int32_t> batch_indexes(
            host_indexes.begin() + static_cast<size_t>(batch_index) * top_k,
            host_indexes.begin() + static_cast<size_t>(batch_index + 1) * top_k);
        std::vector<float> batch_scores(
            host_scores.begin() + static_cast<size_t>(batch_index) * top_k,
            host_scores.begin() + static_cast<size_t>(batch_index + 1) * top_k);

        writeValueLines(build_dir / ("engine_" + prefix + "_indexes_batch" + std::to_string(batch_index) + ".txt"),
                        batch_indexes);
        writeValueLines(build_dir / ("engine_" + prefix + "_scores_batch" + std::to_string(batch_index) + ".txt"),
                        batch_scores);

        std::ofstream summary_ofs(build_dir / ("engine_" + prefix + "_summary_batch" + std::to_string(batch_index) + ".txt"));
        summary_ofs << "num_selected " << selected_count << "\n";
        for (int32_t i = 0; i < clamped_count; ++i) {
            summary_ofs << i << " " << batch_indexes[static_cast<size_t>(i)]
                        << " " << batch_scores[static_cast<size_t>(i)] << "\n";
        }

        std::ostringstream summary_log;
        summary_log << "batch " << batch_index << " num_selected=" << selected_count;
        const int32_t preview_count = std::min<int32_t>(clamped_count, 8);
        for (int32_t i = 0; i < preview_count; ++i) {
            summary_log << " | [" << i
                        << "] idx=" << batch_indexes[static_cast<size_t>(i)]
                        << " score=" << batch_scores[static_cast<size_t>(i)];
        }
        LOG_INFO("{} {}", prefix, summary_log.str());
    }
}

void dumpSuppressionMaskSummary(const std::filesystem::path &build_dir,
                                const uint32_t *device_suppression_mask,
                                const int32_t *device_num_selected,
                                int batch_size,
                                int top_k) {
    const int mask_words_per_batch = (top_k + 31) / 32;
    const auto host_num_selected =
        copyTensorToHost(device_num_selected, static_cast<size_t>(batch_size));
    const auto host_mask =
        copyTensorToHost(device_suppression_mask, static_cast<size_t>(batch_size) * static_cast<size_t>(mask_words_per_batch));

    writeValueLines(build_dir / "engine_nms_mask_all_batches.txt", host_mask);

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        std::vector<uint32_t> batch_mask(
            host_mask.begin() + static_cast<size_t>(batch_index) * mask_words_per_batch,
            host_mask.begin() + static_cast<size_t>(batch_index + 1) * mask_words_per_batch);
        writeValueLines(build_dir / ("engine_nms_mask_batch" + std::to_string(batch_index) + ".txt"),
                        batch_mask);

        std::ofstream summary_ofs(build_dir / ("engine_nms_summary_batch" + std::to_string(batch_index) + ".txt"));
        summary_ofs << "num_selected " << host_num_selected[batch_index] << "\n";

        std::ostringstream summary_log;
        summary_log << "batch " << batch_index << " num_selected=" << host_num_selected[batch_index];
        int preview = 0;
        for (int rank = 0; rank < std::min<int32_t>(host_num_selected[batch_index], top_k); ++rank) {
            const bool suppressed = (batch_mask[rank / 32] & (1u << (rank % 32))) != 0;
            if (suppressed) {
                summary_ofs << rank << "\n";
                if (preview < 8) {
                    summary_log << " | suppress=" << rank;
                    ++preview;
                }
            }
        }
        LOG_INFO("nms {}", summary_log.str());
    }
}

cv::Rect clampBoxToSlice(const float4 &bbox, int width, int height) {
    const int x1 = std::clamp(static_cast<int>(std::floor(bbox.x)), 0, std::max(width - 1, 0));
    const int y1 = std::clamp(static_cast<int>(std::floor(bbox.y)), 0, std::max(height - 1, 0));
    const int x2 = std::clamp(static_cast<int>(std::ceil(bbox.z)), 0, width);
    const int y2 = std::clamp(static_cast<int>(std::ceil(bbox.w)), 0, height);

    const int box_width = std::max(1, x2 - x1);
    const int box_height = std::max(1, y2 - y1);
    return cv::Rect(x1, y1, box_width, box_height);
}

void dumpFinalSliceVisualizations(const std::filesystem::path &build_dir,
                                  const uint8_t *host_frame_bgr,
                                  int frame_width,
                                  int frame_height,
                                  int slice_width,
                                  int slice_height,
                                  const std::vector<Slice> &slices,
                                  const int32_t *device_final_num_detections,
                                  const float *device_final_scores,
                                  const float4 *device_final_bboxes,
                                  const float2 *device_final_landmarks,
                                  int batch_size,
                                  int top_k) {
    cv::Mat frame(frame_height, frame_width, CV_8UC3, const_cast<uint8_t *>(host_frame_bgr));

    const auto host_counts =
        copyTensorToHost(device_final_num_detections, static_cast<size_t>(batch_size));
    const auto host_scores =
        copyTensorToHost(device_final_scores, static_cast<size_t>(batch_size) * static_cast<size_t>(top_k));
    const auto host_bboxes =
        copyTensorToHost(device_final_bboxes, static_cast<size_t>(batch_size) * static_cast<size_t>(top_k));
    const auto host_landmarks =
        copyTensorToHost(device_final_landmarks, static_cast<size_t>(batch_size) * static_cast<size_t>(top_k) * 5);

    for (int batch_index = 0; batch_index < batch_size && batch_index < static_cast<int>(slices.size()); ++batch_index) {
        const auto &slice = slices[batch_index];
        const cv::Rect roi(slice.x1, slice.y1, slice_width, slice_height);
        cv::Mat slice_image = frame(roi).clone();
        const int32_t final_count = std::min<int32_t>(host_counts[batch_index], top_k);

        for (int32_t det_index = 0; det_index < final_count; ++det_index) {
            const size_t flat_index = static_cast<size_t>(batch_index) * static_cast<size_t>(top_k) +
                                      static_cast<size_t>(det_index);
            const float4 bbox = host_bboxes[flat_index];
            const cv::Rect rect = clampBoxToSlice(bbox, slice_width, slice_height);
            const float score = host_scores[flat_index];

            cv::rectangle(slice_image, rect, cv::Scalar(0, 255, 0), 2);

            const std::string label = cv::format("%.3f", score);
            const cv::Point text_origin(rect.x, std::max(rect.y - 6, 14));
            cv::putText(slice_image,
                        label,
                        text_origin,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 0),
                        1,
                        cv::LINE_AA);

            for (int landmark_idx = 0; landmark_idx < 5; ++landmark_idx) {
                const float2 landmark = host_landmarks[flat_index * 5 + static_cast<size_t>(landmark_idx)];
                const int px = std::clamp(static_cast<int>(std::lround(landmark.x)), 0, std::max(slice_width - 1, 0));
                const int py = std::clamp(static_cast<int>(std::lround(landmark.y)), 0, std::max(slice_height - 1, 0));
                cv::circle(slice_image, cv::Point(px, py), 2, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
            }
        }

        cv::putText(slice_image,
                    cv::format("slice=%d final=%d", batch_index, final_count),
                    cv::Point(12, 24),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 200, 255),
                    2,
                    cv::LINE_AA);

        cv::imwrite((build_dir / ("engine_final_slice_batch" + std::to_string(batch_index) + ".png")).string(),
                    slice_image);
    }
}
#endif

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
    cuda_graph_batch_size_ = batch_sizes_[1];
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

    const int nb_tensors = engine->getNbIOTensors();
    if (nb_tensors != kNumRawOutputs + 1) {
        LOG_ERROR("Expected {} IO tensors (1 input + {} raw outputs), got {}",
                  kNumRawOutputs + 1, kNumRawOutputs, nb_tensors);
        std::exit(EXIT_FAILURE);
    }

    input_tensor_name_ = engine->getIOTensorName(0);
    for (int output_index = 0; output_index < kNumRawOutputs; ++output_index) {
        output_tensor_names_[output_index] = engine->getIOTensorName(output_index + 1);
    }

    context_detection_->setInputShape(input_tensor_name_.c_str(),
                                      nvinfer1::Dims4{batch_sizes_[1], 3, model_input_height_, model_input_width_});

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
                          iou_threshold_,
                          top_k_)) {
        LOG_ERROR("Failed to copy slice coordinates to device constant memory");
        std::exit(EXIT_FAILURE);
    }
    LOG_INFO("Slice coordinates copied to device constant memory");

    context_detection_->setTensorAddress(input_tensor_name_.c_str(), device_jetson_input_buffer_float_);
    for (int output_index = 0; output_index < kNumRawOutputs; ++output_index) {
        context_detection_->setTensorAddress(output_tensor_names_[output_index].c_str(),
                                             device_model_output_buffers_[output_index]);
    }

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

#ifdef LOGRESULTS
void DetectionModelInferenceHelper::dumpLogResults(int batch_size) {
    const std::filesystem::path build_dir = resolveBuildDir();
    const size_t input_elements_per_batch =
        static_cast<size_t>(3) * model_input_height_ * model_input_width_;

    dumpRawTensorAllBatches(build_dir / "engine_input_all_batches.txt",
                            device_jetson_input_buffer_float_,
                            static_cast<size_t>(batch_size) * input_elements_per_batch);
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        dumpRawTensorBatch(build_dir / ("engine_input_batch" + std::to_string(batch_index) + ".txt"),
                           device_jetson_input_buffer_float_,
                           static_cast<size_t>(batch_index),
                           input_elements_per_batch);
    }

    for (int output_index = 0; output_index < kNumRawOutputs; ++output_index) {
        dumpRawTensorAllBatches(build_dir / ("engine_raw_output" + std::to_string(output_index) + "_all_batches.txt"),
                                device_model_output_buffers_[output_index],
                                output_element_counts_[output_index]);

        if (output_has_batch_dim_[output_index]) {
            const int num_batches = std::min(batch_size, output_batch_dims_[output_index]);
            for (int batch_index = 0; batch_index < num_batches; ++batch_index) {
                dumpRawTensorBatch(build_dir / ("engine_raw_output" + std::to_string(output_index) +
                                                "_batch" + std::to_string(batch_index) + ".txt"),
                                   device_model_output_buffers_[output_index],
                                   static_cast<size_t>(batch_index),
                                   output_elements_per_batch_[output_index]);
            }
        }
    }
    dumpScoreSelectionSummary(build_dir,
                              "filtered",
                              device_num_selected_,
                              device_filtered_indexes_,
                              device_filtered_scores_,
                              batch_size,
                              top_k_);
    dumpScoreSelectionSummary(build_dir,
                              "sorted",
                              device_num_selected_,
                              device_sorted_indexes_,
                              device_sorted_scores_,
                              batch_size,
                              top_k_);
    dumpRawTensorAllBatches(build_dir / "engine_sorted_bboxes_all_batches.txt",
                            device_sorted_bboxes_,
                            static_cast<size_t>(batch_size) * static_cast<size_t>(top_k_));
    dumpRawTensorAllBatches(build_dir / "engine_sorted_landmarks_all_batches.txt",
                            device_sorted_landmarks_,
                            static_cast<size_t>(batch_size) * static_cast<size_t>(top_k_) * 5);
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        dumpRawTensorBatch(build_dir / ("engine_sorted_bboxes_batch" + std::to_string(batch_index) + ".txt"),
                           device_sorted_bboxes_,
                           static_cast<size_t>(batch_index),
                           static_cast<size_t>(top_k_));
        dumpRawTensorBatch(build_dir / ("engine_sorted_landmarks_batch" + std::to_string(batch_index) + ".txt"),
                           device_sorted_landmarks_,
                           static_cast<size_t>(batch_index),
                           static_cast<size_t>(top_k_) * 5);
    }
    dumpSuppressionMaskSummary(build_dir,
                               device_suppression_mask_,
                               device_num_selected_,
                               batch_size,
                               top_k_);
    dumpScoreSelectionSummary(build_dir,
                              "final",
                              device_final_num_detections_,
                              device_final_indexes_,
                              device_final_scores_,
                              batch_size,
                              top_k_);
    dumpRawTensorAllBatches(build_dir / "engine_final_bboxes_all_batches.txt",
                            device_final_bboxes_,
                            static_cast<size_t>(batch_size) * static_cast<size_t>(top_k_));
    dumpRawTensorAllBatches(build_dir / "engine_final_landmarks_all_batches.txt",
                            device_final_landmarks_,
                            static_cast<size_t>(batch_size) * static_cast<size_t>(top_k_) * 5);
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        dumpRawTensorBatch(build_dir / ("engine_final_bboxes_batch" + std::to_string(batch_index) + ".txt"),
                           device_final_bboxes_,
                           static_cast<size_t>(batch_index),
                           static_cast<size_t>(top_k_));
        dumpRawTensorBatch(build_dir / ("engine_final_landmarks_batch" + std::to_string(batch_index) + ".txt"),
                           device_final_landmarks_,
                           static_cast<size_t>(batch_index),
                           static_cast<size_t>(top_k_) * 5);
    }
    dumpFinalSliceVisualizations(build_dir,
                                 host_jetson_input_buffer_uint8_t_,
                                 camera_input_width_,
                                 camera_input_height_,
                                 model_input_width_,
                                 model_input_height_,
                                 slices_,
                                 device_final_num_detections_,
                                 device_final_scores_,
                                 device_final_bboxes_,
                                 device_final_landmarks_,
                                 batch_size,
                                 top_k_);
    LOG_INFO("LOGRESULTS wrote raw engine tensors to {}", build_dir.string());
    LOG_INFO("LOGRESULTS inputs: engine_input_all_batches.txt and engine_input_batch*.txt");
    LOG_INFO("LOGRESULTS outputs: engine_raw_output{{0..{}}}_all_batches.txt and optional _batch*.txt", kNumRawOutputs - 1);
    LOG_INFO("LOGRESULTS filter/sort/NMS/final results: engine_filtered_*.txt, engine_sorted_*.txt, engine_sorted_bboxes_*.txt, engine_sorted_landmarks_*.txt, engine_nms_*.txt, engine_final_*.txt, engine_final_slice_batch*.png");
}
#endif


size_t DetectionModelInferenceHelper::graphModeIndex(DetectionBenchmarkMode mode) const {
    switch (mode) {
        case DetectionBenchmarkMode::kInferenceOnly:
            return 0;
        case DetectionBenchmarkMode::kUploadPreprocessInference:
            return 1;
        case DetectionBenchmarkMode::kUploadPreprocessInferenceNms:
            return 2;
    }
    return 0;
}

bool DetectionModelInferenceHelper::shouldUseCudaGraph(int batch_size) const {
    return batch_size == cuda_graph_batch_size_;
}

void DetectionModelInferenceHelper::destroyCudaGraphs() {
    for (auto &graph_exec : cuda_graph_execs_) {
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
    }
    for (auto &graph : cuda_graphs_) {
        if (graph) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
    }
}

void DetectionModelInferenceHelper::ensureCudaGraphCaptured(DetectionBenchmarkMode mode, int batch_size) {
    if (!shouldUseCudaGraph(batch_size)) {
        return;
    }

    const size_t graph_index = graphModeIndex(mode);
    if (cuda_graph_execs_[graph_index]) {
        return;
    }

    if (mode == DetectionBenchmarkMode::kUploadPreprocessInferenceNms) {
        ensureSortStorageInitialized();
    }

    auto run_graph_body = [&]() {
        switch (mode) {
            case DetectionBenchmarkMode::kInferenceOnly:
                stageInferenceOnly();
                break;
            case DetectionBenchmarkMode::kUploadPreprocessInference:
                stagePreprocessOnly(batch_size);
                stageInferenceOnly();
                break;
            case DetectionBenchmarkMode::kUploadPreprocessInferenceNms:
                stagePreprocessOnly(batch_size);
                stageInferenceOnly();
                resetPostprocessBuffers(batch_size);
                stagePostprocess(batch_size);
                stageFaceAlignment(batch_size);
                break;
        }
    };

    run_graph_body();
    cudaStreamSynchronize(stream_);

    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    run_graph_body();
    cudaGraph_t graph = nullptr;
    cudaStreamEndCapture(stream_, &graph);

    cudaGraphExec_t graph_exec = nullptr;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    cuda_graphs_[graph_index] = graph;
    cuda_graph_execs_[graph_index] = graph_exec;

    const char *mode_name = "unknown";
    switch (mode) {
        case DetectionBenchmarkMode::kInferenceOnly:
            mode_name = "inference_only";
            break;
        case DetectionBenchmarkMode::kUploadPreprocessInference:
            mode_name = "upload_preprocess_inference";
            break;
        case DetectionBenchmarkMode::kUploadPreprocessInferenceNms:
            mode_name = "upload_preprocess_inference_nms";
            break;
    }
    LOG_INFO("Captured CUDA graph for batch={} mode={}", batch_size, mode_name);
}

void DetectionModelInferenceHelper::launchCapturedGraph(DetectionBenchmarkMode mode) {
    cudaGraphLaunch(cuda_graph_execs_[graphModeIndex(mode)], stream_);
}

void DetectionModelInferenceHelper::stageUploadHostImage(const uint8_t *host_image) {
    std::memcpy(host_jetson_input_buffer_uint8_t_, host_image, input_buffer_size_uint8_t_);
}

void DetectionModelInferenceHelper::stagePreprocessOnly(int batch_size) {
    launchPreprocessKernel(device_jetson_ptr_uint8_t_, device_jetson_input_buffer_float_, batch_size, stream_);
}

void DetectionModelInferenceHelper::stageUploadAndPreprocess(const uint8_t *host_image, int batch_size) {
    stageUploadHostImage(host_image);
    stagePreprocessOnly(batch_size);
}

void DetectionModelInferenceHelper::stageInferenceOnly() {
    context_detection_->enqueueV3(stream_);
}

void DetectionModelInferenceHelper::resetPostprocessBuffers(int batch_size) {
    cudaMemsetAsync(device_num_selected_, 0, static_cast<size_t>(batch_size) * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_filtered_scores_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_filtered_indexes_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_sorted_scores_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_sorted_indexes_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_sorted_bboxes_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(float4), stream_);
    cudaMemsetAsync(device_sorted_landmarks_, 0, static_cast<size_t>(batch_size) * top_k_ * 5 * sizeof(float2), stream_);
    cudaMemsetAsync(device_final_scores_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(float), stream_);
    cudaMemsetAsync(device_final_indexes_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(int32_t), stream_);
    cudaMemsetAsync(device_final_bboxes_, 0, static_cast<size_t>(batch_size) * top_k_ * sizeof(float4), stream_);
    cudaMemsetAsync(device_final_landmarks_, 0, static_cast<size_t>(batch_size) * top_k_ * 5 * sizeof(float2), stream_);
    cudaMemsetAsync(device_suppression_mask_, 0, static_cast<size_t>(batch_size) * ((top_k_ + 31) / 32) * sizeof(uint32_t), stream_);
    cudaMemsetAsync(device_final_num_detections_, 0, static_cast<size_t>(batch_size) * sizeof(int32_t), stream_);
}

void DetectionModelInferenceHelper::stagePostprocess(int batch_size) {
    launchFilterScoresKernel(batch_size, stream_);
    launchSortFilteredScoresKernel(batch_size, stream_);
    launchGatherAllKernel(batch_size, stream_);
    launchBitmaskNMSKernel(batch_size, stream_);
    launchGatherFinalResultKernel(batch_size, stream_);
}

void DetectionModelInferenceHelper::stageFaceAlignment(int batch_size) {
    launchEstimateSimilarityKernel(batch_size, stream_);
    launchWarpAffineKernel(batch_size, stream_);
}

void DetectionModelInferenceHelper::infer(const uint8_t *host_image, int batch_size) {
    stageUploadHostImage(host_image);
    if (shouldUseCudaGraph(batch_size)) {
        ensureCudaGraphCaptured(DetectionBenchmarkMode::kUploadPreprocessInferenceNms, batch_size);
        launchCapturedGraph(DetectionBenchmarkMode::kUploadPreprocessInferenceNms);
    } else {
        stagePreprocessOnly(batch_size);
        stageInferenceOnly();
        resetPostprocessBuffers(batch_size);
        stagePostprocess(batch_size);
        stageFaceAlignment(batch_size);
    }

    cudaStreamSynchronize(stream_);
    dumpAlignedFaceCrops(device_face_crops_,
                         device_final_num_detections_,
                         device_final_scores_,
                         batch_size,
                         top_k_);

#ifdef LOGRESULTS
    dumpLogResults(batch_size);
#endif

}

void DetectionModelInferenceHelper::benchmark(const uint8_t *host_image,
                                              int batch_size,
                                              int iterations,
                                              DetectionBenchmarkMode mode) {
    if (iterations <= 0) {
        throw std::invalid_argument("Benchmark iterations must be positive");
    }

    auto run_once = [&]() {
        switch (mode) {
            case DetectionBenchmarkMode::kInferenceOnly:
                if (shouldUseCudaGraph(batch_size)) {
                    ensureCudaGraphCaptured(mode, batch_size);
                    launchCapturedGraph(mode);
                } else {
                    stageInferenceOnly();
                }
                break;
            case DetectionBenchmarkMode::kUploadPreprocessInference:
                stageUploadHostImage(host_image);
                if (shouldUseCudaGraph(batch_size)) {
                    ensureCudaGraphCaptured(mode, batch_size);
                    launchCapturedGraph(mode);
                } else {
                    stagePreprocessOnly(batch_size);
                    stageInferenceOnly();
                }
                break;
            case DetectionBenchmarkMode::kUploadPreprocessInferenceNms:
                stageUploadHostImage(host_image);
                if (shouldUseCudaGraph(batch_size)) {
                    ensureCudaGraphCaptured(mode, batch_size);
                    launchCapturedGraph(mode);
                } else {
                    stagePreprocessOnly(batch_size);
                    stageInferenceOnly();
                    resetPostprocessBuffers(batch_size);
                    stagePostprocess(batch_size);
                    stageFaceAlignment(batch_size);
                }
                break;
        }
    };

    // Warm up the exact path that will be measured. For inference-only mode we
    // prepare a valid device input once before timing starts.
    if (mode == DetectionBenchmarkMode::kInferenceOnly) {
        stageUploadAndPreprocess(host_image, batch_size);
    }
    run_once();
    cudaStreamSynchronize(stream_);

    const auto start_time = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_once();
    }
    cudaStreamSynchronize(stream_);
    const auto end_time = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed = end_time - start_time;
    const double total_ms = elapsed.count() * 1000.0;
    const double avg_ms = total_ms / static_cast<double>(iterations);
    const double fps = (avg_ms > 0.0) ? (1000.0 / avg_ms) : 0.0;

    const char *mode_name = "unknown";
    switch (mode) {
        case DetectionBenchmarkMode::kInferenceOnly:
            mode_name = "inference_only";
            break;
        case DetectionBenchmarkMode::kUploadPreprocessInference:
            mode_name = "upload_preprocess_inference";
            break;
        case DetectionBenchmarkMode::kUploadPreprocessInferenceNms:
            mode_name = "upload_preprocess_inference_nms";
            break;
    }

    LOG_INFO("Benchmark mode={} iterations={} total={:.3f} s avg={:.3f} ms fps={:.2f}",
             mode_name,
             iterations,
             elapsed.count(),
             avg_ms,
             fps);
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
    

    for (int output_index = 0; output_index < kNumRawOutputs; ++output_index) {
        const auto dims = context_detection_->getTensorShape(output_tensor_names_[output_index].c_str());
        const size_t total_elements = dimsVolume(dims);
        const bool has_batch_dim = dims.nbDims >= 2 && dims.d[0] == batch_sizes_[1];
        const int batch_dim = has_batch_dim ? dims.d[0] : 1;
        const size_t elements_per_batch = total_elements / static_cast<size_t>(batch_dim);
        const size_t buffer_size_bytes = total_elements * sizeof(float);

        output_has_batch_dim_[output_index] = has_batch_dim;
        output_batch_dims_[output_index] = batch_dim;
        output_element_counts_[output_index] = total_elements;
        output_elements_per_batch_[output_index] = elements_per_batch;

        cudaMalloc(&device_model_output_buffers_[output_index], buffer_size_bytes);
        memory_usage += buffer_size_bytes / (1024.0 * 1024.0);

        LOG_DEBUG("\t\tAllocated raw output {} ({}) dims={} total_elements={} elements_per_batch={} has_batch_dim={} size={} mb",
                  output_index,
                  output_tensor_names_[output_index],
                  dimsToString(dims),
                  total_elements,
                  elements_per_batch,
                  has_batch_dim,
                  buffer_size_bytes / (1024.0 * 1024.0));
    }

    const size_t filtered_capacity = static_cast<size_t>(batch_sizes_[2]) * static_cast<size_t>(top_k_);
    cudaMalloc(&device_filtered_scores_, filtered_capacity * sizeof(float));
    cudaMalloc(&device_filtered_indexes_, filtered_capacity * sizeof(int32_t));
    cudaMalloc(&device_sorted_scores_, filtered_capacity * sizeof(float));
    cudaMalloc(&device_sorted_indexes_, filtered_capacity * sizeof(int32_t));
    cudaMalloc(&device_sorted_bboxes_, filtered_capacity * sizeof(float4));
    cudaMalloc(&device_sorted_landmarks_, filtered_capacity * 5 * sizeof(float2));
    cudaMalloc(&device_final_scores_, filtered_capacity * sizeof(float));
    cudaMalloc(&device_final_indexes_, filtered_capacity * sizeof(int32_t));
    cudaMalloc(&device_final_bboxes_, filtered_capacity * sizeof(float4));
    cudaMalloc(&device_final_landmarks_, filtered_capacity * 5 * sizeof(float2));
    cudaMalloc(&device_face_affine_matrices_,
               filtered_capacity * kAffineMatrixElements * sizeof(float));
    cudaMalloc(&device_face_crops_,
               filtered_capacity * kAlignedFaceChannels * kAlignedFaceHeight * kAlignedFaceWidth * sizeof(float));
    cudaMalloc(&device_suppression_mask_,
               static_cast<size_t>(batch_sizes_[2]) * static_cast<size_t>((top_k_ + 31) / 32) * sizeof(uint32_t));
    cudaMalloc(&device_num_selected_, static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t));
    cudaMalloc(&device_final_num_detections_, static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t));

    memory_usage += filtered_capacity * sizeof(float) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * sizeof(int32_t) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * sizeof(float) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * sizeof(int32_t) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * sizeof(float4) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * 5 * sizeof(float2) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * sizeof(float) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * sizeof(int32_t) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * sizeof(float4) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * 5 * sizeof(float2) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * kAffineMatrixElements * sizeof(float) / (1024.0 * 1024.0);
    memory_usage += filtered_capacity * kAlignedFaceChannels * kAlignedFaceHeight * kAlignedFaceWidth * sizeof(float) / (1024.0 * 1024.0);
    memory_usage += static_cast<size_t>(batch_sizes_[2]) * static_cast<size_t>((top_k_ + 31) / 32) * sizeof(uint32_t) / (1024.0 * 1024.0);
    memory_usage += static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t) / (1024.0 * 1024.0);
    memory_usage += static_cast<size_t>(batch_sizes_[2]) * sizeof(int32_t) / (1024.0 * 1024.0);

    
    LOG_DEBUG("\t\tAllocated zero-copy uint8 input buffer: {} mb", input_buffer_size_uint8_t_ / (1024.0 * 1024.0));
    LOG_DEBUG("\t\tAllocated device float input buffer: {} mb", input_buffer_size_float_ / (1024.0 * 1024.0));
    LOG_DEBUG("\t\tTotal anchors across all strides: {}", anchor_count_);
    LOG_DEBUG("\tCalculated size for device float input buffer: {} mb", input_buffer_size_float_ / (1024.0 * 1024.0));
    LOG_INFO("\tTotal GPU memory allocated for buffers: {} mb", memory_usage);
    

}

void DetectionModelInferenceHelper::freeBuffers() {
    LOG_INFO("***** freeBuffers *****");
    destroyCudaGraphs();

    if (host_jetson_input_buffer_uint8_t_) {
        cudaFreeHost(host_jetson_input_buffer_uint8_t_);
        host_jetson_input_buffer_uint8_t_ = nullptr;
        device_jetson_ptr_uint8_t_ = nullptr; // invalidated when host is freed
    }
    if (device_jetson_input_buffer_float_) {
        cudaFree(device_jetson_input_buffer_float_);
        device_jetson_input_buffer_float_ = nullptr;
    }
    for (auto &buffer : device_model_output_buffers_) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    if (device_filtered_scores_) {
        cudaFree(device_filtered_scores_);
        device_filtered_scores_ = nullptr;
    }
    if (device_filtered_indexes_) {
        cudaFree(device_filtered_indexes_);
        device_filtered_indexes_ = nullptr;
    }
    if (device_sorted_scores_) {
        cudaFree(device_sorted_scores_);
        device_sorted_scores_ = nullptr;
    }
    if (device_sorted_indexes_) {
        cudaFree(device_sorted_indexes_);
        device_sorted_indexes_ = nullptr;
    }
    if (device_sorted_bboxes_) {
        cudaFree(device_sorted_bboxes_);
        device_sorted_bboxes_ = nullptr;
    }
    if (device_sorted_landmarks_) {
        cudaFree(device_sorted_landmarks_);
        device_sorted_landmarks_ = nullptr;
    }
    if (device_final_scores_) {
        cudaFree(device_final_scores_);
        device_final_scores_ = nullptr;
    }
    if (device_final_indexes_) {
        cudaFree(device_final_indexes_);
        device_final_indexes_ = nullptr;
    }
    if (device_final_bboxes_) {
        cudaFree(device_final_bboxes_);
        device_final_bboxes_ = nullptr;
    }
    if (device_final_landmarks_) {
        cudaFree(device_final_landmarks_);
        device_final_landmarks_ = nullptr;
    }
    if (device_face_affine_matrices_) {
        cudaFree(device_face_affine_matrices_);
        device_face_affine_matrices_ = nullptr;
    }
    if (device_face_crops_) {
        cudaFree(device_face_crops_);
        device_face_crops_ = nullptr;
    }
    if (device_suppression_mask_) {
        cudaFree(device_suppression_mask_);
        device_suppression_mask_ = nullptr;
    }
    if (device_num_selected_) {
        cudaFree(device_num_selected_);
        device_num_selected_ = nullptr;
    }
    if (device_final_num_detections_) {
        cudaFree(device_final_num_detections_);
        device_final_num_detections_ = nullptr;
    }
    if (device_sort_storage_) {
        cudaFree(device_sort_storage_);
        device_sort_storage_ = nullptr;
        sort_storage_bytes_ = 0;
    }
}
