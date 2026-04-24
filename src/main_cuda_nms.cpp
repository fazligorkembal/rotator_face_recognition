#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include "tensorrt/logging.h"
#include <json.hpp>
#include "spd_logger_helper.h"

#include "detection_model_inference_helper.h"
#include "identification_model_inference_helper.h"
#include "utils.hpp"

// #include "bounded_queue.hpp"
// #include "inference_objects.hpp"
// #include "ui_helper.h"

using json = nlohmann::json;
std::string path_config = "/home/user/Documents/rfr/configs/calib.json";

std::atomic<bool> is_run{true};
std::atomic<bool> is_tracking{false};

// Logger gLogger;
Logger gLogger(Severity::kINFO);

using namespace nvinfer1;
using namespace cv;

void deserializeDetectionEngine(const std::string &model_path_, nvinfer1::IRuntime *&runtime_detection_, nvinfer1::ICudaEngine *&engine_detection_)
{

    LOG_INFO("*******************************************************");
    LOG_INFO("Deserializing TensorRT Engine from file: {}", model_path_);
    if (model_path_.empty())
    {
        LOG_ERROR("Model path is empty or does not end with .engine!");
        throw std::runtime_error("Invalid model path");
    }

    std::ifstream engine_file(model_path_, std::ios::binary);
    if (!engine_file.good())
    {
        LOG_ERROR("Failed to open engine file: {}", model_path_);
        throw std::runtime_error("Failed to open engine file");
    }

    size_t engine_size = 0;
    engine_file.seekg(0, engine_file.end);
    engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    char *serialized_engine = new char[engine_size];
    if (!serialized_engine)
    {
        LOG_ERROR("Failed to allocate memory for serialized engine");
        throw std::runtime_error("Memory allocation failed");
    }
    engine_file.read(serialized_engine, engine_size);
    engine_file.close();

    runtime_detection_ = createInferRuntime(gLogger);
    if (!runtime_detection_)
    {
        LOG_ERROR("Failed to create TensorRT runtime");
        delete[] serialized_engine;
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_detection_ = (ICudaEngine *)runtime_detection_->deserializeCudaEngine(serialized_engine, engine_size);
    if (!engine_detection_)
    {
        LOG_ERROR("Failed to deserialize CUDA engine");
        delete[] serialized_engine;
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }

    delete[] serialized_engine;
    LOG_INFO("TensorRT Engine deserialized successfully, engine size: {} mb", engine_size / (1024.0 * 1024.0));
}

void deserializeInferenceEngine(const std::string model_path, nvinfer1::IRuntime *&runtime_identification, nvinfer1::ICudaEngine *&engine_identification)
{
    LOG_INFO("*******************************************************");
    LOG_INFO("Deserializing TensorRT Engine from file: {}", model_path);

    if (model_path.empty())
    {
        LOG_ERROR("Model path is empty or does not end with .engine!");
        throw std::runtime_error("Invalid model path");
    }

    // Read the engine file
    std::ifstream engine_file(model_path, std::ios::binary);
    if (!engine_file)
    {
        LOG_ERROR("Failed to open engine file: {}", model_path);
        throw std::runtime_error("Failed to open engine file");
    }

    size_t engine_size = 0;
    engine_file.seekg(0, engine_file.end);
    engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    char *serialized_engine = new char[engine_size];
    if (!serialized_engine)
    {
        LOG_ERROR("Failed to allocate memory for serialized engine");
        throw std::runtime_error("Memory allocation failed");
    }

    engine_file.read(serialized_engine, engine_size);
    engine_file.close();

    runtime_identification = createInferRuntime(gLogger);
    if (!runtime_identification)
    {
        LOG_ERROR("Failed to create TensorRT runtime");
        delete[] serialized_engine;
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_identification = (ICudaEngine *)runtime_identification->deserializeCudaEngine(serialized_engine, engine_size);
    if (!engine_identification)
    {
        LOG_ERROR("Failed to deserialize CUDA engine");
        delete[] serialized_engine;
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }

    delete[] serialized_engine;
    LOG_INFO("TensorRT Engine deserialized successfully");
}

int main(int argc, char **argv)
{
    LOG_INFO("Starting face recognition with CUDA NMS");
    const bool face_record_mode = argc >= 2 && std::string_view(argv[1]) == "face_record";

    // ----- Load configuration -----
    nlohmann::json data_config;
    try
    {
        loadConfig(path_config, data_config);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error loading config: {}", e.what());
        return -1;
    }

    spdlog::set_level(spdlog::level::from_str(data_config["general"].value("log_level", "info")));

    // ----- End of configuration loading -----
    nvinfer1::IRuntime *iRuntimeDetection = nullptr;
    nvinfer1::ICudaEngine *iCudaEngineDetection = nullptr;
    nvinfer1::IRuntime *iRuntimeIdentification = nullptr;
    nvinfer1::ICudaEngine *iCudaEngineIdentification = nullptr;

    deserializeDetectionEngine(data_config["detection"]["model_path"], iRuntimeDetection, iCudaEngineDetection);
    deserializeInferenceEngine(data_config["identification"]["model_path"], iRuntimeIdentification, iCudaEngineIdentification);

    std::vector<int> batch_sizes = data_config["detection"]["batch_sizes"].get<std::vector<int>>();
    std::vector<int> identification_batch_sizes =
        data_config["identification"]["batch_sizes"].get<std::vector<int>>();
    std::vector<int> strides = data_config["detection"]["strides"].get<std::vector<int>>();
    int32_t top_k = data_config["detection"]["detection_top_k"];
    int32_t height = data_config["detection"]["input_size"];
    int32_t width = data_config["detection"]["input_size"];
    float confidence_threshold = data_config["detection"]["detection_conf_threshold"];
    float iou_threshold = data_config["detection"]["detection_nms_threshold"];
    int32_t min_box_length = data_config["identification"]["identifier_threshold_min_box_length"];
    int32_t camera_height = data_config["camera"]["height"];
    int32_t camera_width = data_config["camera"]["width"];
    double time_interval_face_record = data_config["camera"].value("time_interval_face_record", 1.0);
    if (time_interval_face_record <= 0.0)
    {
        time_interval_face_record = 1.0;
    }
    int32_t num_slices_x = data_config["detection"]["num_slices_x"];
    int32_t num_slices_y = data_config["detection"]["num_slices_y"];
    int32_t gap_x = data_config["detection"]["gap_x"];
    int32_t gap_y = data_config["detection"]["gap_y"];

    LOG_INFO("Identification batch sizes: min={}, opt={}, max={}",
             identification_batch_sizes[0],
             identification_batch_sizes[1],
             identification_batch_sizes[2]);

    DetectionModelInferenceHelper inferenceHelper(
        iCudaEngineDetection,
        batch_sizes,
        height,
        width,
        strides,
        top_k,
        confidence_threshold,
        iou_threshold,
        camera_height,
        camera_width,
        num_slices_x,
        num_slices_y,
        gap_x,
        gap_y,
        min_box_length);
    
    std::filesystem::path face_record_dir;
    int face_record_index = 0;
    double last_face_record_time = -time_interval_face_record;
    if (face_record_mode)
    {
        auto build_dir = std::filesystem::current_path();
        if (build_dir.filename() != "build")
        {
            build_dir /= "build";
        }
        face_record_dir = build_dir / "face_record";
        std::filesystem::create_directories(face_record_dir);
        LOG_INFO("Face record mode enabled. Saving crops to {}", face_record_dir.string());
    }

    //read opencv video
    std::string video_path = data_config["camera"]["test_video_path"];
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        LOG_ERROR("Failed to open video: {}", video_path);
        return -1;
    }

    const double video_fps = cap.get(cv::CAP_PROP_FPS);
    int frame_index = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            LOG_ERROR("Failed to read frame from video");
            break;
        }
        cv::resize(frame, frame, cv::Size(camera_width, camera_height));
        auto start_time = std::chrono::high_resolution_clock::now();

        if (face_record_mode)
        {
            inferenceHelper.infer(frame.data, DetectionType::SEARCH);

            const double frame_time_sec =
                video_fps > 0.0 ? static_cast<double>(frame_index) / video_fps : static_cast<double>(frame_index);
            if (frame_time_sec - last_face_record_time >= time_interval_face_record)
            {
                int saved_count = 0;
                const auto detections = inferenceHelper.getLastDetections(DetectionType::SEARCH);
                const cv::Rect frame_rect(0, 0, frame.cols, frame.rows);

                for (const auto &detection : detections)
                {
                    cv::Rect crop_rect(
                        detection.x1,
                        detection.y1,
                        detection.x2 - detection.x1,
                        detection.y2 - detection.y1);
                    crop_rect &= frame_rect;

                    if (crop_rect.width < min_box_length || crop_rect.height < min_box_length)
                    {
                        continue;
                    }

                    const auto crop_path =
                        face_record_dir / ("face_" + std::to_string(face_record_index++) + ".jpg");
                    if (cv::imwrite(crop_path.string(), frame(crop_rect).clone()))
                    {
                        ++saved_count;
                    }
                }

                last_face_record_time = frame_time_sec;
                LOG_INFO("Recorded {} face crops at {:.2f}s", saved_count, frame_time_sec);
            }
        }
        else
        {
            const cv::Rect track_crop_rect((frame.cols - width) / 2,
                                           (frame.rows - height) / 2,
                                           width,
                                           height);
            cv::Mat track_crop = frame(track_crop_rect).clone();
            inferenceHelper.infer(track_crop.data,
                                  DetectionType::TRACK,
                                  frame.data,
                                  frame.cols,
                                  frame.rows,
                                  track_crop_rect.x,
                                  track_crop_rect.y);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        LOG_INFO("Inference Mode: {}, Elp Time: {:.2f} ms", face_record_mode ? "SEARCH" : "TRACK", elapsed.count() * 1000);
        ++frame_index;
    }

    /*
    std::string image_path = data_config["detection"]["test_image_path"];
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        LOG_ERROR("Failed to load image: {}", image_path);
        return -1;
    }
    cv::resize(img, img, cv::Size(camera_width, camera_height));
    inferenceHelper.infer(img.data, DetectionType::SEARCH);
    */

    /*
    const cv::Rect track_crop_rect(1152, 440, width, height);
    if (track_crop_rect.x < 0 ||
        track_crop_rect.y < 0 ||
        track_crop_rect.x + track_crop_rect.width > img.cols ||
        track_crop_rect.y + track_crop_rect.height > img.rows)
    {
        LOG_ERROR("Track crop rect is outside image bounds");
        return -1;
    }

    cv::Mat track_crop = img(track_crop_rect).clone();

    auto build_dir = std::filesystem::current_path();
    if (build_dir.filename() != "build")
    {
        build_dir /= "build";
    }
    const auto log_dir = build_dir / "logresults";
    std::filesystem::create_directories(log_dir);
    cv::imwrite((log_dir / "track_input_crop.jpg").string(), track_crop);

    auto start_time = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 5000; ++i)
    inferenceHelper.infer(track_crop.data,
                          DetectionType::TRACK,
                          img.data,
                          img.cols,
                          img.rows,
                          track_crop_rect.x,
                          track_crop_rect.y);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    LOG_INFO("Average inference time per frame: {:.2f} ms", (elapsed.count() * 1000) / 5000);
    */
    LOG_INFO("Done");
}
