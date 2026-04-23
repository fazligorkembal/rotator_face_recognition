#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include <opencv2/opencv.hpp>
#include "tensorrt/logging.h"
#include <json.hpp>
#include "spd_logger_helper.h"

#include "detection_model_inference_helper.h"
#include "utils.hpp"
// #include "identification_model_inference_helper.h"




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

namespace {

void printUsage(const char *program_name)
{
    LOG_INFO("Usage: {} [mode] [iterations]", program_name);
    LOG_INFO("  mode 1 | raw   : model inference only");
    LOG_INFO("  mode 2 | prep  : cpu upload + preprocessing + inference");
    LOG_INFO("  mode 3 | full  : cpu upload + preprocessing + inference + GPU NMS + warpAffine");
    LOG_INFO("  iterations defaults to 5000");
}

DetectionBenchmarkMode parseBenchmarkMode(std::string_view mode_arg)
{
    if (mode_arg == "1" || mode_arg == "raw" || mode_arg == "inference" || mode_arg == "inference_only")
    {
        return DetectionBenchmarkMode::kInferenceOnly;
    }
    if (mode_arg == "2" || mode_arg == "prep" || mode_arg == "preprocess" || mode_arg == "upload_preprocess")
    {
        return DetectionBenchmarkMode::kUploadPreprocessInference;
    }
    if (mode_arg == "3" || mode_arg == "full" || mode_arg == "nms" || mode_arg == "full_pipeline")
    {
        return DetectionBenchmarkMode::kUploadPreprocessInferenceNms;
    }

    throw std::invalid_argument("Unknown benchmark mode");
}

} // namespace

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

int main(int argc, char **argv)
{
    LOG_INFO("Starting face recognition with CUDA NMS");

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

    // nvinfer1::IRuntime *iRuntimeIdentifier = nullptr;
    // nvinfer1::ICudaEngine *iCudaEngineIdentifier = nullptr;

    deserializeDetectionEngine(data_config["detection"]["model_path"], iRuntimeDetection, iCudaEngineDetection);
    //

    std::vector<int> batch_sizes = data_config["detection"]["batch_sizes"].get<std::vector<int>>();
    std::vector<int> strides = data_config["detection"]["strides"].get<std::vector<int>>();
    int32_t top_k = data_config["general"]["top_k"];
    int32_t height = data_config["detection"]["input_size"];
    int32_t width = data_config["detection"]["input_size"];
    float confidence_threshold = data_config["detection"]["conf_threshold"];
    float iou_threshold = data_config["detection"]["nms_threshold"];
    int32_t camera_height  = data_config["camera"]["height"];
    int32_t camera_width   = data_config["camera"]["width"];
    int32_t num_slices_x   = data_config["detection"]["num_slices_x"];
    int32_t num_slices_y   = data_config["detection"]["num_slices_y"];
    int32_t gap_x          = data_config["detection"]["gap_x"];
    int32_t gap_y          = data_config["detection"]["gap_y"];

    

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
        gap_y);

    std::string image_path = data_config["detection"]["test_image_path"];
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        LOG_ERROR("Failed to load image: {}", image_path);
        return -1;
    }
    cv::resize(img, img, cv::Size(camera_width, camera_height));

    const int batch = num_slices_x * num_slices_y;

    if (argc >= 2) {
        try {
            const DetectionBenchmarkMode mode = parseBenchmarkMode(argv[1]);
            const int iterations = (argc >= 3) ? std::stoi(argv[2]) : 5000;
            inferenceHelper.benchmark(img.data, batch, iterations, mode);
        } catch (const std::exception &e) {
            LOG_ERROR("{}", e.what());
            printUsage(argv[0]);
            return -1;
        }
    } else {
        inferenceHelper.infer(img.data, batch);
    }

    LOG_INFO("Done");
}
