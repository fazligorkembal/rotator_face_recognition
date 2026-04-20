#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "detection_model_inference_helper.h"
// #include "identification_model_inference_helper.h"
#include "source_manager.hpp"
#include "tensorrt/logging.h"
#include <json.hpp>
#include "utils.hpp"
// #include "bounded_queue.hpp"
// #include "inference_objects.hpp"
// #include "ui_helper.h"

using json = nlohmann::json;
std::string path_config = "/home/user/Documents/rfr/configs/calib.json";

std::atomic<bool> is_run{true};
std::atomic<bool> is_tracking{false};

//Logger gLogger;
Logger gLogger(Severity::kINFO);

using namespace nvinfer1;
using namespace cv;

void deserializeDetectionEngine(const std::string &model_path_, nvinfer1::IRuntime *&runtime_detection_, nvinfer1::ICudaEngine *&engine_detection_)
{

    LOG_INFO(gLogger) << "*******************************************************" << std::endl;
    LOG_INFO(gLogger) << "Deserializing TensorRT Engine from file: " << model_path_ << std::endl;
    if (model_path_.empty())
    {
        LOG_ERROR(gLogger) << "Model path is empty or does not end with .engine!" << std::endl;
        throw std::runtime_error("Invalid model path");
    }

    std::ifstream engine_file(model_path_, std::ios::binary);
    if (!engine_file.good())
    {
        LOG_ERROR(gLogger) << "Failed to open engine file: " << model_path_ << std::endl;
        throw std::runtime_error("Failed to open engine file");
    }

    size_t engine_size = 0;
    engine_file.seekg(0, engine_file.end);
    engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    char *serialized_engine = new char[engine_size];
    if (!serialized_engine)
    {
        LOG_ERROR(gLogger) << "Failed to allocate memory for serialized engine" << std::endl;
        throw std::runtime_error("Memory allocation failed");
    }
    engine_file.read(serialized_engine, engine_size);
    engine_file.close();

    runtime_detection_ = createInferRuntime(gLogger);
    if (!runtime_detection_)
    {
        LOG_ERROR(gLogger) << "Failed to create TensorRT runtime" << std::endl;
        delete[] serialized_engine;
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_detection_ = (ICudaEngine *)runtime_detection_->deserializeCudaEngine(serialized_engine, engine_size);
    if (!engine_detection_)
    {
        LOG_ERROR(gLogger) << "Failed to deserialize CUDA engine" << std::endl;
        delete[] serialized_engine;
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }

    delete[] serialized_engine;
    LOG_INFO(gLogger) << "TensorRT Engine deserialized and context created successfully, engine size: " << engine_size / (1024.0 * 1024.0) << " mb" << std::endl;
}


int main()
{
    LOG_INFO(gLogger) << "Starting face recognition with CUDA NMS" << std::endl;
    
    // ----- Load configuration -----
    nlohmann::json data_config;
    try
    {
        loadConfig(path_config, data_config);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR(gLogger) << "Error loading config: " << e.what() << std::endl;
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

    
    auto thread_inference = std::thread([&]()
    {
        DetectionModelInferenceHelper inferenceHelper(
                                                iCudaEngineDetection,
                                                batch_sizes,
                                                height,
                                                width,
                                                strides,
                                                top_k,
                                                confidence_threshold,
                                                iou_threshold);
    });

    
    

    if (thread_inference.joinable())
        thread_inference.join();

    LOG_INFO(gLogger) << "Done" << std::endl;
}