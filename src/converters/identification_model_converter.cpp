#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            spdlog::info("[TensorRT] {}", msg);
    }
} gLogger;

bool convertToEngine(const std::string &onnxFile, const std::string &enginePath)

{
    // Initialize CUDA device
    int deviceCount = 0;
    cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
    if (cudaErr != cudaSuccess || deviceCount == 0)
    {
        spdlog::error("No CUDA devices found or CUDA error: {}", cudaGetErrorString(cudaErr));
        return false;
    }

    cudaErr = cudaSetDevice(0);
    if (cudaErr != cudaSuccess)
    {
        spdlog::error("Failed to set CUDA device: {}", cudaGetErrorString(cudaErr));
        return false;
    }
    spdlog::info("CUDA device initialized successfully");

    std::unique_ptr<IBuilder> builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));

    if (!builder)
    {
        spdlog::error("Failed to create TensorRT builder");
        return false;
    }
    uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }
    else
    {
        spdlog::info("Network created successfully");
    }

    initLibNvInferPlugins(&gLogger, "");
    spdlog::info("Plugin registry initialized");

    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        return false;
    }
    else
    {
        spdlog::info("Parser created successfully");
    }

    std::ifstream file(onnxFile, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    spdlog::info("Parsing ONNX model from file: {}, size: {} mb", onnxFile, size / (1024 * 1024));

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        auto msg = "Error, unable to read engine file";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed)
    {
        return false;
    }

    auto nbInputs = network->getNbInputs();
    spdlog::info("ONNX model has {} inputs", nbInputs);

    for (int input = 0; input < nbInputs; ++input)
    {
        auto inputTensor = network->getInput(input);
        auto dims = inputTensor->getDimensions();
        std::string shape = "[";
        for (int i = 0; i < dims.nbDims; ++i)
        {
            shape += std::to_string(dims.d[i]);
            if (i < dims.nbDims - 1)
                shape += ", ";
        }
        shape += "]";
        spdlog::info("{} - Input Name: {}, Shape: {}, Type: {}", input, inputTensor->getName(), shape, static_cast<int>(inputTensor->getType()));
    }

    auto nbOutputs = network->getNbOutputs();
    spdlog::info("ONNX model has {} outputs", nbOutputs);

    for (int output = 0; output < nbOutputs; ++output)
    {
        auto outputTensor = network->getOutput(output);
        auto dims = outputTensor->getDimensions();
        std::string shape = "[";
        for (int i = 0; i < dims.nbDims; ++i)
        {
            shape += std::to_string(dims.d[i]);
            if (i < dims.nbDims - 1)
                shape += ", ";
        }
        shape += "]";
        spdlog::info("{} - Output Name: {}, Shape: {}, Type: {}", output, outputTensor->getName(), shape, static_cast<int>(outputTensor->getType()));
    }

    const auto numInputs = network->getNbInputs();
    if (numInputs < 1)
    {
        auto msg = "Error, model needs at least 1 input!";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    spdlog::info("Model has {} inputs", numInputs);

    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i)
    {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch)
        {
            auto msg = "Error, the model has multiple inputs, each with differing batch sizes!";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }
    if (input0Batch == -1)
    {
        spdlog::info("Model supports dynamic batch size");
    }
    else
    {
        spdlog::info("Model only supports fixed batch size of {}", input0Batch);
    }

    std::unique_ptr<IBuilderConfig> config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    if (input0Batch == -1)
    {
        spdlog::info("Configuring optimization profile: min=1, opt=1, max=8");
        nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
        if (!profile)
        {
            return false;
        }

        for (int32_t i = 0; i < numInputs; ++i)
        {
            auto inputTensor = network->getInput(i);
            auto dims = inputTensor->getDimensions();
            std::string inputName = inputTensor->getName();

            nvinfer1::Dims minDims = dims;
            nvinfer1::Dims optDims = dims;
            nvinfer1::Dims maxDims = dims;
            minDims.d[0] = 1;
            optDims.d[0] = 1;
            maxDims.d[0] = 8;

            profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, minDims);
            profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, optDims);
            profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, maxDims);
        }

        config->addOptimizationProfile(profile);
    }

    if (!builder->platformHasFastFp16())
    {
        spdlog::warn("FP16 not supported on this platform");
    }
    else if (!builder->platformHasFastInt8())
    {
        // TODO add INT8 support
        spdlog::warn("INT8 not supported on this platform");
    }

    config->setFlag(BuilderFlag::kFP16);
    spdlog::warn("For now, only FP16 is enabled"); // TODO: make this configurable and add other types support

    // Create and set CUDA stream for profiling
    cudaStream_t profileStream;
    cudaError_t err = cudaStreamCreate(&profileStream);
    if (err != cudaSuccess)
    {
        spdlog::error("Failed to create CUDA stream: {}", cudaGetErrorString(err));
        return false;
    }
    config->setProfileStream(profileStream);

    spdlog::info("Building TensorRT engine, this may take several minutes...");
    std::unique_ptr<nvinfer1::IHostMemory> plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    // Clean up CUDA stream
    cudaStreamDestroy(profileStream);

    if (!plan)
    {
        spdlog::error("Failed to build serialized network");
        return false;
    }

    spdlog::info("Engine built successfully, saving to file...");
    std::ofstream outfile(enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    spdlog::info("Success, saved engine to {}", enginePath);

    return true;
}

int main()
{
    spdlog::info("Starting model conversion...");
    std::string onnxFile = "/home/user/Documents/rfr/models/r50.onnx";
    std::string engineFile = "/home/user/Documents/rfr/models/r50.engine";
    if (convertToEngine(onnxFile, engineFile))
    {
        spdlog::info("Model conversion completed successfully.");
    }
    else
    {
        spdlog::error("Model conversion failed.");
    }
    return 0;
}