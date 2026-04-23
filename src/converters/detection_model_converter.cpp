#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <array>
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

namespace
{

struct OutputLayoutSpec
{
    int anchors;
    int channels;
};

ITensor *normalizeRawHeadOutput(INetworkDefinition &network,
                                ITensor &raw_output,
                                const std::string &raw_name,
                                int anchors,
                                int channels)
{
    const int anchor_stack = 2;
    if (anchors % anchor_stack != 0)
    {
        throw std::runtime_error("Anchor count must be divisible by anchor_stack");
    }

    const auto dims = raw_output.getDimensions();
    if (dims.nbDims != 2 || dims.d[1] != channels)
    {
        throw std::runtime_error("Unexpected raw SCRFD output shape for " + raw_name);
    }

    const int cells = anchors / anchor_stack;

    auto *shuffle_to_4d = network.addShuffle(raw_output);
    if (!shuffle_to_4d)
    {
        throw std::runtime_error("Failed to create shuffle_to_4d for " + raw_name);
    }
    shuffle_to_4d->setName((raw_name + "_cell_batch_anchor").c_str());
    shuffle_to_4d->setReshapeDimensions(Dims4{cells, -1, anchor_stack, channels});
    Permutation batch_major_permutation{};
    batch_major_permutation.order[0] = 1;
    batch_major_permutation.order[1] = 0;
    batch_major_permutation.order[2] = 2;
    batch_major_permutation.order[3] = 3;
    shuffle_to_4d->setSecondTranspose(batch_major_permutation);

    auto *shuffle_to_3d = network.addShuffle(*shuffle_to_4d->getOutput(0));
    if (!shuffle_to_3d)
    {
        throw std::runtime_error("Failed to create shuffle_to_3d for " + raw_name);
    }
    shuffle_to_3d->setName((raw_name + "_batch_major").c_str());
    shuffle_to_3d->setReshapeDimensions(Dims3{-1, anchors, channels});

    return shuffle_to_3d->getOutput(0);
}

} // namespace

bool convertToEngine(const std::string &onnxFile, const std::string &enginePath)

{
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
    spdlog::info("ONNX parser graph has {} inputs", nbInputs);

    auto nbOutputs = network->getNbOutputs();
    spdlog::info("ONNX parser graph has {} outputs", nbOutputs);

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
        spdlog::info("Raw ONNX output {} - Name: {}, Shape: {}, Type: {}", output, outputTensor->getName(), shape, static_cast<int>(outputTensor->getType()));
    }

    if (nbOutputs != 9)
    {
        throw std::runtime_error("Expected 9 raw SCRFD outputs from ONNX parser");
    }

    const std::array<OutputLayoutSpec, 9> output_specs = {{
        {12800, 1},
        {3200, 1},
        {800, 1},
        {12800, 4},
        {3200, 4},
        {800, 4},
        {12800, 10},
        {3200, 10},
        {800, 10},
    }};

    spdlog::info("Normalizing {} raw ONNX outputs to batch-major [batch, anchors, channels]", nbOutputs);

    std::array<ITensor *, 9> raw_outputs{};
    std::array<ITensor *, 9> normalized_outputs{};
    std::array<std::string, 9> raw_names{};

    for (int output = 0; output < nbOutputs; ++output)
    {
        raw_outputs[output] = network->getOutput(output);
        raw_names[output] = raw_outputs[output]->getName();
    }

    for (int output = 0; output < nbOutputs; ++output)
    {
        normalized_outputs[output] = normalizeRawHeadOutput(
            *network,
            *raw_outputs[output],
            raw_names[output],
            output_specs[output].anchors,
            output_specs[output].channels);
    }

    for (int output = 0; output < nbOutputs; ++output)
    {
        network->unmarkOutput(*raw_outputs[output]);
    }

    for (int output = 0; output < nbOutputs; ++output)
    {
        normalized_outputs[output]->setName(raw_names[output].c_str());
        network->markOutput(*normalized_outputs[output]);
    }

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
        spdlog::info("TensorRT output {} - Name: {}, Shape: {}, Type: {}", output, outputTensor->getName(), shape, static_cast<int>(outputTensor->getType()));
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

    // Check to see if the model supports dynamic batch size or not
    if (input0Batch == -1)
    {
        spdlog::info("Model supports dynamic batch size");
    }
    else
    {
        spdlog::info("Model only supports fixed batch size of {}", input0Batch);
    }

    const auto input3Batch = network->getInput(0)->getDimensions().d[3];
    if (input3Batch == -1)
    {
        spdlog::info("Model supports dynamic width. Using Options.maxInputWidth, Options.minInputWidth, and Options.optInputWidth to set the input width.");
    }
    else
    {
        spdlog::info("Model only supports fixed width of {}", input3Batch);
    }

    std::unique_ptr<IBuilderConfig> config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    if (!profile)
    {
        return false;
    }

    const char *inputName = network->getInput(0)->getName();
    spdlog::info("Setting optimization profile for input: {}", inputName);
    profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4{1, 3, 640, 640});
    profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4{6, 3, 640, 640});
    profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4{6, 3, 640, 640});
    config->addOptimizationProfile(profile);
    spdlog::info("Optimization profile added: min=1x3x640x640, opt=6x3x640x640, max=6x3x640x640");

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
    cudaStreamCreate(&profileStream);
    config->setProfileStream(profileStream);

    std::unique_ptr<nvinfer1::IHostMemory> plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        return false;
    }

    std::ofstream outfile(enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    spdlog::info("Success, saved engine to {}", enginePath);

    return true;
}

int main()
{
    spdlog::info("Starting model conversion...");
    std::string onnxFile = "/home/user/Documents/rfr/models/det_10g_dynamic.onnx";
    std::string engineFile = "/home/user/Documents/rfr/models/det_10g_dynamic.engine";
    if (convertToEngine(onnxFile, engineFile))
    {
        spdlog::info("Model conversion completed successfully.");
    }
    else
    {
        spdlog::error("Model conversion failed.");
    }

    /*
    int stride = 8;
    int height = 640 / stride; // 80
    int width = 640 / stride;  // 80
    int num_anchors = 2;       // veya kaç anchor varsa

    // İlk önce base anchor centers'ı hesapla
    std::vector<float> base_centers(height * width * 2);
    int idx = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            base_centers[idx++] = j * stride; // x
            base_centers[idx++] = i * stride; // y
        }
    }

    // Şimdi num_anchors kere tekrarla
    std::vector<float> anchor_centers(height * width * num_anchors * 2);
    idx = 0;
    for (int hw = 0; hw < height * width; hw++)
    {
        for (int a = 0; a < num_anchors; a++)
        {
            anchor_centers[idx++] = base_centers[hw * 2 + 0]; // x
            anchor_centers[idx++] = base_centers[hw * 2 + 1]; // y
        }
    }

    spdlog::info("Anchor centers calculated. Total centers: {}", anchor_centers.size() / 2);
    for (int i = 0; i < 12800; ++i)
    {
        spdlog::info("Anchor {}: ({}, {})", i, anchor_centers[i * 2], anchor_centers[i * 2 + 1]);
    }
    */
    return 0;
}
