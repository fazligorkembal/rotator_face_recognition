#include "spd_logger_helper.h"
#include "identification_model_inference_helper.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

#ifdef GENERATE_TXT
#include <fstream>
#endif

#include <cmath>

void cosineSimilarityCPU(
    const float *db,    // 64 x 512
    const float *query, // 512
    float *scores       // 64
)
{
    const int N = 64;
    const int D = 512;

    float qn = 0.f;
    for (int j = 0; j < D; j++)
        qn += query[j] * query[j];
    qn = std::sqrt(qn);

    for (int i = 0; i < N; i++)
    {
        const float *v = db + i * D;
        float dot = 0.f, vn = 0.f;

        for (int j = 0; j < D; j++)
        {
            dot += v[j] * query[j];
            vn += v[j] * v[j];
        }

        scores[i] = dot / (std::sqrt(vn) * qn + 1e-8f);
    }
}

namespace fs = std::filesystem;

std::vector<std::string> getAllFilesInDirectory(const std::string &directory_path)
{
    std::vector<std::string> images;

    for (const auto &entry : fs::recursive_directory_iterator(directory_path))
    {
        if (!entry.is_regular_file())
            continue;

        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp")
        {
            images.push_back(fs::absolute(entry.path()).string());
        }
    }
    std::cout << "Found " << images.size() << " images in " << directory_path << std::endl;
    return images;
}

IdentificationModelInferenceHelper::IdentificationModelInferenceHelper(
    ICudaEngine *&engine,
    int32_t batch_size,
    int32_t input_height,
    int32_t input_width,
    int32_t feature_dim,
    const std::string &root_folder,
    cudaStream_t stream)
{
    batch_size_ = batch_size;
    input_height_ = input_height;
    input_width_ = input_width;
    feature_dim_ = feature_dim;
    stream_ = stream;

    // Deserialize the TensorRT engine
    allocateBuffers();

    context_ = engine->createExecutionContext();
    const char *input_tensor_name = engine->getIOTensorName(0);
    const char *output_tensor_name = engine->getIOTensorName(1);
    context_->setTensorAddress(input_tensor_name, device_input_);
    context_->setTensorAddress(output_tensor_name, device_output_features_);
    LOG_INFO("TensorRT tensor addresses binded successfully");

    cublasCreate(&handle_cublas_);
    cublasSetStream(handle_cublas_, stream_);
    generateDB(root_folder);

    LOG_INFO("*******************************************************");
    LOG_INFO("Constructor called");
    LOG_INFO("Root folder for DB generation: {}", root_folder);
    LOG_INFO("Batch size: {}", batch_size_);
    LOG_INFO("Input dimensions: {}x{} ", input_height_, input_width_);
    LOG_INFO("Feature dimension: {}", feature_dim_);
    LOG_INFO("IdentificationModelInferenceHelper initialized successfully");
}

IdentificationModelInferenceHelper::~IdentificationModelInferenceHelper()
{
    if (device_input_)
    {
        cudaFreeAsync(device_input_, stream_);
        device_input_ = nullptr;
    }

    if (device_output_features_)
    {
        cudaFreeAsync(device_output_features_, stream_);
        device_output_features_ = nullptr;
    }

    if (context_)
    {
        delete context_;
        context_ = nullptr;
    }

    LOG_INFO("*******************************************************");
    LOG_INFO("Destructor called, resources freed.");
}

void IdentificationModelInferenceHelper::generateDB(const std::string &root_folder)
{
    std::vector<std::string> image_files = getAllFilesInDirectory(root_folder);
    for (int i = 0; i < image_files.size(); i++)
    {
        cv::Mat img = cv::imread(image_files[i]);
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 128.0, cv::Size(input_width_, input_height_), cv::Scalar(127.5, 127.5, 127.5), true, false);

        cudaMemcpy(device_input_, blob.ptr<float>(), batch_size_ * input_height_ * input_width_ * 3 * sizeof(float), cudaMemcpyHostToDevice);
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(
            device_db_ + i * feature_dim_,
            device_output_features_,
            feature_dim_ * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream_);
    }

    for (int i = 0; i < 64; i++)
    {
        float *v = device_db_ + i * 512;
        float norm;
        cublasSnrm2(handle_cublas_, 512, v, 1, &norm);
        float inv = 1.0f / norm;
        cublasSscal(handle_cublas_, 512, &inv, v, 1);
    }

    LOG_INFO("Database generated with {} images from {}", image_files.size(), root_folder);

    /* TEST */
    /*
    float *test_host_db = new float[feature_dim_ * 64];
    cudaMemcpyAsync(
        test_host_db,
        device_db_,
        feature_dim_ * 64 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_);
    */
}

std::vector<int> IdentificationModelInferenceHelper::infer(float *device_images_face, SCRFDResults &results)
{
    std::vector<int> matched_ids;

    // Check for null pointer
    if (!device_images_face)
    {
        LOG_ERROR("device_images_face is nullptr");
        return matched_ids;
    }

    // std::cout << "Identifying " << results.detected_count_ << " faces." << std::endl;
    for (int i = 0; i < results.detected_count_; i++)
    {
        // Copy face data to input buffe
        cudaError_t copy_err = cudaMemcpyAsync(
            device_input_,
            device_images_face + i * input_height_ * input_width_ * 3,
            input_height_ * input_width_ * 3 * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream_);

        if (copy_err != cudaSuccess)
        {
            LOG_ERROR("cudaMemcpyAsync failed for face {}: {}", i, cudaGetErrorString(copy_err));
            continue;
        }

        context_->enqueueV3(stream_);

        float *host_scores = new float[64];
        float *device_scores = nullptr;
        cudaMalloc(&device_scores, 64 * sizeof(float));

        const int N = 64;
        const int D = 512;

        const float alpha = 1.0f;
        const float beta = 0.0f;

        float qnorm;
        cublasSnrm2(handle_cublas_, 512, device_output_features_, 1, &qnorm);
        float invq = 1.0f / qnorm;
        cublasSscal(handle_cublas_, 512, &invq, device_output_features_, 1);

        cublasSgemv(
            handle_cublas_,
            CUBLAS_OP_T, // transpose
            D,           // rows = 512
            N,           // cols = 64
            &alpha,
            device_db_,
            D, // leading dimension
            device_output_features_,
            1,
            &beta,
            device_scores,
            1);

        cudaMemcpyAsync(
            host_scores,
            device_scores,
            64 * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_);

        float target1 = (host_scores[0] + host_scores[1] + host_scores[2] + host_scores[3] + host_scores[4]) / 5.0f;
        float target2 = (host_scores[5] + host_scores[6] + host_scores[7] + host_scores[8] + host_scores[9]) / 5.0f;
        float target3 = (host_scores[10] + host_scores[11] + host_scores[12] + host_scores[13] + host_scores[14]) / 5.0f;
        if (target1 > 0.4f)
        {
            // std::cout << "Face " << i << " - Target1: " << target1 << ", Target2: " << target2 << std::endl;
            matched_ids.push_back(1);
        }
        else if (target2 > 0.4f)
        {
            // std::cout << "Face " << i << " - Target1: " << target1 << ", Target2: " << target2 << std::endl;
            matched_ids.push_back(2);
        }
        else if (target3 > 0.4f)
        {
            // std::cout << "Face " << i << " - Target1: " << target1 << ", Target2: " << target2 << std::endl;
            matched_ids.push_back(3);
        }
        else
        {
            matched_ids.push_back(-1);
        }
        cudaFree(device_scores);
        delete[] host_scores;
    }

    // Wait for all operations to complete
    cudaError_t sync_err = cudaStreamSynchronize(stream_);
    if (sync_err != cudaSuccess)
    {
        LOG_ERROR("cudaStreamSynchronize failed: {}", cudaGetErrorString(sync_err));
    }

    return matched_ids;
}

void IdentificationModelInferenceHelper::allocateBuffers()
{
    float total = 0;

    int32_t size_device_input = batch_size_ * input_height_ * input_width_ * 3 * sizeof(float);
    cudaError_t err1 = cudaMallocAsync(&device_input_, size_device_input, stream_);
    if (err1 != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate device_input_: {}", cudaGetErrorString(err1));
        throw std::runtime_error("CUDA allocation failed for device_input_");
    }
    total += size_device_input / 1024.0 / 1024.0;

    int32_t size_device_output_features = batch_size_ * feature_dim_ * sizeof(float);
    cudaError_t err2 = cudaMallocAsync(&device_output_features_, size_device_output_features, stream_);
    if (err2 != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate device_output_features_: {}", cudaGetErrorString(err2));
        throw std::runtime_error("CUDA allocation failed for device_output_features_");
    }
    total += size_device_output_features / 1024.0 / 1024.0;

    int32_t size_device_db = 1 * feature_dim_ * 64 * sizeof(float); // Placeholder size
    cudaError_t err3 = cudaMallocAsync(&device_db_, size_device_db, stream_);
    if (err3 != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate device_db_: {}", cudaGetErrorString(err3));
        throw std::runtime_error("CUDA allocation failed for device_db_");
    }
    total += size_device_db / 1024.0 / 1024.0;
    LOG_WARN("device_db_ face count set to static 64 for now");

    cudaError_t sync_err = cudaStreamSynchronize(stream_);
    if (sync_err != cudaSuccess)
    {
        LOG_ERROR("cudaStreamSynchronize failed: {}", cudaGetErrorString(sync_err));
        throw std::runtime_error("CUDA stream synchronization failed");
    }

    LOG_INFO("*******************************************************");
    LOG_INFO("Allocation Done");
    LOG_DEBUG("Allocated Device Memories:");
    LOG_DEBUG("\tdevice_input_: {} mb", (size_device_input) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_output_features_: {} mb", (size_device_output_features) / (1024.0 * 1024.0));
    LOG_DEBUG("\tdevice_db_: {} mb", (size_device_db) / (1024.0 * 1024.0));
    LOG_INFO("Total allocated device memory: {} mb", total);
}