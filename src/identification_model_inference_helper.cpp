#include "identification_model_inference_helper.h"

#include "spd_logger_helper.h"

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

void launchNormalizeIdentificationFeatures(float *features, int feature_dim, int count, cudaStream_t stream);
void launchFindBestIdentificationMatches(const float *scores,
                                         int db_count,
                                         int query_count,
                                         const int32_t *label_group_offsets,
                                         const int32_t *label_group_counts,
                                         const int32_t *label_group_indexes,
                                         int label_count,
                                         float threshold,
                                         int32_t *best_label_indexes,
                                         float *best_label_scores,
                                         cudaStream_t stream);

namespace
{

void checkCuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char *what)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(status)));
    }
}

int32_t featureDimFromTensor(const nvinfer1::Dims &dims)
{
    int32_t feature_dim = 1;
    const int start_index = dims.nbDims > 1 ? 1 : 0;
    for (int i = start_index; i < dims.nbDims; ++i)
    {
        if (dims.d[i] > 0)
        {
            feature_dim *= dims.d[i];
        }
    }

    if (feature_dim <= 0)
    {
        throw std::runtime_error("Failed to resolve identification feature dimension");
    }

    return feature_dim;
}

std::vector<std::filesystem::path> collectImagePaths(const std::string &root)
{
    namespace fs = std::filesystem;

    if (root.empty())
    {
        throw std::runtime_error("selected_faces root is empty");
    }

    if (!fs::exists(root))
    {
        throw std::runtime_error("selected_faces root does not exist: " + root);
    }

    std::vector<fs::path> image_paths;
    for (const auto &entry : fs::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
        {
            image_paths.push_back(entry.path());
        }
    }

    std::sort(image_paths.begin(), image_paths.end());
    return image_paths;
}

std::string labelFromPath(const std::filesystem::path &path)
{
    const auto parent = path.parent_path();
    if (!parent.empty() && parent.filename() != "selected_faces")
    {
        return parent.filename().string();
    }
    return path.stem().string();
}

} // namespace

IdentificationModelInferenceHelper::IdentificationModelInferenceHelper(
    nvinfer1::ICudaEngine *engine,
    int32_t max_batch_size,
    int32_t input_height,
    int32_t input_width,
    const std::string &selected_faces_root,
    cudaStream_t stream)
    : max_batch_size_(max_batch_size),
      input_height_(input_height),
      input_width_(input_width),
      engine_(engine)
{
    if (!engine_)
    {
        throw std::runtime_error("Identification engine is null");
    }
    if (max_batch_size_ <= 0)
    {
        throw std::runtime_error("Identification max_batch_size must be positive");
    }

    if (stream)
    {
        stream_ = stream;
        owns_stream_ = false;
    }
    else
    {
        checkCuda(cudaStreamCreate(&stream_), "Failed to create identification CUDA stream");
        owns_stream_ = true;
    }

    context_ = engine_->createExecutionContext();
    if (!context_)
    {
        throw std::runtime_error("Failed to create identification TensorRT execution context");
    }

    const char *input_tensor_name = engine_->getIOTensorName(0);
    const char *output_tensor_name = engine_->getIOTensorName(1);
    if (!input_tensor_name || !output_tensor_name)
    {
        throw std::runtime_error("Failed to get identification tensor names");
    }

    feature_dim_ = featureDimFromTensor(engine_->getTensorShape(output_tensor_name));

    allocateBuffers();

    if (!context_->setInputShape(input_tensor_name, nvinfer1::Dims4{max_batch_size_, 3, input_height_, input_width_}))
    {
        throw std::runtime_error("Failed to set identification max input shape");
    }
    if (!context_->setTensorAddress(input_tensor_name, device_input_))
    {
        throw std::runtime_error("Failed to bind identification input tensor");
    }
    if (!context_->setTensorAddress(output_tensor_name, device_output_features_))
    {
        throw std::runtime_error("Failed to bind identification output tensor");
    }

    checkCublas(cublasCreate(&cublas_handle_), "Failed to create cuBLAS handle");
    checkCublas(cublasSetStream(cublas_handle_, stream_), "Failed to set cuBLAS stream");

    buildDatabase(selected_faces_root);
    buildLabelGroups();
    current_input_batch_ = -1;

    LOG_INFO("***** Constructor IdentificationModelInferenceHelper *****");
    LOG_INFO("\tSelected faces root: {}", selected_faces_root);
    LOG_INFO("\tMax batch size: {}", max_batch_size_);
    LOG_INFO("\tInput size: {}x{}", input_width_, input_height_);
    LOG_INFO("\tFeature dimension: {}", feature_dim_);
    LOG_INFO("\tDB count: {}", db_count_);
    LOG_INFO("\tIdentity count: {}", unique_labels_.size());
}

IdentificationModelInferenceHelper::~IdentificationModelInferenceHelper()
{
    if (stream_)
    {
        cudaStreamSynchronize(stream_);
    }

    if (device_db_)
    {
        cudaFreeAsync(device_db_, stream_);
        device_db_ = nullptr;
    }
    if (device_similarity_scores_)
    {
        cudaFreeAsync(device_similarity_scores_, stream_);
        device_similarity_scores_ = nullptr;
    }
    if (device_label_group_offsets_)
    {
        cudaFreeAsync(device_label_group_offsets_, stream_);
        device_label_group_offsets_ = nullptr;
    }
    if (device_label_group_counts_)
    {
        cudaFreeAsync(device_label_group_counts_, stream_);
        device_label_group_counts_ = nullptr;
    }
    if (device_label_group_indexes_)
    {
        cudaFreeAsync(device_label_group_indexes_, stream_);
        device_label_group_indexes_ = nullptr;
    }
    if (device_best_label_indexes_)
    {
        cudaFreeAsync(device_best_label_indexes_, stream_);
        device_best_label_indexes_ = nullptr;
    }
    if (device_best_label_scores_)
    {
        cudaFreeAsync(device_best_label_scores_, stream_);
        device_best_label_scores_ = nullptr;
    }
    if (device_output_features_)
    {
        cudaFreeAsync(device_output_features_, stream_);
        device_output_features_ = nullptr;
    }
    if (device_input_)
    {
        cudaFreeAsync(device_input_, stream_);
        device_input_ = nullptr;
    }
    if (stream_)
    {
        cudaStreamSynchronize(stream_);
    }
    if (context_)
    {
        delete context_;
        context_ = nullptr;
    }
    if (cublas_handle_)
    {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    if (stream_ && owns_stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void IdentificationModelInferenceHelper::allocateBuffers()
{
    const size_t input_elements =
        static_cast<size_t>(max_batch_size_) * 3ULL * static_cast<size_t>(input_height_) * static_cast<size_t>(input_width_);
    const size_t output_elements =
        static_cast<size_t>(max_batch_size_) * static_cast<size_t>(feature_dim_);

    checkCuda(cudaMallocAsync(&device_input_, input_elements * sizeof(float), stream_),
              "Failed to allocate identification input buffer");
    checkCuda(cudaMallocAsync(&device_output_features_, output_elements * sizeof(float), stream_),
              "Failed to allocate identification output buffer");
    checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize identification buffer allocation");

    host_best_label_indexes_.assign(static_cast<size_t>(max_batch_size_), -1);
    host_best_label_scores_.assign(static_cast<size_t>(max_batch_size_), 0.0f);
}

void IdentificationModelInferenceHelper::buildDatabase(const std::string &selected_faces_root)
{
    const auto image_paths = collectImagePaths(selected_faces_root);
    db_count_ = static_cast<int32_t>(image_paths.size());

    if (db_count_ == 0)
    {
        throw std::runtime_error("No images found under selected_faces root: " + selected_faces_root);
    }

    checkCuda(cudaMallocAsync(&device_db_,
                              static_cast<size_t>(db_count_) * static_cast<size_t>(feature_dim_) * sizeof(float),
                              stream_),
              "Failed to allocate identification DB buffer");
    checkCuda(cudaMallocAsync(&device_similarity_scores_,
                              static_cast<size_t>(max_batch_size_) * static_cast<size_t>(db_count_) * sizeof(float),
                              stream_),
              "Failed to allocate identification similarity score buffer");
    checkCuda(cudaMallocAsync(&device_best_label_indexes_,
                              static_cast<size_t>(max_batch_size_) * sizeof(int32_t),
                              stream_),
              "Failed to allocate identification best label buffer");
    checkCuda(cudaMallocAsync(&device_best_label_scores_,
                              static_cast<size_t>(max_batch_size_) * sizeof(float),
                              stream_),
              "Failed to allocate identification best score buffer");
    checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize identification DB allocation");

    db_labels_.reserve(image_paths.size());

    const char *input_tensor_name = engine_->getIOTensorName(0);

    for (size_t start = 0; start < image_paths.size(); start += static_cast<size_t>(max_batch_size_))
    {
        const int32_t batch_count = static_cast<int32_t>(
            std::min(static_cast<size_t>(max_batch_size_), image_paths.size() - start));

        std::vector<cv::Mat> images;
        images.reserve(batch_count);

        for (int32_t batch_index = 0; batch_index < batch_count; ++batch_index)
        {
            const auto &image_path = image_paths[start + static_cast<size_t>(batch_index)];
            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (image.empty())
            {
                throw std::runtime_error("Failed to load selected face image: " + image_path.string());
            }
            if (image.cols != input_width_ || image.rows != input_height_)
            {
                cv::resize(image, image, cv::Size(input_width_, input_height_));
            }

            images.push_back(image);
            db_labels_.push_back(labelFromPath(image_path));
        }

        cv::Mat blob = cv::dnn::blobFromImages(
            images,
            1.0 / 128.0,
            cv::Size(input_width_, input_height_),
            cv::Scalar(127.5, 127.5, 127.5),
            true,
            false);

        const size_t batch_input_elements =
            static_cast<size_t>(batch_count) * 3ULL * static_cast<size_t>(input_height_) * static_cast<size_t>(input_width_);

        if (current_input_batch_ != batch_count &&
            !context_->setInputShape(input_tensor_name, nvinfer1::Dims4{batch_count, 3, input_height_, input_width_}))
        {
            throw std::runtime_error("Failed to set identification batch input shape");
        }
        current_input_batch_ = batch_count;
        checkCuda(cudaMemcpyAsync(device_input_,
                                  blob.ptr<float>(),
                                  batch_input_elements * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream_),
                  "Failed to copy selected face batch to identification input");

        if (!context_->enqueueV3(stream_))
        {
            throw std::runtime_error("Failed to enqueue identification TensorRT context while building DB");
        }

        checkCuda(cudaMemcpyAsync(device_db_ + start * static_cast<size_t>(feature_dim_),
                                  device_output_features_,
                                  static_cast<size_t>(batch_count) * static_cast<size_t>(feature_dim_) * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_),
                  "Failed to copy identification features into DB buffer");

        launchNormalizeIdentificationFeatures(device_db_ + start * static_cast<size_t>(feature_dim_),
                                              feature_dim_,
                                              batch_count,
                                              stream_);
        checkCuda(cudaGetLastError(), "Failed to launch DB feature normalization kernel");
    }

    checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize identification DB build");
    LOG_INFO("Built identification DB on GPU with {} entries", db_count_);
}

void IdentificationModelInferenceHelper::buildLabelGroups()
{
    std::unordered_map<std::string, size_t> group_lookup;
    unique_labels_.clear();
    label_groups_.clear();

    for (size_t db_index = 0; db_index < db_labels_.size(); ++db_index)
    {
        const auto &label = db_labels_[db_index];
        const auto found = group_lookup.find(label);
        if (found == group_lookup.end())
        {
            group_lookup.emplace(label, unique_labels_.size());
            unique_labels_.push_back(label);
            label_groups_.push_back({static_cast<int32_t>(db_index)});
            continue;
        }

        label_groups_[found->second].push_back(static_cast<int32_t>(db_index));
    }

    std::vector<int32_t> group_offsets;
    std::vector<int32_t> group_counts;
    std::vector<int32_t> group_indexes;
    group_offsets.reserve(label_groups_.size());
    group_counts.reserve(label_groups_.size());
    group_indexes.reserve(db_labels_.size());

    for (const auto &group : label_groups_)
    {
        group_offsets.push_back(static_cast<int32_t>(group_indexes.size()));
        group_counts.push_back(static_cast<int32_t>(group.size()));
        group_indexes.insert(group_indexes.end(), group.begin(), group.end());
    }

    checkCuda(cudaMallocAsync(&device_label_group_offsets_,
                              group_offsets.size() * sizeof(int32_t),
                              stream_),
              "Failed to allocate identification label group offsets");
    checkCuda(cudaMallocAsync(&device_label_group_counts_,
                              group_counts.size() * sizeof(int32_t),
                              stream_),
              "Failed to allocate identification label group counts");
    checkCuda(cudaMallocAsync(&device_label_group_indexes_,
                              group_indexes.size() * sizeof(int32_t),
                              stream_),
              "Failed to allocate identification label group indexes");

    checkCuda(cudaMemcpyAsync(device_label_group_offsets_,
                              group_offsets.data(),
                              group_offsets.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice,
                              stream_),
              "Failed to copy identification label group offsets");
    checkCuda(cudaMemcpyAsync(device_label_group_counts_,
                              group_counts.data(),
                              group_counts.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice,
                              stream_),
              "Failed to copy identification label group counts");
    checkCuda(cudaMemcpyAsync(device_label_group_indexes_,
                              group_indexes.data(),
                              group_indexes.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice,
                              stream_),
              "Failed to copy identification label group indexes");
    checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize identification label group setup");
}

std::vector<IdentificationMatch> IdentificationModelInferenceHelper::matchWarpedFaces(
    const float *device_warped_faces,
    const int32_t *batch_counts,
    int detection_batch_count,
    int top_k,
    float face_threshold)
{
    const size_t face_elements =
        3ULL * static_cast<size_t>(input_height_) * static_cast<size_t>(input_width_);

    if (!device_warped_faces || !batch_counts || detection_batch_count <= 0)
    {
        return {};
    }

    host_source_face_indices_.clear();
    const int max_possible_faces = detection_batch_count * top_k;
    if (static_cast<int>(host_source_face_indices_.capacity()) < max_possible_faces)
    {
        host_source_face_indices_.reserve(static_cast<size_t>(max_possible_faces));
    }
    int face_count = 0;
    for (int batch_index = 0; batch_index < detection_batch_count; ++batch_index)
    {
        const int count = std::min(batch_counts[batch_index], top_k);
        face_count += count;
        for (int rank = 0; rank < count; ++rank)
        {
            host_source_face_indices_.push_back(batch_index * top_k + rank);
        }
    }

    if (face_count <= 0)
    {
        return {};
    }

    const char *input_tensor_name = engine_->getIOTensorName(0);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::vector<IdentificationMatch> matches;

    for (int start = 0; start < face_count;)
    {
        const int remaining = face_count - start;
        // Run the remaining faces in a single TRT batch whenever they fit.
        // The previous logic collapsed every tail batch smaller than max_batch_size_
        // to size 1, which turned multi-face frames into several serial R50 launches.
        const int r50_batch_count = std::min(remaining, max_batch_size_);

        for (int batch_index = 0; batch_index < r50_batch_count; ++batch_index)
        {
            const size_t src_face_offset =
                static_cast<size_t>(host_source_face_indices_[static_cast<size_t>(start + batch_index)]) * face_elements;
            const size_t dst_face_offset =
                static_cast<size_t>(batch_index) * face_elements;

            checkCuda(cudaMemcpyAsync(device_input_ + dst_face_offset,
                                      device_warped_faces + src_face_offset,
                                      face_elements * sizeof(float),
                                      cudaMemcpyDeviceToDevice,
                                      stream_),
                      "Failed to copy device warped face to identification input");
        }

        if (current_input_batch_ != r50_batch_count &&
            !context_->setInputShape(input_tensor_name, nvinfer1::Dims4{r50_batch_count, 3, input_height_, input_width_}))
        {
            throw std::runtime_error("Failed to set identification runtime batch input shape");
        }
        current_input_batch_ = r50_batch_count;

        if (!context_->enqueueV3(stream_))
        {
            throw std::runtime_error("Failed to enqueue identification TensorRT context for runtime faces");
        }

        launchNormalizeIdentificationFeatures(device_output_features_,
                                              feature_dim_,
                                              r50_batch_count,
                                              stream_);
        checkCuda(cudaGetLastError(), "Failed to launch runtime feature normalization kernel");

        checkCublas(cublasSgemm(cublas_handle_,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                db_count_,
                                r50_batch_count,
                                feature_dim_,
                                &alpha,
                                device_db_,
                                feature_dim_,
                                device_output_features_,
                                feature_dim_,
                                &beta,
                                device_similarity_scores_,
                                db_count_),
                    "Failed to compute identification similarity matrix");

        launchFindBestIdentificationMatches(device_similarity_scores_,
                                            db_count_,
                                            r50_batch_count,
                                            device_label_group_offsets_,
                                            device_label_group_counts_,
                                            device_label_group_indexes_,
                                            static_cast<int>(unique_labels_.size()),
                                            face_threshold,
                                            device_best_label_indexes_,
                                            device_best_label_scores_,
                                            stream_);
        checkCuda(cudaGetLastError(), "Failed to launch identification best-match kernel");

        checkCuda(cudaMemcpyAsync(host_best_label_indexes_.data(),
                                  device_best_label_indexes_,
                                  static_cast<size_t>(r50_batch_count) * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream_),
                  "Failed to copy identification best labels to host");
        checkCuda(cudaMemcpyAsync(host_best_label_scores_.data(),
                                  device_best_label_scores_,
                                  static_cast<size_t>(r50_batch_count) * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream_),
                  "Failed to copy identification best scores to host");
        checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize identification best-match result");

        for (int batch_index = 0; batch_index < r50_batch_count; ++batch_index)
        {
            const int32_t label_index = host_best_label_indexes_[static_cast<size_t>(batch_index)];
            if (label_index < 0)
            {
                continue;
            }

            IdentificationMatch match;
            // buildFaceOverlays() consumes detections in a compact, push_back order.
            // Using the original flat `batch * top_k + rank` index breaks that mapping
            // whenever an earlier slice has fewer than `top_k` detections.
            match.face_index = start + batch_index;
            match.label = unique_labels_[static_cast<size_t>(label_index)];
            match.average_similarity = host_best_label_scores_[static_cast<size_t>(batch_index)];
            matches.push_back(match);
        }

        start += r50_batch_count;
    }

    return matches;
}
