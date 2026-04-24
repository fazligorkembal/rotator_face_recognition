#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace
{

__global__ void normalizeFeatureRowsKernel(float *features, int feature_dim, int count)
{
    const int row = blockIdx.x;
    if (row >= count)
    {
        return;
    }

    extern __shared__ float shared_sum[];
    float local_sum = 0.0f;
    float *feature = features + static_cast<size_t>(row) * static_cast<size_t>(feature_dim);

    for (int i = threadIdx.x; i < feature_dim; i += blockDim.x)
    {
        const float value = feature[i];
        local_sum += value * value;
    }

    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_norm = rsqrtf(shared_sum[0] + 1e-12f);
    for (int i = threadIdx.x; i < feature_dim; i += blockDim.x)
    {
        feature[i] *= inv_norm;
    }
}

__global__ void findBestIdentificationMatchesKernel(
    const float *scores,
    int db_count,
    int query_count,
    const int32_t *label_group_offsets,
    const int32_t *label_group_counts,
    const int32_t *label_group_indexes,
    int label_count,
    float threshold,
    int32_t *best_label_indexes,
    float *best_label_scores)
{
    const int query_index = blockIdx.x;
    if (query_index >= query_count)
    {
        return;
    }

    int best_label = -1;
    float best_score = -1.0f;

    for (int label_index = 0; label_index < label_count; ++label_index)
    {
        const int offset = label_group_offsets[label_index];
        const int count = label_group_counts[label_index];
        float sum = 0.0f;

        for (int i = 0; i < count; ++i)
        {
            const int db_index = label_group_indexes[offset + i];
            sum += scores[query_index * db_count + db_index];
        }

        const float average = count > 0 ? sum / static_cast<float>(count) : -1.0f;
        if (average > best_score)
        {
            best_score = average;
            best_label = label_index;
        }
    }

    best_label_indexes[query_index] = best_score >= threshold ? best_label : -1;
    best_label_scores[query_index] = best_score;
}

} // namespace

void launchNormalizeIdentificationFeatures(
    float *features,
    int feature_dim,
    int count,
    cudaStream_t stream)
{
    if (!features || feature_dim <= 0 || count <= 0)
    {
        return;
    }

    constexpr int kThreads = 256;
    normalizeFeatureRowsKernel<<<count, kThreads, kThreads * sizeof(float), stream>>>(
        features,
        feature_dim,
        count);
}

void launchFindBestIdentificationMatches(
    const float *scores,
    int db_count,
    int query_count,
    const int32_t *label_group_offsets,
    const int32_t *label_group_counts,
    const int32_t *label_group_indexes,
    int label_count,
    float threshold,
    int32_t *best_label_indexes,
    float *best_label_scores,
    cudaStream_t stream)
{
    if (!scores || db_count <= 0 || query_count <= 0 || label_count <= 0)
    {
        return;
    }

    findBestIdentificationMatchesKernel<<<query_count, 1, 0, stream>>>(
        scores,
        db_count,
        query_count,
        label_group_offsets,
        label_group_counts,
        label_group_indexes,
        label_count,
        threshold,
        best_label_indexes,
        best_label_scores);
}
