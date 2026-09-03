
#include "bonsai/bin_mappers.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/predict.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <limits>
#include <memory>
#include <new>
#include <print>
#include <span>
#include <stdexcept>
#include <vector>
#include <vector_types.h>

#include "detail/device_buffer.cuh"
#include "detail/ingest_plane.cuh"
#include "detail/profile.cuh"

namespace bonsai
{

using namespace cuda_detail;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters,readability-identifier-naming)

namespace
{

template <typename BinT>
__global__ void predict_walk_kernel(
    BinT const *bins, uint32_t const *last_bin, uint32_t n_rows, uint32_t n_feats,
    uint32_t const *roots, uint32_t n_trees, uint32_t const *feature,
    uint32_t const *split_bin, uint32_t const *left, uint32_t const *right,
    uint32_t const *default_left, uint32_t const *is_leaf, float const *value, float lr,
    float init, uint32_t n, uint32_t const *rows, float *out)
{
    uint32_t const k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k >= n)
    {
        return;
    }
    uint32_t const r   = mapped_row(rows, k);
    float          acc = 0.0F;
    for (uint32_t t = 0; t < n_trees; ++t)
    {
        uint32_t const base = roots[t];
        uint32_t       idx  = base;
        while (is_leaf[idx] == 0)
        {
            uint32_t const f    = feature[idx];
            uint32_t const last = last_bin[f];
            uint32_t const b    = bins[tiled_cell(f, r, n_rows, n_feats)];
            bool const     l =
                (b == last) ? (default_left[idx] != 0) : (b <= split_bin[idx]);
            idx = base + (l ? left[idx] : right[idx]);
        }
        acc += value[idx];
    }
    out[k] = __fadd_rn(init, __fmul_rn(lr, acc));
}

struct PackedTrees
{
    std::vector<uint32_t> roots, feature, split_bin, left, right, default_left, is_leaf;
    std::vector<float>    value;
};

PackedTrees pack(std::span<DenseTree const> trees, BinMappers const &mappers)
{
    PackedTrees p;
    p.roots.reserve(trees.size());
    for (DenseTree const &tree : trees)
    {
        p.roots.push_back(static_cast<uint32_t>(p.feature.size()));
        for (DenseTree::Node const &nd : tree.nodes())
        {
            bool const leaf = DenseTree::is_leaf(nd);
            p.feature.push_back(leaf ? 0U : nd.feature_id);
            p.split_bin.push_back(
                leaf ? 0U
                     : mappers[nd.feature_id].bin_of_threshold(nd.threshold_or_value));
            p.left.push_back(nd.left);
            p.right.push_back(nd.right);
            p.default_left.push_back(nd.default_left ? 1U : 0U);
            p.is_leaf.push_back(leaf ? 1U : 0U);
            p.value.push_back(leaf ? nd.threshold_or_value : 0.0F);
        }
    }
    return p;
}

} // namespace

class CudaPredictPlan
{
  public:
    DeviceBuffer<uint32_t> last_bin;
    DeviceBuffer<uint32_t> roots;
    DeviceBuffer<uint32_t> feature;
    DeviceBuffer<uint32_t> split_bin;
    DeviceBuffer<uint32_t> left;
    DeviceBuffer<uint32_t> right;
    DeviceBuffer<uint32_t> default_left;
    DeviceBuffer<uint32_t> is_leaf;
    DeviceBuffer<float>    value;

    size_t n_trees       = 0;
    size_t n_feats       = 0;
    float  learning_rate = 0.0F;
    float  init_score    = 0.0F;
    double pack_s = 0.0, upload_s = 0.0;
};

std::shared_ptr<CudaPredictPlan const>
cuda_predict_plan(std::span<DenseTree const> trees, BinMappers const &mappers,
                  float learning_rate, float init_score)
{
    if (!cuda_available() || trees.empty() || mappers.size() == 0)
    {
        return nullptr;
    }
    std::vector<uint32_t> last(mappers.size());
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        if (mappers[f].n_bins() == 0 || mappers[f].n_bins() > 65535)
        {
            return nullptr;
        }
        last[f] = static_cast<uint32_t>(mappers[f].n_bins() - 1);
    }
    ProfileCounters::Lap lap{.enabled = profile_on()};
    PackedTrees          p;
    try
    {
        p = pack(trees, mappers);
    }
    catch (std::bad_alloc const &)
    {
        return nullptr;
    }
    if (p.feature.size() > std::numeric_limits<uint32_t>::max())
    {
        return nullptr;
    }
    auto plan = std::make_shared<CudaPredictPlan>();
    lap(plan->pack_s);
    try
    {
        plan->last_bin.upload(last.data(), last.size());
        plan->roots.upload(p.roots.data(), p.roots.size());
        plan->feature.upload(p.feature.data(), p.feature.size());
        plan->split_bin.upload(p.split_bin.data(), p.split_bin.size());
        plan->left.upload(p.left.data(), p.left.size());
        plan->right.upload(p.right.data(), p.right.size());
        plan->default_left.upload(p.default_left.data(), p.default_left.size());
        plan->is_leaf.upload(p.is_leaf.data(), p.is_leaf.size());
        plan->value.upload(p.value.data(), p.value.size());
    }
    catch (std::runtime_error const &)
    {
        return nullptr;
    }
    lap(plan->upload_s);
    plan->n_trees       = trees.size();
    plan->n_feats       = mappers.size();
    plan->learning_rate = learning_rate;
    plan->init_score    = init_score;
    return plan;
}

bool cuda_predict(CudaPredictPlan const &plan, IngestPlane const &plane, size_t n_rows,
                  size_t n_features, row_index_view rows, size_t n_trees,
                  std::span<float> out)
{
    auto const *cp = matching_plane(plane, n_rows, n_features, plan.n_feats);
    if (cp == nullptr || out.size() != (rows.empty() ? n_rows : rows.size()))
    {
        return false;
    }
    auto const k = static_cast<uint32_t>(
        n_trees == 0 ? plan.n_trees : std::min(n_trees, plan.n_trees));
    auto const stride = static_cast<uint32_t>(n_rows);
    auto const f      = static_cast<uint32_t>(n_features);
    try
    {
        ProfileCounters::Lap lap{.enabled = profile_on()};
        RowMap const         map{rows, n_rows};
        auto const           n = static_cast<uint32_t>(map.n);
        DeviceBuffer<float>  scores;
        scores.reserve(map.n);
        dim3 const grid(static_cast<uint32_t>((map.n + 255) / 256));
        auto const launch = [&](auto const *bins)
        {
            predict_walk_kernel<<<grid, dim3(256)>>>(
                bins, plan.last_bin.data(), stride, f, plan.roots.data(), k,
                plan.feature.data(), plan.split_bin.data(), plan.left.data(),
                plan.right.data(), plan.default_left.data(), plan.is_leaf.data(),
                plan.value.data(), plan.learning_rate, plan.init_score, n, map.data(),
                scores.data());
        };
        cp->with_bins(launch);
        check(cudaGetLastError(), "predict walk launch");
        double walk_s = 0.0;
        double d2h_s  = 0.0;
        if (profile_on())
        {
            check(cudaDeviceSynchronize(), "predict walk sync");
            lap(walk_s);
        }
        check(cudaMemcpy(out.data(), scores.data(), out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "predict scores fetch");
        lap(d2h_s);
        if (profile_on())
        {
            std::println(stderr,
                         "cuda-predict: pack={:.3f}s upload={:.3f}s walk={:.3f}s "
                         "d2h={:.3f}s rows={} plane={} trees={}",
                         plan.pack_s, plan.upload_s, walk_s, d2h_s, map.n, n_rows,
                         static_cast<size_t>(k));
        }
    }
    catch (std::runtime_error const &)
    {
        return false;
    }
    return true;
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters,readability-identifier-naming)

} // namespace bonsai
