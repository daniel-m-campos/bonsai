
#include "bonsai/bin_mappers.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/shap.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/shap.hpp"
#include "bonsai/shap_paths.hpp"
#include "bonsai/tree.hpp"

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

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters,readability-identifier-naming,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

namespace
{

constexpr size_t k_max_path = 32;

template <typename BinT, uint32_t K>
__global__ void shap_walk_kernel(BinT const *bins, uint8_t const *last_bin,
                                 uint32_t n_rows, uint32_t n_feats,
                                 ShapPathElem const *elems, ShapPathHead const *heads,
                                 uint32_t n_paths, float const *weights,
                                 uint32_t n_walk, uint32_t const *rows, float *out)
{
    uint64_t const total  = static_cast<uint64_t>(n_walk) * n_paths;
    uint64_t const stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t task = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         task < total; task += stride)
    {
        auto const         pos  = static_cast<uint32_t>(task / n_paths);
        auto const         r    = mapped_row(rows, pos);
        auto const         p    = static_cast<uint32_t>(task % n_paths);
        ShapPathHead const head = heads[p];
        uint32_t const     n    = head.n_elems;
        if (n == 0)
        {
            continue;
        }

        float poly[K + 1];
        poly[0] = 1.0F;
#pragma unroll
        for (uint32_t i = 1; i <= K; ++i)
        {
            poly[i] = 0.0F;
        }

        uint32_t satisfied = 0;
#pragma unroll
        for (uint32_t j = 0; j < K; ++j)
        {
            float z   = 1.0F;
            bool  one = false;
            if (j < n)
            {
                ShapPathElem const e = elems[head.first + j];
                auto const         f =
                    static_cast<uint32_t>(e.feature & ~ShapPathElem::k_missing_ok);
                auto const b =
                    static_cast<uint32_t>(bins[tiled_cell(f, r, n_rows, n_feats)]);
                one = b == last_bin[f] ? (e.feature & ShapPathElem::k_missing_ok) != 0
                                       : (b >= e.lo && b <= e.hi);
                satisfied |= static_cast<uint32_t>(one) << j;
                z = e.zero_fraction;
            }
#pragma unroll
            for (uint32_t i = j + 1; i >= 1; --i)
            {
                poly[i] = (z * poly[i]) + (one ? poly[i - 1] : 0.0F);
            }
            poly[0] *= z;
        }

        float const *w = weights + (static_cast<size_t>(n - 1) * n / 2);

        float unsatisfied_sum = 0.0F;
#pragma unroll
        for (uint32_t i = 0; i < K; ++i)
        {
            if (i < n)
            {
                unsatisfied_sum += w[i] * poly[i];
            }
        }

        float const  value   = head.value;
        float *const row_out = out + (static_cast<size_t>(pos) * (n_feats + 1));
        for (uint32_t k = 0; k < n; ++k)
        {
            ShapPathElem const e = elems[head.first + k];
            auto const         f =
                static_cast<uint32_t>(e.feature & ~ShapPathElem::k_missing_ok);
            if (((satisfied >> k) & 1U) == 0U)
            {
                atomicAdd(&row_out[f], -value * unsatisfied_sum);
                continue;
            }
            float const z = e.zero_fraction;
            float       deflated[K];
            float       d = 0.0F;
#pragma unroll
            for (uint32_t i = K; i-- > 0;)
            {
                d           = poly[i + 1] - (z * d);
                deflated[i] = d;
            }
            float sum = 0.0F;
#pragma unroll
            for (uint32_t i = 0; i < K; ++i)
            {
                if (i < n)
                {
                    sum += w[i] * deflated[i];
                }
            }
            atomicAdd(&row_out[f], value * (1.0F - z) * sum);
        }
    }
}

} // namespace

class CudaShapPlan
{
  public:
    DeviceBuffer<ShapPathElem> elems;
    DeviceBuffer<ShapPathHead> heads;
    DeviceBuffer<uint8_t>      last_bin;
    DeviceBuffer<float>        weights;

    size_t   n_paths       = 0;
    size_t   n_feats       = 0;
    uint32_t max_path_len  = 0;
    float    learning_rate = 0.0F;
    float    init_score    = 0.0F;
    double   bias_total    = 0.0;
    double   pack_s = 0.0, upload_s = 0.0;
};

namespace
{

template <uint32_t K, typename BinT>
void launch_walk(BinT const *bins, CudaShapPlan const &plan, uint32_t n_rows,
                 uint32_t n_feats, uint32_t n_paths, dim3 grid, dim3 block,
                 uint32_t n_walk, uint32_t const *rows, float *out)
{
    shap_walk_kernel<BinT, K><<<grid, block>>>(
        bins, plan.last_bin.data(), n_rows, n_feats, plan.elems.data(),
        plan.heads.data(), n_paths, plan.weights.data(), n_walk, rows, out);
}

} // namespace

std::shared_ptr<CudaShapPlan const> cuda_shap_plan(std::span<DenseTree const> trees,
                                                   BinMappers const          &mappers,
                                                   float learning_rate,
                                                   float init_score)
{
    if (!cuda_available() || trees.empty() || mappers.size() == 0)
    {
        return nullptr;
    }
    ProfileCounters::Lap lap{.enabled = profile_on()};
    ShapPaths            paths;
    double               bias_total = 0.0;
    try
    {
        paths = pack_shap_paths(trees, mappers, 1);
        for (DenseTree const &tree : trees)
        {
            bias_total += tree_expected_value(tree);
        }
    }
    catch (std::invalid_argument const &)
    {
        return nullptr;
    }
    catch (std::bad_alloc const &)
    {
        return nullptr;
    }
    if (paths.max_path_len > k_max_path ||
        paths.heads.size() > std::numeric_limits<uint32_t>::max() ||
        paths.elems.size() > std::numeric_limits<uint32_t>::max())
    {
        return nullptr;
    }
    auto const         wide = shap_path_weights(k_max_path);
    std::vector<float> weights(wide.size(), 0.0F);
    std::ranges::transform(wide, weights.begin(),
                           [](double v) { return static_cast<float>(v); });

    auto plan = std::make_shared<CudaShapPlan>();
    lap(plan->pack_s);
    try
    {
        plan->elems.upload(paths.elems.data(), std::max<size_t>(paths.elems.size(), 1));
        plan->heads.upload(paths.heads.data(), paths.heads.size());
        plan->last_bin.upload(paths.last_bin.data(), paths.last_bin.size());
        plan->weights.upload(weights.data(), weights.size());
    }
    catch (std::runtime_error const &)
    {
        return nullptr;
    }
    lap(plan->upload_s);
    plan->n_paths       = paths.heads.size();
    plan->n_feats       = mappers.size();
    plan->max_path_len  = static_cast<uint32_t>(paths.max_path_len);
    plan->learning_rate = learning_rate;
    plan->init_score    = init_score;
    plan->bias_total    = bias_total;
    return plan;
}

bool cuda_pred_contribs(CudaShapPlan const &plan, IngestPlane const &plane,
                        size_t n_rows, size_t n_features, row_index_view rows,
                        std::span<double> out)
{
    size_t const cols     = n_features + 1;
    size_t const out_rows = rows.empty() ? n_rows : rows.size();
    auto const  *cp       = matching_plane(plane, n_rows, n_features, plan.n_feats);
    if (cp == nullptr || out.size() != out_rows * cols)
    {
        return false;
    }
    auto const stride = static_cast<uint32_t>(n_rows);
    auto const f      = static_cast<uint32_t>(n_features);
    auto const p      = static_cast<uint32_t>(plan.n_paths);
    try
    {
        ProfileCounters::Lap lap{.enabled = profile_on()};
        RowMap const         map{rows, n_rows};
        auto const           n = static_cast<uint32_t>(map.n);
        DeviceBuffer<float>  phi;
        phi.reserve(out_rows * cols);
        check(cudaMemset(phi.data(), 0, out_rows * cols * sizeof(float)),
              "shap contribs clear");
        uint64_t const total = static_cast<uint64_t>(n) * p;
        dim3 const     block(256);
        dim3 const     grid(static_cast<uint32_t>(
            std::min<uint64_t>((total + block.x - 1) / block.x, 65535)));
        auto const     launch = [&](auto const *bins)
        {
            if (plan.max_path_len <= 8)
            {
                launch_walk<8>(bins, plan, stride, f, p, grid, block, n, map.data(),
                               phi.data());
            }
            else if (plan.max_path_len <= 16)
            {
                launch_walk<16>(bins, plan, stride, f, p, grid, block, n, map.data(),
                                phi.data());
            }
            else
            {
                launch_walk<32>(bins, plan, stride, f, p, grid, block, n, map.data(),
                                phi.data());
            }
        };
        if (total > 0)
        {
            cp->with_bins(launch);
            check(cudaGetLastError(), "shap walk launch");
        }
        double walk_s = 0.0;
        double d2h_s  = 0.0;
        if (profile_on())
        {
            check(cudaDeviceSynchronize(), "shap walk sync");
            lap(walk_s);
        }
        std::vector<float> raw(out_rows * cols, 0.0F);
        check(cudaMemcpy(raw.data(), phi.data(), raw.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "shap contribs fetch");
        lap(d2h_s);
        auto const   lr = static_cast<double>(plan.learning_rate);
        double const bias =
            (plan.bias_total * lr) + static_cast<double>(plan.init_score);
        for (size_t r = 0; r < out_rows; ++r)
        {
            size_t const base = r * cols;
            for (size_t c = 0; c < n_features; ++c)
            {
                out[base + c] = static_cast<double>(raw[base + c]) * lr;
            }
            out[base + n_features] = bias;
        }
        if (profile_on())
        {
            std::println(stderr,
                         "cuda-shap: pack={:.3f}s upload={:.3f}s walk={:.3f}s "
                         "d2h={:.3f}s rows={} plane={} paths={}",
                         plan.pack_s, plan.upload_s, walk_s, d2h_s, out_rows, n_rows,
                         plan.n_paths);
        }
    }
    catch (std::runtime_error const &)
    {
        return false;
    }
    return true;
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters,readability-identifier-naming,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

} // namespace bonsai
