// Device TreeSHAP: the packed leaf paths and the closed-form walk that
// evaluates them against a resident ingest plane. Clang CUDA C++, same
// libc++/C++23 as the rest of the build. Design: docs/architecture/10-cuda.md,
// and include/bonsai/shap_paths.hpp for the closed form itself.

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
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <limits>
#include <memory>
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

// The packed tables are flat arrays offset by hand, and the kernel's
// same-typed pointer parameters are the shape every kernel in this backend
// has (docs/architecture/10-cuda.md).
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters,readability-identifier-naming,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

namespace
{

// The widest merged path the kernel walks. Both coefficient arrays are sized
// by the template parameter and must live in registers, so the length is a
// compile-time constant and a longer path declines at plan time.
constexpr size_t k_max_path = 32;

// One thread per (row, path), grid-stride over row-major tasks. The thread
// reads its row's bin for each of the path's merged elements, builds
// P(t) = prod_j (z_j + o_j t) in registers, and settles the path's elements:
// every unsatisfied element shares one weighted sum of P, and each satisfied
// element deflates P by its monic factor first. Contributions land with
// atomicAdd because paths of one row share features.
//
// Every index into the coefficient arrays comes from a fully unrolled loop, so
// both stay in registers. The build step is written to be an identity when the
// step is past the path's length (z = 1, o = 0), which keeps the guard out of
// the index arithmetic. The weighted sums run i ascending, the order the host
// evaluator uses, so only fp32 rounding separates the two.
template <typename BinT, uint32_t K>
__global__ void shap_walk_kernel(BinT const *bins, uint8_t const *last_bin,
                                 uint32_t n_rows, uint32_t n_feats,
                                 ShapPathElem const *elems, ShapPathHead const *heads,
                                 uint32_t n_paths, float const *weights, float *out)
{
    uint64_t const total  = static_cast<uint64_t>(n_rows) * n_paths;
    uint64_t const stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t task = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         task < total; task += stride)
    {
        auto const         r    = static_cast<uint32_t>(task / n_paths);
        auto const         p    = static_cast<uint32_t>(task % n_paths);
        ShapPathHead const head = heads[p];
        uint32_t const     n    = head.n_elems;
        if (n == 0)
        {
            continue; // a root leaf attributes nothing
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

        // Deflating an unsatisfied element divides P by the scalar z_k, and
        // the (o_k - z_k) prefactor multiplies it straight back, so every
        // unsatisfied element on this path shares one weighted sum.
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
        float *const row_out = out + (static_cast<size_t>(r) * (n_feats + 1));
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
            // Synthetic division by the monic factor (t + z_k). Started from
            // K rather than n, which costs the same identity steps the build
            // pays and keeps every array index a constant: the coefficients
            // above degree n are zero, so the recursion reaches the same
            // deflated[n - 1] = poly[n].
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

bool profile_on()
{
    static bool const on = std::getenv("BONSAI_CUDA_PROFILE") != nullptr;
    return on;
}

} // namespace

// The packed leaf paths, device-resident for as long as the ensemble they were
// packed from is unchanged. Buffers free on the default stream with every
// other DeviceBuffer.
class CudaShapPlan
{
  public:
    DeviceBuffer<ShapPathElem> elems;
    DeviceBuffer<ShapPathHead> heads;
    DeviceBuffer<uint8_t>      last_bin; // per feature: n_bins - 1, the missing bin
    DeviceBuffer<float>        weights;  // Algorithm 2's permutation weights, flattened

    size_t   n_paths       = 0;
    size_t   n_feats       = 0;
    uint32_t max_path_len  = 0;
    float    learning_rate = 0.0F;
    float    init_score    = 0.0F;
    // The per-tree expected values summed once, in tree order and in double,
    // exactly as the host walk accumulates them into the bias column.
    double bias_total = 0.0;
    // Profile-only: the build's own laps, reported beside the walk's.
    double pack_s = 0.0, upload_s = 0.0;
};

namespace
{

// The K dispatch, kept beside the plan so the launch site names the kernel
// once. The bin type is the plane's, the path length the plan's.
template <uint32_t K, typename BinT>
void launch_walk(BinT const *bins, CudaShapPlan const &plan, uint32_t n_rows,
                 uint32_t n_feats, uint32_t n_paths, dim3 grid, dim3 block, float *out)
{
    shap_walk_kernel<BinT, K><<<grid, block>>>(
        bins, plan.last_bin.data(), n_rows, n_feats, plan.elems.data(),
        plan.heads.data(), n_paths, plan.weights.data(), out);
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
        // Single-output ensembles only, matching the device predict rung: a
        // multiclass model rides the host walk, so the class stride is 1.
        paths = pack_shap_paths(trees, mappers, 1);
        for (DenseTree const &tree : trees)
        {
            bias_total += tree_expected_value(tree);
        }
    }
    catch (std::invalid_argument const &)
    {
        return nullptr; // no covers, a split feature wider than the 8-bit interval
    }
    if (paths.max_path_len > k_max_path ||
        paths.heads.size() > std::numeric_limits<uint32_t>::max() ||
        paths.elems.size() > std::numeric_limits<uint32_t>::max())
    {
        return nullptr;
    }
    // w_i^(n) for every n the kernel may walk, built in double and stored in
    // float beside the fp32 coefficients that consume it. 528 floats at the
    // widest, so it rides a DeviceBuffer rather than the constant bank, which
    // is a scarcer resource this backend spends on nothing yet.
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
                        size_t n_rows, size_t n_features, std::span<double> out)
{
    size_t const cols = n_features + 1;
    // The backend tag proves the concrete type without RTTI, exactly as
    // ensure_dataset's adoption does.
    if (plane.backend_tag() != cuda_backend_tag() || out.size() != n_rows * cols ||
        n_rows == 0 || n_rows > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    auto const &cp = static_cast<CudaIngestPlane const &>(plane);
    if (cp.n_rows != n_rows || cp.n_feats != n_features || n_features != plan.n_feats ||
        cp.tile_w != k_bin_tile_width)
    {
        return false;
    }
    auto const n = static_cast<uint32_t>(n_rows);
    auto const f = static_cast<uint32_t>(n_features);
    auto const p = static_cast<uint32_t>(plan.n_paths);
    try
    {
        ProfileCounters::Lap lap{.enabled = profile_on()};
        DeviceBuffer<float>  phi;
        phi.reserve(n_rows * cols);
        check(cudaMemset(phi.data(), 0, n_rows * cols * sizeof(float)),
              "shap contribs clear");
        uint64_t const total = static_cast<uint64_t>(n) * p;
        dim3 const     block(256);
        // Capped so a wide ensemble over many rows launches a grid the device
        // can retire in one wave per SM rather than one block per task.
        dim3 const grid(static_cast<uint32_t>(
            std::min<uint64_t>((total + block.x - 1) / block.x, 65535)));
        auto const launch = [&](auto const *bins)
        {
            if (plan.max_path_len <= 8)
            {
                launch_walk<8>(bins, plan, n, f, p, grid, block, phi.data());
            }
            else if (plan.max_path_len <= 16)
            {
                launch_walk<16>(bins, plan, n, f, p, grid, block, phi.data());
            }
            else
            {
                launch_walk<32>(bins, plan, n, f, p, grid, block, phi.data());
            }
        };
        if (total > 0)
        {
            if (cp.bins_are_u8)
            {
                launch(cp.bins8.data());
            }
            else
            {
                launch(cp.bins16.data());
            }
            check(cudaGetLastError(), "shap walk launch");
        }
        double walk_s = 0.0;
        double d2h_s  = 0.0;
        if (profile_on())
        {
            check(cudaDeviceSynchronize(), "shap walk sync");
            lap(walk_s);
        }
        std::vector<float> raw(n_rows * cols, 0.0F);
        check(cudaMemcpy(raw.data(), phi.data(), raw.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "shap contribs fetch");
        lap(d2h_s);
        // The composition the host walk performs after its per-tree
        // accumulation, reproduced term for term: contributions accumulate
        // raw, then the whole row scales by the learning rate, then the bias
        // column takes init_score. The packed heads carry raw leaf values and
        // the kernel never touches the bias column, so the only thing left
        // here is the scale and the base.
        auto const   lr = static_cast<double>(plan.learning_rate);
        double const bias =
            (plan.bias_total * lr) + static_cast<double>(plan.init_score);
        for (size_t r = 0; r < n_rows; ++r)
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
                         "d2h={:.3f}s rows={} paths={}",
                         plan.pack_s, plan.upload_s, walk_s, d2h_s, n_rows,
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
