#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include "bonsai/config/errors.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/metal/hist_fill_msl.hpp"
#include "bonsai/metal/histogram_engine.hpp"
#include "bonsai/parallel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsai
{

namespace
{

constexpr size_t   k_threadgroup_ceiling_bytes = 32768;
constexpr uint32_t k_threads_per_group         = 256;
constexpr uint32_t k_rows_per_chunk_target     = 4096;
constexpr uint32_t k_max_chunks                = 4096;
constexpr double   k_scale_headroom            = static_cast<double>(1U << 30);
constexpr size_t   k_min_device_rows_default   = size_t{1} << 21;

size_t min_device_rows()
{
    static size_t const value = []
    {
        char const *env = std::getenv("BONSAI_METAL_MIN_ROWS");
        return env != nullptr ? static_cast<size_t>(std::strtoull(env, nullptr, 10))
                              : k_min_device_rows_default;
    }();
    return value;
}

struct NodeParams
{
    uint32_t count;
    uint32_t n_chunks;
    uint32_t use_rows;
    uint32_t has_hess;
    uint32_t plane_rows;
    uint32_t stride;
    float    scale;
};

struct AutoreleaseScope
{
    NS::AutoreleasePool *pool                  = NS::AutoreleasePool::alloc()->init();
    AutoreleaseScope()                         = default;
    AutoreleaseScope(AutoreleaseScope const &) = delete;
    AutoreleaseScope &operator=(AutoreleaseScope const &) = delete;
    ~AutoreleaseScope()
    {
        pool->release();
    }
};

template <typename T> void release_and_clear(T *&object)
{
    if (object != nullptr)
    {
        object->release();
        object = nullptr;
    }
}

MTL::Device *shared_device()
{
    static MTL::Device *const device = MTL::CreateSystemDefaultDevice();
    return device;
}

MTL::ComputePipelineState *compile_fill(MTL::Device *device, char const *bin_type)
{
    AutoreleaseScope const scope;
    std::string const      prefixed =
        std::string("#define BIN_T ") + bin_type + "\n" + detail::k_hist_fill_msl;
    NS::String *src =
        NS::String::string(prefixed.c_str(), NS::StringEncoding::UTF8StringEncoding);
    NS::Error    *error = nullptr;
    MTL::Library *lib   = device->newLibrary(src, nullptr, &error);
    if (lib == nullptr)
    {
        throw std::runtime_error(std::string("metal shader compile failed: ") +
                                 error->localizedDescription()->utf8String());
    }
    MTL::Function *fn = lib->newFunction(
        NS::String::string("hist_fill", NS::StringEncoding::UTF8StringEncoding));
    MTL::ComputePipelineState *pso = device->newComputePipelineState(fn, &error);
    fn->release();
    lib->release();
    if (pso == nullptr)
    {
        throw std::runtime_error("metal pipeline creation failed");
    }
    return pso;
}

MTL::Buffer *ensure_capacity(MTL::Device *device, MTL::Buffer *&buffer, size_t bytes)
{
    if (buffer == nullptr || buffer->length() < bytes)
    {
        release_and_clear(buffer);
        buffer = device->newBuffer(std::max(bytes, size_t{64}),
                                   MTL::ResourceStorageModeShared);
    }
    return buffer;
}

uint32_t chunks_for(size_t count)
{
    size_t const wanted = count / k_rows_per_chunk_target;
    return static_cast<uint32_t>(std::clamp<size_t>(wanted, 1, k_max_chunks));
}

float scale_for(size_t count, uint32_t n_chunks, float max_abs)
{
    double const rows_per_group =
        std::ceil(static_cast<double>(count) / static_cast<double>(n_chunks));
    double const largest = std::max(static_cast<double>(max_abs), 1e-30);
    return static_cast<float>(k_scale_headroom / (rows_per_group * largest));
}

} // namespace

struct MetalHistogramEngine::Impl
{
    MTL::Device               *device  = nullptr;
    MTL::CommandQueue         *queue   = nullptr;
    MTL::ComputePipelineState *pso_u8  = nullptr;
    MTL::ComputePipelineState *pso_u16 = nullptr;

    MTL::Buffer *bins  = nullptr;
    MTL::Buffer *grad  = nullptr;
    MTL::Buffer *hess  = nullptr;
    MTL::Buffer *sel   = nullptr;
    MTL::Buffer *nbins = nullptr;
    MTL::Buffer *rows  = nullptr;
    MTL::Buffer *out   = nullptr;

    void const         *bins_key   = nullptr;
    bool                bins_u8    = true;
    size_t              plane_rows = 0;
    size_t              n_features = 0;
    bool                has_hess   = false;
    float               max_abs    = 0.0F;
    std::vector<size_t> bin_counts;
    CpuHistogramEngine  host;

    Impl()                        = default;
    Impl(Impl const &)            = delete;
    Impl &operator=(Impl const &) = delete;

    ~Impl()
    {
        release_and_clear(out);
        release_and_clear(rows);
        release_and_clear(nbins);
        release_and_clear(sel);
        release_and_clear(hess);
        release_and_clear(grad);
        release_and_clear(bins);
        release_and_clear(pso_u16);
        release_and_clear(pso_u8);
        release_and_clear(queue);
    }

    void ensure_pipeline()
    {
        if (queue != nullptr)
        {
            return;
        }
        device = shared_device();
        if (device == nullptr)
        {
            throw std::runtime_error("metal_depthwise requires an Apple GPU; "
                                     "no Metal device is present");
        }
        queue   = device->newCommandQueue();
        pso_u8  = compile_fill(device, "uchar");
        pso_u16 = compile_fill(device, "ushort");
    }

    void stage_bins(Dataset const &ds)
    {
        void const *key =
            ds.visit_bins(0, [](auto col) -> void const * { return col.data(); });
        if (key == bins_key)
        {
            return;
        }
        plane_rows = ds.plane_n_rows();
        n_features = ds.n_features();
        bins_u8    = ds.bins_are_u8();
        bin_counts.resize(n_features);
        size_t max_bins = 0;
        for (size_t f = 0; f < n_features; ++f)
        {
            bin_counts[f] = ds.n_bins(f);
            max_bins      = std::max(max_bins, bin_counts[f]);
        }
        if (4 * max_bins * sizeof(float) > k_threadgroup_ceiling_bytes)
        {
            throw ConfigError(
                "metal_depthwise: " + std::to_string(max_bins) +
                " bins per feature exceeds the 2048-bin threadgroup ceiling; "
                "lower data.max_bins or use a cpu grower");
        }
        size_t const width = bins_u8 ? 1 : 2;
        release_and_clear(bins);
        bins       = device->newBuffer(plane_rows * n_features * width,
                                       MTL::ResourceStorageModeShared);
        auto *base = static_cast<uint8_t *>(bins->contents());
        for (size_t f = 0; f < n_features; ++f)
        {
            ds.visit_bins(f,
                          [&](auto col)
                          {
                              std::memcpy(base + (f * plane_rows * width), col.data(),
                                          col.size_bytes());
                          });
        }
        auto *slots = static_cast<uint32_t *>(
            ensure_capacity(device, nbins, n_features * sizeof(uint32_t))->contents());
        for (size_t f = 0; f < n_features; ++f)
        {
            slots[f] = static_cast<uint32_t>(bin_counts[f]);
        }
        bins_key = key;
    }

    void stage_gradients(floats_view grad_in, floats_view hess_in)
    {
        std::memcpy(ensure_capacity(device, grad, grad_in.size_bytes())->contents(),
                    grad_in.data(), grad_in.size_bytes());
        has_hess = !hess_in.empty();
        if (has_hess)
        {
            std::memcpy(ensure_capacity(device, hess, hess_in.size_bytes())->contents(),
                        hess_in.data(), hess_in.size_bytes());
        }
        float top = has_hess ? 0.0F : 1.0F;
        for (float const g : grad_in)
        {
            top = std::max(top, std::fabs(g));
        }
        for (float const h : hess_in)
        {
            top = std::max(top, std::fabs(h));
        }
        max_abs = top;
    }
};

bool metal_available()
{
    return shared_device() != nullptr;
}

MetalHistogramEngine::MetalHistogramEngine() : impl_(std::make_unique<Impl>()) {}
MetalHistogramEngine::~MetalHistogramEngine()                                = default;
MetalHistogramEngine::MetalHistogramEngine(MetalHistogramEngine &&) noexcept = default;
MetalHistogramEngine &
MetalHistogramEngine::operator=(MetalHistogramEngine &&) noexcept = default;

void MetalHistogramEngine::begin_tree(Dataset const &ds, floats_view grad,
                                      floats_view hess)
{
    impl_->ensure_pipeline();
    impl_->stage_bins(ds);
    impl_->stage_gradients(grad, hess);
    impl_->host.begin_tree(ds, grad, hess);
}

void MetalHistogramEngine::populate(Dataset const &ds, floats_view grad,
                                    floats_view hess, SplitInput &split_input,
                                    std::span<feature_id_t const> selected)
{
    std::array one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
}

void MetalHistogramEngine::populate_many(Dataset const &ds, floats_view grad,
                                         floats_view hess, split_input_refs nodes,
                                         std::span<feature_id_t const> selected)
{
    Impl &m = *impl_;

    size_t level_rows = 0;
    for (auto const &node_ref : nodes)
    {
        SplitInput const &node = node_ref.get();
        level_rows +=
            node.shape.identity && node.rows.empty() ? m.plane_rows : node.rows.size();
    }
    if (level_rows / nodes.size() < min_device_rows())
    {
        m.host.populate_many(ds, grad, hess, nodes, selected);
        return;
    }

    ArenaLayout const layout{m.bin_counts, m.bins_u8};
    if (selected.empty())
    {
        for (auto const &node_ref : nodes)
        {
            node_ref.get().hists.carve(layout, selected, ds.n_features(), true);
        }
        return;
    }

    uint32_t const n_sel     = static_cast<uint32_t>(selected.size());
    auto          *sel_slots = static_cast<uint32_t *>(
        ensure_capacity(m.device, m.sel, n_sel * sizeof(uint32_t))->contents());
    uint32_t stride = 0;
    for (uint32_t j = 0; j < n_sel; ++j)
    {
        sel_slots[j] = selected[j];
        stride = std::max(stride, static_cast<uint32_t>(m.bin_counts[selected[j]]));
    }

    std::vector<size_t> row_starts(nodes.size(), 0);
    size_t              gathered = 0;
    for (size_t n = 0; n < nodes.size(); ++n)
    {
        row_starts[n]          = gathered;
        SplitInput const &node = nodes[n].get();
        if (!node.shape.identity)
        {
            gathered += node.rows.size();
        }
    }
    auto *row_base = static_cast<uint32_t *>(
        ensure_capacity(m.device, m.rows, gathered * sizeof(uint32_t))->contents());
    size_t const node_floats = static_cast<size_t>(n_sel) * 2 * stride;
    auto        *out_buffer =
        ensure_capacity(m.device, m.out, nodes.size() * node_floats * sizeof(float));

    parallel::for_each_index(
        nodes.size(),
        [&](size_t n)
        {
            SplitInput &node = nodes[n].get();
            node.hists.carve(layout, selected, ds.n_features(), nodes.size() == 1);
            if (!node.shape.identity)
            {
                std::memcpy(row_base + row_starts[n], node.rows.data(),
                            node.rows.size() * sizeof(uint32_t));
            }
        });

    {
        AutoreleaseScope const   scope;
        MTL::CommandBuffer      *command = m.queue->commandBuffer();
        MTL::BlitCommandEncoder *blit    = command->blitCommandEncoder();
        blit->fillBuffer(out_buffer,
                         NS::Range::Make(0, nodes.size() * node_floats * sizeof(float)),
                         0);
        blit->endEncoding();

        MTL::ComputeCommandEncoder *enc = command->computeCommandEncoder();
        enc->setComputePipelineState(m.bins_u8 ? m.pso_u8 : m.pso_u16);
        enc->setBuffer(m.bins, 0, 0);
        enc->setBuffer(m.grad, 0, 1);
        enc->setBuffer(m.has_hess ? m.hess : m.grad, 0, 2);
        enc->setBuffer(m.sel, 0, 4);
        enc->setBuffer(m.nbins, 0, 5);
        enc->setThreadgroupMemoryLength(4UL * stride * sizeof(float), 0);

        for (size_t n = 0; n < nodes.size(); ++n)
        {
            SplitInput const &node  = nodes[n].get();
            size_t const      count = node.shape.identity && node.rows.empty()
                                          ? m.plane_rows
                                          : node.rows.size();
            if (count == 0)
            {
                continue;
            }
            bool const       use_rows = !node.shape.identity;
            uint32_t const   n_chunks = chunks_for(count);
            NodeParams const params{
                .count      = static_cast<uint32_t>(count),
                .n_chunks   = n_chunks,
                .use_rows   = use_rows ? 1U : 0U,
                .has_hess   = m.has_hess ? 1U : 0U,
                .plane_rows = static_cast<uint32_t>(m.plane_rows),
                .stride     = stride,
                .scale      = scale_for(count, n_chunks, m.max_abs),
            };
            enc->setBuffer(m.rows, row_starts[n] * sizeof(uint32_t), 3);
            enc->setBuffer(out_buffer, n * node_floats * sizeof(float), 6);
            enc->setBytes(&params, sizeof(params), 7);
            enc->dispatchThreadgroups(MTL::Size::Make(n_sel, n_chunks, 1),
                                      MTL::Size::Make(k_threads_per_group, 1, 1));
        }
        enc->endEncoding();
        command->commit();
        command->waitUntilCompleted();
        if (command->status() == MTL::CommandBufferStatusError)
        {
            throw std::runtime_error("metal histogram dispatch failed");
        }
    }

    static_assert(sizeof(HistCell) == 2 * sizeof(float));
    auto const *result = static_cast<float const *>(out_buffer->contents());
    parallel::for_each_index(
        nodes.size(),
        [&](size_t n)
        {
            SplitInput  &node = nodes[n].get();
            float const *base = result + (n * node_floats);
            for (uint32_t j = 0; j < n_sel; ++j)
            {
                std::span<HistCell> const cells = node.hists[selected[j]].cells();
                std::memcpy(cells.data(), base + (static_cast<size_t>(j) * 2 * stride),
                            cells.size() * sizeof(HistCell));
            }
        });
}

} // namespace bonsai
