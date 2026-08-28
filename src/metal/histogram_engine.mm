#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "bonsai/config/errors.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/metal/histogram_engine.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

constexpr char const *k_kernel_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct NodeParams
{
    uint  count;
    uint  n_chunks;
    uint  use_rows;
    uint  has_hess;
    uint  plane_rows;
    uint  stride;
    float scale;
};

kernel void hist_fill(device const BIN_T      *bins    [[buffer(0)]],
                      device const float      *grad    [[buffer(1)]],
                      device const float      *hess    [[buffer(2)]],
                      device const uint       *rows    [[buffer(3)]],
                      device const uint       *sel     [[buffer(4)]],
                      device const uint       *nbins   [[buffer(5)]],
                      device atomic_float     *out     [[buffer(6)]],
                      constant NodeParams     &p       [[buffer(7)]],
                      threadgroup atomic_uint *sh      [[threadgroup(0)]],
                      uint3 gid  [[threadgroup_position_in_grid]],
                      uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint const slot  = gid.x;
    uint const chunk = gid.y;
    uint const tid   = tid3.x;
    uint const fid   = sel[slot];
    uint const nb    = nbins[fid];
    for (uint i = tid; i < 4 * nb; i += 256)
    {
        atomic_store_explicit(&sh[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup atomic_uint *mine = sh + (((tid >> 5) & 1u) * 2 * nb);
    device const BIN_T      *col  = bins + (ulong)fid * p.plane_rows;
    uint const               span = p.n_chunks * 256;
    for (uint k = (chunk * 256) + tid; k < p.count; k += span)
    {
        uint const  r = p.use_rows != 0 ? rows[k] : k;
        uint const  b = col[r];
        float const g = grad[r];
        float const h = p.has_hess != 0 ? hess[r] : 1.0f;
        atomic_fetch_add_explicit(&mine[2 * b], (uint)(int)rint(g * p.scale),
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&mine[(2 * b) + 1], (uint)(int)rint(h * p.scale),
                                  memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device atomic_float *o = out + ((ulong)slot * 2 * p.stride);
    for (uint i = tid; i < 2 * nb; i += 256)
    {
        int const   lo = (int)atomic_load_explicit(&sh[i], memory_order_relaxed);
        int const   hi = (int)atomic_load_explicit(&sh[(2 * nb) + i],
                                                   memory_order_relaxed);
        float const v  = (float)(lo + hi) / p.scale;
        if (v != 0.0f)
        {
            atomic_fetch_add_explicit(&o[i], v, memory_order_relaxed);
        }
    }
}
)MSL";

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

id<MTLDevice> shared_device()
{
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLComputePipelineState> compile_fill(id<MTLDevice> device, char const *bin_type)
{
    NSString *src = [NSString
        stringWithFormat:@"#define BIN_T %s\n%s", bin_type, k_kernel_source];
    NSError       *error = nil;
    id<MTLLibrary> lib   = [device newLibraryWithSource:src options:nil error:&error];
    if (lib == nil)
    {
        throw std::runtime_error(
            std::string("metal shader compile failed: ") +
            [[error localizedDescription] UTF8String]);
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"hist_fill"]
                                              error:&error];
    if (pso == nil)
    {
        throw std::runtime_error("metal pipeline creation failed");
    }
    return pso;
}

id<MTLBuffer> ensure_capacity(id<MTLDevice> device, id<MTLBuffer> __strong &buffer,
                              size_t bytes)
{
    if (buffer == nil || [buffer length] < bytes)
    {
        buffer = [device newBufferWithLength:std::max(bytes, size_t{64})
                                     options:MTLResourceStorageModeShared];
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
    id<MTLDevice>               device  = nil;
    id<MTLCommandQueue>         queue   = nil;
    id<MTLComputePipelineState> pso_u8  = nil;
    id<MTLComputePipelineState> pso_u16 = nil;

    id<MTLBuffer> bins  = nil;
    id<MTLBuffer> grad  = nil;
    id<MTLBuffer> hess  = nil;
    id<MTLBuffer> sel   = nil;
    id<MTLBuffer> nbins = nil;
    id<MTLBuffer> rows  = nil;
    id<MTLBuffer> out   = nil;

    void const         *bins_key   = nullptr;
    bool                bins_u8    = true;
    size_t              plane_rows = 0;
    size_t              n_features = 0;
    bool                has_hess   = false;
    float               max_abs    = 0.0F;
    std::vector<size_t> bin_counts;

    void ensure_pipeline()
    {
        if (queue != nil)
        {
            return;
        }
        device = shared_device();
        if (device == nil)
        {
            throw std::runtime_error("metal_depthwise requires an Apple GPU; "
                                     "no Metal device is present");
        }
        queue   = [device newCommandQueue];
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
        bins = [device newBufferWithLength:plane_rows * n_features * width
                                   options:MTLResourceStorageModeShared];
        auto *base = static_cast<uint8_t *>([bins contents]);
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
            [ensure_capacity(device, nbins, n_features * sizeof(uint32_t)) contents]);
        for (size_t f = 0; f < n_features; ++f)
        {
            slots[f] = static_cast<uint32_t>(bin_counts[f]);
        }
        bins_key = key;
    }

    void stage_gradients(floats_view grad_in, floats_view hess_in)
    {
        std::memcpy([ensure_capacity(device, grad, grad_in.size_bytes()) contents],
                    grad_in.data(), grad_in.size_bytes());
        has_hess = !hess_in.empty();
        if (has_hess)
        {
            std::memcpy([ensure_capacity(device, hess, hess_in.size_bytes()) contents],
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
    return shared_device() != nil;
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
}

void MetalHistogramEngine::populate(Dataset const &ds, floats_view grad,
                                    floats_view hess, SplitInput &split_input,
                                    std::span<feature_id_t const> selected)
{
    std::array one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
}

void MetalHistogramEngine::populate_many(Dataset const &ds, floats_view /*grad*/,
                                         floats_view /*hess*/, split_input_refs nodes,
                                         std::span<feature_id_t const> selected)
{
    Impl &m = *impl_;

    ArenaLayout const layout{m.bin_counts, m.bins_u8};
    for (auto const &node_ref : nodes)
    {
        node_ref.get().hists.carve(layout, selected, ds.n_features(),
                                   nodes.size() == 1);
    }
    if (selected.empty())
    {
        return;
    }

    uint32_t const n_sel = static_cast<uint32_t>(selected.size());
    auto *sel_slots      = static_cast<uint32_t *>(
        [ensure_capacity(m.device, m.sel, n_sel * sizeof(uint32_t)) contents]);
    uint32_t stride = 0;
    for (uint32_t j = 0; j < n_sel; ++j)
    {
        sel_slots[j] = selected[j];
        stride = std::max(stride, static_cast<uint32_t>(m.bin_counts[selected[j]]));
    }

    size_t gathered = 0;
    for (auto const &node_ref : nodes)
    {
        SplitInput const &node = node_ref.get();
        if (!node.shape.identity)
        {
            gathered += node.rows.size();
        }
    }
    auto *row_base = static_cast<uint32_t *>(
        [ensure_capacity(m.device, m.rows, gathered * sizeof(uint32_t)) contents]);
    size_t const node_floats = static_cast<size_t>(n_sel) * 2 * stride;
    auto *out_buffer =
        ensure_capacity(m.device, m.out, nodes.size() * node_floats * sizeof(float));

    id<MTLCommandBuffer>      command = [m.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit    = [command blitCommandEncoder];
    [blit fillBuffer:out_buffer
               range:NSMakeRange(0, nodes.size() * node_floats * sizeof(float))
               value:0];
    [blit endEncoding];

    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    [enc setComputePipelineState:m.bins_u8 ? m.pso_u8 : m.pso_u16];
    [enc setBuffer:m.bins offset:0 atIndex:0];
    [enc setBuffer:m.grad offset:0 atIndex:1];
    [enc setBuffer:m.has_hess ? m.hess : m.grad offset:0 atIndex:2];
    [enc setBuffer:m.sel offset:0 atIndex:4];
    [enc setBuffer:m.nbins offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:4UL * stride * sizeof(float) atIndex:0];

    size_t row_offset = 0;
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
        bool const use_rows = !node.shape.identity;
        if (use_rows)
        {
            std::memcpy(row_base + row_offset, node.rows.data(),
                        node.rows.size() * sizeof(uint32_t));
        }
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
        [enc setBuffer:m.rows offset:row_offset * sizeof(uint32_t) atIndex:3];
        [enc setBuffer:out_buffer offset:n * node_floats * sizeof(float) atIndex:6];
        [enc setBytes:&params length:sizeof(params) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(n_sel, n_chunks, 1)
            threadsPerThreadgroup:MTLSizeMake(k_threads_per_group, 1, 1)];
        if (use_rows)
        {
            row_offset += node.rows.size();
        }
    }
    [enc endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if ([command status] == MTLCommandBufferStatusError)
    {
        throw std::runtime_error("metal histogram dispatch failed");
    }

    auto const *result = static_cast<float const *>([out_buffer contents]);
    for (size_t n = 0; n < nodes.size(); ++n)
    {
        SplitInput  &node = nodes[n].get();
        float const *base = result + (n * node_floats);
        for (uint32_t j = 0; j < n_sel; ++j)
        {
            std::span<HistCell> const cells = node.hists[selected[j]].cells();
            float const *src = base + (static_cast<size_t>(j) * 2 * stride);
            for (size_t b = 0; b < cells.size(); ++b)
            {
                cells[b].sum_grad = src[2 * b];
                cells[b].sum_hess = src[(2 * b) + 1];
            }
        }
    }
}

} // namespace bonsai
