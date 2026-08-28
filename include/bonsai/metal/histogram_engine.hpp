#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <memory>
#include <span>

namespace bonsai
{

// True when this build carries the Metal backend AND a usable device is
// present. metal_depthwise is registered in every build; only training
// needs this to be true.
bool metal_available();

// HistogramEngine that offloads the histogram fill to the Apple GPU and
// nothing else: the grower's level loop, partition, and split finding stay
// on the proven host path, which unified memory makes affordable (there is
// no bus for rows or decisions to cross). This engine therefore satisfies
// only the base HistogramEngine concept, never GPULevelEngine, and every
// host-path feature (monotone constraints, interaction constraints,
// sampling, leaf renewal) works unchanged because only the fill moved.
//
// What it guarantees: populate/populate_many produce cells equal to the CPU
// engine's to tolerance, not bit-exactly. Threadgroup accumulation is
// fixed-point on 32-bit integer atomics (Apple8 has no threadgroup float
// atomics and no double anywhere), so within a chunk the sum is
// order-independent; the per-node scale is derived from the tree's max
// |gradient| against the worst case of every row of a chunk landing in one
// bin, so the integer sum cannot overflow on any input. Chunks merge into
// float device atomics, which reintroduces order dependence at that one
// seam.
//
// What breaks it: a dataset whose largest bin count needs more than the
// 32KB threadgroup ceiling (4 * n_bins * 4 bytes) is refused at begin_tree
// with ConfigError rather than silently moved to the host.
//
// Pinned by tests/unit/test_metal_engine.cpp (cell parity vs the CPU
// engine, both bin widths, subset rows, empty hessian).
class MetalHistogramEngine
{
  public:
    MetalHistogramEngine();
    ~MetalHistogramEngine();
    MetalHistogramEngine(MetalHistogramEngine &&) noexcept;
    MetalHistogramEngine &operator=(MetalHistogramEngine &&) noexcept;
    MetalHistogramEngine(MetalHistogramEngine const &)            = delete;
    MetalHistogramEngine &operator=(MetalHistogramEngine const &) = delete;

    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &split_input, std::span<feature_id_t const> selected);
    void populate_many(Dataset const &ds, floats_view grad, floats_view hess,
                       split_input_refs nodes, std::span<feature_id_t const> selected);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

static_assert(HistogramEngine<MetalHistogramEngine>);

} // namespace bonsai
