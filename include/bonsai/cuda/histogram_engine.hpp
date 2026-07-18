#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective_traits.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace bonsai
{

// True when this build carries the CUDA backend AND a usable device is
// present. cuda_depthwise is registered in every build; only training
// needs this to be true. See docs/architecture/10-cuda.md.
bool cuda_available();

// Select the CUDA device for subsequent device work on the CALLING thread
// (parallel.device_id, issue #158). 0 selects the default device when one
// exists and is a no-op otherwise, so the config default changes nothing on
// GPU-less hosts (graceful degradation intact); a nonzero id is validated
// against the visible device count, and out-of-range or a CUDA-less build
// throws ConfigError. Placement only: model bits are unaffected.
void cuda_select_device(uint32_t device_id);

// The CUDA ingest transaction (decision 54, docs 15/16): bins raw features
// on the device against host-fitted cuts and returns the resident plane for
// Dataset::bin to carry. Returns nullptr — leaving the caller on the host
// fill — when the build has no backend, no usable device is present, or the
// dataset's total bins exceed the resident path's shared-memory ceiling
// (grow would decline into the host fallback anyway). Bin ids are
// bit-identical to the host fill over the same mappers.
std::shared_ptr<IngestPlane const> cuda_ingest(detail::ColumnBatch const &batch,
                                               BinMappers const          &mappers);
std::shared_ptr<IngestPlane const> cuda_ingest(features_view     X,
                                               BinMappers const &mappers);

// HistogramEngine that offloads histogram construction to the GPU
// (src/cuda/histogram_engine.cu; a throwing stub backs it when built
// without BONSAI_CUDA). GPU cells match the CPU engine to tolerance, not
// bit-exactly: atomics accumulate in arbitrary order. Design and precision
// scheme: docs/architecture/10-cuda.md.
class CudaHistogramEngine
{
  public:
    CudaHistogramEngine();
    ~CudaHistogramEngine();
    CudaHistogramEngine(CudaHistogramEngine &&) noexcept;
    CudaHistogramEngine &operator=(CudaHistogramEngine &&) noexcept;
    CudaHistogramEngine(CudaHistogramEngine const &)            = delete;
    CudaHistogramEngine &operator=(CudaHistogramEngine const &) = delete;

    // --- HistogramEngine concept (required): begin_tree stages the per-tree
    // gradient upload; populate is the CPU fallback for trees begin_root
    // declines (the GPU copy-back path was retired by decision 41).
    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &split_input, std::span<feature_id_t const> selected);

    // --- GPULevelEngine (optional, phase 3,
    // docs/architecture/11-gpu-resident.md). Level histograms stay on the device, keyed
    // by the node's index in the grower's frontier ("slot"); splits are found on the
    // device and only decisions and child sums cross the bus. The depthwise grower
    // gates this whole cluster on the GPULevelEngine concept; the splitter
    // policy remains the host fallback when begin_root declines.

    // One child-level derivation: the smaller child's histogram builds from
    // its device row segment; the larger derives on-device as parent minus
    // smaller. Rows never cross the bus: partition_level wrote the segments.
    struct LevelOp
    {
        uint32_t parent_slot;
        uint32_t small_slot;
        uint32_t large_slot;
    };

    // One split's routing: partition the parent slot's row segment into the
    // children, stably (left rows keep ascending order, then right rows).
    struct PartitionOp
    {
        uint32_t     parent_slot;
        uint32_t     left_slot;
        uint32_t     right_slot;
        feature_id_t feature_id;
        bin_id_t     bin_id;
        bool         default_left;
    };

    struct LeafStamp
    {
        uint32_t  slot;
        node_id_t node_id;
    };

    // One flattened tree node the device-resident epilogue walks in bin space
    // to fuse the per-row score update (docs/architecture/10-cuda.md). Internal
    // nodes carry the split (feature, bin, missing routing, children); leaves
    // carry the contribution. Dense trees map their nodes one-to-one; oblivious
    // trees synthesize the perfect-tree numbering (children 2i+1 / 2i+2).
    struct ResidentNode
    {
        feature_id_t feature_id   = 0;
        bin_id_t     split_bin    = 0;
        node_id_t    left         = 0;
        node_id_t    right        = 0;
        float        value        = 0.0F;
        bool         default_left = false;
        bool         is_leaf      = false;
    };

    // Starts the resident path for this tree: builds the root histogram into
    // slot 0 and fills root.sums/row_count. Returns false (leaving root
    // untouched) when the resident path cannot run — no device, or a feature's
    // bins exceed the shared-memory budget — and the caller uses populate.
    bool begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                    SplitInput &root, std::span<feature_id_t const> selected);
    // Records final leaf assignment for every row in the given slots'
    // segments (call before the level advances past them).
    void stamp_leaves(std::span<LeafStamp const> stamps);
    // Routes every split slot's rows into its children on the device;
    // child_counts receives (n_left, n_right) per op — the only partition
    // data that returns to the host.
    void partition_level(Dataset const &ds, std::span<PartitionOp const> ops,
                         std::span<uint32_t> child_counts);
    // Populates all smaller children and subtracts all larger ones, then
    // makes the child level current.
    void advance_level(Dataset const &ds, std::span<LevelOp const> ops);
    // Last level of a tree: children are leaves, their histograms unread;
    // performs only the segment-layout flip that stamping depends on.
    void advance_layout_only();
    // End of tree: the per-row leaf assignment (indexed by row id; only rows
    // this tree trained on carry fresh values).
    void finalize_rows(std::span<node_id_t> leaf_by_row);
    // Tree epilogue, engine-owned (decision 53 step 3): maps the resident
    // per-row leaf assignment through node_values on device and downloads
    // values and leaf ids in two bulk copies — replacing the per-tree
    // host stamping loop over every row.
    void finalize_tree(std::span<float const> node_values, std::span<float> values,
                       std::span<node_id_t> leaf_ids);
    // Best split per frontier node from the current level's device
    // histograms; child_sums receives the winning cut's (left, right) totals,
    // 2 cells per node. level[i] corresponds to slot i.
    void find_splits_many(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums);
    // Oblivious: ONE split for the whole frontier, chosen to maximize the gain
    // summed across all nodes and feasible for every node. out is filled with
    // that split for every slot, and child_sums with each node's (left, right)
    // totals at that cut — they seed the children's SplitInput.sums, which the
    // next level's find reads. Enables ObliviousGrower<CudaHistogramEngine>.
    void find_level_split(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums);

    // --- Device-resident objective (docs/architecture/10-cuda.md). Labels and
    // the per-row score vector live on the device for the whole fit. Per tree
    // the engine derives grad/hess from them on device (no host objective, no
    // gh upload); the resident finalize walks the finished tree in bin space
    // and fuses scores[r] += lr * value on device (no values/leaf_ids D2H).
    // resident_begin uploads labels and, when the dataset is weighted, the row
    // weights (both keyed by dataset identity) plus the initial scores once, and
    // returns false when the objective is unsupported or the
    // capacity that lets every tree stay device-resident does not hold: the
    // caller then trains on the host path unchanged. resident_end downloads the
    // scores so the host copy is authoritative again.
    bool resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                        std::span<float const> initial_scores, float learning_rate);
    bool resident_armed() const;
    void resident_finalize(std::span<ResidentNode const> nodes);
    void resident_end(std::span<float> scores_out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

static_assert(HistogramEngine<CudaHistogramEngine>);
static_assert(GPULevelEngine<CudaHistogramEngine>);

} // namespace bonsai
