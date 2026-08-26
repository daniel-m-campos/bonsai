#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "device_buffer.cuh"
#include "ingest_plane.cuh"
#include "profile.cuh"

namespace bonsai
{
namespace cuda_detail
{
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

struct CudaDeviceContext
{
    struct DatasetKey
    {
        BinStore const *store                                = nullptr;
        void const     *bins0                                = nullptr;
        size_t          n_rows                               = 0;
        size_t          n_feats                              = 0;
        bool            operator==(DatasetKey const &) const = default;
    };

    struct DeviceData
    {
        DeviceBuffer<uint8_t>  bins8;
        DeviceBuffer<uint16_t> bins16;
        bool                   bins_are_u8 = false;
        DeviceBuffer<uint32_t> n_bins;

        std::shared_ptr<CudaIngestPlane const> adopted;

        DatasetKey key;

        uint32_t const *n_bins_ptr() const
        {
            return adopted ? adopted->n_bins.data() : n_bins.data();
        }

        template <typename F> void dispatch_bins(F &&fn)
        {
            if (bins_are_u8)
            {
                std::forward<F>(fn)(adopted ? adopted->bins8.data() : bins8.data());
            }
            else
            {
                std::forward<F>(fn)(adopted ? adopted->bins16.data() : bins16.data());
            }
        }
    };

    struct GradientPlane
    {
        DeviceBuffer<float>  grad_raw;
        DeviceBuffer<float>  hess_raw;
        DeviceBuffer<float2> gh;
    };

    struct LevelPipeline
    {
        DeviceBuffer<uint32_t> rows;
        DeviceBuffer<float2>   gh_ordered;
        Staged<uint32_t>       row_offsets;
        Staged<uint32_t>       row_counts;
        Staged<uint32_t>       features;
        Staged<uint32_t>       sel_slot;

        DeviceBuffer<double>   level_a;
        DeviceBuffer<double>   level_b;
        bool                   cur_is_a   = true;
        uint32_t               n_selected = 0;
        uint32_t               stride     = 0;
        Staged<uint32_t>       slots;
        Staged<uint32_t>       triples;
        Staged<double>         node_sums;
        Staged<double>         node_bounds;
        Staged<char>           allowed;
        Staged<int>            monotone;
        DeviceBuffer<FeatBest> feat_best;
        Staged<FeatBest>       node_best;
        Staged<double>         level_child;
        DeviceBuffer<double>   level_score;
        Staged<uint32_t>       small_offsets;
        Staged<uint32_t>       small_counts;
        Staged<uint32_t>       small_slots;

        DeviceBuffer<uint32_t> rows_b;
        DeviceBuffer<float2>   gh_b;
        DeviceBuffer<uint8_t>  flags;
        Staged<PartOpDev>      part_ops;
        DeviceBuffer<uint32_t> block_counts;
        Staged<uint32_t>       nl_dev;
        Staged<uint32_t>       stamp_ids;
        DeviceBuffer<uint32_t> leaf_by_row;
        // perf: Pristine root row list for full-data fits: partitioning ping-pongs
        // the working rows buffer, so the identity permutation is cached once
        // and restored device-to-device per tree instead of re-uploaded.
        // 0 = invalid; only ever the identity, which is what every sampler
        // returns when its size equals n_rows.
        DeviceBuffer<uint32_t> root_rows;
        size_t                 root_rows_cached_n = 0;
        DeviceBuffer<double2>  sum_partial;
        DeviceBuffer<double2>  sum_out;
        DeviceBuffer<float>    epi_node_vals;
        DeviceBuffer<float>    epi_values;
        std::vector<uint32_t>  slot_offsets;
        std::vector<uint32_t>  slot_counts;
        std::vector<uint32_t>  next_offsets;
        std::vector<uint32_t>  next_counts;

        enum : int
        {
            ev_before_memset = 0,
            ev_after_memset,
            ev_after_hist,
            ev_after_subtract,
        };
        cudaEvent_t prof_ev[4]       = {};
        bool        prof_ev_ready    = false;
        bool        prof_ev_recorded = false;
        bool        prof_ev_root     = false;
        cudaEvent_t part_ev[4]       = {};
        bool        part_ev_ready    = false;

        void prof_record_begin(bool root);
        void prof_read(ProfileCounters &prof);
        ~LevelPipeline();

        DeviceBuffer<uint32_t> &cur_rows()
        {
            return cur_is_a ? rows : rows_b;
        }
        DeviceBuffer<uint32_t> &other_rows()
        {
            return cur_is_a ? rows_b : rows;
        }
        DeviceBuffer<float2> &cur_gh()
        {
            return cur_is_a ? gh_ordered : gh_b;
        }
        DeviceBuffer<float2> &other_gh()
        {
            return cur_is_a ? gh_b : gh_ordered;
        }
        DeviceBuffer<double> &cur()
        {
            return cur_is_a ? level_a : level_b;
        }
        DeviceBuffer<double> &other()
        {
            return cur_is_a ? level_b : level_a;
        }
        size_t slot_doubles() const
        {
            return static_cast<size_t>(n_selected) * stride;
        }

        size_t stage_children(std::span<CudaHistogramEngine::LevelOp const> ops);

        void layout_children(std::span<CudaHistogramEngine::PartitionOp const> ops,
                             std::span<uint32_t> child_counts);

        bool stage_find_inputs(std::span<SplitInput const> level,
                               TreeConfig const &config, Dataset const &ds);

        void unpack_splits(std::span<SplitInput const> level, TreeConfig const &config,
                           std::span<SplitOutput> out, std::span<HistCell> child_sums);

        void stage_level_sums(std::span<SplitInput const> level);
    };

    struct LeafPipeline
    {
        DeviceBuffer<double>  pool;
        std::vector<uint32_t> slot_offsets;
        std::vector<uint32_t> slot_counts;
        // perf: Per-round staging. Pinned and asynchronous because the round's whole
        // host residue is these uploads: a pageable copy stream-syncs before it
        // starts, so 8 of them per round drain the pipeline 8 times.
        // Each buffer is written once per round, downstream of the blocking
        // fetch that fenced the previous round's upload of it.
        PinnedStaged<PartOpDev> part_op;
        PinnedStaged<uint32_t>  build_seg;
        PinnedStaged<double>    find_stats;
        PinnedStaged<uint32_t>  find_slots;
        Staged<int>             monotone;
        uint32_t                next_slot = 0;
        uint32_t                max_slots = 0;
    };

    struct NodeTable
    {
        Staged<uint32_t> feature;
        Staged<uint32_t> split_bin;
        Staged<uint32_t> left;
        Staged<uint32_t> right;
        Staged<uint32_t> default_left;
        Staged<uint32_t> is_leaf;
        Staged<float>    value;
        void stage(std::span<CudaHistogramEngine::ResidentNode const> nodes);
    };

    struct ResidentPlane
    {
        DeviceBuffer<float>    labels;
        DeviceBuffer<float>    scores;
        DeviceBuffer<float>    weights;
        LabelsId               labels_key{};
        NodeTable              nodes;
        DeviceObjectiveKind    kind          = DeviceObjectiveKind::none;
        bool                   weighted      = false;
        bool                   armed         = false;
        float                  learning_rate = 0.0F;
        size_t                 n_rows        = 0;
        DeviceBuffer<row_id_t> rows;
        size_t                 n_view_rows   = 0;
        bool                   view_identity = true;
    };

    struct EvalPlane
    {
        DeviceBuffer<uint8_t>  bins;
        DeviceBuffer<uint32_t> n_bins;
        DeviceBuffer<float>    scores;
        DeviceBuffer<float>    labels;
        Staged<double>         loss_partial;
        DeviceObjectiveKind    kind = DeviceObjectiveKind::none;
        NodeTable              nodes;
        bool                   armed   = false;
        size_t                 n_rows  = 0;
        size_t                 n_feats = 0;
    };

    DeviceData      data;
    GradientPlane   grads;
    LevelPipeline   lvl;
    LeafPipeline    leaf;
    ResidentPlane   resident;
    EvalPlane       veval;
    ProfileCounters prof_counters;

    // perf: Runtime shared-memory ceiling for the hist kernels: the opt-in limit
    // when the device grants one (both BinT instantiations opted in), else
    // the 48 KiB static budget. Resolved lazily on first use so engine
    // construction never touches the CUDA runtime.
    size_t shared_limit  = k_max_shared_bytes;
    int    sm_count      = 0;
    bool   shared_probed = false;
    bool   plane_noted   = false;

    // perf: The one histogram-capacity predicate: a node's per-feature scratch is
    // 4 * bins floats in shared memory. begin_root refuses a tree that fails
    // it, and resident_begin must apply the SAME test (once per fit, on the
    // worst-case feature) so no tree can fail it after the resident mode
    // armed. Any new capacity condition must land here, visible to both
    // callers.
    bool hist_budget_ok(size_t max_bins) const
    {
        return 4 * max_bins * sizeof(float) <= shared_limit;
    }

    bool leaf_pool_ok(size_t bytes) const;

    bool leaf_budget_ok(TreeConfig const &config, size_t n_selected,
                        size_t max_bins) const;

    void     init_shared_limit();
    void     ensure_dataset(Dataset const &dataset);
    void     stage_selection(std::span<feature_id_t const> selected, size_t n_feats);
    void     note_plane(bool tiled, size_t shared);
    void     launch_hist(uint32_t ds_rows, uint32_t ds_feats, uint32_t n_nodes,
                         uint32_t max_rows, float2 const *gh, uint32_t const *rows,
                         uint32_t const *offsets, uint32_t const *counts, double *out,
                         uint32_t const *slots);
    uint32_t stage_root_rows(SplitInput const &root, bool identity);
    void     begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    void     begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                        SplitInput &root, std::span<feature_id_t const> selected);
    void     stamp_leaves(std::span<CudaHistogramEngine::LeafStamp const> stamps);
    void     partition_level(Dataset const                                    &ds,
                             std::span<CudaHistogramEngine::PartitionOp const> ops,
                             std::span<uint32_t> child_counts);
    void     finalize_tree(std::span<float const> node_values, std::span<float> values,
                           std::span<node_id_t> leaf_ids);
    void     advance_level(Dataset const                                &ds,
                           std::span<CudaHistogramEngine::LevelOp const> ops);
    void     advance_layout_only();
    void     find_splits_many(Dataset const &ds, TreeConfig const &config,
                              std::span<SplitInput const> level, std::span<SplitOutput> out,
                              std::span<HistCell> child_sums);
    void     find_level_split(Dataset const &ds, TreeConfig const &config,
                              std::span<SplitInput const> level, std::span<SplitOutput> out,
                              std::span<HistCell> child_sums);

    void leaf_begin_root(Dataset const &ds, TreeConfig const &config, floats_view grad,
                         floats_view hess, SplitInput &root,
                         std::span<feature_id_t const> selected);
    CudaHistogramEngine::LeafRound
         leaf_split(Dataset const &ds, CudaHistogramEngine::LeafPartOp const &op);
    void leaf_build(Dataset const &ds, uint32_t small_slot, uint32_t large_slot);
    bool leaf_stage_find(std::span<SplitInput const> nodes,
                         std::span<uint32_t const>   slots);
    void leaf_find(Dataset const &ds, TreeConfig const &config,
                   std::span<SplitInput const> nodes, std::span<uint32_t const> slots,
                   std::span<SplitOutput> out, std::span<HistCell> child_sums);
    void leaf_stamp(std::span<CudaHistogramEngine::LeafStamp const> stamps);

    bool resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                        std::span<float const> initial_scores, float learning_rate);
    bool resident_begin_leaf(Dataset const &ds, TreeConfig const &config,
                             DeviceObjectiveKind    kind,
                             std::span<float const> initial_scores,
                             float                  learning_rate);
    bool resident_armed() const
    {
        return resident.armed;
    }
    void resident_finalize(std::span<CudaHistogramEngine::ResidentNode const> nodes);
    void resident_end(std::span<float> scores_out);

    bool eval_begin(Dataset const &valid, DeviceObjectiveKind kind,
                    std::span<float const> initial_scores);
    std::optional<float>
    eval_accumulate(std::span<CudaHistogramEngine::ResidentNode const> nodes, float lr,
                    std::span<float> scores_out);
};

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace cuda_detail
} // namespace bonsai
