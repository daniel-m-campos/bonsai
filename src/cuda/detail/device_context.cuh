#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <driver_types.h>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
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
        DeviceBuffer<float>   grad_raw;
        DeviceBuffer<float>   hess_raw;
        DeviceBuffer<float2>  gh;
        DeviceBuffer<uint2>   absmax;
        DeviceBuffer<GhQuant> quant;
    };

    struct LevelPipeline
    {
        DeviceBuffer<uint32_t> rows;
        DeviceBuffer<float2>   gh_ordered;
        Staged<uint32_t>       row_offsets;
        Staged<uint32_t>       row_counts;
        Staged<uint32_t>       features;
        Staged<uint32_t>       sel_slot;

        DeviceBuffer<hist_int_t> level_a;
        DeviceBuffer<hist_int_t> level_b;
        bool                     cur_is_a   = true;
        uint32_t                 depth      = 0;
        uint32_t                 n_selected = 0;
        uint32_t                 stride     = 0;
        Staged<uint32_t>         slots;
        Staged<uint32_t>         triples;
        Staged<SiblingDerive>    derive;
        Staged<double>           node_sums;
        Staged<NodeScreen>       node_screen;
        Staged<double>           node_bounds;
        Staged<char>             allowed;
        Staged<int>              monotone;
        DeviceBuffer<FeatBest>   feat_best;
        Staged<FeatBest>         node_best;
        Staged<double>           level_child;
        Staged<uint32_t>         small_offsets;
        Staged<uint32_t>         small_counts;
        Staged<uint32_t>         small_slots;

        DeviceBuffer<uint32_t> rows_b;
        DeviceBuffer<float2>   gh_b;
        Staged<PartOpDev>      part_ops;
        PartTiles              part_tiles;
        Staged<uint32_t>       nl_dev;
        Staged<uint32_t>       stamp_ids;
        DeviceBuffer<uint32_t> leaf_by_row;
        // perf: Pristine root row list for full-data fits: partitioning ping-pongs
        // the working rows buffer, so the identity permutation is cached once
        // and restored device-to-device per tree instead of re-uploaded.
        // Only ever the identity permutation (invariants:
        // device-root-spends-its-host-rows).
        DeviceBuffer<uint32_t> root_rows;
        size_t                 root_rows_cached_n = 0;
        DeviceBuffer<double2>  sum_partial;
        DeviceBuffer<double2>  sum_out;
        DeviceBuffer<float>    epi_node_vals;
        DeviceBuffer<float>    epi_values;
        std::vector<RowSeg>    segs;
        std::vector<RowSeg>    next_segs;

        bool root_rows_cached(size_t n_rows) const
        {
            return root_rows_cached_n == n_rows;
        }

        KernelTimer memset_timer;
        KernelTimer hist_timer;
        KernelTimer small_timer;
        KernelTimer part_timer;
        bool        fill_timed = false;
        uint32_t    fill_level = 0;

        void fill_done(uint32_t level);
        void prof_read(ProfileCounters &prof);

        DeviceBuffer<uint32_t> &rows_of(bool in_b)
        {
            return in_b ? rows_b : rows;
        }
        DeviceBuffer<float2> &gh_of(bool in_b)
        {
            return in_b ? gh_b : gh_ordered;
        }
        DeviceBuffer<uint32_t> &cur_rows()
        {
            return rows_of(!cur_is_a);
        }
        DeviceBuffer<uint32_t> &other_rows()
        {
            return rows_of(cur_is_a);
        }
        DeviceBuffer<float2> &cur_gh()
        {
            return gh_of(!cur_is_a);
        }
        DeviceBuffer<float2> &other_gh()
        {
            return gh_of(cur_is_a);
        }
        DeviceBuffer<hist_int_t> &cur()
        {
            return cur_is_a ? level_a : level_b;
        }
        DeviceBuffer<hist_int_t> &other()
        {
            return cur_is_a ? level_b : level_a;
        }
        size_t slot_cells() const
        {
            return static_cast<size_t>(n_selected) * stride;
        }

        size_t stage_children(std::span<CudaHistogramEngine::LevelOp const> ops);

        void layout_children(std::span<CudaHistogramEngine::PartitionOp const> ops,
                             std::span<uint32_t> child_counts);

        bool stage_find_inputs(std::span<SplitInput const> level,
                               TreeConfig const &config, Dataset const &ds);

        bool stage_allowed(std::span<SplitInput const> nodes);

        void unpack_splits(std::span<SplitInput const> level, TreeConfig const &config,
                           std::span<FeatBest const> best, std::span<SplitOutput> out,
                           std::span<NodeTotals> child_sums);

        void stage_level_sums(std::span<SplitInput const> level);
    };

    struct LeafPipeline
    {
        DeviceBuffer<hist_int_t> pool;
        std::vector<RowSeg>      segs;
        std::vector<uint8_t>     slot_in_b;
        DeviceBuffer<BuildSeg>   build_seg;
        Staged<int>              monotone;
        MappedBuffer<uint32_t>   n_left;
        MappedBuffer<FeatBest>   node_best;
        StreamFence              fence;
        Once                     queued_fill_noted;
        Once                     launch_args_noted;
        uint32_t                 next_slot     = 0;
        uint32_t                 max_slots     = 0;
        uint32_t                 pending_small = k_not_selected;
        uint32_t                 pending_large = k_not_selected;
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
        DeviceBuffer<float> labels;
        DeviceBuffer<float> scores;
        DeviceBuffer<float> weights;
        LabelsId            labels_key{};
        NodeTable           nodes;
        DeviceObjectiveKind kind          = DeviceObjectiveKind::none;
        bool                weighted      = false;
        bool                armed         = false;
        float               learning_rate = 0.0F;
        size_t              n_rows        = 0;
        RowMap              rows;
    };

    struct EvalPlane
    {
        DeviceBuffer<uint8_t>              bins;
        DeviceBuffer<uint32_t>             n_bins;
        DeviceBuffer<float>                scores;
        DeviceBuffer<float>                labels;
        Staged<double>                     loss_partial;
        DeviceObjectiveKind                kind = DeviceObjectiveKind::none;
        NodeTable                          nodes;
        RowMap                             rows;
        std::shared_ptr<IngestPlane const> adopted;
        uint8_t const                     *bin_source = nullptr;
        bool                               armed      = false;
        size_t                             plane_rows = 0;
        size_t                             n_feats    = 0;
    };

    DeviceData      data;
    GradientPlane   grads;
    LevelPipeline   lvl;
    LeafPipeline    leaf;
    ResidentPlane   resident;
    EvalPlane       veval;
    ProfileCounters prof_counters;
    bool finder_exhaustive = std::getenv("BONSAI_CUDA_FINDER_EXHAUSTIVE") != nullptr;

    // perf: Runtime shared-memory ceiling for the hist kernels: the opt-in limit
    // when the device grants one (both BinT instantiations opted in), else
    // the 48 KiB static budget. Resolved lazily on first use so engine
    // construction never touches the CUDA runtime.
    size_t shared_limit  = k_max_shared_bytes;
    int    sm_count      = 0;
    bool   shared_probed = false;
    Once   plane_noted;
    Once   quant_noted;

    // perf: The one histogram-capacity predicate: a node's per-feature scratch is
    // 2 * bins int64 cells in shared memory. begin_root refuses a tree that
    // fails it, and leaf_begin_root, leaf_budget_ok and resident_begin apply
    // the SAME test, resident once per fit on the worst-case feature, so no
    // tree can fail it after the resident mode armed (invariants:
    // device-oversized-tree-refuses-not-falls-back). Any new capacity
    // condition lands here.
    bool hist_budget_ok(size_t max_bins) const
    {
        return hist_shared_bytes(max_bins) <= shared_limit;
    }

    bool leaf_pool_ok(size_t bytes) const;

    bool leaf_budget_ok(TreeConfig const &config, size_t n_selected,
                        size_t max_bins) const;

    void   init_shared_limit();
    void   ensure_dataset(Dataset const &dataset);
    size_t stage_selection(Dataset const &ds, std::span<feature_id_t const> selected);
    void   require_hist_fits(size_t max_sel_bins) const;
    void   launch_root_sums(float2 const *gh, uint32_t n);
    void   launch_stamp(std::span<CudaHistogramEngine::LeafStamp const> stamps,
                        std::span<RowSeg const> segs, uint32_t const *rows,
                        char const *label);
    NodeTotals                      fetch_root_sums();
    void                            wait_for_profile(ProfileCounters::Lap &lap);
    void                            note_plane(bool tiled, size_t shared);
    void                            note_quant();
    void                            note_once(Once &noted, std::string_view line);
    bool                            unit_hessian() const;
    size_t                          tiled_shared_bytes() const;
    bool                            tiled_plane() const;
    template <typename Launch> void dispatch_tiled(Launch const &launch)
    {
        data.dispatch_bins(
            [&](auto const *bins)
            {
                if (unit_hessian())
                {
                    launch(bins, std::true_type{});
                }
                else
                {
                    launch(bins, std::false_type{});
                }
            });
    }
    void       launch_small_fill(Dataset const &ds, bool stored);
    FillLaunch launch_hist(uint32_t ds_rows, uint32_t ds_feats, uint32_t n_nodes,
                           uint32_t max_rows, float2 const *gh, uint32_t const *rows,
                           uint32_t const *offsets, uint32_t const *counts,
                           hist_int_t *out, uint32_t const *slots);
    uint32_t   stage_root_rows(SplitInput const &root, bool identity);
    void       begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    void       begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                          SplitInput &root, std::span<feature_id_t const> selected);
    void       stamp_leaves(std::span<CudaHistogramEngine::LeafStamp const> stamps);
    void       partition_level(Dataset const                                    &ds,
                               std::span<CudaHistogramEngine::PartitionOp const> ops,
                               std::span<uint32_t> child_counts);
    void finalize_tree(std::span<float const> node_values, std::span<float> values,
                       std::span<node_id_t> leaf_ids);
    void advance_level(Dataset const                                &ds,
                       std::span<CudaHistogramEngine::LevelOp const> ops);
    void advance_layout_only();
    void find_splits_many(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<NodeTotals> child_sums);
    void find_level_split(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<NodeTotals> child_sums);

    void leaf_begin_root(Dataset const &ds, TreeConfig const &config, floats_view grad,
                         floats_view hess, SplitInput &root,
                         std::span<feature_id_t const> selected);
    CudaHistogramEngine::LeafRound
         leaf_split(Dataset const &ds, CudaHistogramEngine::LeafPartOp const &op);
    void leaf_enqueue_fill(Dataset const &ds, bool in_b, uint32_t max_rows);
    CudaHistogramEngine::LeafRound
    leaf_children(CudaHistogramEngine::LeafPartOp const &op, uint32_t offset,
                  uint32_t count, uint32_t nl, bool in_b);
    LeafFindNodes leaf_find_nodes(std::span<SplitInput const> nodes,
                                  std::span<uint32_t const>   slots,
                                  TreeConfig const           &config);
    void          leaf_find(Dataset const &ds, TreeConfig const &config,
                            std::span<SplitInput const> nodes, std::span<uint32_t const> slots,
                            std::span<SplitOutput> out, std::span<NodeTotals> child_sums);
    void          leaf_stamp(std::span<CudaHistogramEngine::LeafStamp const> stamps);

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
