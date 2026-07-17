#pragma once

// Per-device state and operations for one CUDA context. Three planes divide the
// resident state by lifetime:
//   DeviceData:    the dataset-resident binned matrix, uploaded once per fit.
//   GradientPlane: the per-tree gradients, refreshed once per tree.
//   LevelPipeline: the per-level resident rows, histograms, and staging buffers.
// ProfileCounters lives in profile.cuh, CudaIngestPlane and the backend tag in
// ingest_plane.cuh; this header includes both.
//
// A real header: the types live in namespace bonsai::cuda_detail with external
// linkage and the heavy method bodies compile once in device_context.cu, so a
// second translation unit can include this without an ODR clash. Every launch
// and every cuda_runtime call stays in the .cu; this header carries only
// declarations, data members, and small member-only accessors, and names no
// entity from kernels.cuh.

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
// Flat device/host buffers throughout this file are offset by hand (docs/
// architecture/10-cuda.md); grad/hess travel as an adjacent pair everywhere
// in this API, matching the gradient-boosting literature's convention.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

// Per-device state and its operations. CudaHistogramEngine owns exactly one and
// forwards through it; the CPU fallback engine stays in the wrapper.
struct CudaDeviceContext
{
    // Identity of the uploaded dataset. The pointers are identity cookies
    // compared by address and never dereferenced; a mismatch is harmless, the
    // matrix re-uploads.
    struct DatasetKey
    {
        Dataset const *dataset                              = nullptr;
        void const    *bins0                                = nullptr;
        size_t         n_rows                               = 0;
        size_t         n_feats                              = 0;
        bool           operator==(DatasetKey const &) const = default;
    };

    // Dataset-resident plane: the binned matrix and its identity, uploaded once
    // per ensure_dataset and read by every launch.
    struct DeviceData
    {
        // One of bins8/bins16 per dataset (uint8 iff every feature fits 256
        // bins); feature-major, n_features * n_rows.
        DeviceBuffer<uint8_t>  bins8;
        DeviceBuffer<uint16_t> bins16;
        bool                   bins_are_u8 = false;
        DeviceBuffer<uint32_t> n_bins; // per-feature bin counts

        // Adopted ingest plane: when the dataset was device-binned, the
        // matrix already lives in the plane and the local buffers stay
        // empty. Accessors below pick the live storage.
        std::shared_ptr<CudaIngestPlane const> adopted;

        DatasetKey key;

        uint32_t const *n_bins_ptr() const
        {
            return adopted ? adopted->n_bins.data() : n_bins.data();
        }

        // Calls fn with the active binned-matrix pointer (uint8 when every feature
        // fits 256 bins, else uint16) — the one branch every histogram and
        // partition launch shares.
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

    // Per-tree gradient plane: raw uploads interleaved into (grad, hess)
    // pairs once per tree by populate.
    struct GradientPlane
    {
        DeviceBuffer<float>  grad_raw; // per-tree raw uploads
        DeviceBuffer<float>  hess_raw;
        DeviceBuffer<float2> gh; // interleaved (grad, hess) per row
    };

    // Per-level pipeline: resident rows, gathered gradients, and histograms
    // ping-pong between parent and child sides; the Staged<> buffers feed each
    // level's find, partition, and stamp launches. gh_ordered lives here rather
    // than in the gradient plane because it is the level-row-ordered gather and
    // ping-pongs with gh_b.
    struct LevelPipeline
    {
        DeviceBuffer<uint32_t> rows;        // concatenated node row lists
        DeviceBuffer<float2>   gh_ordered;  // gathered into level row order
        Staged<uint32_t>       row_offsets; // per batched node: offset into rows
        Staged<uint32_t>       row_counts;  // per batched node: row count
        Staged<uint32_t>       features;

        // Resident level state: ping-pong per-level histogram buffers,
        // slot-indexed [slot][sel][2 * max_sel_bins] like `out`. cur() holds the
        // frontier the next find reads; advance_level writes children into
        // other() and swaps.
        DeviceBuffer<double> level_a;
        DeviceBuffer<double> level_b;
        bool                 cur_is_a   = true;
        uint32_t             n_selected = 0;
        uint32_t             stride = 0;  // doubles per (slot, feature): 2*max_sel_bins
        Staged<uint32_t>     slots;       // hist out_slot per batched small
        Staged<uint32_t>     triples;     // (parent, small, large) per op
        Staged<double>       node_sums;   // 2 per frontier node
        Staged<double>       node_bounds; // lo, hi per frontier node
        Staged<char>         allowed;     // n_nodes * n_sel, only when constrained
        Staged<int>          monotone;    // per feature
        DeviceBuffer<FeatBest> feat_best;
        Staged<FeatBest>       node_best;
        Staged<double>         level_child; // oblivious: 4 per node [gL,hL,gR,hR]
        DeviceBuffer<double>   level_score; // oblivious: per (feature, dl, bin) scores
        Staged<uint32_t>       small_offsets; // small-node subset: offset/count/slot
        Staged<uint32_t>       small_counts;
        Staged<uint32_t>       small_slots;

        // Resident rows. rows/gh_ordered are the "a" side; children scatter into
        // the "b" side and the pair swaps with the hist buffers.
        DeviceBuffer<uint32_t> rows_b;
        DeviceBuffer<float2>   gh_b;
        DeviceBuffer<uint8_t>  flags; // per-row route flag, scatter reuse
        Staged<PartOpDev>      part_ops;
        DeviceBuffer<uint32_t> block_counts; // per (op, chunk), scanned in place
        Staged<uint32_t>       nl_dev;       // per op: total left count
        Staged<uint32_t>       stamp_ids;
        DeviceBuffer<uint32_t> leaf_by_row; // per row id: final leaf node
        // Pristine root row list for full-data fits: partitioning ping-pongs
        // the working rows buffer, so the identity permutation is cached once
        // and restored device-to-device per tree instead of re-uploaded.
        // 0 = invalid; only ever the identity, which is what every sampler
        // returns when its size equals n_rows.
        DeviceBuffer<uint32_t> root_rows;
        size_t                 root_rows_cached_n = 0;
        DeviceBuffer<double2>  sum_partial;   // root-sum pass-1 block partials
        DeviceBuffer<double2>  sum_out;       // root-sum result (1 element)
        DeviceBuffer<float>    epi_node_vals; // per-tree epilogue value table
        DeviceBuffer<float>    epi_values;    // per-row mapped values
        std::vector<uint32_t>  slot_offsets;  // current level's segment layout
        std::vector<uint32_t>  slot_counts;
        std::vector<uint32_t>  next_offsets; // children layout, live after partition
        std::vector<uint32_t>  next_counts;

        // Profile-only: event pairs bracketing the async histogram-build phases,
        // recorded at launch and read at the next profile sync so measuring
        // never serializes the pipeline. ev[0..3]: memset start, memset end /
        // hist start, hist end / subtract start, subtract end. Root builds
        // record ev[1]..ev[2] only.
        cudaEvent_t prof_ev[4]       = {};
        bool        prof_ev_ready    = false; // events created
        bool        prof_ev_recorded = false; // a build awaits reading
        bool        prof_ev_root     = false; // recorded span is a root build

        void prof_record_begin(bool root);
        // Call only after a sync that guarantees the recorded events are past.
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

        // Buckets each child op by the smaller child's size: nodes above the row
        // cutoff stage into the shared-memory kernel's (row_offsets, row_counts,
        // slots), the rest into the direct-global kernel's (small_offsets,
        // small_counts, small_slots); every op's (parent, small, large) triple
        // stages for the subtract. Returns the largest small-child row count
        // (sizes the shared kernel's chunk grid).
        size_t stage_children(std::span<CudaHistogramEngine::LevelOp const> ops);

        // Fills the child slots' (offset, count) layout from the device left-counts
        // and echoes (n_left, n_right) per op back to the caller.
        void layout_children(std::span<CudaHistogramEngine::PartitionOp const> ops,
                             std::span<uint32_t> child_counts);

        // Stages the per-node totals, monotone-bound box, optional allowed-feature
        // mask, and per-feature monotone directions the find kernel reads. Returns
        // whether any node carried a mask (the kernel gets nullptr otherwise).
        bool stage_find_inputs(std::span<SplitInput const> level,
                               TreeConfig const &config, Dataset const &ds);

        // Translates each node's device-side best split into a host SplitOutput and
        // its (left, right) child sums; a node with no valid split or too few rows
        // to split emits an empty output.
        void unpack_splits(std::span<SplitInput const> level, TreeConfig const &config,
                           std::span<SplitOutput> out, std::span<HistCell> child_sums);

        // Oblivious level-find staging: node_sums only. The level kernel reads no
        // bounds/monotone/interaction state (the oblivious grower rejects those
        // constraints at construction), so the full stage_find_inputs is waste.
        void stage_level_sums(std::span<SplitInput const> level);
    };

    // Device-resident objective plane: labels and the per-row score vector live
    // here for the whole fit. begin_tree derives gh from them; the resident
    // finalize walks the finished tree (SoA node arrays) and fuses the score
    // update. Labels are keyed by dataset identity so a re-fit skips re-upload.
    struct ResidentPlane
    {
        DeviceBuffer<float> labels;
        DeviceBuffer<float> scores;
        DatasetKey          labels_key;
        Staged<uint32_t>    node_feature;
        Staged<uint32_t>    node_split_bin;
        Staged<uint32_t>    node_left;
        Staged<uint32_t>    node_right;
        Staged<uint32_t>    node_default_left;
        Staged<uint32_t>    node_is_leaf;
        Staged<float>       node_value;
        bool                armed         = false;
        float               learning_rate = 0.0F;
        size_t              n_rows        = 0;
    };

    DeviceData      data;
    GradientPlane   grads;
    LevelPipeline   lvl;
    ResidentPlane   resident;
    ProfileCounters prof_counters;

    // Runtime shared-memory ceiling for the hist kernels: the opt-in limit
    // when the device grants one (both BinT instantiations opted in), else
    // the 48 KiB static budget. Resolved lazily on first use so engine
    // construction never touches the CUDA runtime.
    size_t shared_limit  = k_max_shared_bytes;
    bool   shared_probed = false;

    void init_shared_limit();
    void ensure_dataset(Dataset const &dataset);
    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    bool begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                    SplitInput &root, std::span<feature_id_t const> selected);
    void stamp_leaves(std::span<CudaHistogramEngine::LeafStamp const> stamps);
    void partition_level(Dataset const                                    &ds,
                         std::span<CudaHistogramEngine::PartitionOp const> ops,
                         std::span<uint32_t> child_counts);
    void finalize_rows(std::span<node_id_t> leaf_by_row);
    void finalize_tree(std::span<float const> node_values, std::span<float> values,
                       std::span<node_id_t> leaf_ids);
    void advance_level(Dataset const                                &ds,
                       std::span<CudaHistogramEngine::LevelOp const> ops);
    // Final-level advance: the children of the last level are leaves, so their
    // histograms are never read by any find; only the layout flip survives so
    // stamping sees the final segments.
    void advance_layout_only();
    void find_splits_many(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums);
    void find_level_split(Dataset const &ds, TreeConfig const &config,
                          std::span<SplitInput const> level, std::span<SplitOutput> out,
                          std::span<HistCell> child_sums);

    bool resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                        std::span<float const> initial_scores, float learning_rate);
    bool resident_armed() const
    {
        return resident.armed;
    }
    void resident_finalize(std::span<CudaHistogramEngine::ResidentNode const> nodes);
    void resident_end(std::span<float> scores_out);
};

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace cuda_detail
} // namespace bonsai
