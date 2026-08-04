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

    // Per-tree leaf pipeline (docs/architecture/20-cuda-leafwise.md): best-first
    // growth expands one leaf at a time, so histograms live in a slot pool
    // zeroed once per tree instead of the level plane's ping-pong. The root
    // takes slot 0, every split builds the smaller child into the next free
    // slot and derives the larger in place in the parent's, which it inherits.
    // Rows, gradients, flags, the leaf assignment array, and the per-tree knobs
    // (features, n_selected, stride) are the level pipeline's, reused as they
    // are; the pool and its segment map are all this plane adds.
    struct LeafPipeline
    {
        DeviceBuffer<double>  pool;         // max_slots * n_selected * stride
        std::vector<uint32_t> slot_offsets; // per slot: segment start in rows
        std::vector<uint32_t> slot_counts;
        // Per-round staging. Pinned and asynchronous because the round's whole
        // host residue is these uploads: a pageable copy stream-syncs before it
        // starts, so eight of them per round drain the pipeline eight times.
        // Each buffer is written once per round, downstream of the blocking
        // fetch that fenced the previous round's upload of it.
        PinnedStaged<PartOpDev> part_op;    // the popped leaf's partition op
        PinnedStaged<uint32_t>  build_seg;  // hist inputs: offset, count, slot
        PinnedStaged<double>    find_stats; // per node: 2 sums, then 2 bounds
        PinnedStaged<uint32_t>  find_slots; // find's slot indirection table
        // Per-feature monotone directions, fit-constant and uploaded once per
        // tree: the level plane restages them per level, which on a plane that
        // finds twice per split would be one upload per round.
        Staged<int> monotone;
        uint32_t    next_slot = 0;
        uint32_t    max_slots = 0;
    };

    // Device-resident objective plane: labels and the per-row score vector live
    // here for the whole fit. begin_tree derives gh from them by the kind's
    // gradient kernel; the resident finalize walks the finished tree (SoA node
    // arrays) and fuses the score update. Labels (and weights, when the dataset
    // carries them) are keyed by dataset identity so a re-fit skips re-upload.
    struct ResidentPlane
    {
        DeviceBuffer<float> labels;
        DeviceBuffer<float> scores;
        DeviceBuffer<float> weights; // uploaded only when the dataset is weighted
        DatasetKey          labels_key;
        Staged<uint32_t>    node_feature;
        Staged<uint32_t>    node_split_bin;
        Staged<uint32_t>    node_left;
        Staged<uint32_t>    node_right;
        Staged<uint32_t>    node_default_left;
        Staged<uint32_t>    node_is_leaf;
        Staged<float>       node_value;
        DeviceObjectiveKind kind          = DeviceObjectiveKind::none;
        bool                weighted      = false;
        bool                armed         = false;
        float               learning_rate = 0.0F;
        size_t              n_rows        = 0;
    };

    DeviceData      data;
    GradientPlane   grads;
    LevelPipeline   lvl;
    LeafPipeline    leaf;
    ResidentPlane   resident;
    ProfileCounters prof_counters;

    // Runtime shared-memory ceiling for the hist kernels: the opt-in limit
    // when the device grants one (both BinT instantiations opted in), else
    // the 48 KiB static budget. Resolved lazily on first use so engine
    // construction never touches the CUDA runtime.
    size_t shared_limit  = k_max_shared_bytes;
    bool   shared_probed = false;

    // Features per histogram block (research probe): 0 or 1 is the
    // one-feature kernel, G > 1 the grouped one whenever G histograms fit
    // hist_group_limit. The grouped kernel carries static shared arrays
    // beside its dynamic histograms, so it opts in for less than the device
    // maximum and its ceiling is its own, not shared_limit.
    uint32_t hist_group       = hist_group_env();
    size_t   hist_group_limit = k_max_shared_bytes;

    // The one histogram-capacity predicate: a node's per-feature scratch is
    // 4 * bins floats in shared memory. begin_root declines a tree with it,
    // and resident_begin must apply the SAME test (once per fit, on the
    // worst-case feature) so no tree can decline into a host-gradient path
    // after the resident mode armed. Any new decline condition must land
    // here, visible to both callers.
    bool hist_budget_ok(size_t max_bins) const
    {
        return 4 * max_bins * sizeof(float) <= shared_limit;
    }

    // The leaf plane's second gate, applied beside hist_budget_ok in
    // leaf_budget_ok: the histogram pool is that plane's only new allocation
    // and an oversized leaf budget must decline to the host plane rather than
    // fail the fit. A quarter of the device's free memory is the bound; the
    // buffer is grow-only and doubles on reallocation.
    bool leaf_pool_ok(size_t bytes) const;

    // The leaf plane's whole decline predicate, both gates in one place:
    // leaf_begin_root applies it per tree over the selected features, and
    // resident_begin_leaf once per fit over every feature and the full leaf
    // budget. A tree's selection is a subset of every feature, so passing the
    // conservative form guarantees the per-tree form; the pool's free-memory
    // bound additionally cannot be retested per tree once armed, since the
    // pool the first tree allocates shrinks the free memory the later trees
    // would measure. Any new leaf-plane decline condition lands here.
    bool leaf_budget_ok(TreeConfig const &config, size_t n_selected,
                        size_t max_bins) const;

    void init_shared_limit();
    void ensure_dataset(Dataset const &dataset);
    // Fills lvl.rows with the tree's root row segment and returns its length: a
    // full-data fit restores the cached identity permutation device-to-device
    // (built once by iota_kernel), any other row list uploads. Shared by the
    // level and leaf planes' root opens.
    uint32_t stage_root_rows(SplitInput const &root, bool identity);
    void     begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    bool     begin_root(Dataset const &ds, floats_view grad, floats_view hess,
                        SplitInput &root, std::span<feature_id_t const> selected);
    void     stamp_leaves(std::span<CudaHistogramEngine::LeafStamp const> stamps);
    void     partition_level(Dataset const                                    &ds,
                             std::span<CudaHistogramEngine::PartitionOp const> ops,
                             std::span<uint32_t> child_counts);
    void     finalize_rows(std::span<node_id_t> leaf_by_row);
    void     finalize_tree(std::span<float const> node_values, std::span<float> values,
                           std::span<node_id_t> leaf_ids);
    void     advance_level(Dataset const                                &ds,
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

    // --- Leaf plane (docs/architecture/20-cuda-leafwise.md) -------------------
    // Opens the tree's slot pool, seeds the root segment, and builds slot 0.
    // Returns false — leaving root untouched — when the histogram budget or
    // the pool budget declines, and the caller trains on the host plane.
    bool leaf_begin_root(Dataset const &ds, TreeConfig const &config, floats_view grad,
                         floats_view hess, SplitInput &root,
                         std::span<feature_id_t const> selected);
    // Routes one leaf's row segment into two adjacent subranges in place and
    // assigns the smaller child the next free slot (the larger inherits the
    // parent's). A partition that empties one side takes no slot: the caller
    // demotes the split back to a leaf.
    CudaHistogramEngine::LeafRound
    leaf_split(Dataset const &ds, CudaHistogramEngine::LeafPartOp const &op);
    // Builds the smaller child into its fresh slot and derives the larger by
    // in-place subtraction in the slot it inherited. small_slot's segment
    // (offset, count) is read from leaf.slot_offsets/slot_counts.
    void leaf_build(Dataset const &ds, uint32_t small_slot, uint32_t large_slot);
    // The leaf plane's own find staging: the level plane's stage_find_inputs
    // in packed pinned form, minus the monotone vector the tree open already
    // uploaded. Returns whether any node carried an allowed-feature mask.
    bool leaf_stage_find(std::span<SplitInput const> nodes,
                         std::span<uint32_t const>   slots);
    // Best split per named pool slot; child_sums receives the winning cut's
    // (left, right) totals, 2 cells per node, as find_splits_many does.
    void leaf_find(Dataset const &ds, TreeConfig const &config,
                   std::span<SplitInput const> nodes, std::span<uint32_t const> slots,
                   std::span<SplitOutput> out, std::span<HistCell> child_sums);
    // Records final leaf assignment for the named slots' segments.
    void leaf_stamp(std::span<CudaHistogramEngine::LeafStamp const> stamps);

    bool resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                        std::span<float const> initial_scores, float learning_rate);
    // Arms for best-first growth: the leaf plane's conservative capacity test
    // first, then the level plane's arming unchanged.
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
};

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,bugprone-easily-swappable-parameters)

} // namespace cuda_detail
} // namespace bonsai
