#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/bin_store.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/row_mirror.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// Monotone identity tokens, minted once and never reused, so equality means
// "the same thing" with no allocator-reuse caveat. Strong enums rather than
// pointers: they cannot be dereferenced, cannot be built from a stray
// pointer, and zero is the never-minted sentinel.
enum class LabelsId : uint64_t
{
};
enum class FitId : uint64_t
{
};

// The binned training matrix plus everything a fit reads beside it: labels,
// weights, cuts, names, and the row view. Copying a Dataset copies pointers,
// not bins: the plane lives in a shared BinStore, which is how subset(rows=)
// views cost a row descriptor instead of a matrix. Identity is minted, never
// inferred: FitId/LabelsId tokens say when two Datasets share a plane or
// labels, and anything caching against a Dataset (device upload, resident
// labels) must key on a token, not an address. Row identity is the wire
// contract: what a row id means never changes across views (decision 103),
// so a reordered fit is a rewrite, not a remap.
class Dataset
{
  public:
    // The optional plane is a completed ingest transaction for this data:
    // when present the host fill is skipped and host columns materialize
    // lazily from the plane on first host consumer.
    static Dataset bin(detail::ColumnBatch const &batch, BinMappers const &mappers,
                       DataConfig const                  &cfg,
                       std::shared_ptr<IngestPlane const> plane = nullptr);
    // Row-major matrix path: transforms strided columns directly, no
    // column-major float materialization. Bin ids identical to the
    // ColumnBatch overload.
    static Dataset bin(features_view X, floats_view labels, BinMappers const &mappers,
                       DataConfig const                  &cfg,
                       std::shared_ptr<IngestPlane const> plane   = nullptr,
                       floats_view                        weights = {});
    // Plane-only path: the raw matrix was never host-addressable (device-
    // resident input), so there is nothing to fall back to and the plane is
    // required. Host consumers materialize from it like any other plane.
    static Dataset bin(size_t n_rows, size_t n_features, floats_view labels,
                       BinMappers const &mappers, DataConfig const &cfg,
                       std::shared_ptr<IngestPlane const> plane,
                       floats_view                        weights = {});

    // Bins that already exist, adopted as the host columns of a new dataset:
    // exactly one width is populated, the one `bins_are_u8` names. Nothing is
    // re-binned and no plane is involved, so the caller owns the pairing of
    // columns to mappers. The lazily minted caches are this dataset's own,
    // never the ones a dataset the bins were read out of is sharing.
    static Dataset from_bins(BinColumns cols, BinMappers mappers, floats_view labels,
                             floats_view weights = {});

    // The PLANE's rows: the mirror's stride, the device geometry key, and the
    // range every row id in this dataset is drawn from. A row view narrows
    // the fit's row list and leaves this alone.
    size_t n_rows() const;
    size_t n_features() const;

    // Which of the plane's rows a fit visits. Every row id is a global id
    // into this dataset's storage, so grad, hess, labels and the row-major
    // mirror stay full length whatever the view selects.
    RowView const &row_view() const
    {
        return rows_;
    }

    // The view's row count, which is what a caller means by "how many rows".
    size_t view_n_rows() const
    {
        return rows_.size();
    }

    // The same data, visited through `rows`. Nothing is copied: the plane,
    // the host columns, the row-major mirror, and the labels, weights and
    // cuts are all shared with this dataset. That is what makes k folds cost
    // one dataset rather than k, and "a view copies nothing, not even its
    // labels" pins it by pointer identity.
    Dataset with_rows(RowView rows) const;

    // The same rows under the features `keep` names, in the order it names
    // them and renumbered densely from zero. A rewrite, not a view: the kept
    // columns are gathered into a dataset that owns them, so the result
    // carries no plane and shares no cache with this one.
    //
    // Dense renumbering is what makes the rewrite worth its cost. A device
    // plane groups features into tiles of a fixed width and reads a tile
    // whole, so scattered survivors leave every tile part-live and part-dead
    // while the same count renumbered fills the first tiles completely. That
    // is worth 21-36% of the histogram fill at any keep fraction, and the
    // rewrite pays for itself within a few boosting iterations.
    //
    // Since the rewrite mints a plane either way, a row view on this dataset
    // is spent here rather than carried: the result holds the view's rows,
    // renumbered from zero, and views nothing. Selecting rows then columns
    // therefore trains the same model as columns then rows, materialized on
    // one path and viewed on the other.
    Dataset select_features(std::span<feature_id_t const> keep) const;

    // This dataset's rows, in this dataset's order, gathered into a plane it
    // owns: a view spent rather than followed. On a dataset that is not a
    // view this is a deep copy with caches of its own.
    //
    // What it buys is contiguity. A view's rows are wherever the parent put
    // them, so a scattered selection pays the gather on every histogram fill,
    // for every tree; materialized once, the same rows are a range, which the
    // fill reads as a subspan and the device reads at full coalescing. Worth
    // it when the same selection is fit repeatedly, which is what a caller
    // reordering rows into fold order is arranging for.
    Dataset materialize() const;

    floats_view       labels() const;
    floats_view       weights() const; // empty if uniform
    BinMappers const &mappers() const;
    size_t            n_bins(size_t fid) const;

    // The store: the binned matrix and its cuts, shared by every view over
    // it. Exposed because identity matters (the device cache keys uploads
    // off the store's address); the accessors below forward so the ~150
    // existing call sites keep reading through the Dataset.
    BinStore const &store() const
    {
        return *store_;
    }

    // Identity of this dataset's labels and weights: equal exactly when the
    // labels block is shared, so a view skips the device re-upload and a
    // different-labels twin can never inherit one. A minted counter, not an
    // address: a freed Meta's address comes back from the allocator, and a
    // label cache keyed on it would silently serve the previous fit's labels
    // to a same-shaped successor.
    LabelsId labels_identity() const
    {
        return meta_->id;
    }

    // Identity of this fit specification: which labels over which rows.
    // Distinct for every factory product and every view, shared by copies,
    // so "is this the dataset the resident state was armed for" is a token
    // comparison that no address reuse, stack or heap, can answer wrongly.
    FitId fit_identity() const
    {
        return id_;
    }

    std::span<float const> cuts(feature_id_t f) const
    {
        return store_->cuts(f);
    }

    bin_id_t bin_of_threshold(feature_id_t f, float threshold) const
    {
        return store_->bin_of_threshold(f, threshold);
    }

    bool bins_are_u8() const
    {
        return store_->bins_are_u8();
    }

    template <typename F> decltype(auto) visit_bins(size_t fid, F &&f) const
    {
        return store_->visit_bins(fid, std::forward<F>(f));
    }

    bin_id_t bin_at(size_t fid, size_t row) const
    {
        return store_->bin_at(fid, row);
    }

    std::shared_ptr<IngestPlane const> const &ingest_plane() const
    {
        return store_->ingest_plane();
    }

    RowMirror const &mirror() const
    {
        return store_->mirror();
    }

    // Features per mirror block, as the layout defines it.
    static constexpr size_t mirror_tile_width()
    {
        return RowMirror::tile_width;
    }

  private:
    // Labels and weights. Fixed at bin time and never mutated, so a row view
    // shares them the way it shares the store rather than deep-copying: at
    // 16M rows the labels alone are 64MB, and a fold loop that copied them
    // per fold would undo what the view is for.
    struct Meta
    {
        std::vector<float> labels;
        std::vector<float> weights;
        LabelsId           id{};
    };

    static LabelsId mint_labels_id();
    static FitId    mint_fit_id();

    std::shared_ptr<BinStore const> store_ =
        std::make_shared<BinStore const>(BinStore::Key{});
    std::shared_ptr<Meta const> meta_ = std::make_shared<Meta const>();
    RowView                     rows_ = RowView::all(0);
    FitId                       id_{};
};

// The one host routing truth: which child a row takes at an internal node.
// The last bin holds missing values and follows default_left; every other bin
// routes left iff it is at or below the split bin. The device kernels
// (goes_left_dev and route_add_kernel's walk in kernels.cuh) mirror this and
// the [cuda] parity suite enforces lockstep.
inline bool routes_left(bin_id_t bin, bin_id_t last_bin, bin_id_t split_bin,
                        bool default_left)
{
    return bin == last_bin ? default_left : bin <= split_bin;
}

} // namespace bonsai
