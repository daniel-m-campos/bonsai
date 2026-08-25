#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <variant>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/row_mirror.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// The binned columns, at their one width. Which alternative is held IS the
// width: 8-bit when every feature fits 256 bins, 16-bit otherwise, and no
// flag exists to disagree with the storage. A plane-backed store holds an
// empty vector of the right alternative until the columns materialize, so
// the width is readable without forcing the materialization.
using U8Columns  = std::vector<std::vector<uint8_t>>;
using U16Columns = std::vector<std::vector<uint16_t>>;
using BinColumns = std::variant<U8Columns, U16Columns>;

// Product of a backend's ingest transaction (decision 54, doc 15): the
// binned columns live wherever the backend put them — Dataset carries the
// plane as an opaque receipt and asks it to materialize host columns only
// when a host consumer needs them. Host-pure: concrete planes are defined
// by their backend (the CUDA TU); this header never names device types.
//
// Deliberate dynamic dispatch, the IBooster precedent: this TU boundary
// makes compile-time dispatch impossible by construction, and the cost is
// two indirect calls per fit. The algorithmic narrative stays static.
class IngestPlane
{
  public:
    // backend_tag identifies the minting backend by address (a TU-local
    // anchor only that backend knows), so an engine can recognize and
    // adopt its own plane by pointer equality instead of RTTI.
    explicit IngestPlane(void const *backend_tag = nullptr) : backend_tag_(backend_tag)
    {
    }
    IngestPlane(IngestPlane const &)            = default;
    IngestPlane &operator=(IngestPlane const &) = default;
    IngestPlane(IngestPlane &&)                 = default;
    IngestPlane &operator=(IngestPlane &&)      = default;
    virtual ~IngestPlane()                      = default;

    // One-time host materialization: fill the HELD alternative with the
    // plane's binned columns, feature-major, byte-identical to the host fill
    // over the same cuts. The caller constructed `cols` at the plane's own
    // width; an implementation must fill in place and never reassign the
    // alternative, which is what lets the width be read concurrently with
    // the materialization.
    virtual void materialize(BinColumns &cols) const = 0;

    // A plane holding the rows `rows` names under the features `keep` names,
    // renumbered densely from zero, gathered inside this backend's memory. An
    // empty `rows` means every row in order. Returns null when the backend has
    // no such gather, and the caller falls back to materializing on the host
    // and gathering there.
    //
    // This exists because the fallback is the expensive one: a column rewrite
    // of a device-resident dataset otherwise pulls the whole plane home and
    // ships the survivors back, once per round of a feature-selection loop.
    virtual std::shared_ptr<IngestPlane const>
    select_columns(std::span<feature_id_t const> /*keep*/,
                   std::span<row_id_t const> /*rows*/) const
    {
        return nullptr;
    }

    void const *backend_tag() const
    {
        return backend_tag_;
    }

  private:
    void const *backend_tag_ = nullptr;
};

// The binned matrix wherever it lives, plus the cuts that make it readable.
//
// This is the sharing unit: one store per binned matrix, held by shared_ptr
// from every Dataset over it, so a row view costs a pointer and the device
// cache can key uploads off an address that lives as long as the bins do.
// A Dataset is a fit specification over a store: which rows, which labels,
// which weights. The store neither knows nor cares which fit reads it.
//
// The cuts live here and not with the labels because bins are unreadable
// without them: a plane plus its mappers is the binned matrix, while labels
// are only ever needed by a fit. Everything inside is fixed once built; the
// two lazy caches (host columns of a plane, the row-major mirror) mutate
// under call_once through shared state, so a const store stays shareable
// across threads.
class BinStore
{
  public:
    size_t n_rows() const
    {
        return n_rows_;
    }

    size_t n_features() const
    {
        return n_features_;
    }

    BinMappers const &mappers() const
    {
        return mappers_;
    }

    size_t n_bins(size_t fid) const
    {
        return mappers_[fid].n_bins();
    }

    // Feature f's strictly increasing bin cut points.
    std::span<float const> cuts(feature_id_t f) const
    {
        return mappers_[f].cuts();
    }

    // Feature f's threshold inversion, on the mapper that owns the cuts.
    bin_id_t bin_of_threshold(feature_id_t f, float threshold) const
    {
        return mappers_[f].bin_of_threshold(threshold);
    }

    // 8-bit halves the memory traffic of the histogram fill, the dominant
    // fit stage. The answer is the variant's own alternative, readable
    // without materializing a plane-backed store's columns.
    bool bins_are_u8() const
    {
        return std::holds_alternative<U8Columns>(declared_columns());
    }

    // Readers dispatch once per column; the callable is monomorphized per
    // width, so the per-row loop never branches.
    template <typename F> decltype(auto) visit_bins(size_t fid, F &&f) const
    {
        return std::visit([&](auto const &cols) -> decltype(auto)
                          { return f(std::span{std::as_const(cols[fid])}); },
                          columns());
    }

    // Single-element read for tree-routing loops (feature varies per step, so
    // a per-column visitor buys nothing there); the alternative check
    // predicts as perfectly as the flag it replaced.
    bin_id_t bin_at(size_t fid, size_t row) const
    {
        return std::visit([&](auto const &cols) -> bin_id_t { return cols[fid][row]; },
                          columns());
    }

    // The completed ingest transaction, if any; backends recognize and adopt
    // their own plane instead of re-uploading host columns.
    std::shared_ptr<IngestPlane const> const &ingest_plane() const
    {
        return plane_;
    }

    // The row-major mirror of these bins, minted on first use and shared by
    // every Dataset over this store. Empty when the bins are not u8: the
    // mirror exists for the byte-wide row walk and a u16 store has no such
    // layout.
    RowMirror const &mirror() const;

    // A store holding the rows `rows` names (empty means every row in
    // order) under the features `keep` names, in the order it names them
    // and renumbered densely from zero. This is the operation a store IS
    // FOR a matrix: copy-with-columns. The backend's own gather runs first
    // when a plane is present; a backend without one declines and the host
    // gather produces the identical store, only slower.
    std::shared_ptr<BinStore const>
    select_columns(std::span<feature_id_t const> keep,
                   std::span<row_id_t const>     rows) const;

    // Passkey: constructible only by Dataset and the store itself, so
    // make_shared builds stores in place while "only Dataset builds stores"
    // stays compiler-checked. The constructors are where the plane/lazy
    // pairing and the one-width rule live, so no factory can get them wrong
    // independently.
    class Key
    {
        Key() = default;
        friend class Dataset;
        friend class BinStore;
    };

    explicit BinStore(Key) {}

    // Host-columns store: the width is whichever alternative `cols` holds,
    // which is how from_bins lets the caller own the pairing of columns to
    // mappers.
    BinStore(Key, size_t n_rows, BinMappers mappers, BinColumns cols);

    // Plane-backed store: the lazy host cache is minted here and the width
    // follows the mappers, so `plane_ != nullptr` implies `lazy_ != nullptr`
    // by construction and no columns are allocated that will never be read.
    BinStore(Key, size_t n_rows, BinMappers mappers,
             std::shared_ptr<IngestPlane const> plane);

  private:
    // Lazily materialized host columns of a plane-backed store, held at the
    // right (empty) alternative from construction. The first host consumer
    // can be a parallel loop (route_unsampled walks bin_at from every
    // worker), so materialization synchronizes via call_once, the same
    // reason RowMirror does.
    struct LazyColumns
    {
        BinColumns     cols;
        std::once_flag once;
    };

    // Fills the mirror's buffer from the host columns; the mirror decides
    // the layout and calls this at most once.
    void mint_into(std::span<uint8_t> out_bins) const;

    // The live columns: materialized on first touch for a plane-backed
    // store, the host columns otherwise.
    BinColumns const &columns() const
    {
        if (!plane_)
        {
            return *cols_;
        }
        std::call_once(lazy_->once, [this] { plane_->materialize(lazy_->cols); });
        return lazy_->cols;
    }

    // The columns WITHOUT forcing materialization: right alternative, maybe
    // empty. What width questions read.
    BinColumns const &declared_columns() const
    {
        return plane_ ? lazy_->cols : *cols_;
    }

    // The inner pointers stay non-const pointees so the two lazy caches can
    // mint under a const store; sharing one materialization across every
    // holder is the point of the heap allocation.
    std::shared_ptr<BinColumns>        cols_      = std::make_shared<BinColumns>();
    std::shared_ptr<RowMirror>         row_major_ = std::make_shared<RowMirror>();
    std::shared_ptr<IngestPlane const> plane_;
    std::shared_ptr<LazyColumns>       lazy_;
    BinMappers                         mappers_;
    size_t                             n_rows_     = 0;
    size_t                             n_features_ = 0;
};

} // namespace bonsai
