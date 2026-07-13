#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

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

    // One-time host materialization: fill exactly one of u8/u16 with the
    // plane's binned columns, feature-major, byte-identical to the host
    // fill over the same cuts.
    virtual void materialize(std::vector<std::vector<uint8_t>>  &u8,
                             std::vector<std::vector<uint16_t>> &u16) const = 0;

    void const *backend_tag() const
    {
        return backend_tag_;
    }

  private:
    void const *backend_tag_ = nullptr;
};

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

    size_t n_rows() const;
    size_t n_features() const;

    floats_view       labels() const;
    floats_view       weights() const; // empty if uniform
    BinMappers const &mappers() const;
    size_t            n_bins(size_t fid) const;
    bool              is_categorical(size_t fid) const;

    // Binned columns store 8-bit when every feature fits 256 bins (the
    // max_bin=255 default) — halving the memory traffic of the histogram
    // fill, the dominant fit stage — and 16-bit otherwise. Readers dispatch
    // once per column via visit_bins; the callable is monomorphized per
    // width, so the per-row loop never branches.
    bool bins_are_u8() const
    {
        return bins_are_u8_;
    }

    template <typename F> decltype(auto) visit_bins(size_t fid, F &&f) const
    {
        if (plane_)
        {
            auto const &hb = host_bins();
            if (bins_are_u8_)
            {
                return f(std::span<uint8_t const>{hb.u8[fid]});
            }
            return f(std::span<uint16_t const>{hb.u16[fid]});
        }
        if (bins_are_u8_)
        {
            return f(std::span<uint8_t const>{features_u8_[fid]});
        }
        return f(std::span<uint16_t const>{features_u16_[fid]});
    }

    // Single-element read for tree-routing loops (feature varies per step, so
    // a per-column visitor buys nothing there); the branch predicts perfectly.
    bin_id_t bin_at(size_t fid, size_t row) const
    {
        if (plane_)
        {
            auto const &hb = host_bins();
            return bins_are_u8_ ? hb.u8[fid][row] : hb.u16[fid][row];
        }
        return bins_are_u8_ ? features_u8_[fid][row] : features_u16_[fid][row];
    }

    // The completed ingest transaction, if any; backends recognize and adopt
    // their own plane instead of re-uploading host columns.
    std::shared_ptr<IngestPlane const> const &ingest_plane() const
    {
        return plane_;
    }

    // Row-major mirror of the u8 columns (n_rows x n_features), built on
    // first use by the row-wise histogram fill; empty when bins are u16
    // (that fill stays feature-parallel). Lazy so CUDA and predict-only
    // workflows never pay the +n_rows*n_features bytes. Safe unguarded:
    // boosters grow one tree at a time, so first use is single-threaded.
    std::span<uint8_t const> row_major_bins() const;

  private:
    // Lazily materialized host columns of a plane-backed dataset. The first
    // host consumer can be a parallel loop (route_unsampled walks bin_at
    // from every worker), so materialization synchronizes via call_once —
    // the row_major_ single-threaded-first-use assumption does NOT hold
    // here. Heap-allocated so Dataset copies share one materialization.
    struct HostBins
    {
        std::vector<std::vector<uint8_t>>  u8;
        std::vector<std::vector<uint16_t>> u16;
        std::once_flag                     once;
    };

    HostBins const &host_bins() const
    {
        std::call_once(lazy_->once,
                       [this] { plane_->materialize(lazy_->u8, lazy_->u16); });
        return *lazy_;
    }

    std::vector<std::vector<uint8_t>>             features_u8_;
    std::vector<std::vector<uint16_t>>            features_u16_;
    mutable std::shared_ptr<std::vector<uint8_t>> row_major_;
    std::shared_ptr<IngestPlane const>            plane_;
    std::shared_ptr<HostBins>                     lazy_;
    bool                                          bins_are_u8_ = false;
    std::vector<float>                            labels_;
    std::vector<float>                            weights_;
    BinMappers                                    mappers_;
    std::vector<bool>                             is_categorical_;
    size_t                                        n_rows_     = 0;
    size_t                                        n_features_ = 0;
};

} // namespace bonsai
