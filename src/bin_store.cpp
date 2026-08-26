#include "bonsai/bin_store.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/row_mirror.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

BinStore::BinStore(Key /*key*/, size_t n_rows, BinMappers mappers, BinColumns cols)
    : cols_(std::make_shared<BinColumns>(std::move(cols))),
      row_major_(std::make_shared<RowMirror>(n_rows, mappers.size())),
      mappers_(std::move(mappers)), n_rows_(n_rows), n_features_(mappers_.size())
{
}

BinStore::BinStore(Key /*key*/, size_t n_rows, BinMappers mappers,
                   std::shared_ptr<IngestPlane const> plane)
    : row_major_(std::make_shared<RowMirror>(n_rows, mappers.size())),
      plane_(std::move(plane)), lazy_(std::make_shared<LazyColumns>()),
      mappers_(std::move(mappers)), n_rows_(n_rows), n_features_(mappers_.size())
{
    if (!mappers_.all_fit_u8())
    {
        lazy_->cols = U16Columns{};
    }
}

std::shared_ptr<BinStore const>
BinStore::select_columns(std::span<feature_id_t const> keep,
                         std::span<row_id_t const>     rows) const
{
    if (keep.empty())
    {
        throw std::invalid_argument("BinStore::select_columns: no features kept; a "
                                    "store with no columns has nothing to serve");
    }
    std::vector<BinMapper>   kept;
    std::vector<std::string> names;
    kept.reserve(keep.size());
    names.reserve(keep.size());
    auto const parent_names = mappers_.feature_names();
    bool const named        = parent_names.size() == n_features_;
    for (feature_id_t const f : keep)
    {
        if (f >= n_features_)
        {
            throw std::invalid_argument("BinStore::select_columns: feature " +
                                        std::to_string(f) +
                                        " is past the last of this store's " +
                                        std::to_string(n_features_) + " features");
        }
        kept.push_back(mappers_[f]);
        if (named)
        {
            names.emplace_back(parent_names[f]);
        }
    }
    BinMappers kept_mappers =
        BinMappers::from_mappers(std::move(kept), std::move(names));
    size_t const out_rows = rows.empty() ? n_rows_ : rows.size();

    if (plane_)
    {
        if (auto sub = plane_->select_columns(keep, rows))
        {
            return std::make_shared<BinStore const>(
                Key{}, out_rows, std::move(kept_mappers), std::move(sub));
        }
    }

    BinColumns out = std::visit(
        [&](auto const &src) -> BinColumns
        {
            std::remove_cvref_t<decltype(src)> gathered(keep.size());
            for (size_t k = 0; k < keep.size(); ++k)
            {
                auto const &col = src[keep[k]];
                auto       &dst = gathered[k];
                dst.resize(out_rows);
                for (size_t i = 0; i < out_rows; ++i)
                {
                    dst[i] = col[rows.empty() ? i : rows[i]];
                }
            }
            return gathered;
        },
        columns());
    return std::make_shared<BinStore const>(Key{}, out_rows, std::move(kept_mappers),
                                            std::move(out));
}

RowMirror const &BinStore::mirror() const
{
    if (!bins_are_u8())
    {
        return *row_major_;
    }
    row_major_->mint_once([this](std::span<uint8_t> out_bins) { mint_into(out_bins); });
    return *row_major_;
}

void BinStore::mint_into(std::span<uint8_t> out_bins) const
{
    auto const  &cols = std::get<U8Columns>(columns());
    size_t const f    = cols.size();
    assert(f == n_features_);
    assert(out_bins.size() == n_rows_ * f);
    // sync: each worker owns a row block in the parallel::for_each_index
    // below, so writes never overlap; overlapping them races and breaks
    // the byte-identity model_hash pins.
    size_t const     width = RowMirror::tile_width;
    uint8_t         *out   = out_bins.data();
    constexpr size_t tile  = 64;
    parallel::for_each_index((n_rows_ + tile - 1) / tile,
                             [&](size_t block)
                             {
                                 size_t const r0 = block * tile;
                                 size_t const r1 = std::min(r0 + tile, n_rows_);
                                 for (size_t c0 = 0; c0 < f; c0 += tile)
                                 {
                                     size_t const c1 = std::min(c0 + tile, f);
                                     for (size_t c = c0; c < c1; ++c)
                                     {
                                         size_t const mb = c / width;
                                         size_t const width_b =
                                             std::min(width, f - (mb * width));
                                         size_t const   base = n_rows_ * mb * width;
                                         size_t const   in_b = c - (mb * width);
                                         uint8_t       *dst  = out + base + in_b;
                                         uint8_t const *col  = cols[c].data();
                                         for (size_t r = r0; r < r1; ++r)
                                         {
                                             dst[r * width_b] = col[r];
                                         }
                                     }
                                 }
                             });
}

} // namespace bonsai
