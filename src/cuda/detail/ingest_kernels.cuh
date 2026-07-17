#pragma once

// The ingest-arm device kernels (decision 54), split from kernels.cuh so the
// ingest TU (histogram_engine.cu) includes only the kernels it launches: once
// the level/find/partition kernels moved to the device-context TU, a shared
// all-kernels header left each TU with unreferenced anonymous kernels, which
// -Wunused-function rejects. These stay anonymous and private to the including
// TU, like the rest of the kernels; they reference no cuda_detail entity.

#include <cstdint>
#include <cuda.h>

namespace bonsai
{
namespace
{

// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-pointer-arithmetic)

// Device twin of BinMapper::transform (decision 54): NaN -> last bin, else
// the count of cuts strictly below x (std::lower_bound). Same comparisons
// over the same host-fitted cuts => bit-identical bin ids to the host fill.
template <typename BinT>
inline __device__ BinT transform_bin(float x, float const *cuts, uint32_t n_cuts)
{
    if (isnan(x))
    {
        return static_cast<BinT>(n_cuts - 1);
    }
    uint32_t lo = 0;
    uint32_t n  = n_cuts;
    while (n > 0)
    {
        uint32_t const half = n / 2;
        if (cuts[lo + half] < x)
        {
            lo += half + 1;
            n -= half + 1;
        }
        else
        {
            n = half;
        }
    }
    return static_cast<BinT>(lo);
}

// Ingest, row-major arm: bin a raw chunk (rows_in_chunk x n_feats, row-major)
// into the feature-major binned matrix. Feature varies fastest across
// threads, so raw reads coalesce; the byte-wide writes scatter at n_rows
// stride — the pass is bounded by the raw H2D either way.
template <typename BinT>
__global__ void bin_rows_kernel(float const *chunk, uint32_t rows_in_chunk,
                                uint32_t row0, uint32_t n_feats, uint32_t n_rows,
                                float const *cuts, uint32_t const *cut_ofs, BinT *out)
{
    uint32_t const i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= rows_in_chunk * n_feats)
    {
        return;
    }
    uint32_t const r  = i / n_feats;
    uint32_t const f  = i % n_feats;
    uint32_t const c0 = cut_ofs[f];
    out[(static_cast<size_t>(f) * n_rows) + row0 + r] =
        transform_bin<BinT>(chunk[i], cuts + c0, cut_ofs[f + 1] - c0);
}

// Ingest, feature-major arm (ColumnBatch): bin one column chunk in place.
template <typename BinT>
__global__ void bin_col_kernel(float const *col, uint32_t n, uint32_t row0,
                               float const *cuts, uint32_t n_cuts, BinT *out_col)
{
    uint32_t const i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    out_col[row0 + i] = transform_bin<BinT>(col[i], cuts, n_cuts);
}

// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-pointer-arithmetic)

} // namespace
} // namespace bonsai
