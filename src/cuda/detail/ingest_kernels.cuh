#pragma once

// The ingest-arm device kernels, split from kernels.cuh so the
// ingest TU (histogram_engine.cu) includes only the kernels it launches: once
// the level/find/partition kernels moved to the device-context TU, a shared
// all-kernels header left each TU with unreferenced anonymous kernels, which
// -Wunused-function rejects. These stay anonymous and private to the including
// TU, like the rest of the kernels. The only cuda_detail entity they name is
// the plane's layout arithmetic (device_buffer.cuh), which every reader and
// writer of the binned matrix shares.

#include <cstdint>
#include <cuda.h>

#include "device_buffer.cuh"

namespace bonsai
{
namespace
{

// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-pointer-arithmetic)

// Device twin of BinMapper::transform: NaN -> last bin, else
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
// into the tile-blocked binned matrix. Feature varies fastest across threads,
// so raw reads coalesce and a warp's writes land inside a handful of strips;
// the pass is bounded by the raw H2D either way.
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
    out[cuda_detail::tiled_cell(f, row0 + r, n_rows, n_feats)] =
        transform_bin<BinT>(chunk[i], cuts + c0, cut_ofs[f + 1] - c0);
}

// Ingest, device-resident arm: gather whole rows of a device matrix into a
// compact row-major block, the sample the host bin mappers cut on. One thread
// per output cell; rows[] is the shared sample, ascending.
__global__ void gather_rows_kernel(float const *X, uint32_t const *rows, uint32_t cells,
                                   uint32_t n_feats, float *out)
{
    uint32_t const i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= cells)
    {
        return;
    }
    out[i] = X[(static_cast<size_t>(rows[i / n_feats]) * n_feats) + (i % n_feats)];
}

// Ingest, column arm (ColumnBatch): bin one column chunk. The column's cells
// are a strip position inside its tile, so these writes stride by the strip
// width where the row-major arm's coalesce.
template <typename BinT>
__global__ void bin_col_kernel(float const *col, uint32_t n, uint32_t row0, uint32_t f,
                               uint32_t n_rows, uint32_t n_feats, float const *cuts,
                               uint32_t n_cuts, BinT *out)
{
    uint32_t const i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    out[cuda_detail::tiled_cell(f, row0 + i, n_rows, n_feats)] =
        transform_bin<BinT>(col[i], cuts, n_cuts);
}

// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-pointer-arithmetic)

} // namespace
} // namespace bonsai
