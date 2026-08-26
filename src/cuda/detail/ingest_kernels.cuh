#pragma once

#include <cstdint>
#include <cuda.h>

#include "device_buffer.cuh"

namespace bonsai
{
namespace
{

// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-pointer-arithmetic)

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
