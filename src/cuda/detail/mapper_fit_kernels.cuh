#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace bonsai
{
namespace cuda_detail
{

constexpr uint32_t k_nan_key            = 0xFFFFFFFFu;
constexpr uint32_t k_transpose_tile     = 32;
constexpr uint32_t k_transpose_row_step = 8;
constexpr uint32_t k_sort_threads       = 256;
constexpr uint32_t k_sort_warps         = k_sort_threads / 32;
constexpr uint32_t k_radix_digits       = 256;
constexpr uint32_t k_radix_passes       = 4;
constexpr uint32_t k_cuts_warps         = 8;
constexpr uint32_t k_cuts_threads       = k_cuts_warps * 32;

inline __device__ uint32_t sortable_key(float f)
{
    uint32_t const b = __float_as_uint(f);
    return b ^ ((b >> 31) != 0u ? 0xFFFFFFFFu : 0x80000000u);
}

inline __device__ float key_to_float(uint32_t k)
{
    return __uint_as_float((k >> 31) != 0u ? k ^ 0x80000000u : ~k);
}

inline __device__ uint32_t digit_of(uint32_t key, uint32_t pass)
{
    return (key >> (8 * pass)) & 0xFFu;
}

inline __device__ uint32_t lane_id()
{
    return threadIdx.x & 31u;
}

struct SampleMatrix
{
    float const    *X;
    uint32_t        n_feats;
    uint32_t const *rows;
    uint32_t        m;

    __device__ uint32_t key_at(uint32_t r, uint32_t c) const
    {
        uint32_t const src_row = rows != nullptr ? rows[r] : r;
        float const    v       = X[static_cast<size_t>(src_row) * n_feats + c];
        return isnan(v) ? k_nan_key : sortable_key(v);
    }
};

inline __device__ void store_column_tile(uint32_t const *tile_column, uint32_t c,
                                         uint32_t r, uint32_t m, bool in,
                                         uint32_t *__restrict__ keys,
                                         uint32_t *__restrict__ nan_counts)
{
    uint32_t const key  = in ? *tile_column : 0u;
    uint32_t const nans = __ballot_sync(0xFFFFFFFFu, in && key == k_nan_key);
    if (in)
    {
        keys[static_cast<size_t>(c) * m + r] = key;
    }
    if (lane_id() == 0 && nans != 0u)
    {
        atomicAdd(&nan_counts[c], static_cast<uint32_t>(__popc(nans)));
    }
}

__global__ void gather_keys_kernel(SampleMatrix X, uint32_t col0, uint32_t w,
                                   uint32_t *__restrict__ keys,
                                   uint32_t *__restrict__ nan_counts)
{
    __shared__ uint32_t tile[k_transpose_tile][k_transpose_tile + 1];
    uint32_t const      r0 = blockIdx.x * k_transpose_tile;
    uint32_t const      c0 = blockIdx.y * k_transpose_tile;
    uint32_t const      x  = threadIdx.x;
    for (uint32_t j = threadIdx.y; j < k_transpose_tile; j += k_transpose_row_step)
    {
        uint32_t const r = r0 + j;
        uint32_t const c = c0 + x;
        tile[j][x]       = r < X.m && c < w ? X.key_at(r, col0 + c) : k_nan_key;
    }
    __syncthreads();
    for (uint32_t j = threadIdx.y; j < k_transpose_tile; j += k_transpose_row_step)
    {
        uint32_t const c = c0 + j;
        uint32_t const r = r0 + x;
        store_column_tile(&tile[x][j], c, r, X.m, r < X.m && c < w, keys, nan_counts);
    }
}

inline __device__ void exclusive_scan_digits(uint32_t const *hist, uint32_t *running)
{
    constexpr uint32_t per_lane = k_radix_digits / 32;
    uint32_t const     lane     = lane_id();
    uint32_t           s[per_lane];
    uint32_t           sum = 0;
    for (uint32_t k = 0; k < per_lane; ++k)
    {
        s[k] = hist[lane * per_lane + k];
        sum += s[k];
    }
    uint32_t incl = sum;
    for (uint32_t off = 1; off < 32; off <<= 1)
    {
        uint32_t const t = __shfl_up_sync(0xFFFFFFFFu, incl, off);
        if (lane >= off)
        {
            incl += t;
        }
    }
    uint32_t run = incl - sum;
    for (uint32_t k = 0; k < per_lane; ++k)
    {
        running[lane * per_lane + k] = run;
        run += s[k];
    }
}

using WarpCounts = uint16_t[k_sort_warps][k_radix_digits];

inline __device__ bool digit_is_uniform(uint32_t const *src, uint32_t m, uint32_t pass,
                                        uint32_t *hist)
{
    uint32_t const tid = threadIdx.x;
    hist[tid]          = 0;
    __syncthreads();
    for (uint32_t i = tid; i < m; i += k_sort_threads)
    {
        atomicAdd(&hist[digit_of(src[i], pass)], 1u);
    }
    __syncthreads();
    bool const uniform = hist[digit_of(src[0], pass)] == m;
    __syncthreads();
    return uniform;
}

inline __device__ void scatter_tile(uint32_t const *src, uint32_t *dst, uint32_t m,
                                    uint32_t pass, uint32_t base, WarpCounts &cnt,
                                    WarpCounts &nxt, uint32_t *running)
{
    uint32_t const tid  = threadIdx.x;
    uint32_t const lane = lane_id();
    uint32_t const warp = tid >> 5;
    uint32_t const i    = base + tid;
    bool const     in   = i < m;
    uint32_t const key  = in ? src[i] : 0u;
    uint32_t const d    = digit_of(key, pass);
    for (uint32_t w = 0; w < k_sort_warps; ++w)
    {
        nxt[w][tid] = 0;
    }
    uint32_t const peers = __match_any_sync(0xFFFFFFFFu, in ? d : k_radix_digits);
    uint32_t const rank  = __popc(peers & ((1u << lane) - 1u));
    if (in && rank == 0)
    {
        cnt[warp][d] = static_cast<uint16_t>(__popc(peers));
    }
    __syncthreads();
    uint32_t total = 0;
    for (uint32_t w = 0; w < k_sort_warps; ++w)
    {
        total += cnt[w][tid];
    }
    if (in)
    {
        uint32_t off = running[d] + rank;
        for (uint32_t w = 0; w < warp; ++w)
        {
            off += cnt[w][d];
        }
        dst[off] = key;
    }
    __syncthreads();
    running[tid] += total;
}

__global__ void __launch_bounds__(k_sort_threads)
    radix_sort_columns_kernel(uint32_t *__restrict__ keys, uint32_t *__restrict__ swap,
                              uint32_t m, uint8_t *__restrict__ parity)
{
    __shared__ uint32_t   hist[k_radix_digits];
    __shared__ uint32_t   running[k_radix_digits];
    __shared__ WarpCounts warp_cnt[2];
    uint32_t const        tid   = threadIdx.x;
    uint32_t             *src   = keys + static_cast<size_t>(blockIdx.x) * m;
    uint32_t             *dst   = swap + static_cast<size_t>(blockIdx.x) * m;
    uint32_t              swaps = 0;
    uint32_t              tile  = 0;
    for (uint32_t w = 0; w < k_sort_warps; ++w)
    {
        warp_cnt[0][w][tid] = 0;
        warp_cnt[1][w][tid] = 0;
    }
    for (uint32_t pass = 0; pass < k_radix_passes && m != 0; ++pass)
    {
        if (digit_is_uniform(src, m, pass, hist))
        {
            continue;
        }
        if (tid < 32)
        {
            exclusive_scan_digits(hist, running);
        }
        __syncthreads();
        for (uint32_t base = 0; base < m; base += k_sort_threads, ++tile)
        {
            scatter_tile(src, dst, m, pass, base, warp_cnt[tile & 1u],
                         warp_cnt[(tile + 1u) & 1u], running);
        }
        uint32_t *const tmp = src;
        src                 = dst;
        dst                 = tmp;
        ++swaps;
        __syncthreads();
    }
    if (tid == 0)
    {
        parity[blockIdx.x] = static_cast<uint8_t>(swaps & 1u);
    }
}

struct SortedColumn
{
    uint32_t const *keys;
    uint32_t        n;
    uint32_t        held      = 0;
    uint32_t        held_base = 0xFFFFFFFFu;

    __device__ float at(uint32_t i)
    {
        uint32_t const base = i & ~31u;
        if (base != held_base)
        {
            uint32_t const j = base + lane_id();
            held             = j < n ? keys[j] : 0u;
            held_base        = base;
        }
        return key_to_float(__shfl_sync(0xFFFFFFFFu, held, i & 31u));
    }
};

struct Run
{
    float    value;
    uint32_t count;
};

struct RunCursor
{
    SortedColumn &col;
    uint32_t      i = 0;

    __device__ bool next(Run &run)
    {
        if (i >= col.n)
        {
            return false;
        }
        float f   = col.at(i++);
        run.count = 1;
        while (i < col.n)
        {
            float const g = col.at(i);
            if (f < g)
            {
                break;
            }
            f = g;
            ++run.count;
            ++i;
        }
        run.value = f;
        return true;
    }
};

struct RunSummary
{
    uint32_t n_runs;
    uint32_t max_count;
    uint32_t n_heavy;
    uint64_t heavy_sum;
};

struct CutSink
{
    float   *out;
    uint32_t n    = 0;
    float    last = 0.0f;

    __device__ void push(float v)
    {
        if (lane_id() == 0)
        {
            out[n] = v;
        }
        ++n;
        last = v;
    }

    __device__ void push_if_above_last(float v)
    {
        if (n == 0 || last < v)
        {
            push(v);
        }
    }
};

inline __device__ float float_midpoint(float a, float b)
{
    constexpr float lo    = FLT_MIN * 2.0f;
    constexpr float hi    = FLT_MAX / 2.0f;
    float const     abs_a = a < 0.0f ? -a : a;
    float const     abs_b = b < 0.0f ? -b : b;
    if (abs_a <= hi && abs_b <= hi)
    {
        return __fmul_rn(__fadd_rn(a, b), 0.5f);
    }
    if (abs_a < lo)
    {
        return __fadd_rn(a, __fmul_rn(b, 0.5f));
    }
    if (abs_b < lo)
    {
        return __fadd_rn(__fmul_rn(a, 0.5f), b);
    }
    return __fadd_rn(__fmul_rn(a, 0.5f), __fmul_rn(b, 0.5f));
}

inline __device__ bool closes_a_bin(bool heavy, bool next_heavy, uint64_t in_bin,
                                    double bin_size)
{
    double const filled = static_cast<double>(in_bin);
    return heavy || filled >= bin_size || (next_heavy && filled >= bin_size / 2.0);
}

inline __device__ bool is_heavy(uint32_t count, double mean_bin)
{
    return !(static_cast<double>(count) < mean_bin);
}

inline __device__ RunSummary summarize_runs(SortedColumn &col, double mean_bin)
{
    RunSummary s{};
    RunCursor  cur{col};
    for (Run r; cur.next(r);)
    {
        ++s.n_runs;
        s.max_count = max(s.max_count, r.count);
        if (is_heavy(r.count, mean_bin))
        {
            ++s.n_heavy;
            s.heavy_sum += r.count;
        }
    }
    return s;
}

inline __device__ void push_every_run(SortedColumn &col, CutSink &sink)
{
    RunCursor cur{col};
    for (Run r; cur.next(r);)
    {
        sink.push(r.value);
    }
}

inline __device__ void greedy_weighted_cuts(SortedColumn &col, uint32_t budget,
                                            double mean_bin, RunSummary const &runs,
                                            CutSink &sink)
{
    uint64_t  rest_sum    = col.n - runs.heavy_sum;
    uint64_t  rest_groups = static_cast<uint64_t>(budget) + 1 - runs.n_heavy;
    double    bin_size    = rest_groups != 0 ? static_cast<double>(rest_sum) /
                                             static_cast<double>(rest_groups)
                                             : mean_bin;
    uint64_t  in_bin      = 0;
    RunCursor cur{col};
    Run       run{};
    Run       next{};
    cur.next(run);
    while (sink.n < budget && cur.next(next))
    {
        bool const heavy      = is_heavy(run.count, mean_bin);
        bool const next_heavy = is_heavy(next.count, mean_bin);
        if (!heavy)
        {
            rest_sum -= run.count;
        }
        in_bin += run.count;
        if (closes_a_bin(heavy, next_heavy, in_bin, bin_size))
        {
            sink.push_if_above_last(float_midpoint(run.value, next.value));
            in_bin = 0;
            if (!heavy && rest_groups > 1)
            {
                --rest_groups;
                bin_size =
                    static_cast<double>(rest_sum) / static_cast<double>(rest_groups);
            }
        }
        run = next;
    }
}

inline __device__ void cuts_for_column(SortedColumn col, uint32_t budget, float *out,
                                       uint32_t *n_out)
{
    uint32_t const n        = col.n;
    double const   mean_bin = static_cast<double>(n) / static_cast<double>(budget + 1);
    RunSummary const runs   = summarize_runs(col, mean_bin);
    CutSink          sink{out};
    if (runs.n_runs <= budget)
    {
        push_every_run(col, sink);
    }
    else if (is_heavy(runs.max_count, mean_bin))
    {
        greedy_weighted_cuts(col, budget, mean_bin, runs, sink);
    }
    else
    {
        uint64_t step =
            (static_cast<uint64_t>(n) + budget) / (static_cast<uint64_t>(budget) + 1);
        step = step == 0 ? 1 : step;
        for (uint64_t k = step; k < n; k += step)
        {
            sink.push_if_above_last(col.at(static_cast<uint32_t>(k)));
        }
    }
    sink.push_if_above_last(FLT_MAX);
    sink.push(__int_as_float(0x7F800000));
    if (lane_id() == 0)
    {
        *n_out = sink.n;
    }
}

__global__ void __launch_bounds__(k_cuts_threads)
    column_cuts_kernel(uint32_t const *__restrict__ keys,
                       uint32_t const *__restrict__ swap,
                       uint8_t const *__restrict__ parity,
                       uint32_t const *__restrict__ nan_counts, uint32_t m, uint32_t w,
                       uint32_t budget, float *__restrict__ cuts,
                       uint32_t *__restrict__ n_cuts)
{
    uint32_t const col = blockIdx.x * k_cuts_warps + (threadIdx.x >> 5);
    if (col >= w)
    {
        return;
    }
    uint32_t const *const sorted =
        (parity[col] != 0 ? swap : keys) + static_cast<size_t>(col) * m;
    SortedColumn column{sorted, m - nan_counts[col]};
    cuts_for_column(column, budget, cuts + static_cast<size_t>(col) * (budget + 2),
                    n_cuts + col);
}

} // namespace cuda_detail
} // namespace bonsai
