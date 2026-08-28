#include <metal_stdlib>
using namespace metal;

struct NodeParams
{
    uint  count;
    uint  n_chunks;
    uint  use_rows;
    uint  has_hess;
    uint  plane_rows;
    uint  stride;
    float scale;
};

kernel void hist_fill(device const BIN_T      *bins    [[buffer(0)]],
                      device const float      *grad    [[buffer(1)]],
                      device const float      *hess    [[buffer(2)]],
                      device const uint       *rows    [[buffer(3)]],
                      device const uint       *sel     [[buffer(4)]],
                      device const uint       *nbins   [[buffer(5)]],
                      device atomic_float     *out     [[buffer(6)]],
                      constant NodeParams     &p       [[buffer(7)]],
                      threadgroup atomic_uint *sh      [[threadgroup(0)]],
                      uint3 gid  [[threadgroup_position_in_grid]],
                      uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint const slot  = gid.x;
    uint const chunk = gid.y;
    uint const tid   = tid3.x;
    uint const fid   = sel[slot];
    uint const nb    = nbins[fid];
    for (uint i = tid; i < 4 * nb; i += 256)
    {
        atomic_store_explicit(&sh[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup atomic_uint *mine = sh + (((tid >> 5) & 1u) * 2 * nb);
    device const BIN_T      *col  = bins + (ulong)fid * p.plane_rows;
    uint const               span = p.n_chunks * 256;
    for (uint k = (chunk * 256) + tid; k < p.count; k += span)
    {
        uint const  r = p.use_rows != 0 ? rows[k] : k;
        uint const  b = col[r];
        float const g = grad[r];
        float const h = p.has_hess != 0 ? hess[r] : 1.0f;
        atomic_fetch_add_explicit(&mine[2 * b], (uint)(int)rint(g * p.scale),
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&mine[(2 * b) + 1], (uint)(int)rint(h * p.scale),
                                  memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device atomic_float *o = out + ((ulong)slot * 2 * p.stride);
    for (uint i = tid; i < 2 * nb; i += 256)
    {
        int const   lo = (int)atomic_load_explicit(&sh[i], memory_order_relaxed);
        int const   hi = (int)atomic_load_explicit(&sh[(2 * nb) + i],
                                                   memory_order_relaxed);
        float const v  = (float)(lo + hi) / p.scale;
        if (v != 0.0f)
        {
            atomic_fetch_add_explicit(&o[i], v, memory_order_relaxed);
        }
    }
}
