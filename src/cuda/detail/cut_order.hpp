#pragma once

#include <bit>
#include <cstdint>

#if defined(__CUDACC__)
#define BONSAI_HOST_DEVICE __host__ __device__
#else
#define BONSAI_HOST_DEVICE
#endif

namespace bonsai
{
namespace cuda_detail
{

struct FeatBest
{
    double  gain, gL, hL, gR, hR;
    int32_t bin, dl, valid, sel;
};

inline BONSAI_HOST_DEVICE bool feat_better(double ga, int ba, int da, int va, double gb,
                                           int bb, int db, int vb)
{
    if (va != vb)
    {
        return va > vb;
    }
    if (va == 0)
    {
        return false;
    }
    if (ga != gb)
    {
        return ga > gb;
    }
    if (ba != bb)
    {
        return ba < bb;
    }
    return da > db;
}

inline BONSAI_HOST_DEVICE bool feat_better(FeatBest const &a, FeatBest const &b)
{
    return feat_better(a.gain, a.bin, a.dl, a.valid, b.gain, b.bin, b.dl, b.valid);
}

inline BONSAI_HOST_DEVICE long long key_of_positive_gain(double gain)
{
    return std::bit_cast<long long>(gain);
}

inline BONSAI_HOST_DEVICE bool split_better(FeatBest const &a, FeatBest const &b)
{
    if (a.valid != b.valid)
    {
        return a.valid > b.valid;
    }
    if (a.valid == 0)
    {
        return false;
    }
    long long const ka = key_of_positive_gain(a.gain);
    long long const kb = key_of_positive_gain(b.gain);
    if (ka != kb)
    {
        return ka > kb;
    }
    if (a.bin != b.bin)
    {
        return a.bin < b.bin;
    }
    return a.dl > b.dl;
}

} // namespace cuda_detail
} // namespace bonsai
