#pragma once

// Minimal stand-in for bonsai/split.hpp, providing ONLY the constexpr gain
// math the device kernels call (score / bounded_leaf_weight / l1_thresholded).
// Lets the isolated kernel bench include the real src/cuda/detail/kernels.cuh
// verbatim without dragging in TreeConfig/SplitInput/Histogram and the C++23
// host code that nvcc can't compile. Kept byte-compatible with the real
// definitions so codegen is a fair comparison.

#include <algorithm>

namespace bonsai
{

constexpr double l1_thresholded(double g, double l1)
{
    if (g > l1)
    {
        return g - l1;
    }
    if (g < -l1)
    {
        return g + l1;
    }
    return 0.0;
}

constexpr double score(double g, double h, double lambda)
{
    return (g * g) / (h + lambda);
}

constexpr double score(double g, double h, double l1, double l2)
{
    double const t = l1_thresholded(g, l1);
    return (t * t) / (h + l2);
}

constexpr double bounded_leaf_weight(double g, double h, double l1, double l2,
                                     double lo, double hi)
{
    double const w = -l1_thresholded(g, l1) / (h + l2);
    return std::clamp(w, lo, hi);
}

} // namespace bonsai
