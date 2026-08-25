#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <cstddef>
#include <random>

namespace bonsai
{

// grad/hess are mutable: samplers that reweight the kept rows (GOSS) scale
// them in place; the booster recomputes both from the objective every
// iteration, so the scaling never outlives the tree it was drawn for.
//
// `candidates` is the row universe the draw may pick from, which is the
// Dataset's row view: every row of the plane for a plain fit, the selected
// ones for a view. A sampler emits ids out of that list, never positions into
// it, so grad, hess and the bins stay globally indexed. `out` holds the draw
// and is at least as long as the candidate list; the return is how much of it
// was written.
template <typename T>
concept Sampler =
    std::constructible_from<T, Config const &> &&
    requires(T const &s, floats_out grad, floats_out hess, std::mt19937 &rng,
             row_index_view candidates, row_index_out out) {
        { s.sample(grad, hess, rng, candidates, out) } -> std::same_as<size_t>;
    };

struct AllRowsSampler
{
    explicit AllRowsSampler(Config const &);

    static size_t sample(floats_out grad, floats_out hess, std::mt19937 &rng,
                         row_index_view candidates, row_index_out out);
};

struct BernoulliSampler
{
    explicit BernoulliSampler(Config const &cfg);

    size_t sample(floats_out grad, floats_out hess, std::mt19937 &rng,
                  row_index_view candidates, row_index_out out) const;

  private:
    float p_;
};

// Gradient one-side sampling (LightGBM): keep the top_rate fraction of rows
// by |grad|, sample other_rate of the total uniformly from the rest, and
// amplify the sampled rest's grad/hess by (1 - top_rate) / other_rate so
// histogram sums stay unbiased.
struct GossSampler
{
    explicit GossSampler(Config const &cfg);

    size_t sample(floats_out grad, floats_out hess, std::mt19937 &rng,
                  row_index_view candidates, row_index_out out) const;

  private:
    float top_rate_;
    float other_rate_;
};

// What the booster needs to know about a sampler without naming one. Two
// independent questions, both answered at compile time:
//
//   copies_view_verbatim  the emitted list IS the dataset's row view, in
//                         order, so the view's own descriptor already
//                         describes it: identity is a constant-time read, its
//                         runs let the column fill take subspans, and the
//                         per-tree refill can be skipped entirely.
//   reads_gradients       sample() inspects grad/hess, so a fit that derives
//                         them on the device cannot leave them there.
//
// The primary template answers both conservatively, so a sampler added later
// gets the correct slow path by default and opts in deliberately rather than
// by being forgotten in a chain of type comparisons.
template <typename Sa> struct sampler_traits
{
    static constexpr bool copies_view_verbatim = false;
    static constexpr bool reads_gradients      = true;
};

template <> struct sampler_traits<AllRowsSampler>
{
    static constexpr bool copies_view_verbatim = true;
    static constexpr bool reads_gradients      = false;
};

template <> struct sampler_traits<BernoulliSampler>
{
    static constexpr bool copies_view_verbatim = false;
    static constexpr bool reads_gradients      = false;
};

} // namespace bonsai
