#pragma once

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cstddef>
#include <mdspan>
#include <span>
#include <stdexcept>
#include <vector>

namespace bonsai
{

// K trees per round over an n x K score matrix, shared by a fit on K classes
// and a fit on K targets. Trees are round-major (tree t serves output t % K
// of round t / K) and load_state refuses a count that is not whole rounds
// (pins: "a multiclass tree count off the class grid refuses to load").
template <TreeGrower Gr, Sampler Sa> class MultiOutputEnsemble : public Ensemble<Gr, Sa>
{
  public:
    using tree_type = typename Gr::Tree;
    using Ensemble<Gr, Sa>::n_iters;
    using Ensemble<Gr, Sa>::n_outputs;
    using Ensemble<Gr, Sa>::trees;

    size_t score_width() const override
    {
        return n_outputs();
    }

    void seed_validation_scores(features_view X, std::span<float> out,
                                size_t n_rounds) const override
    {
        if (n_rounds > 0)
        {
            auto const scores = raw_scores(X, n_rounds);
            std::ranges::copy(scores, out.begin());
            return;
        }
        for (size_t i = 0; i < out.size(); ++i)
        {
            out[i] = this->init_score_at(i % n_outputs());
        }
    }

    std::vector<float> const &init_scores() const
    {
        return this->init_scores_per_output();
    }
    void load_state(std::vector<tree_type> trees, std::vector<float> init_scores)
    {
        if (trees.size() % n_outputs() != 0)
        {
            throw std::invalid_argument(
                "load_state: tree count is not a multiple of the output count");
        }
        this->set_init_scores(std::move(init_scores));
        mutable_trees() = std::move(trees);
    }

  protected:
    using Ensemble<Gr, Sa>::config;
    using Ensemble<Gr, Sa>::init_score_at;
    using Ensemble<Gr, Sa>::mutable_trees;

    MultiOutputEnsemble(Config const &config, size_t n_outputs)
        : Ensemble<Gr, Sa>(config, n_outputs)
    {
    }

    void broadcast_init_scores(size_t n)
    {
        size_t const n_k = n_outputs();
        scores_.assign(n * n_k, 0.0F);
        for (size_t i = 0; i < n; ++i)
        {
            std::ranges::copy(init_scores(),
                              std::span{scores_}.subspan(i * n_k, n_k).begin());
        }
    }

    // Warm start: replay every loaded tree's train contribution into the
    // scores; a no-op for a fresh booster.
    void replay_warm_start(Dataset const &train, size_t n)
    {
        size_t const n_k = n_outputs();
        for (size_t t = 0; t < trees().size(); ++t)
        {
            std::vector<float> raw(n, 0.0F);
            detail::accumulate_train_contribution(trees()[t], train, raw);
            size_t const k      = t % n_k;
            auto const   scores = std::mdspan(scores_.data(), n, n_k);
            for (size_t i = 0; i < n; ++i)
            {
                scores[i, k] += config().learning_rate * raw[i];
            }
        }
    }

    // Raw scores (init + lr * tree sums), n x K, over the first n_rounds
    // rounds (0 = all); accumulate(tree, raw) adds one tree over the rows.
    template <typename Accumulate>
    std::vector<float> raw_scores_over(size_t n, size_t n_rounds,
                                       Accumulate const &accumulate) const
    {
        size_t const rounds = n_rounds == 0 ? n_iters() : std::min(n_rounds, n_iters());
        std::vector<float> scores(n * n_outputs(), 0.0F);
        std::vector<float> raw(n);
        for (size_t k = 0; k < n_outputs(); ++k)
        {
            std::ranges::fill(raw, 0.0F);
            auto const round_trees = std::mdspan(trees().data(), rounds, n_outputs());
            for (size_t r = 0; r < rounds; ++r)
            {
                accumulate(round_trees[r, k], raw);
            }
            for (size_t i = 0; i < n; ++i)
            {
                scores[(i * n_outputs()) + k] =
                    init_score_at(k) + (config().learning_rate * raw[i]);
            }
        }
        return scores;
    }

    std::vector<float> raw_scores(features_view X, size_t n_rounds) const
    {
        return raw_scores_over(X.extent(0), n_rounds,
                               [&](tree_type const &tree, std::vector<float> &raw)
                               { tree.predict(X, raw); });
    }

    std::vector<float> raw_scores_binned(Dataset const &bins, size_t n_rounds) const
    {
        return raw_scores_over(
            bins.view_n_rows(), n_rounds,
            [&](tree_type const &tree, std::vector<float> &raw)
            { detail::accumulate_view_contribution(tree, bins, raw); });
    }

    std::vector<float> scores_;
};

} // namespace bonsai
