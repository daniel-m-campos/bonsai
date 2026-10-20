#pragma once

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <concepts>
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
        fill_init_scores(out);
    }

    // Staged prediction is predict_at after each round, stacked round-major
    // at the derived class's own width.
    void predict_staged(features_view X, floats_out out) const override
    {
        size_t const stride = X.extent(0) * this->prediction_width();
        for (size_t r = 0; r < n_iters(); ++r)
        {
            this->predict_at(X, out.subspan(r * stride, stride), r + 1);
        }
    }

    void predict_staged_binned(Dataset const &bins, floats_out out) const override
    {
        size_t const stride = bins.view_n_rows() * this->prediction_width();
        for (size_t r = 0; r < n_iters(); ++r)
        {
            this->predict_at_binned(bins, out.subspan(r * stride, stride), r + 1);
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
        scores_.assign(n * n_outputs(), 0.0F);
        fill_init_scores(scores_);
    }

    void fill_init_scores(std::span<float> block) const
    {
        auto const rows =
            std::mdspan(block.data(), block.size() / n_outputs(), n_outputs());
        for (size_t i = 0; i < rows.extent(0); ++i)
        {
            for (size_t k = 0; k < rows.extent(1); ++k)
            {
                rows[i, k] = init_score_at(k);
            }
        }
    }

    auto score_block()
    {
        return std::mdspan(scores_.data(), scores_.size() / n_outputs(), n_outputs());
    }

    auto round_trees() const
    {
        return std::mdspan(trees().data(), n_iters(), n_outputs());
    }

    // Warm start: replay every loaded tree's train contribution into the
    // scores; a no-op for a fresh booster.
    void replay_warm_start(Dataset const &train, size_t n)
    {
        auto const         by_round = round_trees();
        auto const         scores   = score_block();
        std::vector<float> raw(n);
        for (size_t r = 0; r < by_round.extent(0); ++r)
        {
            for (size_t k = 0; k < by_round.extent(1); ++k)
            {
                std::ranges::fill(raw, 0.0F);
                detail::accumulate_train_contribution(by_round[r, k], train, raw);
                for (size_t i = 0; i < n; ++i)
                {
                    scores[i, k] += config().learning_rate * raw[i];
                }
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
        auto const         block    = std::mdspan(scores.data(), n, n_outputs());
        auto const         by_round = round_trees();
        std::vector<float> raw(n);
        for (size_t k = 0; k < n_outputs(); ++k)
        {
            std::ranges::fill(raw, 0.0F);
            for (size_t r = 0; r < rounds; ++r)
            {
                accumulate(by_round[r, k], raw);
            }
            for (size_t i = 0; i < n; ++i)
            {
                block[i, k] = init_score_at(k) + (config().learning_rate * raw[i]);
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

    void advance_output(size_t k, size_t n, std::vector<float> const &leaf_values)
    {
        auto const scores = score_block();
        parallel::for_each_index(
            n,
            [&](size_t i) { scores[i, k] += config().learning_rate * leaf_values[i]; });
    }

    std::vector<float> &scores()
    {
        return scores_;
    }

    // Output k's trees as one dense, round-ordered vector, every output
    // rebuilt once per epoch and held together so a reader keeps a
    // superseded set alive while it walks it.
    std::shared_ptr<std::vector<std::vector<DenseTree>> const> dense_columns() const
    {
        return columns_.get(this->epoch(),
                            [&]
                            {
                                auto const by_round = round_trees();
                                std::vector<std::vector<DenseTree>> columns(
                                    n_outputs());
                                for (size_t r = 0; r < by_round.extent(0); ++r)
                                {
                                    for (size_t k = 0; k < by_round.extent(1); ++k)
                                    {
                                        columns[k].push_back(dense_of(by_round[r, k]));
                                    }
                                }
                                return columns;
                            });
    }

  private:
    static DenseTree dense_of(tree_type const &tree)
    {
        if constexpr (std::same_as<tree_type, ObliviousTree>)
        {
            return dense_equivalent(tree);
        }
        else
        {
            return tree;
        }
    }

    std::vector<float>                                      scores_;
    detail::EpochCache<std::vector<std::vector<DenseTree>>> columns_;
};

} // namespace bonsai
