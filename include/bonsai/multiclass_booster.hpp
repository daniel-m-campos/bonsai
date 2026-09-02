#pragma once

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace bonsai
{

// K-class softmax boosting: each round grows one tree per class on the
// softmax gradients (grad_k = p_k - 1[y == k], hess_k = p_k (1 - p_k), the
// true diagonal Hessian, not the factor-2 convention that halves every
// Newton step; invariants: softmax-true-hessian). The Ensemble holds the
// K-wide trees and answers every query over them; this class owns the round.
// predict() emits argmax class ids and eval() is the multiclass logloss. The
// 1-D Objective concept can't
// express the K-output shape, which is why this is its own booster
// dispatched via BoosterFor<{softmax, G, Sa}>.
template <TreeGrower Gr, Sampler Sa>
class MulticlassBooster final : public Ensemble<Gr, Sa>
{
    using Ensemble<Gr, Sa>::config;
    using Ensemble<Gr, Sa>::grad;
    using Ensemble<Gr, Sa>::grower;
    using Ensemble<Gr, Sa>::hess;
    using Ensemble<Gr, Sa>::init_score_at;
    using Ensemble<Gr, Sa>::mutable_trees;
    using Ensemble<Gr, Sa>::select_rows;

  public:
    using grower_type  = Gr;
    using sampler_type = Sa;
    using tree_type    = typename Gr::Tree;
    using Ensemble<Gr, Sa>::n_iters;
    using Ensemble<Gr, Sa>::n_outputs;
    using Ensemble<Gr, Sa>::trees;

    explicit MulticlassBooster(Config const &config)
        : Ensemble<Gr, Sa>(config, config.objective.n_classes)
    {
        if (n_outputs() < 2)
        {
            throw ConfigError("objective.n_classes must be >= 2 for softmax");
        }
    }

    void update_one_iter(Dataset const &train) override
    {
        size_t const n   = train.plane_n_rows();
        size_t const n_k = n_outputs();
        if (scores_.empty())
        {
            grad().resize(n);
            hess().resize(n);
            if (trees().empty())
            {
                seed_class_priors(train);
            }
            broadcast_init_scores(n);
            replay_warm_start(train, n);
        }

        std::vector<float> probs(n * n_k);
        softmax_rows(scores_, n, n_k, probs);

        // Sample weights scale grad/hess, matching the single-output booster;
        // w.empty() multiplies by exactly 1.0F, keeping unweighted fits
        // bit-identical.
        auto const w = train.weights();
        for (size_t k = 0; k < n_k; ++k)
        {
            parallel::for_each_index(
                n,
                [&](size_t i)
                {
                    float const p = probs[(i * n_k) + k];
                    float const y = class_of(train.labels()[i], n_k) == k ? 1.0F : 0.0F;
                    float const wi = w.empty() ? 1.0F : w[i];
                    grad()[i]      = (p - y) * wi;
                    hess()[i]      = std::max(p * (1.0F - p), 1e-6F) * wi;
                });
            auto [tree, leaf_values, leaf_ids] =
                grower().grow(train, grad(), hess(), select_rows(train));
            // Over the plane, one slot per (row, class), for the same reason
            // the binary booster gives: a repeated row id must not have its
            // prediction advanced once per occurrence.
            parallel::for_each_index(
                n, [&](size_t i)
                { scores_[(i * n_k) + k] += config().learning_rate * leaf_values[i]; });
            mutable_trees().push_back(std::move(tree));
            grower().recycle(std::move(leaf_values), std::move(leaf_ids));
        }
    }

    // Argmax class id per row.
    void predict(features_view X, floats_out y_hat) const override
    {
        predict_at(X, y_hat, 0);
    }

    void predict_at(features_view X, floats_out y_hat, size_t n_rounds) const override
    {
        assert(y_hat.size() == X.extent(0));
        label_rows(raw_scores(X, n_rounds), y_hat);
    }

    // Row-wise softmax of the class logits: out is n_rows * K, row-major
    // (the per-class probabilities predict() argmaxes over).
    void predict_proba(features_view X, std::span<double> out) const override
    {
        assert(out.size() == X.extent(0) * n_outputs());
        softmax_rows(raw_scores(X, 0), X.extent(0), n_outputs(), out);
    }

    void predict_at_binned(Dataset const &bins, floats_out y_hat,
                           size_t n_rounds) const override
    {
        assert(y_hat.size() == bins.view_n_rows());
        label_rows(raw_scores_binned(bins, n_rounds), y_hat);
    }

    void predict_proba_binned(Dataset const &bins, std::span<double> out) const override
    {
        assert(out.size() == bins.view_n_rows() * n_outputs());
        softmax_rows(raw_scores_binned(bins, 0), bins.view_n_rows(), n_outputs(), out);
    }

    // Multiclass logloss on raw scores.
    float eval(features_view X, floats_view labels) const override
    {
        return logloss_from_scores(raw_scores(X, 0), labels);
    }

    // --- Early-stopping seam (IBooster): width-K raw-score matrix.
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

    float validation_loss(std::span<float const> scores,
                          floats_view            labels) const override
    {
        return logloss_from_scores(scores, labels);
    }

    void predict_staged(features_view X, floats_out out) const override
    {
        size_t const n = X.extent(0);
        for (size_t r = 0; r < n_iters(); ++r)
        {
            predict_at(X, floats_out{out.data() + (r * n), n}, r + 1);
        }
    }

    void predict_staged_binned(Dataset const &bins, floats_out out) const override
    {
        size_t const n = bins.view_n_rows();
        for (size_t r = 0; r < n_iters(); ++r)
        {
            predict_at_binned(bins, floats_out{out.data() + (r * n), n}, r + 1);
        }
    }

    // Save/load accessors: the per-class log priors are the extra the
    // multiclass envelope carries beyond the trees.
    std::vector<float> const &init_scores() const
    {
        return this->init_scores_per_output();
    }
    void load_state(std::vector<tree_type> trees, std::vector<float> init_scores)
    {
        if (trees.size() % n_outputs() != 0)
        {
            throw std::invalid_argument(
                "load_state: tree count is not a multiple of n_classes");
        }
        this->set_init_scores(std::move(init_scores));
        mutable_trees() = std::move(trees);
    }

  private:
    // Log class priors over the rows this fit visits (a view counts its own
    // labels), like lightgbm's boost_from_average. Only for a fresh booster:
    // a warm start keeps the priors it was loaded with.
    void seed_class_priors(Dataset const &train)
    {
        size_t const        n_k = n_outputs();
        std::vector<double> counts(n_k, 0.0);
        floats_view const   labels = train.labels();
        for (row_id_t const r : train.row_view().materialize())
        {
            counts[class_of(labels[r], n_k)] += 1.0;
        }
        auto const         n_fit = static_cast<double>(train.view_n_rows());
        std::vector<float> priors(n_k);
        for (size_t k = 0; k < n_k; ++k)
        {
            priors[k] = static_cast<float>(std::log(std::max(counts[k], 1.0) / n_fit));
        }
        this->set_init_scores(std::move(priors));
    }

    void broadcast_init_scores(size_t n)
    {
        size_t const n_k = n_outputs();
        scores_.assign(n * n_k, 0.0F);
        auto const scores = std::mdspan(scores_.data(), n, n_k);
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t k = 0; k < n_k; ++k)
            {
                scores[i, k] = init_score_at(k);
            }
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
            internal::accumulate_train_contribution(trees()[t], train, raw);
            size_t const k      = t % n_k;
            auto const   scores = std::mdspan(scores_.data(), n, n_k);
            for (size_t i = 0; i < n; ++i)
            {
                scores[i, k] += config().learning_rate * raw[i];
            }
        }
    }

    static size_t class_of(float label, size_t n_k)
    {
        auto const k = std::lround(label);
        return k >= 0 && static_cast<size_t>(k) < n_k ? static_cast<size_t>(k) : 0;
    }

    // Numerically-stable softmax of one row's K logits: max-subtract, then
    // exp each class (in double) into sink(k, e). Returns {maxv, sum(e)} for
    // the caller to normalize (probs) or form a loss. scores may be float or
    // double; the exp argument promotes to double either way, so every call
    // site's op order is preserved. base is the row's flat offset (row * K).
    struct RowSoftmax
    {
        double maxv;
        double sum;
    };
    template <typename Scores, typename Sink>
    static RowSoftmax row_softmax_exp(Scores const &scores, size_t base, size_t k_count,
                                      Sink &&sink)
    {
        double maxv = scores[base];
        for (size_t k = 1; k < k_count; ++k)
        {
            maxv = std::max(maxv, static_cast<double>(scores[base + k]));
        }
        double sum = 0.0;
        for (size_t k = 0; k < k_count; ++k)
        {
            double const e = std::exp(scores[base + k] - maxv);
            sink(k, e);
            sum += e;
        }
        return {maxv, sum};
    }

    // Row-wise normalized softmax of an n x k logit matrix into out, whose
    // element type sets the precision each probability is stored at (float
    // for the training accumulator, double for predict_proba).
    template <typename Scores, typename Out>
    static void softmax_rows(Scores const &scores, size_t n, size_t k, Out &out)
    {
        using value_type = std::remove_reference_t<decltype(out[0])>;
        parallel::for_each_index(
            n,
            [&](size_t i)
            {
                size_t const base = i * k;
                auto const [maxv, sum] =
                    row_softmax_exp(scores, base, k, [&](size_t c, double e)
                                    { out[base + c] = static_cast<value_type>(e); });
                for (size_t c = 0; c < k; ++c)
                {
                    out[base + c] /= static_cast<value_type>(sum);
                }
            });
    }

    // Each row's predicted label: the class of its largest logit, first
    // index on ties.
    void label_rows(std::vector<float> const &scores, floats_out y_hat) const
    {
        size_t const k_count = n_outputs();
        parallel::for_each_index(y_hat.size(),
                                 [&](size_t i)
                                 {
                                     size_t best = 0;
                                     for (size_t k = 1; k < k_count; ++k)
                                     {
                                         if (scores[(i * k_count) + k] >
                                             scores[(i * k_count) + best])
                                         {
                                             best = k;
                                         }
                                     }
                                     y_hat[i] = static_cast<float>(best);
                                 });
    }

    float logloss_from_scores(std::span<float const> scores, floats_view labels) const
    {
        size_t const n     = labels.size();
        double       total = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            size_t const base = i * n_outputs();
            // Loss needs the exp-sum and max but no per-class probs, so the
            // sink is a no-op.
            auto const [maxv, sum] =
                row_softmax_exp(scores, base, n_outputs(), [](size_t, double) {});
            size_t const y = class_of(labels[i], n_outputs());
            total -= (scores[base + y] - maxv) - std::log(sum);
        }
        return static_cast<float>(total / static_cast<double>(n));
    }

    // Raw (init + lr * tree sums) scores, n_rows x K, using the first
    // n_rounds rounds (0 = all). accumulate(tree, raw) adds one tree's
    // contribution over the rows; the two callers differ only in whether
    // that routing reads features or bins.
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
            { internal::accumulate_view_contribution(tree, bins, raw); });
    }

    // n_rows x K training accumulator.
    std::vector<float> scores_;
};

} // namespace bonsai
