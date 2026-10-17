#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/multi_output_ensemble.hpp"
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
// Newton step; invariants: softmax-true-hessian). MultiOutputEnsemble holds
// the K-wide trees and scores and answers every raw query over them; this
// class owns the round. predict() emits argmax class ids and eval() is the
// multiclass logloss. The 1-D Objective concept can't express the K-output
// shape, which is why this is its own booster dispatched via
// BoosterFor<{softmax, G, Sa}>.
template <TreeGrower Gr, Sampler Sa>
class MulticlassBooster final : public MultiOutputEnsemble<Gr, Sa>
{
    using Base = MultiOutputEnsemble<Gr, Sa>;

  public:
    using tree_type = typename Gr::Tree;
    using Base::n_iters;
    using Base::n_outputs;
    using Base::trees;

    explicit MulticlassBooster(Config const &config)
        : Base(config, config.objective.n_classes)
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
        if (this->scores().empty())
        {
            this->grad().resize(n);
            this->hess().resize(n);
            if (trees().empty())
            {
                seed_class_priors(train);
            }
            this->broadcast_init_scores(n);
            this->replay_warm_start(train, n);
        }

        std::vector<float> probs(n * n_k);
        softmax_rows(this->scores(), n, n_k, probs);

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
                    float const wi  = w.empty() ? 1.0F : w[i];
                    this->grad()[i] = (p - y) * wi;
                    this->hess()[i] = std::max(p * (1.0F - p), 1e-6F) * wi;
                });
            auto [tree, leaf_values, leaf_ids, leaf_bounds] = this->grower().grow(
                train, this->grad(), this->hess(), this->select_rows(train));
            // Over the plane, one slot per (row, class), for the same reason
            // the binary booster gives: a repeated row id must not have its
            // prediction advanced once per occurrence.
            this->advance_output(k, n, leaf_values);
            this->mutable_trees().push_back(std::move(tree));
            this->grower().recycle(std::move(leaf_values), std::move(leaf_ids));
        }
    }

    // Argmax class id per row.
    void predict_at(features_view X, floats_out y_hat, size_t n_rounds) const override
    {
        assert(y_hat.size() == X.extent(0));
        label_rows(this->raw_scores(X, n_rounds), y_hat);
    }

    // Row-wise softmax of the class logits: out is n_rows * K, row-major
    // (the per-class probabilities predict() argmaxes over).
    void predict_proba(features_view X, std::span<double> out) const override
    {
        assert(out.size() == X.extent(0) * n_outputs());
        softmax_rows(this->raw_scores(X, 0), X.extent(0), n_outputs(), out);
    }

    void predict_at_binned(Dataset const &bins, floats_out y_hat,
                           size_t n_rounds) const override
    {
        assert(y_hat.size() == bins.view_n_rows());
        label_rows(this->raw_scores_binned(bins, n_rounds), y_hat);
    }

    void predict_proba_binned(Dataset const &bins, std::span<double> out) const override
    {
        assert(out.size() == bins.view_n_rows() * n_outputs());
        softmax_rows(this->raw_scores_binned(bins, 0), bins.view_n_rows(), n_outputs(),
                     out);
    }

    // Multiclass logloss on raw scores.
    float eval(features_view X, floats_view labels) const override
    {
        return logloss_from_scores(this->raw_scores(X, 0), labels);
    }

    float validation_loss(std::span<float const> scores,
                          floats_view            labels) const override
    {
        return logloss_from_scores(scores, labels);
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
        parallel::for_each_index(
            y_hat.size(),
            [&](size_t i)
            {
                auto const row = std::span{scores}.subspan(i * k_count, k_count);
                y_hat[i] =
                    static_cast<float>(std::ranges::max_element(row) - row.begin());
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

    // n_rows x K training accumulator.
};

} // namespace bonsai
