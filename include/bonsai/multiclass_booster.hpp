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
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bonsai
{

// K-class softmax boosting: each round grows one tree per class on the
// softmax gradients (grad_k = p_k - 1[y == k], hess_k = p_k (1 - p_k) —
// the true diagonal Hessian; the factor-2 'xgboost convention' halves
// every Newton step and cost 2x the iterations to match lightgbm at the
// same learning rate). trees_ is flat, round-major: tree for class k of
// round r sits at index r * K + k. predict() emits argmax class ids;
// eval() is the multiclass logloss. The 1-D Objective concept can't
// express the K-output shape, which is why this is its own IBooster
// implementation dispatched via BoosterFor<{softmax, G, Sa}>.
template <TreeGrower Gr, Sampler Sa>
class MulticlassBooster final : public ITrainableBooster
{
  public:
    using grower_type  = Gr;
    using sampler_type = Sa;
    using tree_type    = typename Gr::Tree;

    explicit MulticlassBooster(Config const &config)
        : config_(config.booster_config), n_classes_(config.objective.n_classes),
          grower_(config.tree_config), sampler_(config),
          rng_(config.booster_config.random_seed)
    {
        if (n_classes_ < 2)
        {
            throw ConfigError("objective.n_classes must be >= 2 for softmax");
        }
    }

    void update_one_iter(Dataset const &train) override
    {
        size_t const n   = train.plane_n_rows();
        size_t const n_k = n_classes_;
        if (scores_.empty())
        {
            grad_.resize(n);
            hess_.resize(n);
            if (trees_.read().empty())
            {
                seed_class_priors(train);
            }
            broadcast_init_scores(n);
            replay_warm_start(train, n);
        }

        // Per-row softmax probabilities feed every class's gradients.
        std::vector<float> probs(n * n_k);
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     size_t const base      = i * n_k;
                                     auto const [maxv, sum] = row_softmax_exp(
                                         scores_, base, n_k, [&](size_t k, double e)
                                         { probs[base + k] = static_cast<float>(e); });
                                     for (size_t k = 0; k < n_k; ++k)
                                     {
                                         probs[base + k] /= static_cast<float>(sum);
                                     }
                                 });

        // The fit's candidate rows: the Dataset's view, which is every row
        // unless the caller subset it.
        RowView const &view = train.row_view();
        if (row_indices_.size() != view.size())
        {
            row_indices_.resize(view.size());
        }
        if constexpr (sampler_traits<sampler_type>::copies_view_verbatim)
        {
            view.materialize_into(row_indices_);
        }
        // A drawing sampler picks out of the view's rows, so it is handed them
        // as the candidate list and emits ids, not positions.
        else if (candidates_.size() != view.size())
        {
            view.materialize_into(candidates_);
        }
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
                    grad_[i]       = (p - y) * wi;
                    hess_[i]       = std::max(p * (1.0F - p), 1e-6F) * wi;
                });
            size_t   n_selected = 0;
            RowShape shape;
            // The view's shape describes the row list only when the sampler
            // copied it verbatim; a drawing sampler's list keeps gathering.
            if constexpr (sampler_traits<sampler_type>::copies_view_verbatim)
            {
                n_selected = row_indices_.size();
                shape      = {.identity = view.is_identity(), .runs = view.runs()};
            }
            else
            {
                n_selected =
                    sampler_.sample(grad_, hess_, rng_, candidates_, row_indices_);
                shape.identity =
                    view.is_identity() && n_selected == view.size() &&
                    rows_are_identity({row_indices_.data(), n_selected}, view.size());
            }
            auto [tree, leaf_values, leaf_ids] =
                grower_.grow(train, grad_, hess_,
                             RowSelection{{row_indices_.data(), n_selected}, shape});
            // Over the plane, one slot per (row, class), for the same reason
            // the binary booster gives: a repeated row id must not have its
            // prediction advanced once per occurrence.
            parallel::for_each_index(
                n, [&](size_t i)
                { scores_[(i * n_k) + k] += config_.learning_rate * leaf_values[i]; });
            trees_.mutate().push_back(std::move(tree));
            grower_.recycle(std::move(leaf_values), std::move(leaf_ids));
        }
    }

    // Argmax class id per row.
    void predict(features_view X, floats_out y_hat) const override
    {
        predict_at(X, y_hat, 0);
    }

    void predict_at(features_view X, floats_out y_hat, size_t n_rounds) const override
    {
        size_t const n = X.extent(0);
        assert(y_hat.size() == n);
        auto const scores = raw_scores(X, n_rounds);
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     size_t best = 0;
                                     for (size_t k = 1; k < n_classes_; ++k)
                                     {
                                         if (scores[(i * n_classes_) + k] >
                                             scores[(i * n_classes_) + best])
                                         {
                                             best = k;
                                         }
                                     }
                                     y_hat[i] = static_cast<float>(best);
                                 });
    }

    // Row-wise softmax of the class logits: out is n_rows * n_classes_,
    // row-major (the per-class probabilities predict() argmaxes over).
    void predict_proba(features_view X, std::span<double> out) const override
    {
        size_t const k = n_classes_;
        size_t const n = X.extent(0);
        assert(out.size() == n * k);
        auto const scores = raw_scores(X, 0);
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     size_t const base      = i * k;
                                     auto const [maxv, sum] = row_softmax_exp(
                                         scores, base, k, [&](size_t c, double e)
                                         { out[base + c] = e; });
                                     for (size_t c = 0; c < k; ++c)
                                     {
                                         out[base + c] /= sum;
                                     }
                                 });
    }

    void predict_at_binned(Dataset const &bins, floats_out y_hat,
                           size_t n_rounds) const override
    {
        size_t const n = bins.view_n_rows();
        assert(y_hat.size() == n);
        auto const scores = raw_scores_binned(bins, n_rounds);
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     size_t best = 0;
                                     for (size_t k = 1; k < n_classes_; ++k)
                                     {
                                         if (scores[(i * n_classes_) + k] >
                                             scores[(i * n_classes_) + best])
                                         {
                                             best = k;
                                         }
                                     }
                                     y_hat[i] = static_cast<float>(best);
                                 });
    }

    void predict_proba_binned(Dataset const &bins, std::span<double> out) const override
    {
        size_t const k = n_classes_;
        size_t const n = bins.view_n_rows();
        assert(out.size() == n * k);
        auto const scores = raw_scores_binned(bins, 0);
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     size_t const base      = i * k;
                                     auto const [maxv, sum] = row_softmax_exp(
                                         scores, base, k, [&](size_t c, double e)
                                         { out[base + c] = e; });
                                     for (size_t c = 0; c < k; ++c)
                                     {
                                         out[base + c] /= sum;
                                     }
                                 });
    }

    // Multiclass logloss on raw scores.
    float eval(features_view X, floats_view labels) const override
    {
        return logloss_from_scores(raw_scores(X, 0), labels);
    }

    // --- Early-stopping seam (IBooster): width-K raw-score matrix.
    size_t score_width() const override
    {
        return n_classes_;
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
            out[i] = init_scores_.empty() ? 0.0F : init_scores_[i % n_classes_];
        }
    }

    // Row-major over classes: each row's features are read once for all K
    // trees instead of K passes over X, and the per-row add is unchanged.
    void accumulate_last_round(features_view X, floats_out scores) const override
    {
        auto const &trees = trees_.read();
        assert(trees.size() >= n_classes_);
        size_t const first = trees.size() - n_classes_;
        float const  lr    = config_.learning_rate;
        parallel::for_each_index(X.extent(0),
                                 [&](size_t i)
                                 {
                                     for (size_t k = 0; k < n_classes_; ++k)
                                     {
                                         scores[(i * n_classes_) + k] +=
                                             lr * trees[first + k].value_for(
                                                      X, static_cast<row_id_t>(i));
                                     }
                                 });
    }

    void accumulate_last_round_binned(Dataset const &bins,
                                      floats_out     scores) const override
    {
        auto const &trees = trees_.read();
        assert(trees.size() >= n_classes_);
        assert(bins.view_n_rows() * n_classes_ == scores.size());
        assert(!bins.mirror().bins().empty());
        size_t const first = trees.size() - n_classes_;
        float const  lr    = config_.learning_rate;
        // One SplitBins per class, hoisted out of the row loop.
        std::vector<internal::SplitBins> sb;
        sb.reserve(n_classes_);
        for (size_t k = 0; k < n_classes_; ++k)
        {
            sb.push_back(internal::split_bins(trees[first + k], bins));
        }
        auto const    &m  = bins.mirror();
        auto const     rm = m.bins();
        RowIndex const rows{bins.row_view()};
        parallel::for_each_index(
            rows.size(),
            [&](size_t i)
            {
                row_id_t const r      = rows[i];
                auto const     bin_of = [&](size_t f) { return rm[m.index(r, f)]; };
                for (size_t k = 0; k < n_classes_; ++k)
                {
                    scores[(i * n_classes_) + k] +=
                        lr * internal::value_binned(trees[first + k], sb[k], bin_of);
                }
            });
    }

    float validation_loss(std::span<float const> scores,
                          floats_view            labels) const override
    {
        return logloss_from_scores(scores, labels);
    }

    size_t n_iters() const override
    {
        return trees_.read().size() / n_classes_;
    }

    // One tree per class per round.
    size_t n_trees() const override
    {
        return trees_.read().size();
    }

    std::vector<double> feature_importance(ImportanceType type) const override
    {
        std::vector<double> out;
        for (auto const &tree : trees_.read())
        {
            internal::accumulate_importance(tree, type, out);
        }
        return out;
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

    void predict_leaf(features_view X, std::span<node_id_t> out) const override
    {
        internal::predict_leaf_over(trees_.read(), X, out);
    }

    void predict_leaf_binned(Dataset const       &bins,
                             std::span<node_id_t> out) const override
    {
        internal::predict_leaf_over_binned(trees_.read(), bins, out);
    }

    std::string dump(std::span<std::string const> feature_names) const override
    {
        std::string out;
        auto const &trees = trees_.read();
        for (size_t t = 0; t < trees.size(); ++t)
        {
            out += "tree " + std::to_string(t / n_classes_) + " class " +
                   std::to_string(t % n_classes_) + ":\n";
            internal::dump_tree(trees[t], feature_names, out);
        }
        return out;
    }

    // Per-class TreeSHAP: out is (n_rows x n_classes x (n_features + 1)),
    // row-major — class k's contributions for row i start at
    // (i * K + k) * (n_features + 1). Each class's slice sums to that
    // class's raw score (the efficiency property, per class).
    void pred_contribs(features_view X, std::span<double> out,
                       size_t n_features) const override
    {
        if constexpr (std::same_as<tree_type, ObliviousTree>)
        {
            auto const dense = dense_.get(trees_.read(), trees_.epoch());
            contribs_over(*dense, X, out, n_features);
        }
        else
        {
            contribs_over(trees_.read(), X, out, n_features);
        }
    }

    // The shape both contribs paths share: per row and class, zero the slice,
    // walk that class's trees into it (they are flat round-major, so class k
    // strides by n_classes_), scale by the learning rate, and add the class's
    // init score to the bias column. `into(t, row, phi)` is the only
    // difference between them.
    template <typename Trees, typename Into>
    void contribs_impl(Trees const &trees, size_t n, std::span<double> out,
                       size_t n_features, Into const &into) const
    {
        size_t const cols = n_features + 1;
        assert(out.size() == n * n_classes_ * cols);
        parallel::for_each_index(
            n,
            [&](size_t i)
            {
                for (size_t k = 0; k < n_classes_; ++k)
                {
                    std::span<double> const phi =
                        out.subspan(((i * n_classes_) + k) * cols, cols);
                    std::ranges::fill(phi, 0.0);
                    for (size_t t = k; t < trees.size(); t += n_classes_)
                    {
                        into(t, static_cast<row_id_t>(i), phi);
                    }
                    for (double &v : phi)
                    {
                        v *= config_.learning_rate;
                    }
                    phi[n_features] += init_scores_.empty() ? 0.0F : init_scores_[k];
                }
            });
    }

    // Per-class TreeSHAP over any dense-shaped tree range.
    template <typename Trees>
    void contribs_over(Trees const &trees, features_view X, std::span<double> out,
                       size_t n_features) const
    {
        auto const biases = internal::shap_biases(trees);
        contribs_impl(trees, X.extent(0), out, n_features,
                      [&](size_t t, row_id_t row, std::span<double> phi)
                      { tree_shap(trees[t], X, row, phi, biases[t]); });
    }

    void pred_contribs_binned(Dataset const &bins, std::span<double> out,
                              size_t n_features) const override
    {
        if constexpr (std::same_as<tree_type, ObliviousTree>)
        {
            auto const dense = dense_.get(trees_.read(), trees_.epoch());
            contribs_over_binned(*dense, bins, out, n_features);
        }
        else
        {
            contribs_over_binned(trees_.read(), bins, out, n_features);
        }
    }

    // contribs_over's twin: routing reads the row's bins, the class stride and
    // the arithmetic that follows are unchanged.
    template <typename Trees>
    void contribs_over_binned(Trees const &trees, Dataset const &bins,
                              std::span<double> out, size_t n_features) const
    {
        auto const     biases = internal::shap_biases(trees);
        auto const     sb     = internal::tree_split_bins(trees, bins);
        RowIndex const rows{bins.row_view()};
        contribs_impl(trees, rows.size(), out, n_features,
                      [&](size_t t, row_id_t position, std::span<double> phi)
                      {
                          tree_shap_binned(trees[t], sb[t], bins, rows[position], phi,
                                           biases[t]);
                      });
    }

    void truncate(size_t n_rounds) override
    {
        size_t const keep = n_rounds * n_classes_;
        if (keep < trees_.read().size())
        {
            auto &trees = trees_.mutate();
            trees.erase(trees.begin() + static_cast<std::ptrdiff_t>(keep), trees.end());
        }
    }

    // Save/load accessors, mirroring Booster's shape plus the extras the
    // multiclass envelope needs.
    std::vector<tree_type> const &trees() const
    {
        return trees_.read();
    }
    std::vector<float> const &init_scores() const
    {
        return init_scores_;
    }
    size_t n_classes() const
    {
        return n_classes_;
    }
    void load_state(std::vector<tree_type> trees, std::vector<float> init_scores)
    {
        trees_.mutate() = std::move(trees);
        init_scores_    = std::move(init_scores);
    }

  private:
    // Log class priors over the rows this fit visits (a view counts its own
    // labels), like lightgbm's boost_from_average. Only for a fresh booster:
    // a warm start keeps the priors it was loaded with.
    void seed_class_priors(Dataset const &train)
    {
        size_t const n_k = n_classes_;
        init_scores_.assign(n_k, 0.0F);
        std::vector<double> counts(n_k, 0.0);
        floats_view const   labels = train.labels();
        for (row_id_t const r : train.row_view().materialize())
        {
            counts[class_of(labels[r], n_k)] += 1.0;
        }
        auto const n_fit = static_cast<double>(train.view_n_rows());
        for (size_t k = 0; k < n_k; ++k)
        {
            init_scores_[k] =
                static_cast<float>(std::log(std::max(counts[k], 1.0) / n_fit));
        }
    }

    // Every row starts at its class's init score.
    void broadcast_init_scores(size_t n)
    {
        size_t const n_k = n_classes_;
        scores_.assign(n * n_k, 0.0F);
        auto const scores = std::mdspan(scores_.data(), n, n_k);
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t k = 0; k < n_k; ++k)
            {
                scores[i, k] = init_scores_[k];
            }
        }
    }

    // Warm start: replay every loaded tree's train contribution into the
    // scores; a no-op for a fresh booster.
    void replay_warm_start(Dataset const &train, size_t n)
    {
        size_t const n_k = n_classes_;
        for (size_t t = 0; t < trees_.read().size(); ++t)
        {
            std::vector<float> raw(n, 0.0F);
            internal::accumulate_train_contribution(trees_.read()[t], train, raw);
            size_t const k      = t % n_k;
            auto const   scores = std::mdspan(scores_.data(), n, n_k);
            for (size_t i = 0; i < n; ++i)
            {
                scores[i, k] += config_.learning_rate * raw[i];
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

    // Raw (init + lr * tree sums) scores, n_rows x K, using the first
    // n_rounds rounds (0 = all).
    float logloss_from_scores(std::span<float const> scores, floats_view labels) const
    {
        size_t const n     = labels.size();
        double       total = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            size_t const base = i * n_classes_;
            // Loss needs the exp-sum and max but no per-class probs, so the
            // sink is a no-op.
            auto const [maxv, sum] =
                row_softmax_exp(scores, base, n_classes_, [](size_t, double) {});
            size_t const y = class_of(labels[i], n_classes_);
            total -= (scores[base + y] - maxv) - std::log(sum);
        }
        return static_cast<float>(total / static_cast<double>(n));
    }

    std::vector<float> raw_scores(features_view X, size_t n_rounds) const
    {
        size_t const n      = X.extent(0);
        size_t const rounds = n_rounds == 0 ? n_iters() : std::min(n_rounds, n_iters());
        std::vector<float> scores(n * n_classes_, 0.0F);
        std::vector<float> raw(n);
        for (size_t k = 0; k < n_classes_; ++k)
        {
            std::ranges::fill(raw, 0.0F);
            auto const round_trees =
                std::mdspan(trees_.read().data(), rounds, n_classes_);
            for (size_t r = 0; r < rounds; ++r)
            {
                round_trees[r, k].predict(X, raw);
            }
            for (size_t i = 0; i < n; ++i)
            {
                scores[(i * n_classes_) + k] =
                    init_scores_.empty()
                        ? config_.learning_rate * raw[i]
                        : init_scores_[k] + (config_.learning_rate * raw[i]);
            }
        }
        return scores;
    }

    // raw_scores over binned rows: the same per-class accumulation, routed in
    // bin space.
    std::vector<float> raw_scores_binned(Dataset const &bins, size_t n_rounds) const
    {
        size_t const n      = bins.view_n_rows();
        size_t const rounds = n_rounds == 0 ? n_iters() : std::min(n_rounds, n_iters());
        std::vector<float> scores(n * n_classes_, 0.0F);
        std::vector<float> raw(n);
        for (size_t k = 0; k < n_classes_; ++k)
        {
            std::ranges::fill(raw, 0.0F);
            auto const round_trees =
                std::mdspan(trees_.read().data(), rounds, n_classes_);
            for (size_t r = 0; r < rounds; ++r)
            {
                internal::accumulate_view_contribution(round_trees[r, k], bins, raw);
            }
            for (size_t i = 0; i < n; ++i)
            {
                scores[(i * n_classes_) + k] =
                    init_scores_.empty()
                        ? config_.learning_rate * raw[i]
                        : init_scores_[k] + (config_.learning_rate * raw[i]);
            }
        }
        return scores;
    }

    BoosterConfig config_;
    size_t        n_classes_;
    grower_type   grower_;
    sampler_type  sampler_;
    std::mt19937  rng_;
    // The ensemble and its mutation epoch in one place: reads go through
    // read(), every write through mutate(), which is what bumps the epoch the
    // dense SHAP cache and the device plans rebuild on.
    internal::Versioned<std::vector<tree_type>> trees_;
    std::vector<float>    scores_;      // n_rows x K training accumulator
    std::vector<float>    init_scores_; // per-class log prior
    std::vector<float>    grad_;
    std::vector<float>    hess_;
    std::vector<row_id_t> row_indices_;
    // The rows a drawing sampler may pick from; empty when none draws.
    std::vector<row_id_t>  candidates_;
    internal::DensifyCache dense_;
};

} // namespace bonsai
