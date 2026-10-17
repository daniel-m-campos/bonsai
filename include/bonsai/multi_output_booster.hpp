#pragma once

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/monotone.hpp"
#include "bonsai/multi_output_ensemble.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/objective_traits.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsai
{

// K regression targets, one tree per target per round from target k's column
// of the label block and score matrix through Booster's own compute, renewal
// and score update, so a fit with every row selected matches K single-target
// fits bit for bit (pins: "a K-target fit matches K single-target fits bit
// for bit"). Under resident-objective-eligibility the round arms the device
// plane once with the whole block and grows tree k from the device's column
// k, so it matches K resident single-target fits bit for bit too (pins: "a
// K-target resident fit matches K resident fits"). No DART or device
// prediction, as for multiclass.
template <Objective Obj, TreeGrower Gr, Sampler Sa>
class MultiOutputBooster final : public MultiOutputEnsemble<Gr, Sa>
{
    using Base = MultiOutputEnsemble<Gr, Sa>;

  public:
    using objective_type = Obj;
    using tree_type      = typename Gr::Tree;
    using Base::n_iters;
    using Base::n_outputs;
    using Base::trees;

    explicit MultiOutputBooster(Config const &config)
        : Base(config, config.objective.n_targets),
          monotone_constraints_(config.tree_config.monotone_constraints),
          objective_(config)
    {
        if (n_outputs() < 2)
        {
            throw ConfigError(
                "objective.n_targets must be >= 2 for a multi-target fit");
        }
    }

    void update_one_iter(Dataset const &train) override
    {
        size_t const n   = train.plane_n_rows();
        size_t const n_k = n_outputs();
        if (this->scores().empty())
        {
            require_width(train);
            this->grad().resize(n);
            this->hess().resize(n);
            split_targets(train, n);
            if (trees().empty())
            {
                seed_init_scores(train);
            }
            this->broadcast_init_scores(n);
            this->replay_warm_start(train, n);
            column_.resize(n);
        }

        if (try_resident_round(train))
        {
            return;
        }

        for (size_t k = 0; k < n_k; ++k)
        {
            take_column(k, n);
            objective_.compute(column_, targets_[k], this->grad(), this->hess());
            scale_by_weights(train);
            RowSelection const rows = this->select_rows(train);
            auto [tree, leaf_values, leaf_ids, leaf_bounds] =
                this->grower().grow(train, this->grad(), this->hess(), rows);
            if constexpr (requires(std::span<float> r) { objective_.renew_leaf(r); })
            {
                detail::renew_round(objective_, tree, leaf_ids, leaf_bounds,
                                    leaf_values, targets_[k], column_, train.row_view(),
                                    monotone_constraints_);
            }
            this->advance_output(k, n, leaf_values);
            this->mutable_trees().push_back(std::move(tree));
            this->grower().recycle(std::move(leaf_values), std::move(leaf_ids));
        }
    }

    size_t prediction_width() const override
    {
        return n_outputs();
    }

    void predict_at(features_view X, floats_out y_hat, size_t n_rounds) const override
    {
        assert(y_hat.size() == X.extent(0) * n_outputs());
        std::ranges::copy(this->raw_scores(X, n_rounds), y_hat.begin());
    }

    void predict_at_binned(Dataset const &bins, floats_out y_hat,
                           size_t n_rounds) const override
    {
        assert(y_hat.size() == bins.view_n_rows() * n_outputs());
        std::ranges::copy(this->raw_scores_binned(bins, n_rounds), y_hat.begin());
    }

    float eval(features_view X, floats_view labels) const override
    {
        return objective_.eval(this->raw_scores(X, 0), labels);
    }

    float validation_loss(std::span<float const> scores,
                          floats_view            labels) const override
    {
        return objective_.eval(floats_view{scores.data(), scores.size()}, labels);
    }

  private:
    static constexpr bool has_renew_leaf =
        requires(Obj const &o, std::span<float> r) { o.renew_leaf(r); };

    bool try_resident_round(Dataset const &train)
    {
        if constexpr (has_device_objective<Obj> && !sampler_traits<Sa>::reads_gradients)
        {
            bool const open =
                resident_round_open(this->config().dart_drop_rate,
                                    reproject_monotone_on_host<tree_type>(
                                        has_renew_leaf, monotone_constraints_));
            if (resident_.sync(this->grower(), train, open,
                               std::span<float>{this->scores()}, objective_,
                               this->config().learning_rate))
            {
                resident_round(train);
                return true;
            }
        }
        return false;
    }

    void resident_round(Dataset const &train)
    {
        for (size_t k = 0; k < n_outputs(); ++k)
        {
            this->grower().resident_select(k);
            RowSelection const rows = this->select_rows(train);
            auto res = this->grower().grow(train, floats_view{}, floats_view{}, rows);
            this->mutable_trees().push_back(std::move(res.tree));
        }
    }

    void require_width(Dataset const &train) const
    {
        if (train.n_targets() != n_outputs())
        {
            throw std::invalid_argument(
                "objective.n_targets is " + std::to_string(n_outputs()) +
                " and the labels carry " + std::to_string(train.n_targets()) +
                " targets per row");
        }
    }

    void split_targets(Dataset const &train, size_t n)
    {
        floats_view const block = train.labels();
        targets_.assign(n_outputs(), std::vector<float>(n));
        for (size_t k = 0; k < n_outputs(); ++k)
        {
            for (size_t i = 0; i < n; ++i)
            {
                targets_[k][i] = block[(i * n_outputs()) + k];
            }
        }
    }

    void seed_init_scores(Dataset const &train)
    {
        RowView const     &view = train.row_view();
        std::vector<float> init(n_outputs());
        for (size_t k = 0; k < n_outputs(); ++k)
        {
            init[k] =
                view.is_identity()
                    ? objective_.init_score(targets_[k])
                    : objective_.init_score(floats_view{view.gather(targets_[k])});
        }
        this->set_init_scores(std::move(init));
    }

    void take_column(size_t k, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
        {
            column_[i] = this->scores()[(i * n_outputs()) + k];
        }
    }

    void scale_by_weights(Dataset const &train)
    {
        if (train.weights().empty())
        {
            return;
        }
        parallel::for_each_index(this->grad().size(),
                                 [&](size_t i)
                                 {
                                     this->grad()[i] *= train.weights()[i];
                                     this->hess()[i] *= train.weights()[i];
                                 });
    }

    std::vector<int>                monotone_constraints_;
    objective_type                  objective_;
    std::vector<std::vector<float>> targets_;
    std::vector<float>              column_;
    ResidentState                   resident_;
};

} // namespace bonsai
