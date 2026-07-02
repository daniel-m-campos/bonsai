#pragma once

#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace bonsai
{

class IBooster
{
  public:
    virtual ~IBooster()                                             = default;
    virtual void   update_one_iter(Dataset const &train)            = 0;
    virtual float  eval(features_view X, floats_view labels) const  = 0;
    virtual void   predict(features_view X, floats_out y_hat) const = 0;
    virtual size_t n_iters() const                                  = 0;
};

template <Objective Obj, TreeGrower Gr, Sampler Sa>
class Booster final : public IBooster
{
  public:
    using objective_type = Obj;
    using grower_type    = Gr;
    using sampler_type   = Sa;
    using tree_type      = typename Gr::Tree;

    explicit Booster(Config const &config)
        : config_(config.booster_config), grower_(config.tree_config), sampler_(config),
          rng_(config.booster_config.random_seed)
    {
    }

    void update_one_iter(Dataset const &train) override
    {
        if (trees_.empty())
        {
            grad_.resize(train.n_rows());
            hess_.resize(train.n_rows());
            init_score_ = objective_type::init_score(train.labels());
            scores_.assign(train.n_rows(), init_score_);
        }

        objective_type::compute(scores_, train.labels(), grad_, hess_);

        if (!train.weights().empty())
        {
            for (size_t i = 0; i < grad_.size(); ++i)
            {
                grad_[i] *= train.weights()[i];
                hess_[i] *= train.weights()[i];
            }
        }

        if (row_indices_.size() != train.n_rows())
        {
            row_indices_.resize(train.n_rows());
        }
        size_t const n_selected = sampler_.sample(grad_, hess_, rng_, row_indices_);

        auto [tree, leaf_values] =
            grower_.grow(train, grad_, hess_, {row_indices_.data(), n_selected});

        parallel::for_each_index(
            scores_.size(),
            [&](size_t i) { scores_[i] += config_.learning_rate * leaf_values[i]; });

        trees_.push_back(std::move(tree));
    }

    float eval(features_view X, floats_view labels) const override
    {
        std::vector<float> scores(X.extent(0));
        predict(X, scores);
        return objective_type::eval(scores, labels);
    }

    void predict(features_view X, floats_out scores) const override
    {
        assert(X.extent(0) == scores.size());
        std::fill(scores.begin(), scores.end(), 0.0F);
        for (auto const &tree : trees_)
        {
            tree.predict(X, scores);
        }
        for (float &score : scores)
        {
            score = init_score_ + (score * config_.learning_rate);
        }
    }

    size_t n_iters() const override
    {
        return trees_.size();
    }

    // Save/load accessors. Public so io::save_booster / io::load_booster
    // can serialize state without befriending the I/O module.
    std::vector<tree_type> const &trees() const
    {
        return trees_;
    }
    float init_score() const
    {
        return init_score_;
    }
    void load_state(std::vector<tree_type> trees, float init_score)
    {
        trees_      = std::move(trees);
        init_score_ = init_score;
    }

  private:
    BoosterConfig          config_;
    grower_type            grower_;
    sampler_type           sampler_;
    std::mt19937           rng_;
    std::vector<tree_type> trees_;
    std::vector<float>     scores_;
    std::vector<float>     grad_;
    std::vector<float>     hess_;
    std::vector<row_id_t>  row_indices_;
    float                  init_score_ = 0.0F;
};

} // namespace bonsai
