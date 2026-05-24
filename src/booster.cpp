#include "bonsai/booster.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace bonsai
{

template <Objective Obj, TreeGrower Gr, Sampler Sa>
Booster<Obj, Gr, Sa>::Booster(Config const &config)
    : config_(config.booster_config), grower_(config.tree_config),
      rng_(config.booster_config.random_seed)
{
}

template <Objective Obj, TreeGrower Gr, Sampler Sa>
void Booster<Obj, Gr, Sa>::update_one_iter(Dataset const &train)
{
    // init scores
    if (trees_.empty())
    {
        grad_.resize(train.n_rows());
        hess_.resize(train.n_rows());
        init_score_ = objective_type::init_score(train.labels());
        scores_.assign(train.n_rows(), init_score_);
    }

    objective_type::compute(scores_, train.labels(), grad_, hess_);

    // apply weights if they exist
    if (!train.weights().empty())
    {
        for (size_t i = 0; i < grad_.size(); ++i)
        {
            grad_[i] *= train.weights()[i];
            hess_[i] *= train.weights()[i];
        }
    }

    // sample rows
    if (row_indices_.size() != train.n_rows())
    {
        row_indices_.resize(train.n_rows());
    }
    size_t const n_selected = sampler_type::sample(grad_, hess_, rng_, row_indices_);

    // grow tree
    auto [tree, leaf_values] =
        grower_.grow(train, grad_, hess_, {row_indices_.data(), n_selected});

    // update scores
    for (size_t i = 0; i < scores_.size(); ++i)
    {
        scores_[i] += config_.learning_rate * leaf_values[i];
    }

    trees_.push_back(std::move(tree));
}

template <Objective Obj, TreeGrower Gr, Sampler Sa>
float Booster<Obj, Gr, Sa>::eval(features_view X, floats_view labels) const
{
    std::vector<float> scores(X.extent(0));
    predict(X, scores);
    return objective_type::eval(scores, labels);
}

template <Objective Obj, TreeGrower Gr, Sampler Sa>
void Booster<Obj, Gr, Sa>::predict(features_view X, floats_out scores) const
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

template class Booster<MSEObjective, DepthwiseGrower<HistogramNodeSplitFinder>,
                       AllRowsSampler>;
template class Booster<LogLossObjective, DepthwiseGrower<HistogramNodeSplitFinder>,
                       AllRowsSampler>;

} // namespace bonsai
