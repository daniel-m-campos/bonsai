#pragma once

#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <cstddef>
#include <random>
#include <span>
#include <vector>

namespace bonsai
{

class IBooster
{
  public:
    virtual ~IBooster()                                           = default;
    virtual void update_one_iter(Dataset const &train)            = 0;
    virtual float eval(features_view X, floats_view labels) const = 0;
    virtual void predict(features_view X, floats_out y_hat) const = 0;
    virtual size_t n_iters() const                                = 0;
};

template <Objective Obj, TreeGrower Gr, Sampler Sa>
class Booster final : public IBooster
{
  public:
    using objective_type = Obj;
    using grower_type    = Gr;
    using sampler_type   = Sa;

    explicit Booster(Config const &config);

    void update_one_iter(Dataset const &train) override;
    float eval(features_view X, floats_view labels) const override;
    void predict(features_view X, floats_out scores) const override;
    size_t n_iters() const override
    {
        return trees_.size();
    };

    // Save/load accessors. Public so io::save_booster / io::load_booster
    // can serialize state without befriending the I/O module.
    std::vector<typename grower_type::Tree> const &trees() const
    {
        return trees_;
    }
    float init_score() const
    {
        return init_score_;
    }
    void load_state(std::vector<typename grower_type::Tree> trees, float init_score)
    {
        trees_      = std::move(trees);
        init_score_ = init_score;
    }

  private:
    BoosterConfig config_;
    grower_type grower_;
    std::mt19937 rng_;
    std::vector<typename grower_type::Tree> trees_;
    std::vector<float> scores_;
    std::vector<float> grad_;
    std::vector<float> hess_;
    std::vector<row_id_t> row_indices_;
    float init_score_ = 0.0F;
};
} // namespace bonsai
