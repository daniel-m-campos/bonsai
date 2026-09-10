#pragma once

#include <cstdint>

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/dispatch_config.hpp"
#include "bonsai/cuda/grower.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/multiclass_booster.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/typelist.hpp"

namespace bonsai
{

// The cartesian product of all dispatchable (objective, grower, sampler)
// combinations, searched by make_booster and by io/model save/load.
using Configurations = cartesian_product_t<Objectives, Growers, Samplers>;

namespace detail
{
// Default: the uniform single-output booster. Softmax routes to the
// K-output MulticlassBooster, the one objective whose shape doesn't fit
// Booster<O,G,Sa>.
template <typename Combo> struct booster_for
{
    using type = Booster<type_at_t<0, Combo>, type_at_t<1, Combo>, type_at_t<2, Combo>>;
};

template <typename G, typename Sa> struct booster_for<TypeList<SoftmaxObjective, G, Sa>>
{
    using type = MulticlassBooster<G, Sa>;
};
} // namespace detail

template <typename Combo> using BoosterFor = typename detail::booster_for<Combo>::type;

// The triple the table is searched with: a leafwise budget of 0 or of
// 2^max_depth and above cannot bind, so that grower becomes its engine's
// depthwise one; cfg.dispatch stays put (invariants: leaf-budget-route-keeps-the-name).
inline DispatchConfig resolve_dispatch(Config const &cfg)
{
    TreeConfig const &tree        = cfg.tree_config;
    bool const        budget_free = tree.max_leaves == 0;
    bool const        budget_over =
        tree.max_depth < 32 && tree.max_leaves >= (uint32_t{1} << tree.max_depth);
    DispatchConfig disp = cfg.dispatch;
    if (!budget_free && !budget_over)
    {
        return disp;
    }
    if (disp.grower_name == impl_name<LeafwiseGrower<CpuHistogramEngine>>::value)
    {
        disp.grower_name = impl_name<DepthwiseGrower<CpuHistogramEngine>>::value;
    }
    else if (disp.grower_name == impl_name<CudaLeafwiseGrower>::value)
    {
        disp.grower_name = impl_name<CudaDepthwiseGrower>::value;
    }
    return disp;
}

// Invoke `cb.template operator()<Combo>()` for the single Combo matching
// resolve_dispatch(cfg); returns what it returned, or false if none matched.
template <typename Callback> bool with_combo_matching(Config const &cfg, Callback &&cb)
{
    DispatchConfig const disp   = resolve_dispatch(cfg);
    bool                 done   = false;
    bool                 result = false;
    for_each_type<Configurations>(
        [&]<typename Combo>()
        {
            if (done)
            {
                return;
            }
            using O  = type_at_t<0, Combo>;
            using G  = type_at_t<1, Combo>;
            using Sa = type_at_t<2, Combo>;
            if (disp.objective_name != impl_name<O>::value)
            {
                return;
            }
            if (disp.grower_name != impl_name<G>::value)
            {
                return;
            }
            if (disp.sampler_name != impl_name<Sa>::value)
            {
                return;
            }
            result = std::forward<Callback>(cb).template operator()<Combo>();
            done   = true;
        });
    return done && result;
}

} // namespace bonsai
