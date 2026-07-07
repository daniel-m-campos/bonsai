#pragma once

#include "bonsai/booster.hpp"
#include "bonsai/config/dispatch_config.hpp"
#include "bonsai/multiclass_booster.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/names.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/typelist.hpp"

namespace bonsai
{

// The cartesian product of all dispatchable (objective, grower, sampler)
// combinations. Drives factory dispatch in make_booster and tree-type
// resolution in io/model save/load.
using Configurations = cartesian_product_t<Objectives, Growers, Samplers>;

namespace detail
{
// Default: the uniform single-output booster. Softmax routes to the
// K-output MulticlassBooster — the one objective whose shape doesn't fit
// Booster<O,G,Sa> (see multiclass_booster.hpp).
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

// Invoke `cb.template operator()<Combo>()` for the single Combo in
// Configurations whose three impl_names match `disp`. Returns whatever the
// callback returned, or false if no combo matched.
template <typename Callback>
bool with_combo_matching(DispatchConfig const &disp, Callback &&cb)
{
    bool done   = false;
    bool result = false;
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
