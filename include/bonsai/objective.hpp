#pragma once

#include "bonsai/types.hpp"
#include <concepts>

namespace bonsai
{

template <typename T>
concept Objective =
    requires(floats_view preds, floats_view targets, floats_out grad, floats_out hess) {
        { T::compute(preds, targets, grad, hess) } -> std::same_as<void>;
        { T::eval(preds, targets) } -> std::same_as<float>;
        { T::init_score(targets) } -> std::same_as<float>;
    };

struct MSEObjective
{
    static void compute(floats_view preds, floats_view targets, floats_out grad,
                        floats_out hess);
    static float eval(floats_view preds, floats_view targets);
    static float init_score(floats_view targets);
};

struct LogLossObjective
{
    static void compute(floats_view scores, floats_view labels, floats_out grad,
                        floats_out hess);
    static float eval(floats_view scores, floats_view labels);
    static float init_score(floats_view labels);
};

} // namespace bonsai
