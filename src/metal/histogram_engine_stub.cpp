#include "bonsai/dataset.hpp"
#include "bonsai/metal/histogram_engine.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <memory>
#include <span>
#include <stdexcept>

namespace bonsai
{

namespace
{

[[noreturn]] void throw_unavailable()
{
    throw std::runtime_error(
        "metal_depthwise requires a macOS build with -DBONSAI_METAL=ON; "
        "this binary was built without the Metal backend");
}

} // namespace

bool metal_available()
{
    return false;
}

struct MetalHistogramEngine::Impl
{
};

MetalHistogramEngine::MetalHistogramEngine() : impl_(std::make_unique<Impl>()) {}
MetalHistogramEngine::~MetalHistogramEngine()                                = default;
MetalHistogramEngine::MetalHistogramEngine(MetalHistogramEngine &&) noexcept = default;
MetalHistogramEngine &
MetalHistogramEngine::operator=(MetalHistogramEngine &&) noexcept = default;

void MetalHistogramEngine::begin_tree(Dataset const & /*ds*/, floats_view /*grad*/,
                                      floats_view /*hess*/)
{
    throw_unavailable();
}

void MetalHistogramEngine::populate(Dataset const & /*ds*/, floats_view /*grad*/,
                                    floats_view /*hess*/, SplitInput & /*split_input*/,
                                    std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

void MetalHistogramEngine::populate_many(Dataset const & /*ds*/, floats_view /*grad*/,
                                         floats_view /*hess*/,
                                         split_input_refs /*nodes*/,
                                         std::span<feature_id_t const> /*selected*/)
{
    throw_unavailable();
}

} // namespace bonsai
