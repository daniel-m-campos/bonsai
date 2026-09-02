#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// One feature's value-to-bin map: right-edge cuts, the last cut a +inf
// sentinel, NaN routed to the reserved final bin. transform() is the one
// binning rule in the system: the CUDA ingest kernel mirrors it bit for bit
// (docs/invariants.md, device-binning-byte-identity) and predict inverts it
// through bin_of_threshold, so an edit here is a wire-format change even
// though no byte layout moves. Three constructors, one invariant: cuts are
// strictly increasing and bin(v) <= split_bin iff v <= cuts[split_bin].
class BinMapper
{
  public:
    static BinMapper fit(floats_view column, BinMapperConfig const &cfg);
    // Cuts from an already-gathered, NaN-free working set. Precondition
    // (asserted): `sample` contains no NaN, since one poisons the whole
    // column's cuts.
    static BinMapper from_sample(std::vector<float> sample, BinMapperConfig const &cfg);
    static BinMapper from_cuts(std::vector<float> cuts)
    {
        return BinMapper{std::move(cuts)};
    }
    // User-supplied interior cut points: validates (finite, strictly
    // increasing, non-empty; ConfigError otherwise) and appends the FLT_MAX
    // top-band cut plus the +inf missing sentinel, so callers pass only the
    // domain edges and every edge is a live split candidate. from_cuts stays
    // the trusted path for the model loader's own serialized cuts.
    static BinMapper from_edges(std::vector<float> edges);
    // The bin a value belongs to. NaN, and only NaN, bins as missing; +inf,
    // which equals the sentinel cut, bins into the top band. Pinned by the
    // binned-walk-bit-identical-to-raw-walk invariant.
    bin_id_t transform(float x) const;
    // The bin a stored split threshold came from. The grower records
    // threshold = cuts()[bin] and cuts are strictly increasing, so lower_bound
    // recovers that bin exactly. The one inversion of the grower's threshold
    // step (invariants: threshold-to-bin-one-inversion).
    bin_id_t bin_of_threshold(float threshold) const
    {
        return static_cast<bin_id_t>(std::ranges::lower_bound(cuts_, threshold) -
                                     cuts_.begin());
    }
    size_t n_bins() const
    {
        return cuts_.size();
    }
    floats_view cuts() const
    {
        return {cuts_};
    }

  private:
    explicit BinMapper(std::vector<float> cuts) : cuts_{std::move(cuts)} {}

    std::vector<float> cuts_;
};

} // namespace bonsai
