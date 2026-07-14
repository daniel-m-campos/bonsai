#pragma once

// Phase profilers, all in one place: each attributes a pipeline's wall-clock
// across named phases, gated by an env var and printed once at process exit.
// The CUDA engine's ProfileCounters stay device-side in
// src/cuda/histogram_engine.cu — they lap GPU streams, not host phases.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <print>
#include <string>
#include <utility>

namespace bonsai::detail
{

// Adds the time since construction (or the previous lap) into sink.
struct Lap
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void                                  operator()(double &sink)
    {
        auto const now = std::chrono::steady_clock::now();
        sink += std::chrono::duration<double>(now - t0).count();
        t0 = now;
    }
};

// CRTP base: Derived declares double phase members plus
//   static constexpr char const *env, *prefix;
//   static constexpr std::array  fields = {std::pair{"name", &Derived::m_s}, ...};
// and optionally `std::string extra() const` for non-phase counters.
template <typename Derived> struct Profiler
{
    using Lap          = detail::Lap;
    bool const enabled = std::getenv(Derived::env) != nullptr;

    static Derived &instance()
    {
        static Derived prof;
        return prof;
    }

  private:
    friend Derived;
    Profiler() = default;
    ~Profiler()
    {
        auto const &self  = static_cast<Derived const &>(*this);
        double      total = 0.0;
        for (auto const &[name, member] : Derived::fields)
        {
            total += self.*member;
        }
        if (!enabled || total <= 0.0)
        {
            return;
        }
        try
        {
            std::string line{Derived::prefix};
            line += ':';
            for (auto const &[name, member] : Derived::fields)
            {
                line += std::format(" {}={:.2f}s", name, self.*member);
            }
            if constexpr (requires { self.extra(); })
            {
                line += self.extra();
            }
            std::println(stderr, "{}", line);
        }
        catch (...) // NOLINT(bugprone-empty-catch): a throwing destructor at
                    // exit would terminate; losing a profile line beats that.
        {
        }
    }
};

// Booster per-iteration phases (BONSAI_FIT_PROFILE=1). The grow phase's own
// breakdown is the GrowProfiler.
struct FitProfiler : Profiler<FitProfiler>
{
    static constexpr char const *env    = "BONSAI_FIT_PROFILE";
    static constexpr char const *prefix = "fit-profile";

    double objective_s = 0, sample_s = 0, grow_s = 0, renew_s = 0, score_s = 0,
           dart_s = 0;

    static constexpr std::array fields = {
        std::pair{"objective", &FitProfiler::objective_s},
        std::pair{"sample", &FitProfiler::sample_s},
        std::pair{"grow", &FitProfiler::grow_s},
        std::pair{"renew", &FitProfiler::renew_s},
        std::pair{"score", &FitProfiler::score_s},
        std::pair{"dart", &FitProfiler::dart_s},
    };
};

// Grow-loop phases (BONSAI_GROW_PROFILE=1), host-side counterpart of the
// CUDA engine's ProfileCounters. populate_adds counts histogram adds
// scheduled; populate_row_s is the row-wise fill's share of populate.
struct GrowProfiler : Profiler<GrowProfiler>
{
    static constexpr char const *env    = "BONSAI_GROW_PROFILE";
    static constexpr char const *prefix = "grow-profile";

    double find_s = 0, bookkeep_s = 0, partition_s = 0, populate_s = 0, finalize_s = 0;
    // Conservation buckets (doc 16): everything grow spends outside the
    // phase laps above. setup = per-tree output allocs + feature sampling +
    // the LevelStep ctor (begin_tree: gh upload, dataset residency);
    // commit = demote + commit_children between the engine phases;
    // assemble = gains/covers resize + Tree construction + result move-out.
    double setup_s = 0, commit_s = 0, assemble_s = 0;
    // assign = make_root's host copy of the sampler's row list (64MB/tree at
    // 16M full-data fits); a rung-0 conservation split of populate.
    double assign_s      = 0;
    double populate_adds = 0, populate_row_s = 0;

    static constexpr std::array fields = {
        std::pair{"find", &GrowProfiler::find_s},
        std::pair{"bookkeep", &GrowProfiler::bookkeep_s},
        std::pair{"partition", &GrowProfiler::partition_s},
        std::pair{"populate", &GrowProfiler::populate_s},
        std::pair{"finalize", &GrowProfiler::finalize_s},
        std::pair{"setup", &GrowProfiler::setup_s},
        std::pair{"commit", &GrowProfiler::commit_s},
        std::pair{"assemble", &GrowProfiler::assemble_s},
        std::pair{"assign", &GrowProfiler::assign_s},
    };

    std::string extra() const
    {
        return std::format(" adds={:.0f}M row_s={:.2f}s", populate_adds / 1e6,
                           populate_row_s);
    }
};

// CSV-to-Dataset pipeline stages (BONSAI_INGEST_PROFILE=1); accumulates
// across files (train + valid).
struct IngestProfiler : Profiler<IngestProfiler>
{
    static constexpr char const *env    = "BONSAI_INGEST_PROFILE";
    static constexpr char const *prefix = "ingest-profile";

    double read_s = 0, index_s = 0, parse_s = 0, fit_s = 0, bin_s = 0, buffer_s = 0,
           dbin_s = 0;

    static constexpr std::array fields = {
        std::pair{"read", &IngestProfiler::read_s},
        std::pair{"index", &IngestProfiler::index_s},
        std::pair{"parse", &IngestProfiler::parse_s},
        std::pair{"mapper-fit", &IngestProfiler::fit_s},
        std::pair{"bin", &IngestProfiler::bin_s},
        std::pair{"buffer", &IngestProfiler::buffer_s},
        std::pair{"dbin", &IngestProfiler::dbin_s},
    };
};

} // namespace bonsai::detail
