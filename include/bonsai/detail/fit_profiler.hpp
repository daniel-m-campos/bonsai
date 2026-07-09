#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <print>

namespace bonsai::detail
{

// Attributes a fit's wall-clock across the booster's per-iteration phases
// (BONSAI_FIT_PROFILE=1), printed once at process exit. The grow phase's
// own breakdown is the GrowProfiler (BONSAI_GROW_PROFILE=1).
struct FitProfiler
{
    bool const enabled     = std::getenv("BONSAI_FIT_PROFILE") != nullptr;
    double     objective_s = 0, sample_s = 0, grow_s = 0, renew_s = 0, score_s = 0,
           dart_s = 0;

    static FitProfiler &instance()
    {
        static FitProfiler prof;
        return prof;
    }

    ~FitProfiler()
    {
        if (!enabled ||
            objective_s + sample_s + grow_s + renew_s + score_s + dart_s <= 0.0)
        {
            return;
        }
        try
        {
            std::println(stderr,
                         "fit-profile: objective={:.2f}s sample={:.2f}s "
                         "grow={:.2f}s renew={:.2f}s score={:.2f}s dart={:.2f}s",
                         objective_s, sample_s, grow_s, renew_s, score_s, dart_s);
        }
        catch (...) // NOLINT(bugprone-empty-catch): a throwing destructor at
                    // exit would terminate; losing a profile line beats that.
        {
        }
    }

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
};

} // namespace bonsai::detail
