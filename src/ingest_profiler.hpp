#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <print>

namespace bonsai::detail
{

// Ingest counterpart of GrowProfiler (src/level_step.hpp): attributes the
// CSV-to-Dataset pipeline's wall-clock across its stages
// (BONSAI_INGEST_PROFILE=1), printed once at process exit. Accumulates
// across files (train + valid).
struct IngestProfiler
{
    bool const enabled = std::getenv("BONSAI_INGEST_PROFILE") != nullptr;
    double     read_s = 0, index_s = 0, parse_s = 0, fit_s = 0, bin_s = 0, buffer_s = 0;

    static IngestProfiler &instance()
    {
        static IngestProfiler prof;
        return prof;
    }

    ~IngestProfiler()
    {
        if (enabled && read_s + index_s + parse_s + fit_s + bin_s + buffer_s > 0.0)
        {
            std::println(stderr,
                         "ingest-profile: read={:.2f}s index={:.2f}s parse={:.2f}s "
                         "mapper-fit={:.2f}s bin={:.2f}s buffer={:.2f}s",
                         read_s, index_s, parse_s, fit_s, bin_s, buffer_s);
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
