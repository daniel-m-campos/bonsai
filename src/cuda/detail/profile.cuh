#pragma once

// BONSAI_CUDA_PROFILE=1 timing accumulators and the running-stopwatch helper
// they pair with. Shared by the CUDA translation units: everything here has
// external linkage in namespace bonsai::cuda_detail and references no
// internal-linkage entity, so including it from more than one TU is ODR-clean.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <print>

namespace bonsai
{
namespace cuda_detail
{

// BONSAI_CUDA_PROFILE=1 accumulators, printed at engine destruction.
struct ProfileCounters
{
    using clock     = std::chrono::steady_clock;
    bool   enabled  = std::getenv("BONSAI_CUDA_PROFILE") != nullptr;
    double upload_s = 0, gpu_s = 0, unpack_s = 0, cpu_s = 0;
    double part_stage_s = 0, adv_stage_s = 0, find_stage_s = 0, lfind_stage_s = 0;
    double gh_upload_s = 0, root_stage_s = 0, gpu_wait_s = 0;
    double bins_upload_s = 0, fin_wait_s = 0, fin_d2h_s = 0;
    double find_kern_s = 0, find_d2h_s = 0;
    // Marginal-round decomposition: device-side spans from event pairs read at
    // the next profile sync, plus the begin_root host reduction, so every
    // millisecond of the round has a name.
    double root_sums_s = 0, adv_memset_s = 0, adv_hist_s = 0, adv_sub_s = 0;
    double root_hist_s = 0, fin_stamp_s = 0, fin_map_s = 0;
    // Resident-objective laps: the device gradient kernel that replaces the gh
    // upload, and the fused route+score-update that replaces the finalize D2H.
    double obj_kernel_s = 0, score_kernel_s = 0;
    size_t launches = 0, gpu_nodes = 0, cpu_calls = 0;

    ProfileCounters()                                       = default;
    ProfileCounters(ProfileCounters const &)                = delete;
    ProfileCounters &operator=(ProfileCounters const &)     = delete;
    ProfileCounters(ProfileCounters &&) noexcept            = delete;
    ProfileCounters &operator=(ProfileCounters &&) noexcept = delete;

    // Running stopwatch: lap(sink) adds the time since the previous lap into
    // sink, a no-op when profiling is off. One per method call marks off its
    // upload / gpu / unpack phases against the shared accumulators.
    struct Lap
    {
        bool              enabled;
        clock::time_point t0 = clock::now();
        void              operator()(double &sink)
        {
            if (!enabled)
            {
                return;
            }
            auto const t1 = clock::now();
            sink += std::chrono::duration<double>(t1 - t0).count();
            t0 = t1;
        }
    };
    Lap lap()
    {
        return Lap{.enabled = enabled};
    }

    ~ProfileCounters()
    {
        if (!enabled || (gpu_s == 0 && cpu_s == 0))
        {
            return;
        }
        try
        {
            std::println(stderr,
                         "cuda-profile: upload={:.2f}s gpu={:.2f}s unpack={:.2f}s "
                         "cpu_fallback={:.2f}s | {} launches covering {} nodes, {} "
                         "cpu-fallback nodes",
                         part_stage_s + adv_stage_s + find_stage_s + lfind_stage_s,
                         gpu_s, unpack_s, cpu_s, launches, gpu_nodes, cpu_calls);
            std::println(stderr,
                         "cuda-upload-decomp: gh={:.2f}s root_stage={:.2f}s "
                         "part_stage={:.2f}s adv_stage={:.2f}s find_stage={:.2f}s "
                         "lfind_stage={:.2f}s gpu_wait={:.2f}s legacy={:.2f}s "
                         "bins_upload={:.2f}s fin_wait={:.2f}s fin_d2h={:.2f}s "
                         "find_kern={:.2f}s find_d2h={:.2f}s",
                         gh_upload_s, root_stage_s, part_stage_s, adv_stage_s,
                         find_stage_s, lfind_stage_s, gpu_wait_s, upload_s,
                         bins_upload_s, fin_wait_s, fin_d2h_s, find_kern_s, find_d2h_s);
            std::println(stderr,
                         "cuda-round-decomp: root_sums={:.2f}s root_hist={:.2f}s "
                         "adv_memset={:.2f}s adv_hist={:.2f}s adv_sub={:.2f}s "
                         "fin_stamp={:.2f}s fin_map={:.2f}s",
                         root_sums_s, root_hist_s, adv_memset_s, adv_hist_s, adv_sub_s,
                         fin_stamp_s, fin_map_s);
            std::println(stderr,
                         "cuda-resident-decomp: obj_kernel={:.2f}s "
                         "score_kernel={:.2f}s",
                         obj_kernel_s, score_kernel_s);
        }
        catch (...)
        {
            std::fputs("cuda-profile: failed to format profile line\n", stderr);
        }
    }
};

} // namespace cuda_detail
} // namespace bonsai
