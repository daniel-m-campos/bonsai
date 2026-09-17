#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <numeric>
#include <print>
#include <span>
#include <string>

#include "kernel_args.cuh"

namespace bonsai
{
namespace cuda_detail
{

inline bool profile_on()
{
    static bool const on = std::getenv("BONSAI_CUDA_PROFILE") != nullptr;
    return on;
}

inline size_t row_total(std::span<uint32_t const> counts)
{
    return std::reduce(counts.begin(), counts.end(), size_t{0});
}

inline size_t active_fill_blocks(FillLaunch const         &launched,
                                 std::span<uint32_t const> counts)
{
    size_t active = 0;
    for (uint32_t const count : counts)
    {
        active += node_chunk_count(count, launched.chunk_rows, launched.n_chunks);
    }
    return active * launched.grid_x;
}

struct ProfileCounters
{
    using clock    = std::chrono::steady_clock;
    bool   enabled = profile_on();
    double gpu_s = 0, unpack_s = 0;
    double part_stage_s = 0, adv_stage_s = 0, find_stage_s = 0, lfind_stage_s = 0;
    double gh_upload_s = 0, root_stage_s = 0, gpu_wait_s = 0;
    double bins_upload_s = 0, fin_wait_s = 0, fin_d2h_s = 0;
    double find_kern_s = 0, find_d2h_s = 0;
    double root_sums_s = 0, adv_memset_s = 0, adv_hist_s = 0;
    double root_hist_s = 0, fin_stamp_s = 0, fin_map_s = 0;
    double part_kernel_s = 0;
    double obj_kernel_s = 0, score_kernel_s = 0;
    double eval_kernel_s = 0;
    size_t launches = 0, gpu_nodes = 0;

    struct LevelCounters
    {
        double hist_s = 0, small_s = 0, find_s = 0, find_gb = 0;
        size_t rows = 0, blocks = 0, small_rows = 0;
    };
    static constexpr size_t                  k_level_slots = 16;
    std::array<LevelCounters, k_level_slots> levels{};

    LevelCounters &level(uint32_t depth)
    {
        return levels[std::min<size_t>(depth, k_level_slots) - 1];
    }

    ProfileCounters()                                       = default;
    ProfileCounters(ProfileCounters const &)                = delete;
    ProfileCounters &operator=(ProfileCounters const &)     = delete;
    ProfileCounters(ProfileCounters &&) noexcept            = delete;
    ProfileCounters &operator=(ProfileCounters &&) noexcept = delete;

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
    void launched(size_t nodes = 0)
    {
        if (enabled)
        {
            ++launches;
            gpu_nodes += nodes;
        }
    }
    void note_fill(uint32_t depth, FillLaunch const &fill,
                   std::span<uint32_t const> row_counts,
                   std::span<uint32_t const> small_counts)
    {
        if (!enabled)
        {
            return;
        }
        LevelCounters &c = level(depth);
        c.rows += row_total(row_counts);
        c.blocks += active_fill_blocks(fill, row_counts);
        c.small_rows += row_total(small_counts);
    }
    void note_find(uint32_t depth, double kern_s, size_t strip_bytes)
    {
        find_kern_s += kern_s;
        if (depth == 0)
        {
            return;
        }
        LevelCounters &c = level(depth);
        c.find_s += kern_s;
        c.find_gb += static_cast<double>(strip_bytes) * 1e-9;
    }

    std::string level_hist_line() const
    {
        std::string line;
        for (size_t i = 0; i < k_level_slots; ++i)
        {
            LevelCounters const &c = levels[i];
            if (c.hist_s > 0)
            {
                line += std::format(" hist_l{}={:.2f}s rows_l{}={} blocks_l{}={}"
                                    " small_l{}={:.2f}s small_rows_l{}={}"
                                    " find_l{}={:.2f}s find_gb_l{}={:.1f}",
                                    i + 1, c.hist_s, i + 1, c.rows, i + 1, c.blocks,
                                    i + 1, c.small_s, i + 1, c.small_rows, i + 1,
                                    c.find_s, i + 1, c.find_gb);
            }
        }
        return line;
    }

    ~ProfileCounters()
    {
        if (!enabled || gpu_s == 0)
        {
            return;
        }
        try
        {
            std::println(stderr,
                         "cuda-profile: upload={:.2f}s gpu={:.2f}s unpack={:.2f}s "
                         "| {} launches covering {} nodes",
                         part_stage_s + adv_stage_s + find_stage_s + lfind_stage_s,
                         gpu_s, unpack_s, launches, gpu_nodes);
            std::println(stderr,
                         "cuda-upload-decomp: gh={:.2f}s root_stage={:.2f}s "
                         "part_stage={:.2f}s adv_stage={:.2f}s find_stage={:.2f}s "
                         "lfind_stage={:.2f}s gpu_wait={:.2f}s "
                         "bins_upload={:.2f}s fin_wait={:.2f}s fin_d2h={:.2f}s "
                         "find_kern={:.2f}s find_d2h={:.2f}s",
                         gh_upload_s, root_stage_s, part_stage_s, adv_stage_s,
                         find_stage_s, lfind_stage_s, gpu_wait_s, bins_upload_s,
                         fin_wait_s, fin_d2h_s, find_kern_s, find_d2h_s);
            std::println(stderr,
                         "cuda-round-decomp: root_sums={:.2f}s root_hist={:.2f}s "
                         "adv_memset={:.2f}s adv_hist={:.2f}s "
                         "fin_stamp={:.2f}s fin_map={:.2f}s",
                         root_sums_s, root_hist_s, adv_memset_s, adv_hist_s,
                         fin_stamp_s, fin_map_s);
            std::println(stderr, "cuda-level-decomp:{}", level_hist_line());
            std::println(stderr, "cuda-part-decomp: kernel={:.3f}s", part_kernel_s);
            std::println(stderr,
                         "cuda-resident-decomp: obj_kernel={:.2f}s "
                         "score_kernel={:.2f}s eval_kernel={:.2f}s",
                         obj_kernel_s, score_kernel_s, eval_kernel_s);
        }
        catch (...)
        {
            std::fputs("cuda-profile: failed to format profile line\n", stderr);
        }
    }
};

} // namespace cuda_detail
} // namespace bonsai
