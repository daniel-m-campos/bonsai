// Stall decomposition for the CPU histogram fill (issue 355 step 3).
//
// Replicates run_fill's inner loop on synthetic data and isolates its three
// memory streams to name the per-core bound at each node size: strip reads
// (scattered 128B), grad/hess reads, and the histogram read-modify-write.
// An AVX-512 arm prices vectorizing the feature loop (8 cells per gather;
// legal because a row's per-feature targets are disjoint addresses).
//
// Single-threaded on purpose: the production loop parallelizes over nodes
// and blocks, so the per-core add rate is the quantity that decays with
// depth in the populate ledger. Build and drive with run.sh.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace
{

struct Cell
{
    float g = 0.0F;
    float h = 0.0F;
};

constexpr size_t k_cols = 128;
constexpr size_t k_bins = 256;

double now_s()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Ascending subset of size m drawn from [0, n): the shape of a deep node's
// row list (partition_rows is stable, so real lists are ascending).
std::vector<uint32_t> make_rows(size_t n, size_t m, uint64_t seed)
{
    std::mt19937_64                       rng(seed);
    std::vector<uint32_t>                 rows(m);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    if (m == n)
    {
        for (size_t i = 0; i < n; ++i)
        {
            rows[i] = static_cast<uint32_t>(i);
        }
        return rows;
    }
    // Reservoir-free: sample with replacement then dedup-adjust is biased;
    // instead pick a sorted random subset via stride jitter, close enough
    // for access-pattern purposes and O(m).
    double const stride = static_cast<double>(n) / static_cast<double>(m);
    for (size_t i = 0; i < m; ++i)
    {
        auto r  = static_cast<size_t>((static_cast<double>(i) + 0.5) * stride);
        auto j  = static_cast<size_t>(pick(rng) % std::max<size_t>(1, stride));
        rows[i] = static_cast<uint32_t>(std::min(n - 1, r + j));
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

struct Bench
{
    size_t                n;
    std::vector<uint8_t>  strip;   // n x k_cols
    std::vector<float>    grad;
    std::vector<float>    hess;
    std::vector<Cell>     hists;   // k_cols x k_bins
    std::vector<Cell *>   bases;   // per-feature cell base

    explicit Bench(size_t n_rows) : n(n_rows)
    {
        strip.resize(n * k_cols);
        grad.resize(n);
        hess.resize(n);
        std::mt19937_64 rng(42);
        for (auto &b : strip)
        {
            b = static_cast<uint8_t>(rng() & 0xFF);
        }
        for (size_t i = 0; i < n; ++i)
        {
            grad[i] = static_cast<float>((rng() & 0xFFFF)) * 1e-4F;
            hess[i] = 1.0F;
        }
        hists.resize(k_cols * k_bins);
        bases.resize(k_cols);
        for (size_t s = 0; s < k_cols; ++s)
        {
            bases[s] = hists.data() + (s * k_bins);
        }
    }

    void reset() { std::memset(hists.data(), 0, hists.size() * sizeof(Cell)); }

    double checksum() const
    {
        double t = 0.0;
        for (auto const &c : hists)
        {
            t += static_cast<double>(c.g) + static_cast<double>(c.h);
        }
        return t;
    }
};

// The production loop: strip + gh reads, per-feature scatter add, k_ahead
// prefetch as shipped.
void scalar_full(Bench &b, std::vector<uint32_t> const &rows)
{
    uint8_t const *const rm      = b.strip.data();
    Cell *const *const   bases   = b.bases.data();
    uint32_t const      *rp      = rows.data();
    size_t const         m       = rows.size();
    constexpr size_t     k_ahead = 16;
    for (size_t k = 0; k < m; ++k)
    {
        if (k + k_ahead < m)
        {
            size_t const pr = rp[k + k_ahead];
            __builtin_prefetch(rm + (pr * k_cols), 0, 0);
            __builtin_prefetch(rm + (pr * k_cols) + 64, 0, 0);
            __builtin_prefetch(&b.grad[pr], 0, 0);
            __builtin_prefetch(&b.hess[pr], 0, 0);
        }
        size_t const         r  = rp[k];
        uint8_t const *const rb = rm + (r * k_cols);
        float const          g  = b.grad[r];
        float const          h  = b.hess[r];
        for (size_t s = 0; s < k_cols; ++s)
        {
            Cell &c = bases[s][rb[s]];
            c.g += g;
            c.h += h;
        }
    }
}

// Strip stream alone: same scattered 128B reads, no histogram traffic.
uint64_t strip_only(Bench &b, std::vector<uint32_t> const &rows)
{
    uint8_t const *const rm  = b.strip.data();
    uint64_t             acc = 0;
    for (uint32_t const r : rows)
    {
        uint8_t const *const rb = rm + (static_cast<size_t>(r) * k_cols);
        for (size_t s = 0; s < k_cols; s += 8)
        {
            uint64_t v;
            std::memcpy(&v, rb + s, 8);
            acc += v;
        }
    }
    return acc;
}

// Histogram stream alone: adds with synthetic bins (no strip reads), so the
// scatter target's cache behavior is isolated.
void hist_only(Bench &b, std::vector<uint32_t> const &rows)
{
    Cell *const *const bases = b.bases.data();
    for (uint32_t const r : rows)
    {
        float const g   = b.grad[r];
        float const h   = b.hess[r];
        uint64_t    lcg = (static_cast<uint64_t>(r) * 2862933555777941757ULL) + 1;
        for (size_t s = 0; s < k_cols; ++s)
        {
            lcg     = (lcg * 2862933555777941757ULL) + 3037000493ULL;
            Cell &c = bases[s][(lcg >> 32) & 0xFF];
            c.g += g;
            c.h += h;
        }
    }
}

#if defined(__AVX512F__)
// AVX-512 arm: 8 features per iteration; one 64-bit gather lane carries one
// 8-byte cell, the (g, h) pair is broadcast as interleaved f32, and the
// scatter writes back. Per-row targets are disjoint, so lanes never collide.
void simd_full(Bench &b, std::vector<uint32_t> const &rows)
{
    uint8_t const *const rm    = b.strip.data();
    size_t const         m     = rows.size();
    uint32_t const      *rp    = rows.data();
    __m512i              base[k_cols / 8];
    for (size_t s = 0; s < k_cols; s += 8)
    {
        alignas(64) uint64_t a[8];
        for (size_t j = 0; j < 8; ++j)
        {
            a[j] = reinterpret_cast<uint64_t>(b.bases[s + j]);
        }
        base[s / 8] = _mm512_load_si512(a);
    }
    constexpr size_t k_ahead = 16;
    for (size_t k = 0; k < m; ++k)
    {
        if (k + k_ahead < m)
        {
            size_t const pr = rp[k + k_ahead];
            __builtin_prefetch(rm + (pr * k_cols), 0, 0);
            __builtin_prefetch(rm + (pr * k_cols) + 64, 0, 0);
        }
        size_t const         r  = rp[k];
        uint8_t const *const rb = rm + (r * k_cols);
        double               pair_d;
        float const          gh[2] = {b.grad[r], b.hess[r]};
        std::memcpy(&pair_d, gh, 8);
        __m512 const ghv = _mm512_castpd_ps(_mm512_set1_pd(pair_d));
        for (size_t s = 0; s < k_cols; s += 8)
        {
            __m128i const b8   = _mm_loadl_epi64(
                reinterpret_cast<__m128i const *>(rb + s));
            __m512i const bins = _mm512_cvtepu8_epi64(b8);
            __m512i const addr =
                _mm512_add_epi64(base[s / 8], _mm512_slli_epi64(bins, 3));
            __m512i cells = _mm512_i64gather_epi64(addr, nullptr, 1);
            __m512 cf = _mm512_add_ps(_mm512_castsi512_ps(cells), ghv);
            _mm512_i64scatter_epi64(nullptr, addr, _mm512_castps_si512(cf), 1);
        }
    }
}
#endif

double time_arm(char const *name, size_t n, size_t m, double adds,
                auto &&fn)
{
    double const t0 = now_s();
    fn();
    double const dt = now_s() - t0;
    std::printf("ARM %s n=%zu m=%zu adds=%.0f s=%.4f gadds=%.2f\n", name, n, m,
                adds, dt, adds / dt / 1e9);
    return dt;
}

} // namespace

int main(int argc, char **argv)
{
    size_t const n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2'097'152;
    Bench        b(n);
    size_t const sizes[] = {2'048, 8'192, 32'768, 131'072, 524'288, n};
    for (size_t const m0 : sizes)
    {
        if (m0 > n)
        {
            continue;
        }
        auto const   rows = make_rows(n, m0, 7);
        size_t const m    = rows.size();
        // Repeat small nodes so every arm-point runs >= ~200M adds.
        size_t const reps = std::max<size_t>(1, 200'000'000 / (m * k_cols));
        double const adds = static_cast<double>(m) * k_cols *
                            static_cast<double>(reps);
        b.reset();
        time_arm("scalar_full", n, m, adds,
                 [&] { for (size_t i = 0; i < reps; ++i) scalar_full(b, rows); });
        double const ref = b.checksum();
        uint64_t sink = 0;
        time_arm("strip_only", n, m, adds,
                 [&] { for (size_t i = 0; i < reps; ++i) sink += strip_only(b, rows); });
        b.reset();
        time_arm("hist_only", n, m, adds,
                 [&] { for (size_t i = 0; i < reps; ++i) hist_only(b, rows); });
#if defined(__AVX512F__)
        if (__builtin_cpu_supports("avx512f"))
        {
            b.reset();
            time_arm("simd_full", n, m, adds,
                     [&] { for (size_t i = 0; i < reps; ++i) simd_full(b, rows); });
            double const simd = b.checksum();
            if (std::abs(simd - ref) > 1e-3 * std::abs(ref))
            {
                std::printf("SIMD_MISMATCH ref=%.6f simd=%.6f\n", ref, simd);
                return 1;
            }
        }
#endif
        std::printf("OK m=%zu sink=%llu ref=%.3f\n", m,
                    static_cast<unsigned long long>(sink & 1), ref);
    }
    std::puts("STALL_BENCH_DONE");
    return 0;
}
