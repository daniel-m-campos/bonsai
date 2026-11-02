# E7. The lane finder

This round made the split finder (the code that scores every candidate cut in each feature's histogram to pick the best split) score eight features at once with wide vector instructions. On the cpu-wide cell this cut depthwise training time 20% and 26% on two EPYC hosts with every model hash unchanged ([decision 154](../../decisions.md#154-the-host-node-finder-scans-eight-features-in-lockstep-on-vector-lanes-bit-identical-to-the-scalar-scan-and-the-level-step-spreads-each-nodes-find-across-the-team-adopted)). It is also a worked introduction to SIMD (single instruction, multiple data), which this page explains from scratch. The split finder does two IEEE divisions per histogram cell (one bin of one feature's histogram), and before this round it scanned one feature at a time.

## The idea: one instruction, several numbers

A normal (scalar) instruction does one arithmetic operation: `c = a / b` on one double. A SIMD instruction (single instruction, multiple data) does the same operation on a small fixed-size array of numbers held in one wide register. The slots in that array are called lanes. The hardware does the lanes at once, so an 8-lane divide costs about as much as one or two scalar divides, not eight.

Register widths decide how many lanes you get:

| instruction set | register width | lanes of double | lanes of float |
|---|---|---|---|
| SSE2 (every x86-64) | 128 bits | 2 | 4 |
| AVX2 (Haswell 2013 and later, all EPYCs) | 256 bits | 4 | 8 |
| AVX-512 (Zen 4 and later, some Intel) | 512 bits | 8 | 16 |
| NEON (every arm64, including the M2) | 128 bits | 2 | 4 |

A compiler emits only the instruction sets you tell it the CPU has. The wheels target generic x86-64, which guarantees only SSE2, so clang uses at most 2 lanes of double. AVX2 therefore needs a runtime check, described under "Runtime dispatch" later on this page.

## Vector types: telling the compiler the shape

Clang's extension lets you declare "eight doubles that live in registers":

```cpp
using lane_d = double __attribute__((ext_vector_type(8)));
using lane_i = int64_t __attribute__((ext_vector_type(8)));
```

A `lane_d` behaves like a value, and the ordinary operators mean "do this on every lane":

```cpp
lane_d a, b;
lane_d c = (a * a) / b;   // 8 multiplies, then 8 divides, lane by lane
c += 1.0;                 // a scalar broadcasts to all 8 lanes
```

Each lane is an ordinary IEEE double doing an ordinary IEEE operation. That is the whole basis of the bit-identity claim: lane 3 computing `t*t/d` produces exactly the double the scalar code produces for that feature, because it is the same operation on the same inputs, just scheduled beside seven others. The compiler could break this by fusing `a*b+c` into a single fused multiply-add (FMA) instruction, which rounds once instead of twice. If it fused in the lane path but not the scalar path, the two results would differ in the last bit. The build sets `-ffp-contract=off` so neither path fuses.

You declared eight lanes but the machine may have two or four. The compiler splits the type: on NEON an 8-wide `lane_d` becomes four 2-lane registers, so each source divide `c = a / b` compiles to four `fdiv v.2d` instructions. The round counted these in the assembly to confirm the vector path was taken. On AVX2 it becomes two `vdivpd ymm`. Writing the code at 8 lanes lets one source serve every target.

**Comparisons produce masks, not bools.** This is the part that feels strange at first:

```cpp
lane_i too_thin = hess_left < min_hess;
```

There is no single yes/no answer for eight lanes, so a comparison yields an integer vector of the same width where each lane is all ones (-1) if true and all zeros if false. Masks combine with `&`, `|`, `~` lane by lane. In the finder:

```cpp
lane_i const feasible = p.active & ~((hess_left < rule.min_hess) | (hess_right < rule.min_hess));
```

reads "the lane is live and neither child is below the hessian floor", eight answers at once.

**Branches become selects.** You cannot `if` on a mask, since some lanes would take each side. Instead you compute both outcomes and pick per lane:

```cpp
gain = better ? candidate : gain;   // per lane: keep the new gain where better is -1
```

The hardware instruction for this is a blend (`vblendvpd` on AVX2, `bsl` on NEON): take bytes from the second operand where the mask bits are set, from the third elsewhere. That is how `LaneBest::offer` records the best bin for each of eight features with no branch at all. The scalar code's `if (gain > best.gain) best = ...` became exactly this. The scalar scan's "first bin wins a tie" rule survives because the comparison is still strict `>`.

## Why this was the win, and why it lost on SSE2

The scalar scan's cost was not arithmetic volume but dependency. Per cell it did two divisions and a chain of compares; a divide has a latency (the cycles before its result is ready) of around 13 cycles and the next decision waited on it. On the EPYC that works out to 17 cycles per cell, with the divider busy for 9 of them and idle otherwise. Lanes fix this by giving the divider eight independent features to work on per step. The divider's throughput (how many divisions it finishes per cycle) now sets the pace, and the latency of any one divide no longer does. On the EPYC 4564P with AVX2 this cut the scan from 3.20 to 1.76 ns per cell (-45%, one thread, hot cache; [decision 154](../../decisions.md#154-the-host-node-finder-scans-eight-features-in-lockstep-on-vector-lanes-bit-identical-to-the-scalar-scan-and-the-level-step-spreads-each-nodes-find-across-the-team-adopted)).

The SSE2 draft lost for a reason worth remembering. SSE2 has no blend instruction. The compiler emulates every select as three operations (and, and-not, or) per 2-lane register, so an 8-lane select costs twelve instructions. The lane kernel then issued more micro-ops per cell than the scalar scan, and under SMT, where two threads share one core's execution units, the scalar scan's idle latency was being filled by its sibling thread anyway. So the SSE2 lane version matched the scalar scan on one thread and ran 15% slower with 12 threads, where threads share cores. The lesson: SIMD pays when the vector instructions exist natively; emulated ones can cost more than the scalar code they replace.

**Runtime dispatch.** Because the wheel must still run on a pre-2013 CPU, the build contains two paths, the AVX2 lane kernel and the older scalar scan, and the program checks the CPU once and picks one:

```cpp
#ifdef __x86_64__
#define BONSAI_LANE_TARGET __attribute__((target("avx2")))
#endif

template <bool TwoDirs>
BONSAI_LANE_TARGET void scan_lanes(...) { scan_lanes_body<TwoDirs>(...); }
```

The `target("avx2")` attribute tells clang "compile this one function as if AVX2 were available". The helpers carry `[[gnu::always_inline]]` so their bodies get pulled into that function and compiled under the same permission; a non-inlined helper would stay at the SSE2 baseline. Then at runtime:

```cpp
inline bool lanes_available()
{
#ifdef __x86_64__
    static bool const avx2 = []
    {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return avx2;
#else
    return true;
#endif
}
```

On x86-64 this asks the CPU once (it executes the `cpuid` instruction behind the scenes), and `scans_in_lanes` routes features into lane groups only when the answer is yes; otherwise the old scalar scan runs. On arm64 the function returns true without asking, because NEON is part of the baseline. Executing an AVX2 instruction on an x86-64 CPU without it is an illegal-instruction crash, so the check is not optional.

**The ABI mismatch.** CI caught this one. The ABI (the calling convention that fixes where each argument lives) would pass an 8-wide double by value in a 512-bit register, which the generic x86-64 baseline lacks. Clang refuses with `-Wpsabi` because two source files built for different CPUs would disagree on where the argument is. Pass it by reference instead; after inlining that costs nothing.

**The gather.** One thing SIMD does badly: loading eight values that live at eight unrelated addresses. The lane kernel's eight features' histograms sit in different arena runs, so each step does eight scalar loads and inserts them into a vector. When a range held only three features, five lanes loaded and computed padding, and on the cpu-tall cell the find ran 58 to 64% slower than the scalar scan ([decision 154](../../decisions.md#154-the-host-node-finder-scans-eight-features-in-lockstep-on-vector-lanes-bit-identical-to-the-scalar-scan-and-the-level-step-spreads-each-nodes-find-across-the-team-adopted)), so each range now carries at least eight features.

**How to see what you got.** The habit that caught every problem this round: compile the one file to assembly and count instructions.

```
clang++ ... -arch x86_64 -S src/split.cpp -o split.s
grep -c 'vdivpd.*ymm' split.s     # 4-lane divides present: AVX2 took effect
grep -c 'divsd' split.s           # scalar divides: the scalar path
```

If the counts do not change when you think you vectorized something, you did not. An Apple Silicon Mac can emit x86 assembly for its own SDK, which is how the SSE2 and `-Wpsabi` issues were caught without a Linux box.

To experiment, write a harness that links `src/split.cpp` and prints nanoseconds per cell over a random histogram. Change `k_scan_lanes` to 4 or 16, or remove `-ffp-contract=off`, and watch both the timing and the hash change. The hash comes from `scripts/model_hash.py`. The `lane-scan-matches-scalar-scan` test in [tests/unit/test_split_node.cpp](../../../tests/unit/test_split_node.cpp) pins bit-identity: it runs 400 random nodes through `find` and `find_parallel` and checks that each returns the same best split as a scalar reference scan written in the test, with the gain compared as bits and the feature, bin and missing-value direction compared exactly.

## The record

The finder is `scan_lanes` in [src/split.cpp](../../../src/split.cpp). Training time on the cpu-wide cell fell 20% and 26% (depthwise) and 13.5% and 27% (leafwise) on two EPYC hosts, with the time spent finding splits down 31 to 53% and every model hash unchanged ([decision 154](../../decisions.md#154-the-host-node-finder-scans-eight-features-in-lockstep-on-vector-lanes-bit-identical-to-the-scalar-scan-and-the-level-step-spreads-each-nodes-find-across-the-team-adopted)). Each host ran the old and new code on the same rented machine in one session. The tall cell moved less than 3%, since split finding is about 4% of its fit. An x86-64 CPU without AVX2 keeps the scalar scan and sees no change.
