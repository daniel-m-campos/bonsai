# E7. The lane finder

The split finder spends its life dividing: two IEEE divisions per histogram cell, one feature at a time. This case is the round that made it scan eight features in lockstep on vector lanes, bit-identical to the scalar scan, and it is a lesson in SIMD using that finder as the worked example, since every concept shows up there.

## The idea: one instruction, several numbers

A normal (scalar) instruction does one arithmetic operation: `c = a / b` on one double. A SIMD instruction (single instruction, multiple data) does the same operation on a small fixed-size array of numbers held in one wide register. The slots in that array are called lanes. The hardware does the lanes at once, so an 8-lane divide costs about as much as one or two scalar divides, not eight.

Register widths decide how many lanes you get:

| instruction set | register width | lanes of double | lanes of float |
|---|---|---|---|
| SSE2 (every x86-64) | 128 bits | 2 | 4 |
| AVX2 (Haswell 2013 and later, all EPYCs) | 256 bits | 4 | 8 |
| AVX-512 (Zen 4 and later, some Intel) | 512 bits | 8 | 16 |
| NEON (every arm64, including the M2) | 128 bits | 2 | 4 |

The catch is that a compiler only emits instructions for the sets it is told the CPU has. The wheels are built for generic x86-64, which promises only SSE2, so by default clang could use 2-lane doubles and nothing wider. That is why AVX2 needed a runtime dispatch; more on that below.

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

Each lane is an ordinary IEEE double doing an ordinary IEEE operation. That is the whole basis of the bit-identity claim: lane 3 computing `t*t/d` produces exactly the double the scalar code produces for that feature, because it is the same operation on the same inputs, just scheduled beside seven others. The only way to lose that is if the compiler fuses `a*b+c` into one rounded FMA in one path and not the other, which is why the build sets `-ffp-contract=off`.

You declared eight lanes but the machine may have two or four. The compiler splits the type: on NEON an 8-wide `lane_d` becomes four 2-lane registers and `c = a / b` becomes four `fdiv v.2d` instructions, which is exactly the "24 fdiv.2d" count the round kept checking in the assembly. On AVX2 it becomes two `vdivpd ymm`. Writing the code at 8 lanes lets one source serve every target.

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

The scalar scan's cost was not arithmetic volume but dependency. Per cell it did two divisions and a chain of compares; a divide has a latency around 13 cycles and the next decision waited on it. On the EPYC that works out to 17 cycles per cell, with the divider busy for 9 of them and idle otherwise. Lanes fix this by giving the divider eight independent features to work on per step, so throughput, not latency, sets the pace. On the 4564P with AVX2 that read 3.20 to 1.76 ns per cell.

The SSE2 draft lost for a reason worth remembering. SSE2 has no blend instruction. The compiler emulates every select as three operations (and, and-not, or) per 2-lane register, so an 8-lane select costs twelve instructions. The lane kernel then issued more micro-ops per cell than the scalar scan, and under SMT, where two threads share one core's execution units, the scalar scan's idle latency was being filled by its sibling thread anyway. So the lane version was a wash alone and 15% slower paired. The lesson: SIMD pays when the vector instructions exist natively; emulated ones can cost more than the scalar code they replace.

**Runtime dispatch.** Because the wheel must still run on a pre-2013 CPU, the kernel is compiled twice in spirit:

```cpp
#ifdef __x86_64__
#define BONSAI_LANE_TARGET __attribute__((target("avx2")))
#endif

template <bool TwoDirs>
BONSAI_LANE_TARGET void scan_lanes(...) { scan_lanes_body<TwoDirs>(...); }
```

The `target("avx2")` attribute tells clang "compile this one function as if AVX2 were available". The helpers carry `[[gnu::always_inline]]` so their bodies get pulled into that function and compiled under the same permission; a non-inlined helper would stay at the SSE2 baseline. Then at runtime:

```cpp
static bool const avx2 = __builtin_cpu_supports("avx2");
```

asks the CPU once (it executes the `cpuid` instruction behind the scenes) and `scans_in_lanes` routes features into lane groups only when the answer is yes; otherwise the old scalar scan runs. Executing an AVX2 instruction on a CPU without it is an illegal-instruction crash, so the check is not optional.

**The ABI trap.** CI caught this one: a function that takes an 8-wide double by value would pass it in a 512-bit register, which does not exist at the generic baseline. Clang refuses with `-Wpsabi` because two translation units compiled with different targets would disagree on where the argument lives. Passing by reference sidesteps it, and after inlining the reference costs nothing.

**The gather.** One thing SIMD does badly: loading eight values that live at eight unrelated addresses. The lane kernel's eight features' histograms sit in different arena runs, so each step does eight scalar loads and inserts them into a vector. That is a real cost, and the reason the tall cell regressed when a range held only three features: five lanes still paid for loads and math on padding. Hence the rule that a range carries at least eight features.

**How to see what you got.** The habit that caught every problem this round: compile the one file to assembly and count instructions.

```
clang++ ... -arch x86_64 -S src/split.cpp -o split.s
grep -c 'vdivpd.*ymm' split.s     # 4-lane divides present: AVX2 took effect
grep -c 'divsd' split.s           # scalar divides: the scalar path
```

If the counts do not change when you think you vectorized something, you did not. An Apple Silicon Mac can emit x86 assembly for its own SDK, which is how the SSE2 and `-Wpsabi` issues were caught without a Linux box.

If you want to play with it, the quickest experiment is a small harness that links `src/split.cpp` and prints nanoseconds per cell over a random histogram, so you can change `k_scan_lanes` to 4 or 16, or drop `-ffp-contract=off`, and watch both the speed and the hash move. The hash is `scripts/model_hash.py`, and the bit-identity claim is pinned in tree by the `lane-scan-matches-scalar-scan` test in [tests/unit/test_split_node.cpp](../../../tests/unit/test_split_node.cpp), which runs 400 random nodes through both finders and compares every gain as bits.

## The record

The finder is `scan_lanes` in [src/split.cpp](../../../src/split.cpp); the measured tables are decision 154, same-pod on two EPYC hosts: cpu-wide depthwise -20% and -26%, leafwise -13.5% and -27%, with the find bucket -31 to -53% and every model hash unchanged.
