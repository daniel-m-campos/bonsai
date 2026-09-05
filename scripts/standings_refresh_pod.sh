#!/usr/bin/env bash
# On-pod half of the standings refresh (decision 92). Invoked by
# scripts/standings_refresh.py's measure phase (decision 96); manual use:
#   AXES=gpu-tall,gpu-wide GIT_SHA=<sha> PREV_VERSION=1.6.1 ANCHOR_VERSION=1.15.0 bash scripts/standings_refresh_pod.sh
#   AXES=cpu-tall,cpu-wide GIT_SHA=<sha> PLANE=cpu ANCHOR_VERSION=1.15.0 bash scripts/standings_refresh_pod.sh
# Writes dated standings jsonl + the plane's A/B rows to /root/standings/.
#
# One axis, one bundled spec of the same name, one dated output file
# (<axis>-YYYY-MM.jsonl): the scenario matrix of decision 103.
#
# BONSAI_BENCH_DATA_CACHE is exported to a ram disk, which it was not while
# the cache handed back memmaps: those fault pages inside fit(), so they moved
# measured fit_s rather than only saving regeneration (12% for bonsai leafwise
# at 16M, unequally across libraries since only some re-touch the raw arrays
# after ingest). It now loads whole, before worker() opens a timer, and a cell
# too large to hold twice declines to the generator (decision 113). One draw
# per cell instead of one per variant and repeat.
#
# PLANE picks which half of the matrix this pod measures. A gpu pod builds
# python-cuda and runs the device axes; a cpu pod builds the plain python
# module and runs the cpu axes behind the CPU-cap gate below (issue #355).
set -euo pipefail

AXES="${AXES:?comma list of axes}"
# Stripped once so the parity guard and the axis loop read the same list.
AXES="${AXES// /}"
GIT_SHA="${GIT_SHA:?commit to measure}"
PREV_VERSION="${PREV_VERSION:-}"
ANCHOR_VERSION="${ANCHOR_VERSION:-}"
PLANE="${PLANE:-gpu}"
# The axis the parity rows anchor; standings_refresh.py's PARITY_AXIS.
PARITY_AXIS=gpu-tall
# The tag for the rows this pod writes, derived below from what this
# container actually resolved rather than from what was requested: a name
# built out of a purchase is how a 12-thread run got committed under a
# 16-thread tag. Settable by hand for a one-off session.
HOST_TAG="${HOST_TAG:-}"

export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
mkdir -p /root/standings
YM=$(date -u +%Y-%m)
QUOTA_FAIL=/root/standings/quota-fail.txt

if [ ! -d /root/bonsai ]; then
    git clone https://github.com/daniel-m-campos/bonsai.git /root/bonsai
fi
cd /root/bonsai

# `git fetch origin <sha>` asks the server for a bare object id, which GitHub
# refuses unless it is a ref tip, and a non-final command in an && list is
# exempt from set -e. So the old one-liner printed "couldn't find remote ref",
# never checked anything out, measured whatever the clone had landed on, and
# still reported failures=0. Fetch every branch and tag (the refspec is
# explicit because a single-branch clone's is not), then assert that HEAD is
# the commit that was asked for.
checkout_sha() {  # <commit-ish>; nonzero unless HEAD ends up there
    git fetch --tags --quiet origin \
        '+refs/heads/*:refs/remotes/origin/*' || return 1
    git checkout --quiet --force "$1" || return 1
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "$1^{commit}")" ] || return 1
}

if ! checkout_sha "$GIT_SHA"; then
    # The driver polls for DONE and learns of the end no other way, so the
    # abort prints one too, carrying the failure that makes it an abort.
    echo "STANDINGS_REFRESH_FAIL checkout: asked for $GIT_SHA, HEAD is $(git rev-parse --short HEAD); a row measured at an unverified commit is not publishable"
    echo "checkout: $GIT_SHA was never checked out" >> "$QUOTA_FAIL"
    echo "STANDINGS_REFRESH_DONE axes=$AXES ym=$YM host=${HOST_TAG:-unknown} failures=1"
    exit 1
fi
echo "MEASURING $(git rev-parse HEAD)"

if [ "$PLANE" = cpu ]; then
    make python PYTHON=/opt/venv/bin/python
    BUILD=/root/bonsai/build/python
    # nproc is the cpuset the container was handed, which is what the fit
    # can actually run on; the rented vCPU count is only a request.
    [ -n "$HOST_TAG" ] || HOST_TAG="cpupod-$(nproc)cpu"
else
    make python-cuda PYTHON=/opt/venv/bin/python
    BUILD=/root/bonsai/build-cuda/python
    [ -n "$HOST_TAG" ] || HOST_TAG="pod-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')"
fi

DATA_CACHE=/dev/shm/bonsai-bench-data
mkdir -p "$DATA_CACHE"
# BONSAI_BENCH_GIT_SHA states the commit rather than letting the runner infer
# it: runlog.git_sha() shells out to `git rev-parse` in the CURRENT directory,
# so a runner launched from anywhere but the checkout records "unknown" and the
# row cannot be attributed. This script already asserted HEAD is GIT_SHA above.
BENCH=(env PYTHONPATH="$BUILD" BONSAI_BENCH_DATA_CACHE="$DATA_CACHE" \
    BONSAI_BENCH_GIT_SHA="$GIT_SHA" \
    /opt/venv/bin/python -m bonsai.bench)
# OMP_WAIT_POLICY=passive only on the CPU plane: a spin-wait barrier spends
# on waiting whatever the cap withholds, and the device planes are measured
# without it, so the flag never silently crosses into a gpu axis.
CPU_BENCH=(env OMP_WAIT_POLICY=passive PYTHONPATH="$BUILD" \
    BONSAI_BENCH_DATA_CACHE="$DATA_CACHE" BONSAI_BENCH_GIT_SHA="$GIT_SHA" \
    /opt/venv/bin/python -m bonsai.bench)
SPECS=/root/bonsai/python/bonsai/bench/specs
# A bandwidth host must leave one core of its quota unclaimed, because a fit
# at threads == quota sits on the ceiling and the timing describes the
# container rather than the engine. A larger static margin was tried and
# dropped: no rental reliably offers one, twelve consecutive draws across two
# gpu families all metered 13.6 cores, and the throttle counters below
# measure the thing the margin was guessing at.
QUOTA_SPARE_CORES=1
# Throttled share of CFS enforcement periods around the axis's first fit that
# makes the rest of the axis unpublishable. A rented cpu pod caps by cpuset
# and cannot throttle at all (nr_periods stays 0); a gpu pod caps by CFS
# bandwidth, and a 16-thread fit on its 13.6-core quota measured 97%
# throttled (issue #355 step 18). The gate covers both.
THROTTLED_PCT_MAX=5
FAILURES=0

# Same-pod A/B: every arm once per rep, reps interleaved, one venv per
# wheel so no arm displaces another, one file per plane.
AB_ARMS=()
if [ -n "$ANCHOR_VERSION" ]; then AB_ARMS+=(anchor); fi
if [ -n "$PREV_VERSION" ]; then AB_ARMS+=(old); fi
AB_ARMS+=(new)
AB_GPU_CELLS=("1000000 100" "16000000 100")
AB_GPU_GROWERS=(cuda_depthwise cuda_levelwise)
AB_GPU_REPS=2
AB_CPU_GROWERS=(levelwise depthwise leafwise)
AB_CPU_REPS=4
CPU_AB_AXIS=cpu-tall
if [ "$PLANE" = cpu ]; then AB_ENV=(env OMP_WAIT_POLICY=passive); else AB_ENV=(env); fi

wheel_python() {  # <version>; the interpreter of a venv holding that wheel
    venv="/root/venv-$1"
    if [ ! -x "$venv/bin/python" ]; then
        uv venv -q --python /opt/venv/bin/python "$venv" >&2
        uv pip install -q --python "$venv/bin/python" \
            "bonsai-gbt==$1" numpy psutil >&2
    fi
    echo "$venv/bin/python"
}

# A wheel arm imports from its venv (empty PYTHONPATH); new from this build.
ab_arms() {  # <out> <grower> <rows> <cols> <threads>
    for arm in "${AB_ARMS[@]}"; do
        case "$arm" in
            anchor) py=$(wheel_python "$ANCHOR_VERSION"); path= ;;
            old) py=$(wheel_python "$PREV_VERSION"); path= ;;
            new) py=/opt/venv/bin/python; path=$BUILD ;;
        esac
        "${AB_ENV[@]}" PYTHONPATH="$path" "$py" scripts/standings_ab.py \
            --rows "$3" --cols "$4" --grower "$2" --arm "$arm" \
            --threads "$5" >> "$1"
    done
}

if [ "$PLANE" = gpu ] && [ "${#AB_ARMS[@]}" -gt 1 ]; then
    : > /root/standings/ab-gpu.jsonl
    for rep in $(seq "$AB_GPU_REPS"); do
        for grower in "${AB_GPU_GROWERS[@]}"; do
            for cell in "${AB_GPU_CELLS[@]}"; do
                set -- $cell
                ab_arms /root/standings/ab-gpu.jsonl "$grower" "$1" "$2" 16
            done
        done
    done
fi

run_cpu_ab() {  # <threads>
    [ "${#AB_ARMS[@]}" -gt 1 ] || return 0
    rows=$(spec_field "$CPU_AB_AXIS" "s['cells'][0]['rows']")
    cols=$(spec_field "$CPU_AB_AXIS" "s['cells'][0]['cols']")
    : > /root/standings/ab-cpu.jsonl
    for rep in $(seq "$AB_CPU_REPS"); do
        for grower in "${AB_CPU_GROWERS[@]}"; do
            ab_arms /root/standings/ab-cpu.jsonl "$grower" "$rows" "$cols" "$1"
        done
    done
}

# Ingest/train parity: the same anchor cell through bonsai's fused call and
# through the two-step Dataset + train form the runner uses. The published
# split is only honest while these agree, because a two-step Dataset without
# its device hint bins on the host and posts an ingest number for a pipeline
# no cuda grower runs. One process per arm keeps peak RSS per-arm, and two
# interleaved reps keep pod drift out of the comparison. The gpu pod owns
# this file: a cpu session must not land a skipped-only parity.jsonl on top
# of the real one when both sessions pull into the same directory.
#
# Only the session measuring the anchor axis takes these rows, for the same
# reason and one more. They are committed as that axis's companion, so a
# session that does not measure it would spend four runs at the anchor cell
# to produce a second host's parity for the same month, landing on top of
# the real one with nothing downstream able to tell them apart.
if [ "$PLANE" = gpu ] && [[ ",$AXES," == *",$PARITY_AXIS,"* ]]; then
    : > /root/standings/parity.jsonl
    for rep in 1 2; do
        for arm in fused two_step; do
            [ "$arm" = fused ] && FUSED=--fused || FUSED=
            env PYTHONPATH="$BUILD" \
                /opt/venv/bin/python scripts/standings_ab.py \
                --rows 16000000 --cols 100 --grower cuda_depthwise \
                --arm "$arm" --skip-without-cuda $FUSED \
                >> /root/standings/parity.jsonl
        done
    done
fi

# The extreme axis input is 2^34 f32 cells (~71GB). This floor is bonsai's
# requirement plus room to generate, not the fleet's: bonsai peaked at 66.7GB
# there, so a single 188GB card clears it. catboost wants 196.3GB and now
# publishes an OOM instead, which is a result rather than a gap, and the row
# carries the container size it OOMed at (decision 113). Sizing this floor to
# a competitor's appetite is what used to force a two-GPU draw at twice the
# rate. A pod under the floor still says so rather than dying halfway.
EXTREME_RAM_GB=150

# The RAM this container may use, in whole GB. runlog.usable_ram_gb() is the
# one implementation of the cgroup-limit-or-machine rule, including the v1
# unlimited sentinel; a second copy here is a rule that can disagree with the
# host block the rows carry.
container_ram_gb() {
    env PYTHONPATH="$BUILD" /opt/venv/bin/python \
        -c "from bonsai.bench import runlog; print(runlog.usable_ram_gb())"
}

# The cgroup CPU-bandwidth ceiling in cores, or "unlimited". v2 first, v1
# fallback; anything unreadable is unlimited, which is what bare metal is.
quota_cores() {
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r q p < /sys/fs/cgroup/cpu.max
        [ "$q" = max ] && { echo unlimited; return; }
        awk -v q="$q" -v p="$p" 'BEGIN {printf "%.2f", q / p}'
        return
    fi
    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$q" -le 0 ] && { echo unlimited; return; }
        awk -v q="$q" -v p="$p" 'BEGIN {printf "%.2f", q / p}'
        return
    fi
    echo unlimited
}

# "<nr_periods> <nr_throttled>", zeros when the controller exposes neither.
throttle_counters() {
    stat=/sys/fs/cgroup/cpu.stat
    [ -r "$stat" ] || stat=/sys/fs/cgroup/cpu/cpu.stat
    [ -r "$stat" ] || { echo "0 0"; return; }
    awk '/^nr_periods /{p=$2} /^nr_throttled /{t=$2} END {print (p?p:0), (t?t:0)}' "$stat"
}

spec_field() {  # <axis> <expr over the loaded spec dict s>
    /opt/venv/bin/python -c "import json,sys; s=json.load(open(sys.argv[1])); print($2)" \
        "$SPECS/$1.json"
}

# Loud, recoverable, and unpublishable: the marker travels back with the
# results, and the partial rows are renamed out of the supersession glob so
# no one can build a standings PR out of a throttled sweep.
fail_axis() {  # <axis> <why>
    echo "STANDINGS_REFRESH_FAIL $1: $2"
    echo "$1: $2" >> "$QUOTA_FAIL"
    partial="/root/standings/$1-$YM.jsonl"
    [ -f "$partial" ] && mv "$partial" "/root/standings/QUOTAFAIL-$1-$YM.jsonl"
    FAILURES=$((FAILURES + 1))
}

run_axis() {
    case "$1" in
        gpu-extreme)
            ram_gb=$(container_ram_gb)
            if [ "$ram_gb" -lt "$EXTREME_RAM_GB" ]; then
                echo "SKIP gpu-extreme: this container may use ${ram_gb}GB RAM, below the ${EXTREME_RAM_GB}GB floor (bonsai peaks near 66.7GB host on the 2^34-cell input, plus room to generate it)"
                return 0
            fi
            run_spec gpu-extreme ;;
        cpu-tall|cpu-wide)
            run_cpu_axis "$1" ;;
        gpu-tall|gpu-wide|gpu-early-stop|gpu-shap)
            run_spec "$1" ;;
        quality-grinsztajn)
            run_grinsztajn ;;
        *)
            fail_axis "$1" "no branch measures this axis; the driver asked for something this script cannot run"
            return 0 ;;
    esac
}

# The quality division has no spec: it fits a fixed dataset suite rather than
# a cell grid, so it takes its own module and an output path instead of
# run_spec's --spec/--out pair. Same env as BENCH, spelled out rather than
# suffixed onto it, because the array-suffix idiom that would reuse it reads
# as a typo.
run_grinsztajn() {
    env PYTHONPATH="$BUILD" BONSAI_BENCH_DATA_CACHE="$DATA_CACHE" \
        BONSAI_BENCH_GIT_SHA="$GIT_SHA" \
        /opt/venv/bin/python -m bonsai.bench.grinsztajn \
        "/root/standings/quality-grinsztajn-$YM.jsonl"
}

run_spec() {
    "${BENCH[@]}" run --spec "$1" \
        --out "/root/standings/$1-$YM.jsonl" \
        --run-label "standings-$1-$YM" --host-name "$HOST_TAG"
}

# The CPU-cap gate. The spec's thread count is the published claim, so it is
# never adapted: either the container can honour it or the axis fails and is
# re-run on a bigger rental. Sizing is the driver's job; this asserts that the
# rental is what the driver thinks it bought.
#
# Which cap the host enforces decides what the axis needs, because it decides
# whether a spinning barrier costs anything. A cpuset host hands the container
# whole CPUs, so one CPU per thread is enough and a spinning thread only burns
# the core it already owns. A bandwidth host meters core-seconds across all
# threads, so spin-wait eats the shared allowance and the quota has to exceed
# the thread count, with the measured throttle share below as the real check.
run_cpu_axis() {
    axis=$1
    out="/root/standings/$axis-$YM.jsonl"
    spec_threads=$(spec_field "$axis" "max(s.get('threads', [16]))")
    quota=$(quota_cores)
    usable=$(nproc)
    echo "CPUCAP $axis: quota=${quota}, usable=${usable} cpus, spec=${spec_threads}t"
    if [ "$quota" = unlimited ]; then
        if [ "$usable" -lt "$spec_threads" ]; then
            fail_axis "$axis" "no bandwidth quota, so the cpuset is the cap, and its ${usable} cpus are fewer than the ${spec_threads} threads the spec claims; rent a wider cpu pod (--cpu-vcpu) and re-run the axis"
            return 0
        fi
    else
        required=$(awk -v t="$spec_threads" -v s="$QUOTA_SPARE_CORES" \
            'BEGIN {printf "%.2f", t + s}')
        if awk -v q="$quota" -v r="$required" 'BEGIN {exit !(q < r)}'; then
            fail_axis "$axis" "this host meters cpu bandwidth at ${quota} cores, below the ${required} its ${spec_threads}t claim needs (one spare core, so the fit is not pinned to the ceiling); draw another pod or lower the spec's thread count"
            return 0
        fi
    fi

    # The first fit is measured with the throttle counters bracketing it;
    # the rest of the axis resumes against the same file, so the probe row
    # is a real standings row and costs nothing extra.
    probe=$(spec_field "$axis" "s['variants'][0]")
    read -r p0 t0 <<< "$(throttle_counters)"
    "${CPU_BENCH[@]}" run --spec "$axis" --variants "$probe" --repeats 1 \
        --out "$out" --run-label "standings-$axis-$YM" --host-name "$HOST_TAG"
    read -r p1 t1 <<< "$(throttle_counters)"
    # The ternary is parenthesized because `printf fmt, e > x` is redirection
    # in awk, not comparison: unparenthesized, this wrote the period count to
    # a file named 0 and left pct empty, and an empty pct compares as a string
    # and passes, so the gate never fired.
    pct=$(awk -v p="$((p1 - p0))" -v t="$((t1 - t0))" \
        'BEGIN {printf "%.1f", (p > 0 ? 100 * t / p : 0)}')
    echo "QUOTA $axis: throttled ${pct}% of $((p1 - p0)) enforcement periods around the first fit"
    case "$pct" in
        ''|*[!0-9.]*)
            fail_axis "$axis" "the throttle probe produced no percentage, so nothing was measured to gate on; a gate that cannot read its own counters must not pass the axis"
            return 0 ;;
    esac
    if awk -v pct="$pct" -v max="$THROTTLED_PCT_MAX" 'BEGIN {exit !(pct + 0 > max + 0)}'; then
        fail_axis "$axis" "throttled in ${pct}% of CFS periods at ${spec_threads}t (limit ${THROTTLED_PCT_MAX}%); this host cannot time the cpu plane"
        return 0
    fi
    "${CPU_BENCH[@]}" run --spec "$axis" --out "$out" \
        --run-label "standings-$axis-$YM" --host-name "$HOST_TAG"
    if [ "$axis" = "$CPU_AB_AXIS" ]; then
        run_cpu_ab "$spec_threads"
    fi
}

IFS=',' read -ra AXIS_LIST <<< "$AXES"
for axis in "${AXIS_LIST[@]}"; do
    run_axis "$axis"
done
# DONE always prints, even after a quota failure: the driver polls for it,
# and a run that ends without it looks like a dead pod. The exit status and
# the marker file carry the verdict.
echo "STANDINGS_REFRESH_DONE axes=$AXES ym=$YM host=$HOST_TAG failures=$FAILURES"
[ "$FAILURES" -eq 0 ]
