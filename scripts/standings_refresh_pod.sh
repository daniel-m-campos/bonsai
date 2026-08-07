#!/usr/bin/env bash
# On-pod half of the standings refresh (decision 92). Invoked by
# scripts/standings_refresh.py's measure phase (decision 96); manual use:
#   AXES=gpu-tall,gpu-wide GIT_SHA=<sha> PREV_VERSION=1.6.1 bash scripts/standings_refresh_pod.sh
#   AXES=cpu-tall,cpu-wide GIT_SHA=<sha> PLANE=cpu bash scripts/standings_refresh_pod.sh
# Writes dated standings jsonl + the A/B rows to /root/standings/.
#
# One axis, one bundled spec of the same name, one dated output file
# (<axis>-YYYY-MM.jsonl): the scenario matrix of decision 103.
#
# BONSAI_BENCH_DATA_CACHE is not exported here: its memmap reads happen
# inside fit(), not just data generation, so it changes measured fit_s
# rather than only speeding up regeneration (12% slower for bonsai
# leafwise at 16M, and non-uniformly across libraries since only some of
# them re-touch the raw arrays after ingest).
#
# PLANE picks which half of the matrix this pod measures. A gpu pod builds
# python-cuda and runs the device axes; a cpu pod builds the plain python
# module and runs the cpu axes behind the headroom gate below (issue #355).
set -euo pipefail

AXES="${AXES:?comma list of axes}"
GIT_SHA="${GIT_SHA:?commit to measure}"
PREV_VERSION="${PREV_VERSION:-}"
PLANE="${PLANE:-gpu}"
# The tag for the rows this pod writes. The driver passes the rental it
# bought (flavor and vcpu count) for a cpu pod; a gpu pod reads its device.
HOST_TAG="${HOST_TAG:-}"

export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
mkdir -p /root/standings

if [ ! -d /root/bonsai ]; then
    git clone https://github.com/daniel-m-campos/bonsai.git /root/bonsai
fi
cd /root/bonsai
git fetch origin "$GIT_SHA" && git checkout -f "$GIT_SHA"

if [ "$PLANE" = cpu ]; then
    make python PYTHON=/opt/venv/bin/python
    BUILD=/root/bonsai/build/python
    [ -n "$HOST_TAG" ] || HOST_TAG="cpupod-$(nproc)vcpu"
else
    make python-cuda PYTHON=/opt/venv/bin/python
    BUILD=/root/bonsai/build-cuda/python
    [ -n "$HOST_TAG" ] || HOST_TAG="pod-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')"
fi

YM=$(date -u +%Y-%m)
BENCH=(env PYTHONPATH="$BUILD" /opt/venv/bin/python -m bonsai.bench)
# OMP_WAIT_POLICY=passive only on the CPU plane: a spin-wait barrier spends
# on waiting whatever the cap withholds, and the device planes are measured
# without it, so the flag never silently crosses into a gpu axis.
CPU_BENCH=(env OMP_WAIT_POLICY=passive PYTHONPATH="$BUILD" \
    /opt/venv/bin/python -m bonsai.bench)
SPECS=/root/bonsai/python/bonsai/bench/specs
QUOTA_FAIL=/root/standings/quota-fail.txt
# The sizing rule, enforced here as well as at rental time: a cpu axis needs
# 1.5 CPUs per thread it asks for. bonsai's barriers spin, so every thread
# stays runnable through a barrier and the process draws against the cap even
# while it waits; at threads == cap that draw is the cap, and the timing
# describes the container. Headroom is what keeps the fit off the ceiling.
CPU_HEADROOM=1.5
# Throttled share of CFS enforcement periods around the axis's first fit that
# makes the rest of the axis unpublishable. A rented cpu pod caps by cpuset
# and cannot throttle at all (nr_periods stays 0); a gpu pod caps by CFS
# bandwidth, and a 16-thread fit on its 13.6-core quota measured 97%
# throttled (issue #355 step 18). The gate covers both.
THROTTLED_PCT_MAX=5
FAILURES=0

# Same-pod A/B: previous release wheel (no PYTHONPATH) vs HEAD build,
# interleaved per rep so pod thermal/state drift cannot masquerade as a
# code delta. This is the perf-change detector.
if [ "$PLANE" = gpu ] && [ -n "$PREV_VERSION" ]; then
    uv pip install -q --python /opt/venv/bin/python "bonsai-gbt==$PREV_VERSION"
    : > /root/standings/ab.jsonl
    for rep in 1 2; do
        for grower in cuda_depthwise cuda_levelwise; do
            for cell in "1000000 100" "16000000 100"; do
                set -- $cell
                /opt/venv/bin/python scripts/standings_ab.py \
                    --rows "$1" --cols "$2" --grower "$grower" --arm old \
                    >> /root/standings/ab.jsonl
                env PYTHONPATH=/root/bonsai/build-cuda/python \
                    /opt/venv/bin/python scripts/standings_ab.py \
                    --rows "$1" --cols "$2" --grower "$grower" --arm new \
                    >> /root/standings/ab.jsonl
            done
        done
    done
fi

# Ingest/train parity: the same anchor cell through bonsai's fused call and
# through the two-step Dataset + train form the runner uses. The published
# split is only honest while these agree, because a two-step Dataset without
# its device hint bins on the host and posts an ingest number for a pipeline
# no cuda grower runs. One process per arm keeps peak RSS per-arm, and two
# interleaved reps keep pod drift out of the comparison. The gpu pod owns
# this file: a cpu session must not land a skipped-only parity.jsonl on top
# of the real one when both sessions pull into the same directory.
if [ "$PLANE" = gpu ]; then
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

# The extreme axis generates a 2^36-cell float32 input, ~275GB resident. A
# pod without the RAM must say so instead of dying halfway through a
# three-hour sweep.
EXTREME_RAM_GB=320

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

# The cores this container may actually use: the tighter of the bandwidth
# quota and the cpuset. Two caps exist and a host carries either. A gpu pod
# caps bandwidth (13.6 cores against the 128 nproc advertises). A rented cpu
# pod caps the cpuset instead: its cpu.max reads "max", and nproc reports the
# vcpu count that was purchased, so the quota alone would call it unlimited.
cpu_allowance() {
    allowed=$(nproc)
    q=$(quota_cores)
    [ "$q" = unlimited ] && { echo "$allowed"; return; }
    awk -v a="$allowed" -v q="$q" 'BEGIN {printf "%.2f", (q < a ? q : a)}'
}

cpu_stat_path() {
    if [ -r /sys/fs/cgroup/cpu.stat ]; then echo /sys/fs/cgroup/cpu.stat
    else echo /sys/fs/cgroup/cpu/cpu.stat; fi
}

# "<nr_periods> <nr_throttled>", zeros when the controller exposes neither.
throttle_counters() {
    stat=$(cpu_stat_path)
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
            ram_gb=$(free -g | awk '/^Mem:/ {print $2}')
            if [ "$ram_gb" -lt "$EXTREME_RAM_GB" ]; then
                echo "SKIP gpu-extreme: host has ${ram_gb}GB RAM, the 2^36-cell f32 input needs >= ${EXTREME_RAM_GB}GB"
                return 0
            fi
            run_spec gpu-extreme ;;
        cpu-tall|cpu-wide)
            run_cpu_axis "$1" ;;
        gpu-tall|gpu-wide|gpu-early-stop)
            run_spec "$1" ;;
        *)
            echo "unknown axis: $1" >&2; exit 2 ;;
    esac
}

run_spec() {
    "${BENCH[@]}" run --spec "$1" \
        --out "/root/standings/$1-$YM.jsonl" \
        --run-label "standings-$1-$YM" --host-name "$HOST_TAG"
}

# The headroom gate. The spec's thread count is the published claim, so it
# is never adapted: either the rental has 1.5 CPUs per thread or the axis
# fails and is re-run on a bigger one. Sizing the rental is the driver's job;
# this is the assertion that the rental is what the driver thinks it bought.
run_cpu_axis() {
    axis=$1
    out="/root/standings/$axis-$YM.jsonl"
    spec_threads=$(spec_field "$axis" "max(s.get('threads', [16]))")
    allowance=$(cpu_allowance)
    required=$(awk -v t="$spec_threads" -v h="$CPU_HEADROOM" \
        'BEGIN {printf "%d", (t * h == int(t * h)) ? t * h : int(t * h) + 1}')
    echo "QUOTA $axis: allowance=${allowance} cores (quota=$(quota_cores), nproc=$(nproc)), spec=${spec_threads}t, needs>=${required}"
    if awk -v a="$allowance" -v r="$required" 'BEGIN {exit !(a < r)}'; then
        fail_axis "$axis" "needs ${required} cores for its ${spec_threads}t claim (${CPU_HEADROOM} per thread) but this container may use ${allowance}; rent a wider cpu pod (--cpu-vcpu) and re-run the axis"
        return 0
    fi

    # The first fit is measured with the throttle counters bracketing it;
    # the rest of the axis resumes against the same file, so the probe row
    # is a real standings row and costs nothing extra.
    probe=$(spec_field "$axis" "s['variants'][0]")
    read -r p0 t0 <<< "$(throttle_counters)"
    "${CPU_BENCH[@]}" run --spec "$axis" --variants "$probe" --repeats 1 \
        --out "$out" --run-label "standings-$axis-$YM" --host-name "$HOST_TAG"
    read -r p1 t1 <<< "$(throttle_counters)"
    pct=$(awk -v p="$((p1 - p0))" -v t="$((t1 - t0))" \
        'BEGIN {printf "%.1f", p > 0 ? 100 * t / p : 0}')
    echo "QUOTA $axis: throttled ${pct}% of $((p1 - p0)) enforcement periods around the first fit"
    if awk -v pct="$pct" -v max="$THROTTLED_PCT_MAX" 'BEGIN {exit !(pct > max)}'; then
        fail_axis "$axis" "throttled in ${pct}% of CFS periods at ${spec_threads}t (limit ${THROTTLED_PCT_MAX}%); this host cannot time the cpu plane"
        return 0
    fi
    "${CPU_BENCH[@]}" run --spec "$axis" --out "$out" \
        --run-label "standings-$axis-$YM" --host-name "$HOST_TAG"
}

IFS=',' read -ra AXIS_LIST <<< "$AXES"
for axis in "${AXIS_LIST[@]}"; do
    run_axis "$(echo "$axis" | tr -d ' ')"
done
# DONE always prints, even after a quota failure: the driver polls for it,
# and a run that ends without it looks like a dead pod. The exit status and
# the marker file carry the verdict.
echo "STANDINGS_REFRESH_DONE axes=$AXES ym=$YM host=$HOST_TAG failures=$FAILURES"
[ "$FAILURES" -eq 0 ]
