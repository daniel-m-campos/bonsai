#!/usr/bin/env bash
# On-pod half of the standings refresh (decision 92). Invoked by
# scripts/standings_refresh.py's measure phase (decision 96); manual use:
#   AXES=gpu-tall,cpu-tall GIT_SHA=<sha> PREV_VERSION=1.6.1 bash scripts/standings_refresh_pod.sh
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
set -euo pipefail

AXES="${AXES:?comma list of axes}"
GIT_SHA="${GIT_SHA:?commit to measure}"
PREV_VERSION="${PREV_VERSION:-}"

export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
mkdir -p /root/standings

if [ ! -d /root/bonsai ]; then
    git clone https://github.com/daniel-m-campos/bonsai.git /root/bonsai
fi
cd /root/bonsai
git fetch origin "$GIT_SHA" && git checkout -f "$GIT_SHA"
make python-cuda PYTHON=/opt/venv/bin/python

YM=$(date -u +%Y-%m)
HOST_TAG="pod-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')"
BENCH=(env PYTHONPATH=/root/bonsai/build-cuda/python /opt/venv/bin/python -m bonsai.bench)

# Same-pod A/B: previous release wheel (no PYTHONPATH) vs HEAD build,
# interleaved per rep so pod thermal/state drift cannot masquerade as a
# code delta. This is the perf-change detector.
if [ -n "$PREV_VERSION" ]; then
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
# interleaved reps keep pod drift out of the comparison.
: > /root/standings/parity.jsonl
for rep in 1 2; do
    for arm in fused two_step; do
        [ "$arm" = fused ] && FUSED=--fused || FUSED=
        env PYTHONPATH=/root/bonsai/build-cuda/python \
            /opt/venv/bin/python scripts/standings_ab.py \
            --rows 16000000 --cols 100 --grower cuda_depthwise \
            --arm "$arm" --skip-without-cuda $FUSED \
            >> /root/standings/parity.jsonl
    done
done

# The extreme axis generates a 2^36-cell float32 input, ~275GB resident. A
# pod without the RAM must say so instead of dying halfway through a
# three-hour sweep.
EXTREME_RAM_GB=320

run_axis() {
    case "$1" in
        gpu-extreme)
            ram_gb=$(free -g | awk '/^Mem:/ {print $2}')
            if [ "$ram_gb" -lt "$EXTREME_RAM_GB" ]; then
                echo "SKIP gpu-extreme: host has ${ram_gb}GB RAM, the 2^36-cell f32 input needs >= ${EXTREME_RAM_GB}GB"
                return 0
            fi
            run_spec gpu-extreme ;;
        gpu-tall|gpu-wide|cpu-tall|cpu-wide|gpu-early-stop)
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

IFS=',' read -ra AXIS_LIST <<< "$AXES"
for axis in "${AXIS_LIST[@]}"; do
    run_axis "$(echo "$axis" | tr -d ' ')"
done
echo "STANDINGS_REFRESH_DONE axes=$AXES ym=$YM host=$HOST_TAG"
