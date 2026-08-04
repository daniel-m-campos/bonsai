#!/usr/bin/env bash
# On-pod half of the standings refresh (decision 92). Invoked by
# scripts/standings_refresh.py's measure phase (decision 96); manual use:
#   AXES=rows,width GIT_SHA=<sha> PREV_VERSION=1.5.4 bash scripts/standings_refresh_pod.sh
# Writes dated standings jsonl + the A/B rows to /root/standings/.
#
# The width axis runs the standings-cols spec, which drops the 16384-col
# CPU arms (lgbm_cpu, bonsai_oblivious): those two cells pin the wall clock
# at several minutes each with no GPU-side change able to move them. Set
# WIDTH_FULL=1 for a release refresh to run standings-cols-full instead,
# which keeps them; the output file is the same name either way.
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
WIDTH_FULL="${WIDTH_FULL:-}"

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
        for grower in cuda_depthwise cuda_oblivious; do
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

run_axis() {
    case "$1" in
        rows)
            "${BENCH[@]}" run --spec standings-rows \
                --out "/root/standings/rebaseline-$YM.jsonl" \
                --run-label "standings-rows-$YM" --host-name "$HOST_TAG" ;;
        width)
            WIDTH_SPEC="standings-cols"
            [ -n "$WIDTH_FULL" ] && WIDTH_SPEC="standings-cols-full"
            "${BENCH[@]}" run --spec "$WIDTH_SPEC" \
                --out "/root/standings/cols-rebaseline-$YM.jsonl" \
                --run-label "standings-cols-$YM" --host-name "$HOST_TAG" ;;
        shape)
            "${BENCH[@]}" run --spec iso-volume-2026-08 \
                --out "/root/standings/iso-volume-$YM.jsonl" \
                --run-label "standings-shape-$YM" --host-name "$HOST_TAG"
            "${BENCH[@]}" run --spec iso-volume-33-2026-08 \
                --out "/root/standings/iso-volume-$YM.jsonl" \
                --run-label "standings-shape-33-$YM" --host-name "$HOST_TAG" ;;
        frontier)
            "${BENCH[@]}" run --spec gpu-pareto-16M \
                --out "/root/standings/gpu-pareto-16M-$YM.jsonl" \
                --run-label "standings-frontier-$YM" --host-name "$HOST_TAG" ;;
        airline)
            env PYTHONPATH=/root/bonsai/build-cuda/python \
                /opt/venv/bin/python -m bonsai.bench.airline \
                "/root/standings/airline-$YM.jsonl" --sizes 0.1m,1m,10m \
                --host-name "$HOST_TAG" ;;
        *)
            echo "unknown axis: $1" >&2; exit 2 ;;
    esac
}

IFS=',' read -ra AXIS_LIST <<< "$AXES"
for axis in "${AXIS_LIST[@]}"; do
    run_axis "$(echo "$axis" | tr -d ' ')"
done
echo "STANDINGS_REFRESH_DONE axes=$AXES ym=$YM host=$HOST_TAG"
