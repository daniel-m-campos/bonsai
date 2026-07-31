#!/usr/bin/env bash
# On-pod campaign driver: clone or fetch, build, run a committed bench spec.
# Usage (on the pod, after scp'ing this script):
#   HOST_TAG=<tag> BRANCH=main SPEC=<bundled-name-or-path> \
#     OUT=/root/<name>.jsonl RUN_LABEL=<label> bash scripts/pod_bench_driver.sh
# Optional: DATA_CACHE=/dev/shm/bonsai-bench
# Idempotent: re-running after a spot reap resumes (the CLI resumes by
# default in spec mode; ok/unsupported rows are final, failures re-attempt).
set -euo pipefail

USAGE="HOST_TAG=<tag> BRANCH=main SPEC=<bundled-name-or-path> OUT=/root/<name>.jsonl RUN_LABEL=<label> bash scripts/pod_bench_driver.sh"
for var in HOST_TAG SPEC OUT RUN_LABEL; do
    [ -n "${!var:-}" ] || { echo "error: $var is unset" >&2; echo "usage: $USAGE" >&2; exit 2; }
done
BRANCH="${BRANCH:-main}"

# sshd sessions do not inherit the Docker image's ENV (runbook section 3).
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Shallow single-branch clones have no origin/<branch> refs: fetch the branch
# and checkout FETCH_HEAD (docs/ops/runpod-runbook.md, hit three times).
if [ -d /root/bonsai ]; then
    git -C /root/bonsai fetch --depth 1 origin "$BRANCH"
    git -C /root/bonsai checkout -f FETCH_HEAD
else
    git clone --depth 1 https://github.com/daniel-m-campos/bonsai.git /root/bonsai
    (cd /root/bonsai && git fetch --depth 1 origin "$BRANCH" && git checkout -f FETCH_HEAD)
fi
cd /root/bonsai

# Blackwell (compute capability >= 10) needs CUDA >= 12.8; the bonsai-ci image
# ships 12.4, so side-install the 12.8 toolkit (same recipe as
# scripts/setup_gpu_node.sh) and point the configure at it. A pre-existing
# build-cuda/build.ninja satisfies the python-cuda prerequisite, so the extra
# -DCUDAToolkit_ROOT survives the make call below.
CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)
if [ "${CC_MAJOR:-0}" -ge 10 ]; then
    if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
        wget -qO /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(. /etc/os-release && echo "${VERSION_ID/./}")/x86_64/cuda-keyring_1.1-1_all.deb"
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -qq
        apt-get install -y -qq cuda-toolkit-12-8
    fi
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/usr/lib/llvm-21/bin/clang++ \
        -DBONSAI_CUDA=ON \
        -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
        -G Ninja -S . -B build-cuda
fi
make python-cuda PYTHON=/opt/venv/bin/python

BENCH=(env PYTHONPATH=/root/bonsai/build-cuda/python /opt/venv/bin/python -m bonsai.bench)
# --data-cache lands in a sibling PR; pass it only when the CLI knows it.
EXTRA=()
if [ -n "${DATA_CACHE:-}" ] && "${BENCH[@]}" run --help | grep -q -- --data-cache; then
    EXTRA=(--data-cache "$DATA_CACHE")
fi

"${BENCH[@]}" run --spec "$SPEC" --out "$OUT" \
    --run-label "$RUN_LABEL" --host-name "$HOST_TAG" ${EXTRA[@]+"${EXTRA[@]}"}

echo "copy home: scp -P <port> root@<ip>:$OUT benchmarks/results/"
echo "then DELETE THE POD (billing runs while it exists)"
