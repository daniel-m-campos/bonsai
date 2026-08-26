#!/bin/bash
# Round 2: answer the three questions the first probe left open.
#   A  locality, isolated: root_hist only, enough iters to clear the 10ms timer
#   B  column views: same feature count, spread across tiles vs packed into them
#   C  fold memory: k materialized Datasets vs one plane plus k index arrays
set -u
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/seg-probe || { echo "SEG2_FAIL no tree"; exit 1; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cmake -B build-cuda -DBONSAI_CUDA=ON -DBONSAI_PYTHON=ON \
  -DCMAKE_CXX_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/opt/venv/bin/python \
  -G Ninja -S . > build.log 2>&1 || { echo "SEG2_FAIL configure"; tail -20 build.log; exit 1; }
cmake --build build-cuda --target _bonsai bonsai_params -j"$(nproc)" >> build.log 2>&1 \
  || { echo "SEG2_FAIL build"; tail -20 build.log; exit 1; }
echo "BUILD_OK"
export PYTHONPATH=/root/seg-probe/build-cuda/python

# A. 4M with 300 iters: root_hist accumulates ~70 ticks instead of 7, which is
#    enough to resolve the 6% question. root_hist is one node at depth 0, so
#    tree shape cannot contaminate it.
echo "=== A locality isolated (4M x 128, 300 iters, root_hist is the answer)"
timeout 3600 /opt/venv/bin/python probe_view_cost.py --rows 4000000 --cols 128 \
  --iters 300 --depth 8 --reps 3 --segments 1,160 \
  --python /opt/venv/bin/python 2>&1 || echo "SEG2_FAIL A"

# B. Same k features either way; only their distribution across the width-8
#    tiles changes. A spread draw touches every tile, a packed one empties
#    most of them, and the kernel returns early on an empty tile.
echo "=== B column views: spread vs packed, same feature count"
for frac in 1.0 0.5 0.25 0.125; do
  for mode in spread packed; do
    envv=""
    [ "$mode" = packed ] && envv="1"
    out=$(BONSAI_PROBE_FEATCONTIG="$envv" BONSAI_CUDA_PROFILE=1 timeout 1200 \
      /opt/venv/bin/python - "$frac" <<'PY' 2>&1
import sys, numpy as np, bonsai
frac = float(sys.argv[1])
n, f = 4_000_000, 128
rng = np.random.default_rng(0)
X = rng.random((n, f), dtype=np.float32)
y = (X[:, 0] * 2 + X[:, 1] + rng.normal(0, .1, n)).astype(np.float32)
ds = bonsai.Dataset(X, y, device="cuda")
p = {"dispatch.grower_name": "cuda_depthwise", "booster.n_iters": 60,
     "tree.max_depth": 8, "tree.feature_fraction": frac}
bonsai.train(bonsai.Params.from_dict({**p, "booster.n_iters": 2}), ds)
bonsai.train(bonsai.Params.from_dict(p), ds)
PY
    )
    hist=$(echo "$out" | grep -o "cuda-round-decomp: .*" | tail -1)
    echo "  frac=$frac mode=$mode :: $hist"
  done
done

# C. The CPCV memory wall, measured per process from NVML.
echo "=== C fold memory (16M x 128, k=5)"
timeout 1800 /opt/venv/bin/python probe_fold_memory.py --rows 16000000 --cols 128 \
  --folds 5 2>&1 || echo "SEG2_FAIL C"

echo "SEG2_DONE"
