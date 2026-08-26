#!/bin/bash
# Build the probe on a pod and sweep it at standings-class shapes.
# Prints SEG_DONE at the end; every guarded failure prints SEG_FAIL <why>.
set -u
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/seg-probe || { echo "SEG_FAIL no tree"; exit 1; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "SMs: $(nvidia-smi --query-gpu=count --format=csv,noheader) | nproc=$(nproc)"

# The kernel TU is clang CUDA, not nvcc and not GCC: the image ships llvm-21.
cmake -B build-cuda -DBONSAI_CUDA=ON -DBONSAI_PYTHON=ON \
  -DCMAKE_CXX_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/opt/venv/bin/python \
  -G Ninja -S . > build.log 2>&1 || { echo "SEG_FAIL configure"; tail -20 build.log; exit 1; }
cmake --build build-cuda --target _bonsai bonsai_params -j"$(nproc)" >> build.log 2>&1 \
  || { echo "SEG_FAIL build"; tail -20 build.log; exit 1; }
echo "BUILD_OK"

# 4M is the working shape; 16M is the standings tall cell. Both at 128 cols,
# which is where the tile plane has 16 tiles rather than the nano's 8.
for shape in "4000000 128" "16000000 128"; do
  set -- $shape
  echo "=== rows=$1 cols=$2"
  PYTHONPATH=build-cuda/python timeout 3600 /opt/venv/bin/python probe_view_cost.py \
    --rows "$1" --cols "$2" --iters 30 --depth 8 --reps 3 \
    --segments 1,160,10000 --python /opt/venv/bin/python 2>&1 \
    || echo "SEG_FAIL shape $1x$2"
done
echo "SEG_DONE"
