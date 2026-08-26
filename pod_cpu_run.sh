#!/bin/bash
# x86-64 at a realistic thread count, with EVERY pool capped.
#
# The earlier runs on this pod class throttled because only the fit's
# parallel.n_threads was capped: the Dataset constructor carries its own
# n_threads and OpenMP defaults to the advertised core count, so binning ran
# 256-wide against a 27-core quota. Cap all three (OMP_NUM_THREADS, the
# Dataset ctor, the fit config) and nr_throttled should barely move.
set -u
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root/seg-probe || { echo "MC_FAIL no tree"; exit 1; }

echo "arch: $(uname -m) | nproc: $(nproc)"
grep -m1 'model name' /proc/cpuinfo || true
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo none)"
echo "avx512:  $(grep -m1 -o 'avx512[a-z]*' /proc/cpuinfo | head -3 | tr '\n' ' ')"

QUOTA=$(awk '{ if ($1 == "max") print 0; else printf "%.1f", $1/$2 }' /sys/fs/cgroup/cpu.max)
# threads * 1.5 must clear the quota, because bonsai's barriers spin.
THREADS=$(awk -v q="$QUOTA" 'BEGIN { t = int(q / 1.5); print (t < 2 ? 2 : t) }')
echo "quota ${QUOTA} cores -> ${THREADS} threads"
export OMP_WAIT_POLICY=passive OMP_NUM_THREADS="$THREADS"

cmake -B build -DBONSAI_PYTHON=ON -DCMAKE_CXX_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/opt/venv/bin/python \
  -G Ninja -S . > build.log 2>&1 || { echo "MC_FAIL configure"; tail -20 build.log; exit 1; }
cmake --build build --target _bonsai bonsai_params -j"$THREADS" >> build.log 2>&1 \
  || { echo "MC_FAIL build"; tail -20 build.log; exit 1; }
echo "BUILD_OK"

# Sweep the thread count too: the M2 showed the penalty shrinking as threads
# rise (+25% at 2, +18% at 4), so the deployment-relevant number is the one
# at the top of the range, not the bottom.
for t in 2 4 "$THREADS"; do
  echo "=== threads=$t"
  before=$(grep -m1 nr_throttled /sys/fs/cgroup/cpu.stat | awk '{print $2}')
  OMP_NUM_THREADS="$t" PYTHONPATH=/root/seg-probe/build/python timeout 3600 \
    /opt/venv/bin/python probe_cpu_view.py --rows 2000000 --cols 32 --fracs 0.8 \
    --iters 60 --depth 6 --threads "$t" --reps 5 \
    --python /opt/venv/bin/python 2>&1 | grep -vE "^  rep" || echo "MC_FAIL t=$t"
  after=$(grep -m1 nr_throttled /sys/fs/cgroup/cpu.stat | awk '{print $2}')
  echo "  throttled during: $((after - before))"
done
echo "MC_DONE"
