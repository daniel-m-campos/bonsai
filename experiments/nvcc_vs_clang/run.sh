#!/usr/bin/env bash
# Compile bench.cu with BOTH clang (-x cuda) and nvcc, run each, and compare
# per-kernel time + register usage + SASS FP64 density. Run on a CUDA node
# (this repo's macOS host has no NVIDIA GPU). See README.md.
#
#   bash run.sh [arch]      # arch defaults to native; use sm_120 on a 5090
set -euo pipefail

ARCH="${1:-native}"
CUDA="${CUDA_HOME:-/usr/local/cuda}"
CLANG="${CLANG:-clang++-21}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SHIM="$HERE/shim"
SRC="$HERE/bench.cu"
OUT="$HERE/out"
mkdir -p "$OUT"

echo "arch=$ARCH  cuda=$CUDA  clang=$CLANG  nvcc=$($CUDA/bin/nvcc --version | tail -1)"

echo "== clang -x cuda build =="
"$CLANG" -x cuda "$SRC" -o "$OUT/bench_clang" \
    --offload-arch="$ARCH" -O3 -std=c++20 -I "$SHIM" \
    --cuda-path="$CUDA" -L"$CUDA/lib64" -lcudart \
    -Xcuda-ptxas -v 2>"$OUT/clang_ptxas.txt" || { cat "$OUT/clang_ptxas.txt"; exit 1; }
grep -iE "registers|stack frame|spill|kernel" "$OUT/clang_ptxas.txt" || true

echo "== nvcc build =="
"$CUDA/bin/nvcc" "$SRC" -o "$OUT/bench_nvcc" \
    -arch="$ARCH" -O3 -std=c++20 --expt-relaxed-constexpr -I "$SHIM" \
    -Xptxas -v 2>"$OUT/nvcc_ptxas.txt" || { cat "$OUT/nvcc_ptxas.txt"; exit 1; }
grep -iE "registers|stack frame|spill|_kernel" "$OUT/nvcc_ptxas.txt" || true

echo "== run (clang) ==" && "$OUT/bench_clang"
echo "== run (nvcc)  ==" && "$OUT/bench_nvcc"

echo "== SASS instruction profile (rough) =="
for c in clang nvcc; do
    "$CUDA/bin/cuobjdump" -sass "$OUT/bench_$c" >"$OUT/$c.sass" 2>/dev/null || true
    dbl=$(grep -cE '\b(DADD|DMUL|DFMA|DSETP)\b' "$OUT/$c.sass" || true)
    atom=$(grep -cE '\b(ATOM|RED)\b' "$OUT/$c.sass" || true)
    tot=$(wc -l <"$OUT/$c.sass")
    printf "%-6s  fp64(DADD/DMUL/DFMA/DSETP)=%-5s atomics(ATOM/RED)=%-5s sass_lines=%s\n" \
        "$c" "$dbl" "$atom" "$tot"
done
echo "Full detail in $OUT/{clang,nvcc}_ptxas.txt and $OUT/{clang,nvcc}.sass"
