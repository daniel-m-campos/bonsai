#!/usr/bin/env bash
# One measured arm: run the bench worker under perfrun and record everything.
#
#   arm.sh <label> <core|mem|tlb|none> <variant> <threads> <iters>
#
# Env passthrough (OMP_*, GOMP_*, BONSAI_*) is inherited from the caller, so
# an A/B is just two invocations with different exports. Output lands in
# $OUT/<label>.txt: the PERFRUN counter lines plus the worker's RESULT json.
set -u
LABEL=$1; SET=$2; VARIANT=$3; THREADS=$4; ITERS=$5
OUT=${OUT:-/root/out}
ROOT=${ROOT:-/root/bonsai}
ROWS=${ROWS:-2097152}
mkdir -p "$OUT"

SPEC=$(printf '{"variant":"%s","cell":{"axis":"cell","rows":%s,"cols":128,"bins":255,"bins_effective":255,"depth":8,"iters":%s,"lr":0.1,"informative":20,"n_test":100000,"seed":42},"threads":%s}' \
  "$VARIANT" "$ROWS" "$ITERS" "$THREADS")

{
  echo "ARM	$LABEL"
  echo "SPEC	$SPEC"
  echo "ENV	OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-unset} GOMP_SPINCOUNT=${GOMP_SPINCOUNT:-unset} OMP_PROC_BIND=${OMP_PROC_BIND:-unset}"
} >> "$OUT/$LABEL.txt"

cd "$ROOT" || exit 1
echo "$SPEC" | PYTHONPATH="$ROOT/build/python" \
  BONSAI_BENCH_DATA_CACHE=${BONSAI_BENCH_DATA_CACHE:-/root/datacache} \
  /root/perfrun "$SET" /opt/venv/bin/python -m bonsai.bench worker \
  >> "$OUT/$LABEL.txt" 2>> "$OUT/$LABEL.txt"

echo "ARMEND	$LABEL" >> "$OUT/$LABEL.txt"
