#!/usr/bin/env bash
# Issue #355 step 18, phase 1: thread-scaling under cgroup CPU-bandwidth
# instrumentation. perf_event_open is seccomp-blocked in the container, so the
# throttle counters plus cpu-seconds-per-wall carry the attribution.
set -u
export OUT=/root/out
export ROOT=/root/bonsai
mkdir -p "$OUT"

warm() { OUT=/root/warm /root/arm.sh warm none bonsai_depthwise 16 2; }

run() {  # label variant threads  (100 iters, clean env)
  env -u OMP_WAIT_POLICY -u GOMP_SPINCOUNT -u OMP_PROC_BIND -u OMP_NUM_THREADS \
    /root/arm.sh "$1" none "$2" "$3" 100
}

echo "warming data cache"; warm

# Scaling ladder, interleaved so any drift hits both engines alike.
for t in 16 12 8 1; do
  run "bonsai_t$t" bonsai_depthwise "$t"
  run "xgb_t$t"    xgb_hist         "$t"
done

# The spin A/B: libgomp busy-waits at every barrier, which burns quota that a
# throttled cgroup cannot spare. passive makes idle threads sleep instead.
for v in bonsai_depthwise xgb_hist; do
  short=${v%%_*}
  OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 /root/arm.sh "${short}_t16_passive" none "$v" 16 100
done

touch /root/PHASE1_DONE
