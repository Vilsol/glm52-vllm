#!/usr/bin/env bash
# Sweep koush's isolated FP8 E4M3->float test across all 8 GPUs (hw path).
# CLEAN on all 8 = no faulty fp8 conversion unit. Any FAULT/CUDA_ERROR = suspect card.
# NOTE: no /usr/bin/time on this host -- use bash SECONDS.
cd /root/glm52-vllm
fail=0
for g in 0 1 2 3 4 5 6 7; do
  echo "======== GPU $g ========"
  t0=$SECONDS
  out=$(./fp8_scale_test "$g" 10000 256 hw 2>&1); rc=$?
  echo "$out" | grep -E "GPU |RESULT|FAULT|CUDA_ERROR" || echo "$out" | tail -3
  echo "  wall $((SECONDS-t0))s  exit=$rc"
  [ $rc -ne 0 ] && fail=1
done
echo "======== SWEEP COMPLETE (fail=$fail) ========"
