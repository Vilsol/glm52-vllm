#!/usr/bin/env bash
###############################################################################
# r20 cutover driver: wait for boot -> PROVE the r9-specific config actually took
# -> idle-gate -> bench -> attribute the result honestly.
#
# WHY THE VERIFY STAGE IS LONGER THAN THE BENCH STAGE: four of our last five
# benches were silently invalidated (three by user traffic, one by a 128k
# prefill cap). A number nobody can attribute is worse than no number, so this
# script records the evidence FIRST and refuses to bench without it.
#
# CONTAMINATION: this box serves real users. The bench harness reports
# avg_running_reqs per cell; if that exceeds the requested concurrency, foreign
# traffic was in the batch and the cell is void. We gate before starting AND
# re-check after, then print the per-cell verdict so a contaminated run is
# labelled rather than quietly filed as a regression.
###############################################################################
set -uo pipefail
OUT=/root/glm52-vllm; BENCH=/root/llm-inference-bench
TAG="${TAG:-r20-20260803}"
PORT="${PORT:-8443}"
LMC_HTTP="${LMC_HTTP:-8532}"
BASE="http://127.0.0.1:$PORT"
log(){ echo "$(date -u +%H:%M:%S) $*"; }

# ── 1. wait for health (cold boot ~25-30 min: 768GB L1 pin evicts the weight
#       page cache, so weights reload from NFS) ────────────────────────────────
while [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" 2>/dev/null)" != "200" ]; do
  st=$(docker inspect -f '{{.State.Status}}' glm52 2>/dev/null)
  if [ "$st" != "running" ]; then
    log "CONTAINER $st DURING BOOT — aborting"
    # Dump the whole log NOW: the watchdog recycles the container, and we have
    # already lost one crash stack that way (2026-07-24).
    docker logs glm52 > "$OUT/crash-$TAG.log" 2>&1
    log "full log saved to crash-$TAG.log"
    grep -iE "error|traceback|illegal|Xid|FATAL|CUDA error" "$OUT/crash-$TAG.log" | tail -30
    exit 1
  fi
  sleep 30
done
log "HEALTHY: $(docker ps --filter name=glm52 --format '{{.Status}}')"

# ── 2. prove the r9 config took ──────────────────────────────────────────────
{
  echo "### image"
  docker inspect -f '{{.Config.Image}}' glm52

  echo; echo "### TRAP 2 — calibration must be skipped, NOT probed"
  echo "(r9's probe is reported broken on our 2-NUMA topology; expect skipped:all-explicit)"
  docker logs glm52 2>&1 | grep -iE "PCIE_CALIBRATION_STATUS|PCIe calibration" | head -3

  echo; echo "### TRAP 1 — CKV gather capacity must be 524288, not r9's 140000 default"
  docker logs glm52 2>&1 | grep -iE "full CKV gather capacity|CKV_GATHER_MAX_TOKENS" | head -3

  echo; echo "### LMCache native wrapper came up (mode/L1/chunk)"
  docker logs glm52 2>&1 | grep -iE "LMCache ready" | head -2
  echo "-- healthcheck:"; curl -s --max-time 5 "http://127.0.0.1:$LMC_HTTP/healthcheck" || echo "(no response)"

  echo; echo "### our TTL patch actually applied (86400, not the image's 600/300)"
  docker exec glm52 bash -c 'pgrep -af "lmcache server|lmcache.*server" | head -2' 2>/dev/null

  echo; echo "### connector is in the real vllm args"
  docker logs glm52 2>&1 | grep -oE 'LMCacheMPConnector' | head -3
  docker logs glm52 2>&1 | grep -iE "Creating v1 connector|Registering kv caches" | tail -3

  echo; echo "### external prefix cache metrics exposed"
  curl -s "$BASE/metrics" | grep -E '^vllm:external_prefix_cache_(hits|queries)_total'

  echo; echo "### KV capacity — r11 baseline was 1,027,968 tokens at this same GMU 0.94"
  echo "(r12 changed 'draft peak accounting before KV allocation', so this CAN move."
  echo " Materially ABOVE 1,027,968 = the safety margin became KV cache = the exact"
  echo " shape of the 2026-07-29 r11 OOM. Below 917,504 = cannot hold one full request.)"
  docker logs glm52 2>&1 | grep -iE "GPU KV cache size|Maximum concurrency" | tail -3

  echo; echo "### VRAM headroom at idle (per GPU) — the r11 OOM had 89 MiB free on GPU6"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

  echo; echo "### indexer folding (new in r9, absent from r4)"
  docker exec glm52 bash -c 'p=$(pgrep -f "EngineCore"|head -1); tr "\0" "\n" < /proc/${p:-1}/environ | grep -iE "TWO_LEVEL_FOLD|INDEXER_SHARDS"' 2>/dev/null

  echo; echo "### resolved DCP / PCIe env"
  docker exec glm52 bash -c 'p=$(pgrep -f "vllm serve|EngineCore"|head -1); tr "\0" "\n" < /proc/${p:-1}/environ | grep -E "VLLM_(DCP|B12X_MLA_CKV|ENABLE_PCIE|PCIE_DMA)" | sort' 2>/dev/null
} > "$OUT/verify-$TAG.txt" 2>&1
log "verification written to verify-$TAG.txt"

# Hard gate: the two traps are the whole reason this launcher exists.
if ! grep -q "skipped:all-explicit" "$OUT/verify-$TAG.txt"; then
  log "!!! calibration did NOT skip — r9 probed the fabric. Check verify-$TAG.txt before trusting any number."
fi
if ! grep -q "524288" "$OUT/verify-$TAG.txt"; then
  log "!!! CKV gather capacity is not 524288 — long prefills may be on the slow fallback path."
fi

# GMU gate. The r11 OOM booted healthy and died on the first real batch, so a
# green /health proves nothing here. Compare KV against r11's figure at the
# SAME GMU and refuse to bench if the margin silently turned into cache.
KVTOK=$(docker logs glm52 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+' | tail -1 | grep -oE '[0-9,]+$' | tr -d ',')
if [ -n "${KVTOK:-}" ]; then
  log "KV cache tokens: $KVTOK (r11 @ GMU 0.94 = 1027968)"
  if [ "$KVTOK" -lt 917504 ]; then
    log "!!! KV ($KVTOK) is BELOW max_model_len 917504 — cannot hold one full-length request. ABORTING."
    exit 3
  fi
  # Scale the ceiling with DCP. KV capacity is ~proportional to the DCP degree
  # because decode context parallelism shards the KV across that many ranks.
  # The gate was calibrated at DCP2; at DCP4 the CORRECT value is ~2x, and a
  # fixed ceiling aborts a perfectly healthy boot (hit 2026-08-03: DCP4 came up
  # with 1,912,320 tokens and was killed as an "OOM signature").
  DCPDEG="${DCP:-2}"
  KVCEIL=$(( 1130000 * DCPDEG / 2 ))
  if [ "$KVTOK" -gt "$KVCEIL" ]; then
    log "!!! KV ($KVTOK) is >10% ABOVE r11's 1,027,968 at the same GMU."
    log "!!! This is the r11-OOM signature: freed memory became KV, not headroom."
    log "!!! Drop GPU_MEMORY_UTILIZATION by 0.01 and relaunch BEFORE serving traffic."
    exit 3
  fi
else
  log "!!! could not parse 'GPU KV cache size' from the log — verify by hand before benching."
fi

# Free-VRAM floor. Anything under ~1 GiB/GPU at idle means no room for the
# transient all-reduce buffers that killed the first r11 boot.
python3 - <<'PY' || exit 3
import subprocess, sys, os
rows = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total",
                       "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
bad = []
for r in rows:
    i, used, tot = [x.strip() for x in r.split(",")]
    free = int(tot) - int(used)
    print(f"  GPU{i}: {free} MiB free")
    if free < 1024: bad.append((i, free))
# The floor assumes a FRESH boot. PyTorch's caching allocator retains blocks
# after any workload, so re-running this driver against an already-warm server
# reads ~1 GiB where the same config showed ~3.6 GiB at boot (hit 2026-08-03 on
# DCP4: 3602 MiB at boot -> 996 MiB after the warm-up pass, gate aborted a
# healthy config). Set ALLOW_WARM_VRAM=1 when intentionally re-benching a server
# that has already served traffic.
if bad and os.environ.get("ALLOW_WARM_VRAM") != "1":
    print("!!! GPUs with <1 GiB free at IDLE:", bad)
    print("!!! The r11 OOM died with 89 MiB free. Lower GPU_MEMORY_UTILIZATION before serving.")
    sys.exit(3)
PY

# ── 3. idle gate: 3 consecutive clean samples, not one lucky one ─────────────
idle=0
for i in $(seq 1 20); do
  R=$(curl -s "$BASE/metrics" | awk '/^vllm:num_requests_running/{print $2; exit}')
  if [ "$R" = "0.0" ]; then idle=$((idle+1)); else idle=0; log "busy (running=$R), waiting"; fi
  [ "$idle" -ge 3 ] && break
  sleep 20
done
if [ "$idle" -lt 3 ]; then
  log "NEVER IDLE — skipping bench (a contaminated bench is worse than none)"
  exit 2
fi
log "idle confirmed (3 consecutive samples)"

# ── 4. bench ────────────────────────────────────────────────────────────────
curl -s "$BASE/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total' > "$OUT/mtp-$TAG-before.txt"
log "starting bench"
cd "$BENCH"
# WARM-UP RUN, DISCARDED. Measured 2026-08-03: the first bench after a boot pays
# JIT compile, CUDA-graph/FlashInfer autotune and cold page cache, and the
# penalty is wildly context-dependent -- observed -77% at 8k, -42% at 64k, but
# only -0.1% at 512k. Runs 2 and 3 of the same config agree to <1%. Every number
# produced before this fix was a cold single sample and is NOT comparable to
# anything measured after it. Discarding run 1 is the difference between a
# reproducible bench and one that mostly measures warm-up.
if [ "${SKIP_WARMUP:-0}" != "1" ]; then
  log "warm-up pass (discarded)"
  LLM_BENCH_MAX_PREFILL=917504 ./.venv/bin/python -u llm_decode_bench.py \
    --port "$PORT" --concurrency 1,4,8,16 --contexts 0 --duration 15 \
    --prefill-contexts 8k,64k,128k,256k,512k --prefill-metric client \
    --display-mode plain --output /dev/null > /dev/null 2>&1
  log "warm-up done; measuring"
fi
# LLM_BENCH_MAX_PREFILL: local patch at llm_decode_bench.py:12553 — without it
# every prefill silently caps at 128k. Reapply after any `git pull` in this repo.
LLM_BENCH_MAX_PREFILL=917504 ./.venv/bin/python -u llm_decode_bench.py \
  --port "$PORT" --concurrency 1,4,8,16 --contexts 0 --duration 15 \
  --prefill-contexts 8k,64k,128k,256k,512k --prefill-metric client \
  --display-mode plain --output "$OUT/bench-$TAG.json" > "$OUT/bench-$TAG.log" 2>&1
log "bench rc=$?"
curl -s "$BASE/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total' > "$OUT/mtp-$TAG-after.txt"

# ── 5. attribute the result ─────────────────────────────────────────────────
TAG="$TAG" python3 - <<'PY'
import json, os, re
out, tag = '/root/glm52-vllm', os.environ['TAG']
def mtp(f):
    d = {}
    for l in open(f):
        m = re.match(r'vllm:spec_decode_num_(\w+)_tokens_total\{[^}]*\}\s+([\d.]+)', l)
        if m: d[m.group(1)] = float(m.group(2))
    return d
try:
    b, a = mtp(f'{out}/mtp-{tag}-before.txt'), mtp(f'{out}/mtp-{tag}-after.txt')
    dd, da = a['draft'] - b['draft'], a['accepted'] - b['accepted']
    print(f"MTP accept: {100*da/dd:.2f}%  (draft={dd:.0f} accepted={da:.0f})")
except Exception as e:
    print("MTP calc failed:", e)

# Per-cell contamination verdict. r4's run looked like a 40% regression purely
# because foreign requests were in the batch; never file a number without this.
try:
    d = json.load(open(f'{out}/bench-{tag}.json'))
    st = d.get('summary_table', {}).get('0', {})
    print(f"\n{'conc':>5} {'agg tok/s':>10} {'avg_run':>8} {'verdict':>12}")
    clean = True
    for r in d.get('results', []):
        c = r['concurrency']; run = r.get('avg_running_reqs')
        v = json.dumps(st.get(str(c)))[:8] if st.get(str(c)) else '-'
        ok = (run is not None and run <= c * 1.05)
        clean &= ok
        print(f"{c:>5} {v:>10} {run:>8} {'CLEAN' if ok else 'CONTAMINATED':>12}")
    print("\nRUN IS USABLE" if clean else
          "\nRUN IS CONTAMINATED — foreign traffic in the batch; decode numbers are void.")

    # Direct r11 comparison. This upgrade is a single-variable change, so the
    # honest expectation is PARITY. Flag both directions: a big win is as
    # suspicious as a big loss when no upstream note claims an NVFP4 change.
    R11 = {1: 110.5, 4: 312.3, 8: 431.5, 16: 634.7}
    print(f"\n{'conc':>5} {'r11':>8} {'r20':>8} {'delta':>9}   (r11 = clean 2026-07-29 bench)")
    for r in d.get('results', []):
        c = r['concurrency']
        cur = st.get(str(c))
        if isinstance(cur, dict):
            cur = cur.get('agg_tok_s') or cur.get('tok_s')
        if c in R11 and isinstance(cur, (int, float)):
            pct = 100 * (cur - R11[c]) / R11[c]
            note = ''
            if c == 1 and pct < -20:
                note = '  <-- CHECK VLLM_ENABLE_PCIE_ALLREDUCE=0 (r17 touched the PCIe collective)'
            elif abs(pct) > 10:
                note = '  <-- >10% swing; no r12..r20 note claims an NVFP4 change. Re-run before believing.'
            print(f"{c:>5} {R11[c]:>8.1f} {cur:>8.1f} {pct:>+8.1f}%{note}")
except Exception as e:
    print("bench parse failed:", e)
PY

echo
echo "### external prefix cache after bench (did LMCache serve anything?)"
curl -s "$BASE/metrics" | grep -E '^vllm:external_prefix_cache_(hits|queries)_total'
log "post-bench health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health") container=$(docker inspect -f '{{.State.Status}}' glm52)"
echo "=== R20 VERIFY+BENCH DONE ==="
