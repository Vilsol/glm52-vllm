#!/usr/bin/env bash
###############################################################################
# r9 cutover driver: wait for boot -> PROVE the r9-specific config actually took
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
TAG="${TAG:-r11-20260729}"
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

  echo; echo "### KV capacity — r4 baseline was 990,592 tokens"
  docker logs glm52 2>&1 | grep -iE "GPU KV cache size|Maximum concurrency" | tail -3

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
except Exception as e:
    print("bench parse failed:", e)
PY

echo
echo "### external prefix cache after bench (did LMCache serve anything?)"
curl -s "$BASE/metrics" | grep -E '^vllm:external_prefix_cache_(hits|queries)_total'
log "post-bench health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health") container=$(docker inspect -f '{{.State.Status}}' glm52)"
echo "=== R9 VERIFY+BENCH DONE ==="
