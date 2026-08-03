#!/usr/bin/env bash
# Wait for v20 health, gate on idle, then run the FULL bench with MTP brackets.
set -uo pipefail
BENCH=/root/llm-inference-bench
OUT=/root/glm52-vllm
log(){ echo "$(date -u +%H:%M:%S) $*"; }

# 1) wait for health (abort if container dies)
while [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8443/health 2>/dev/null)" != "200" ]; do
  st=$(docker inspect -f '{{.State.Status}}' glm52 2>/dev/null)
  if [ "$st" != "running" ]; then
    log "CONTAINER $st DURING BOOT — bench aborted"
    docker logs glm52 2>&1 | grep -iE "error|traceback|illegal|Xid|die|failed|CUDA error" | tail -20
    exit 1
  fi
  sleep 30
done
log "v20 HEALTHY: $(docker ps --filter name=glm52 --format '{{.Status}}')"

# 2) settle + idle gate (server just booted; give it a moment, then require 0 running)
sleep 20
R=$(curl -s http://127.0.0.1:8443/metrics | awk '/^vllm:num_requests_running/{print $2}')
if [ "$R" != "0.0" ]; then log "NOT IDLE (running=$R) — bench aborted (retry when quiet)"; exit 2; fi
log "idle confirmed (running=0)"

# 2b) capture PCIe calibration decision + topology (festr asked for exactly this)
{
  echo "### PCIe calibration"; docker logs glm52 2>&1 | grep -i "pcie calibration" | head -5
  echo "### resolved DCP env"
  docker exec glm52 bash -c 'p=$(pgrep -f "vllm serve|EngineCore"|head -1); tr "\0" "\n" < /proc/${p:-1}/environ | grep -E "VLLM_(DCP|B12X_MLA_CKV|ENABLE_PCIE|PCIE_DMA_MIN)" | sort'
  echo "### nvidia-smi topo -m"; nvidia-smi topo -m
} > /root/glm52-vllm/calibration-v20-0726-20260727.txt 2>&1
log "calibration + topology captured"

# 3) MTP counters before
curl -s http://127.0.0.1:8443/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total' > "$OUT/mtp-v20-0726-20260727-before.txt"

# 4) FULL bench: full decode matrix + long-prefill sweep (patched cap)
log "starting full bench"
cd "$BENCH"
LLM_BENCH_MAX_PREFILL=917504 ./.venv/bin/python -u llm_decode_bench.py \
  --port 8443 --concurrency 1,4,8,16,32 --contexts 0 --duration 15 \
  --prefill-contexts 8k,64k,128k,256k,512k --prefill-metric client \
  --display-mode plain \
  --output "$OUT/bench-v20-0726-20260727.json" \
  > "$OUT/bench-v20-0726-20260727.log" 2>&1
rc=$?
log "bench exited rc=$rc"

# 5) MTP counters after + accept rate
curl -s http://127.0.0.1:8443/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total' > "$OUT/mtp-v20-0726-20260727-after.txt"
python3 - <<'PY'
import re
def g(f):
    d={}
    for l in open(f):
        m=re.match(r'vllm:spec_decode_num_(\w+)_tokens_total\{[^}]*\}\s+([\d.]+)',l)
        if m: d[m.group(1)]=float(m.group(2))
    return d
try:
    b=g('/root/glm52-vllm/mtp-v20-0726-20260727-before.txt'); a=g('/root/glm52-vllm/mtp-v20-0726-20260727-after.txt')
    dd=a['draft']-b['draft']; da=a['accepted']-b['accepted']
    print(f"MTP v20-0726-20260727: draft={dd:.0f} accepted={da:.0f} accept_rate={100*da/dd:.2f}%")
except Exception as e:
    print("MTP calc failed:", e)
PY
# 6) still healthy after the run? (deep-context bench has crashed us before)
h=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8443/health 2>/dev/null)
log "post-bench health=$h  container=$(docker inspect -f '{{.State.Status}}' glm52 2>/dev/null)"
echo "=== FULL BENCH DONE ==="
