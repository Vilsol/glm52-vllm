#!/usr/bin/env bash
set -uo pipefail
OUT=/root/glm52-vllm; BENCH=/root/llm-inference-bench
TAG=r4-lmcache-20260727
log(){ echo "$(date -u +%H:%M:%S) $*"; }

# 1) wait for health (LMCache cold boot ~25-30min)
while [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8443/health 2>/dev/null)" != "200" ]; do
  st=$(docker inspect -f '{{.State.Status}}' glm52 2>/dev/null)
  if [ "$st" != "running" ]; then
    log "CONTAINER $st DURING BOOT — aborting"
    docker logs glm52 2>&1 | grep -iE "error|traceback|illegal|Xid|FATAL|CUDA error" | tail -20; exit 1
  fi
  sleep 30
done
log "HEALTHY: $(docker ps --filter name=glm52 --format '{{.Status}}')"

# 2) PROVE LMCache is actually wired in, not merely that the server is up
{
  echo "### connector in vllm args"
  docker logs glm52 2>&1 | grep -oE '"kv_connector":"[^"]*"|LMCacheMPConnector' | head -3
  echo "### lmcache connector init"
  docker logs glm52 2>&1 | grep -iE "lmcache.*connector|LMCacheMPConnector|kv_transfer" | tail -5
  echo "### external prefix cache metrics exist?"
  curl -s http://127.0.0.1:8443/metrics | grep -E '^vllm:external_prefix_cache_(hits|queries)_total'
  echo "### calibration"
  docker logs glm52 2>&1 | grep -i "pcie calibration" | head -2
  echo "### resolved DCP env"
  docker exec glm52 bash -c 'p=$(pgrep -f "vllm serve|EngineCore"|head -1); tr "\0" "\n" < /proc/${p:-1}/environ | grep -E "VLLM_(DCP|B12X_MLA_CKV|ENABLE_PCIE)" | sort'
} > "$OUT/verify-$TAG.txt" 2>&1
log "connector verification written"

# 3) idle gate
sleep 15
R=$(curl -s http://127.0.0.1:8443/metrics | awk '/^vllm:num_requests_running/{print $2}')
[ "$R" = "0.0" ] || { log "NOT IDLE (running=$R) — bench skipped"; exit 2; }
log "idle confirmed"

curl -s http://127.0.0.1:8443/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total' > "$OUT/mtp-$TAG-before.txt"
log "starting bench"
cd "$BENCH"
LLM_BENCH_MAX_PREFILL=917504 ./.venv/bin/python -u llm_decode_bench.py \
  --port 8443 --concurrency 1,4,8,16 --contexts 0 --duration 15 \
  --prefill-contexts 8k,64k,128k,256k,512k --prefill-metric client \
  --display-mode plain --output "$OUT/bench-$TAG.json" > "$OUT/bench-$TAG.log" 2>&1
log "bench rc=$?"
curl -s http://127.0.0.1:8443/metrics | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total' > "$OUT/mtp-$TAG-after.txt"
python3 - <<'PY'
import re
def g(f):
    d={}
    for l in open(f):
        m=re.match(r'vllm:spec_decode_num_(\w+)_tokens_total\{[^}]*\}\s+([\d.]+)',l)
        if m: d[m.group(1)]=float(m.group(2))
    return d
try:
    b=g('/root/glm52-vllm/mtp-r4-lmcache-20260727-before.txt'); a=g('/root/glm52-vllm/mtp-r4-lmcache-20260727-after.txt')
    dd=a['draft']-b['draft']; da=a['accepted']-b['accepted']
    print(f"MTP accept: {100*da/dd:.2f}%  (draft={dd:.0f} accepted={da:.0f})")
except Exception as e: print("MTP calc failed:",e)
PY
# external cache activity during the run (did LMCache actually serve anything?)
curl -s http://127.0.0.1:8443/metrics | grep -E '^vllm:external_prefix_cache_(hits|queries)_total'
log "post-bench health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8443/health) container=$(docker inspect -f '{{.State.Status}}' glm52)"
echo "=== R4 LMCACHE BENCH DONE ==="
