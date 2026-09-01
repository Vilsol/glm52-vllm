#!/usr/bin/env bash
# Full stability + speed battery. Run AFTER a topology change (DCP, context,
# quant) to prove nothing regressed. Writes one JSON per phase plus a summary.
#
# Every number is checked for contamination afterwards by agg-clean-cells.py:
# the bench does NOT detect external traffic during a cell (it only flags
# UNDER-filled cells), so on a live box a cell where someone else's request
# landed will otherwise be silently wrong. See BENCHMARKS Phase 18 s9.
set -uo pipefail
cd "$(dirname "$0")"
TAG="${TAG:-$(date -u +%Y%m%dT%H%M)}"
OUT="bench-battery-$TAG"; mkdir -p "$OUT"
PORT="${PORT:-8443}"; MODEL="${MODEL:-GLM-5.2}"; DUR="${DUR:-15}"
B() { (cd ../llm-inference-bench && timeout "${2:-1800}" uv run python3 llm_decode_bench.py \
        --port "$PORT" --model "$MODEL" --duration "$DUR" $1 \
        --output "/root/glm52-vllm/$OUT/$3.json") >"$OUT/$3.log" 2>&1; echo "  $3 exit=$?"; }

echo "=== battery $TAG -> $OUT/"
echo "--- pre-flight"
curl -s "localhost:$PORT/metrics" | grep -E "^vllm:(num_preemptions_total|num_requests_running) " | sed 's/^/    /'
docker logs glm52 2>&1 | grep -E "GPU KV cache size|Maximum concurrency" | tail -2 | sed 's/^/    /'
P0=$(curl -s "localhost:$PORT/metrics" | awk '/^vllm:num_preemptions_total/{print $2}')

# 1. decode sweep @ctx0 — headline throughput + per-user scaling
B "--concurrency 1,4,8,16 --contexts 0"                 1800 01-decode-sweep
# 2. decode vs context @c1 and c4 — the scaling check
B "--concurrency 1,4 --contexts 0,8k,64k,128k"          2400 02-ctx-matrix
# 3. long context — only meaningful once the pool is big (DCP4 / 1M)
B "--concurrency 1 --contexts 256k,512k"                2400 03-long-ctx
# 4. standalone cold prefill, incl. past 128k via our local patch
LLM_BENCH_MAX_PREFILL=1048576 \
  B "--concurrency 1 --contexts 0 --standalone-prefill --prefill-contexts 8k,64k,128k,256k,512k" 3000 04-prefill

echo "--- post-flight"
P1=$(curl -s "localhost:$PORT/metrics" | awk '/^vllm:num_preemptions_total/{print $2}')
echo "    preemptions during battery: $P0 -> $P1"
curl -s "localhost:$PORT/metrics" | grep -E "^vllm:(prefix_cache|external_prefix_cache)_(hits|queries)_total" | sed 's/^/    /'
curl -s "http://10.42.6.29:8532/metrics" 2>/dev/null | grep -v '^#' | grep -E "l1_usage_ratio|l1_evicted_chunks_total" | sed -E 's/\{[^}]*\}//' | sed 's/^/    /'

echo "--- coherence (stability, not speed)"
for p in "Write a Python function that reverses a linked list. Code only." \
         "What is 17 * 23? Answer with the number only."; do
  curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys;print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':200,'temperature':0}))" "$p")" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);c=d['choices'][0]['message']['content'];print('    ok  %d chars: %s' % (len(c),' '.join(c.split())[:90]))" 2>/dev/null \
  || echo "    FAILED"
done

echo "--- CLEAN-ONLY RESULTS (external traffic filtered)"
python3 agg-clean-cells.py "$OUT"/*.json 2>/dev/null || echo "    (aggregator found nothing)"
echo "=== battery done -> $OUT/"
