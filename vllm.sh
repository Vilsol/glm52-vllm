#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 on 8x RTX PRO 6000 Blackwell (96GB, PCIe, 2 NUMA nodes)
#
# Performance- AND correctness-tuned vLLM launcher. Wraps the image's canonical
# /usr/local/bin/serve-glm52.sh and injects host-specific bring-up fixes plus the
# config found by benchmarking + a long-generation correctness gate (Estonia).
# See ./BENCHMARKS.md for the full journal and per-lever attribution.
#
# Usage:
#   ./vllm.sh                       # start (detached "glm52" on :8000)
#   docker logs -f glm52            # follow startup ('Application startup complete')
#   MAX_NUM_SEQS=64 ./vllm.sh       # override any tunable via env
#
# Image: voipmonitor/vllm:dark-devotion-39ae3ed-b12x5b2e018-cu132-20260617
###############################################################################
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:dark-devotion-39ae3ed-b12x5b2e018-cu132-20260617}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8080}"   # same port the old Kimi-K2.6 server used

# ---- Model location (downloaded HF snapshot) --------------------------------
MODEL_HOST="${MODEL_HOST:-$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/)}"
MODEL_HOST="${MODEL_HOST%/}"
if [[ ! -f "${MODEL_HOST}/config.json" ]]; then
  echo "ERROR: GLM-5.2-NVFP4 snapshot not found at: ${MODEL_HOST}" >&2
  echo "Download: HF_TOKEN=... uv run --with 'huggingface_hub[hf_xet]' hf download lukealonso/GLM-5.2-NVFP4" >&2
  exit 1
fi

# ---- Tunables (defaults = benchmarked + correctness-gated best) -------------
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-1}"                       # raise for long-ctx KV capacity (DCP4 needs the newer v11 image for DCP+MTP)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"              # throughput sweet spot (peak ~763 tok/s @ c=32)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"   # MTP k=2: sweet spot; k>=3 risks illegal-memory on long ctx
GLM51_DISABLE_MTP="${GLM51_DISABLE_MTP:-0}"     # 0 = MTP ON (+~50% throughput, validated correct)

# Force the w4a16 MoE decode kernel. *** CORRECTNESS-CRITICAL ***: the default
# NVFP4 w4a4 decode kernel accumulates error and produces token-salad past ~3k
# generated tokens on long context. A16 fixes it (also +6-17% batched decode;
# costs ~7% prefill). Set 0 ONLY for prefill-only/offline work that tolerates the
# long-generation risk.
B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"

# DSA indexer: the checkpoint ships index_topk_pattern=null -> mis-selected tokens
# on long context. use_index_cache + the explicit F/S pattern (78 chars, GLM-5.2)
# fix it and speed up decode. *** CORRECTNESS-CRITICAL ***.
GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
HF_OVERRIDES="${HF_OVERRIDES:-$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "$GLM52_INDEX_TOPK_PATTERN")}"

# More NCCL channels -> better all-reduce bandwidth on this PCIe/SHM path.
NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"

# cudagraph capture MUST cover the full MTP decode batch = seqs*(1+spec); the
# image default (16) only graphs ~5 seqs and throughput falls off a cliff at
# concurrency>=8. This is the single biggest decode win (up to +390%).
if [[ "${GLM51_DISABLE_MTP}" == "1" ]]; then _spec_mult=1; else _spec_mult=$(( 1 + NUM_SPECULATIVE_TOKENS )); fi
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$(( MAX_NUM_SEQS * _spec_mult ))}"

# Persistent host cache so AOT/inductor/triton/cudagraph survive restarts. First
# cold start ~25-40min (NFS weight read); warm restarts are minutes (2TB host RAM
# page-caches the 467GB weights).
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache}"
mkdir -p "$CACHE_ROOT/jit" "$CACHE_ROOT/triton" "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/cutlass_dsl"

# Patched GLM tool-call parser: the stock glm4_moe streaming parser drops
# ZERO-ARGUMENT tool calls (name terminated by </tool_call> with no \n/<arg_key>
# returns None -> tool call never emitted), which breaks streaming agent clients
# (opencode etc.) whenever a no-arg tool is called. Fix mounts a corrected file.
PATCH_PARSER="/root/glm52-vllm/patch/glm4_moe_tool_parser.py"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# The bash entrypoint applies three host-specific fixes the stock serve-glm52.sh
# lacks on this pod (it targets a different host):
#   1. PYTHON_BIN -> /opt/venv/bin/python (script's default .venv is absent).
#   2. NCCL_GRAPH_FILE is baked to ""; the "-noxml" NCCL aborts opening an empty
#      path -> unset it + USE_NCCL_XML=0.
#   3. NCCL P2P over PCIe across 2 NUMA nodes fails ("invalid device ordinal") and
#      CUDA IPC is blocked in this pod -> NCCL_P2P_DISABLE=1 (NCCL over SHM). The
#      b12x PCIe all-reduce can't open IPC here either and falls back to NCCL,
#      which the repo confirms is faster for 8-GPU cross-socket anyway.
docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --ipc=host --network host --privileged \
  --entrypoint /bin/bash \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e MODEL="$MODEL_HOST" -e MTP_MODEL="$MODEL_HOST" -e SERVED_MODEL_NAME=GLM-5.2 \
  -e PORT="$PORT" \
  -e TP_SIZE="$TP_SIZE" -e DCP_SIZE="$DCP_SIZE" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  -e NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" -e GLM51_DISABLE_MTP="$GLM51_DISABLE_MTP" \
  -e B12X_MOE_FORCE_A16="$B12X_MOE_FORCE_A16" \
  -e HF_OVERRIDES="$HF_OVERRIDES" \
  -e NCCL_MIN_NCHANNELS="$NCCL_MIN_NCHANNELS" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -e CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl \
  -v /root/glm52-vllm/patch/glm4_moe_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/glm4_moe_tool_parser.py:ro \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/jit:/cache/jit" \
  -v "$CACHE_ROOT/triton:/root/.cache/triton" \
  -v "$CACHE_ROOT/torchinductor:/root/.cache/torchinductor" \
  -v "$CACHE_ROOT/cutlass_dsl:/root/.cache/cutlass_dsl" \
  "$IMAGE" \
  -lc 'unset NCCL_GRAPH_FILE
       export USE_NCCL_XML=0 NCCL_P2P_DISABLE=1 PYTHON_BIN=/opt/venv/bin/python
       exec /usr/local/bin/serve-glm52.sh \
         --max-cudagraph-capture-size '"${MAX_CUDAGRAPH_CAPTURE_SIZE}"' \
         --linear-backend auto'

echo "Launched '$NAME' on :$PORT"
echo "  model    : $MODEL_HOST"
echo "  TP=$TP_SIZE seqs=$MAX_NUM_SEQS cudagraph=$MAX_CUDAGRAPH_CAPTURE_SIZE MTP_k=$NUM_SPECULATIVE_TOKENS A16=$B12X_MOE_FORCE_A16"
echo "  logs     : docker logs -f $NAME   (wait for 'Application startup complete')"
echo
echo "  ALWAYS gate correctness before trusting throughput numbers:"
echo "    cd /root/llm-inference-bench && uv run --with requests python3 llm_decode_bench.py \\"
echo "      --port $PORT --model GLM-5.2 --test-profile estonia --profile-concurrency 2 \\"
echo "      --profile-runs 4 --max-tokens 8000   # expect correct_rate 1.0, hit_max_tokens 0"

###############################################################################
# TUNING SUMMARY (8x RTX PRO 6000, GLM-5.2-NVFP4, measured 2026-06-18)
# Ranked impact:
#  1. cudagraph capture = seqs*(1+spec): fixes decode cliff at concurrency>=8 (+390%).
#  2. A16 + use_index_cache + index_topk_pattern: REQUIRED for long-gen correctness
#     (token-salad past ~3k tokens otherwise). A16 also +6-17% batched decode, -7% prefill.
#  3. MTP on, k=2: +~50% throughput, stable, clean termination (k>=3 risks illegal mem).
#  4. max_num_seqs 16->32: +10-21% throughput tail. Peak ctx0: 763 tok/s @ c=32.
# Measured: single-stream ~77-81 tok/s; prefill ~2.9-3.0k tok/s (8k-64k).
# Other goals: offline batch -> MAX_NUM_SEQS=64 (capture auto->192); long context ->
#   DCP_SIZE=4 for ~Nx KV (needs the v11 image for DCP+MTP). NCCL_P2P_DISABLE=1 is
#   mandatory on this pod. KV stays fp8 (bf16 breaks the b12x sparse path).
###############################################################################
