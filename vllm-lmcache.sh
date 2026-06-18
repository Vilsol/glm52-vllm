#!/usr/bin/env bash
###############################################################################
# GLM-5.2-v11 + LMCache — prefill accelerator via persistent KV reuse.
# Based on github.com/myshytf/glm-5.2-v11-lmcache (serve.sh + 6 patches),
# adapted to THIS host: NCCL P2P fix, our model snapshot, our tool-call patch.
#
# LMCache caches KV in L1 (RAM) + L2 (disk). On a prefix cache HIT the prefill
# for that prefix is SKIPPED (KV loaded instead of recomputed) -> repeated long
# prompts (agent/codebase context, RAG docs, multi-turn) prefill near-instantly.
# DCP4-aware patches make all 4 DCP shards store/lookup (100% hit).
#
# NOTE: must NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments (crashes LMCache).
###############################################################################
set -euo pipefail

IMG=voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618
SNAP=$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/); SNAP="${SNAP%/}"
CACHE=/root/glm52-vllm/cache-v11
MNT=/root/glm52-vllm/lmcache-mnt
PATCH=/root/glm52-vllm/patch/glm4_moe_tool_parser.py
mkdir -p "$CACHE/jit" "$CACHE/triton" "$CACHE/torchinductor" "$CACHE/cutlass_dsl" /root/glm52-vllm/lmcache_l2

docker rm -f glm52 >/dev/null 2>&1 || true
docker run -d --name glm52 --device nvidia.com/gpu=all --ipc=host --network host --privileged \
  --entrypoint /lmcache-mnt/entrypoint-lmcache.sh \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e MODEL="$SNAP" -e SERVED_MODEL_NAME=GLM-5.2 -e SERVED_MODEL_NAMES=GLM-5.2 -e PORT=8080 \
  -e TP_SIZE=8 -e DCP_SIZE="${DCP_SIZE:-4}" -e MTP="${MTP:-1}" -e NUM_SPECULATIVE_TOKENS=2 \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}" \
  -e MAX_NUM_SEQS=16 -e MAX_NUM_BATCHED_TOKENS=8192 -e MAX_CUDAGRAPH_CAPTURE_SIZE=48 \
  -e B12X_MOE_FORCE_A16=1 -e NCCL_MIN_NCHANNELS=4 \
  -e GLM52_ENABLE_LMCACHE=1 -e GLM52_LMCACHE_CHUNK_SIZE="${CHUNK:-512}" \
  -e GLM52_LMCACHE_L1_GB="${L1_GB:-96}" -e GLM52_LMCACHE_L1_INIT_GB=8 \
  -e GLM52_LMCACHE_L2_GB="${L2_GB:-200}" -e GLM52_LMCACHE_DISK_PATH=/lmcache_l2 \
  -e HF_HOME=/root/.cache/huggingface -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor -e CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl \
  -v "$MNT:/lmcache-mnt" \
  -v "$PATCH:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/glm4_moe_tool_parser.py:ro" \
  -v /root/glm52-vllm/lmcache_l2:/lmcache_l2 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE/jit:/cache/jit" -v "$CACHE/triton:/root/.cache/triton" \
  -v "$CACHE/torchinductor:/root/.cache/torchinductor" -v "$CACHE/cutlass_dsl:/root/.cache/cutlass_dsl" \
  "$IMG"
echo "launched glm52 + LMCache (DCP=${DCP_SIZE:-4}, L1=${L1_GB:-96}GB). docker logs -f glm52"
