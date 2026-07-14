#!/usr/bin/env bash
# Experimental launcher for GLM-5.2-NVFP4 on 8x RTX PRO 6000 Blackwell.
# Wraps the image's pre-tuned /usr/local/bin/serve-glm52.sh, fixing this host's
# GPU numbering (0-7) and persisting AOT/JIT/compile caches across restarts so
# warm restarts are fast. All tuning knobs are overridable via env.
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:dark-devotion-39ae3ed-b12x5b2e018-cu132-20260617}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8000}"

# Resolve the downloaded model snapshot in the HF cache.
MODEL_HOST="${MODEL_HOST:-$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/ )}"
MODEL_HOST="${MODEL_HOST%/}"
if [[ ! -f "${MODEL_HOST}/config.json" ]]; then
  echo "ERROR: model snapshot not found at ${MODEL_HOST}" >&2
  exit 1
fi

# Persistent host cache root so AOT compile / mega-artifact / flashinfer autotune
# survive container restarts (first cold start is slow, later starts are fast).
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache}"
mkdir -p "$CACHE_ROOT/jit" "$CACHE_ROOT/triton" "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/cutlass_dsl"

# --- Tunables (defaults mirror the image's serve-glm52.sh) -------------------
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GLM51_DISABLE_MTP="${GLM51_DISABLE_MTP:-0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
# extra raw vllm flags appended to serve-glm52.sh (argparse: last value wins),
# e.g. EXTRA_ARGS="--max-cudagraph-capture-size 256"
EXTRA_ARGS="${EXTRA_ARGS:-}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --ipc=host --network host --privileged \
  --entrypoint /entrypoint.sh \
  -v /root/glm52-vllm/entrypoint.sh:/entrypoint.sh:ro \
  -v /root/glm52-vllm/certs:/certs:ro \
  ${TLS_ENABLE:+-e TLS_ENABLE=$TLS_ENABLE} \
  -e PYTHON_BIN=/opt/venv/bin/python \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  ${NCCL_P2P_DISABLE:+-e NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE} \
  ${NCCL_SHM_DISABLE:+-e NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE} \
  ${NCCL_MIN_NCHANNELS:+-e NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS} \
  ${VLLM_ENABLE_PCIE_ALLREDUCE:+-e VLLM_ENABLE_PCIE_ALLREDUCE=$VLLM_ENABLE_PCIE_ALLREDUCE} \
  ${B12X_MOE_FORCE_A16:+-e B12X_MOE_FORCE_A16=$B12X_MOE_FORCE_A16} \
  ${VLLM_DCP_GLOBAL_TOPK:+-e VLLM_DCP_GLOBAL_TOPK=$VLLM_DCP_GLOBAL_TOPK} \
  ${VLLM_DCP_SHARD_DRAFT:+-e VLLM_DCP_SHARD_DRAFT=$VLLM_DCP_SHARD_DRAFT} \
  ${VLLM_PCIE_ALLREDUCE_BACKEND:+-e VLLM_PCIE_ALLREDUCE_BACKEND=$VLLM_PCIE_ALLREDUCE_BACKEND} \
  ${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:+-e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=$VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE} \
  ${NCCL_BUFFSIZE:+-e NCCL_BUFFSIZE=$NCCL_BUFFSIZE} \
  ${NCCL_MAX_NCHANNELS:+-e NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS} \
  ${NUMA_INTERLEAVE:+-e NUMA_INTERLEAVE=$NUMA_INTERLEAVE} \
  ${HF_OVERRIDES:+-e HF_OVERRIDES=$HF_OVERRIDES} \
  ${PYTORCH_CUDA_ALLOC_CONF:+-e PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF} \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e MODEL="$MODEL_HOST" \
  -e MTP_MODEL="$MODEL_HOST" \
  -e SERVED_MODEL_NAME=GLM-5.2 \
  -e PORT="$PORT" \
  -e TP_SIZE="$TP_SIZE" \
  -e DCP_SIZE="$DCP_SIZE" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  -e GLM51_DISABLE_MTP="$GLM51_DISABLE_MTP" \
  -e NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" \
  -e HF_HOME=/root/.cache/huggingface \
  -e XDG_CACHE_HOME=/cache/jit \
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
  ${EXTRA_ARGS}

echo "Launched container '$NAME' on port $PORT (model: $MODEL_HOST)"
echo "Follow logs: docker logs -f $NAME"
