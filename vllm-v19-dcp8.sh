#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v19 "Gilded Gnosis" + DCP8, TP8, CKV fast-path. EXPERIMENT.
#
# Purpose: exercise koush's full-CKV DCP prefill (vLLM PR #111) at TP8/DCP8,
# which the v19 image auto-enables (VLLM_DCP_QUERY_SPLIT=1,
# VLLM_B12X_MLA_CKV_GATHER=1) for the 8:8 topology. DCP8 gives a ~4.4M-token KV
# pool; v18/v19 validation showed +85-90% prefill vs v17 at 64k, and v19 also
# narrows the DCP8->DCP1 decode gap (koush's sparse-gather decode follow-up).
#
# DIFFERENCES vs production (vllm-v14-lmcache.sh, DCP2+LMCache+v17) — READ:
#   * NO LMCache. This is the image's native launcher; the myshytf LMCache MP
#     connector + 12 patches are NOT validated on the gilded-gnosis vLLM base.
#     So no warm-restore here. A separate task if we keep DCP8.
#   * DCP8 not DCP2: bigger KV pool, historically slower single-stream decode
#     (v19 claws most of that back, unproven on THIS host).
#   * SERVED_MODEL_NAME single ("GLM-5.2"); the image launcher takes one name,
#     so the "GLM-5.2-NVFP4" alias is dropped (opencode uses GLM-5.2).
#
# HOST FIX preserved: the image launcher (serve-glm52-v16.sh:207) HARDCODES
#   export VLLM_ENABLE_PCIE_ALLREDUCE=1
# which collapsed batch-1 decode on THIS host (c1 ~40 tok/s; documented v13/v14
# — NCCL all-reduce is the fix). A docker -e can't override a script `export`,
# so we extract the image launcher, relax that one line to honor the env, mount
# it back, and default VLLM_ENABLE_PCIE_ALLREDUCE=0. Override PCIE=1 to A/B it
# (the P2P fix + newer b12x MAY have changed this — worth re-measuring once).
###############################################################################
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v19-vllm7ea567a-b12xc7dc733-fi801d57a-cu132-20260719}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8443}"
TP="${TP:-8}"
DCP="${DCP:-8}"
MTP="${MTP:-3}"
MOE_MODE="${MOE_MODE:-a16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024000}"   # 1000*1024, aligned to block(64)*dcp(8)*2=1024
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"  # DCP8 profile unproven here; raise after it holds
PCIE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"     # 0 = NCCL all-reduce (host fix); 1 to A/B
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v19}"
PATCH_OUT="$CACHE_ROOT/serve-glm52-v16.pcie-env.sh"

MODEL_HOST="${MODEL_HOST:-$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/ )}"
MODEL_HOST="${MODEL_HOST%/}"
[[ -f "${MODEL_HOST}/config.json" ]] || { echo "ERROR: model snapshot not found" >&2; exit 1; }

mkdir -p "$CACHE_ROOT/jit" "$CACHE_ROOT/tmp"

# --- extract the image's real launcher and relax the hardcoded PCIe export ---
docker create --name v19extract "$IMAGE" >/dev/null
docker cp v19extract:/usr/local/bin/serve-glm52-v16.sh "$PATCH_OUT.orig" >/dev/null
docker rm -f v19extract >/dev/null
if ! grep -q '^export VLLM_ENABLE_PCIE_ALLREDUCE=1$' "$PATCH_OUT.orig"; then
  echo "WARN: expected PCIe export line not found in image launcher — image layout changed; review before running" >&2
fi
sed 's/^export VLLM_ENABLE_PCIE_ALLREDUCE=1$/export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"/' \
  "$PATCH_OUT.orig" > "$PATCH_OUT"
chmod +x "$PATCH_OUT"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --restart unless-stopped \
  --device nvidia.com/gpu=all --network host --ipc host --init --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e MODEL_FAMILY=glm52 \
  -e GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  -e MODEL="$MODEL_HOST" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2}" \
  -e PORT="$PORT" \
  -e TP="$TP" \
  -e DCP="$DCP" \
  -e MTP="$MTP" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e MOE_MODE="$MOE_MODE" \
  -e QUANTIZATION="${QUANTIZATION:-modelopt_fp4}" \
  -e ONLINE_QUANT="${ONLINE_QUANT:-none}" \
  -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
  -e F8_DMA="${F8_DMA:-0}" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="$PCIE" \
  -v "$PATCH_OUT:/usr/local/bin/serve-glm52-v16.sh:ro" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/jit:/cache" \
  -v "$CACHE_ROOT/tmp:/container-tmp" \
  "$IMAGE"

echo "Launched '$NAME' — v19 DCP$DCP TP$TP MTP$MTP $MOE_MODE, PCIe-allreduce=$PCIE, port $PORT (CKV auto-on at 8:$DCP)"
echo "Follow: docker logs -f $NAME   | expect log lines VLLM_B12X_MLA_CKV_GATHER=1 and VLLM_DCP_QUERY_SPLIT=1"
