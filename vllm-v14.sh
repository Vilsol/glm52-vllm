#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v14 "eldritch v7" image, ~1M-context production-candidate.
#
# v14 = Luke's b12x rework (beats both old-B12X and FlashInfer SM120 — the
# attention-backend question is over, everything is B12X now) + online MXFP8
# conversion of BF16 dense linears + fp8-compressed DMA + hybrid DCP + the
# InstantTensor loader. Recipe: rtx6kpro models/glm5.2_v14.md.
#
# INTERFACE CHANGE vs v13: the image ships its own launcher
# /usr/local/bin/run-glm52-v14-server reading env (TP/DCP/MTP/MOE_MODE/...),
# so this script replaces launch.sh+entrypoint.sh entirely for v14. No
# --privileged (verified unnecessary by community; our pod may not grant it).
#
# Default trial profile (community-consensus mainline, quality-leaning):
#   MOE_MODE=a16       KLD 0.067-0.072 vs a4's 0.107-0.121 (a4 = faster prefill,
#                      measurably dumber; a16 validated by Korean-quality tests)
#   ONLINE_MXFP8=1     BF16 dense linears -> MXFP8 at load: +8% decode, ~+0.005 KLD
#   F8_DMA=ring        fp8-compressed DMA all-reduce: +15-23% prefill, ~+0.01 KLD
#   hybrid DCP         a2a for decode steps (<=64 tok), ag_rs for prefill batches
#   instanttensor      2x faster model load (BUFFERED reuses page cache when warm)
# Reference numbers (festr's rig, DCP2/MTP3/A16/f8=0/131k/gpu0.90):
#   KV 1,079,424 | cc1 111.9 | cc8 493.8 | cc32 1,144 | DCP1 prefill ~6.2k
#
# HOST NOTE: GPU P2P works on this host (see entrypoint.sh header / Phase 10);
# fixes are persistent (modprobe.d + gpu-acs-p2p.service).
# ALL-REDUCE (measured 2026-07-07): the image launcher HARDCODES the b12x PCIe
# all-reduce on, and just like on v13 its small-payload path collapses batch-1
# decode on this host (c1 ~41-46 tok/s across every DCP/MTP/F8 combination; v13
# showed the same 30 tok/s signature). We mount a patched launcher that makes it
# env-overridable and default VLLM_ENABLE_PCIE_ALLREDUCE=0 (NCCL all-reduce).
# CAVEAT: F8_DMA rides the PCIe-DMA path, so disabling it may cost the fp8-DMA
# prefill boost — measured tradeoff documented in BENCHMARKS.md.
#
# A/B on another port:  NAME=glm52-v14 PORT=8081 bash vllm-v14.sh
###############################################################################
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8443}"
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v14}"

MODEL_HOST="${MODEL_HOST:-$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/ )}"
MODEL_HOST="${MODEL_HOST%/}"
[[ -f "${MODEL_HOST}/config.json" ]] || { echo "ERROR: model snapshot not found at ${MODEL_HOST}" >&2; exit 1; }

mkdir -p "$CACHE_ROOT/cache" "$CACHE_ROOT/tmp"

# Context/parallelism. gpu_mem 0.93 (not the old 0.95): v14's memory profile at
# 1M ctx is unproven on this host; reference KV @0.90/DCP2 was 1.08M tokens so
# 0.93 should hold a 1M request with margin. Raise after it proves out.
DCP="${DCP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MTP="${MTP:-3}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --network host --ipc host --init --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  -e GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  -e MODEL="$MODEL_HOST" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2}" \
  -e PORT="$PORT" \
  -e TP="${TP:-8}" \
  -e DCP="$DCP" \
  -e DCP_BACKEND="${DCP_BACKEND:-a2a}" \
  -e DCP_A2A_MAX_TOKENS="${DCP_A2A_MAX_TOKENS:-64}" \
  -e DCP_A2A_LARGE_BACKEND="${DCP_A2A_LARGE_BACKEND:-ag_rs}" \
  -e MTP="$MTP" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e MOE_MODE="${MOE_MODE:-a16}" \
  -e MOE_BACKEND="${MOE_BACKEND:-b12x}" \
  -e LINEAR_BACKEND="${LINEAR_BACKEND:-auto}" \
  -e ONLINE_MXFP8="${ONLINE_MXFP8:-1}" \
  -e F8_DMA="${F8_DMA:-ring}" \
  -e LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}" \
  -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}" \
  -e INSTANTTENSOR_CONCURRENCY="${INSTANTTENSOR_CONCURRENCY:-16}" \
  ${INSTANTTENSOR_IO_DEPTH:+-e INSTANTTENSOR_IO_DEPTH=$INSTANTTENSOR_IO_DEPTH} \
  -e QUANTIZATION="${QUANTIZATION:-modelopt_fp4}" \
  -e GLM52_INDEX_TOPK_PATTERN="FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}" \
  -e TLS_ENABLE="${TLS_ENABLE:-0}" \
  -e TLS_CERT="${TLS_CERT:-/certs/cert.pem}" \
  -e TLS_KEY="${TLS_KEY:-/certs/key.pem}" \
  -v /root/glm52-vllm/patch/run-glm52-v14-server:/usr/local/bin/run-glm52-v14-server:ro \
  -v /root/glm52-vllm/certs:/certs:ro \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/cache:/cache" \
  -v "$CACHE_ROOT/tmp:/container-tmp" \
  "$IMAGE" \
  run-glm52-v14-server ${EXTRA_ARGS:-}

echo "Launched '$NAME' on port $PORT (v14, DCP$DCP MTP$MTP $([ "${ONLINE_MXFP8:-1}" = 1 ] && echo mxfp8) f8=${F8_DMA:-ring})"
echo "Follow logs: docker logs -f $NAME"
