#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v14 + LMCache (community MP-connector port), DCP4 profile.
#
# Purpose: several parallel long-context users. DCP2 (default) keeps the best
# single-stream/prefill speed with a ~1.09M-token GPU KV pool; LMCache L1 in
# host RAM (768GB, ~15M tokens) warm-restores any user's context that gets
# evicted from the GPU pool, turning a multi-minute re-prefill into a seconds
# RAM restore. Override DCP=4 for a larger 2.1M resident pool if users are
# truly-concurrent long-decode streams rather than interleaved.
#
# Based on myshytf/glm-5.2-v11-lmcache serve_glm52_v14_lmcache.sh (v14 port,
# LMCache 0.4.6 pip-installed at container start + 12 patches). Local copy at
# patch/serve_glm52_v14_lmcache.sh with ONE change: fuse_allreduce_rms is
# opt-in (FUSE_ALLREDUCE=1) — it crashed this host pre-P2P-fix; retest later.
#
# HOST OVERRIDES vs the community defaults:
#   VLLM_ENABLE_PCIE_ALLREDUCE=0  b12x PCIe all-reduce collapses batch-1 decode
#                                 here (v13 AND v14, measured) — NCCL instead.
#   L1=768GB RAM-only (~15M tokens @ 53.4KB/token; we have 2TB, weights need
#                                 ~436GB page-cached so ~674GB slack remains —
#                                 the last size where weight-cache eviction is a
#                                 non-issue. init 64GB, ~75s boot pinning).
#   L2=0 (RAM-only per requirement; the disk-tier code path is retained but off).
#   MAX_NUM_SEQS=16, GRAPH=64     community lmcache-tuned values (also matches
#                                 our Phase-8 lesson: lmcache likes fewer seqs).
#
# NOTE from upstream: do NOT combine with PYTORCH_CUDA_ALLOC_CONF=
# expandable_segments (the MP connector fights it; script unsets it).
#
# INSTANTTENSOR_CONCURRENCY (2026-07-09): default 16 (auto would be
# min(32,cpu)//world_size = 4). BENCHMARKED cold: total boot 28 min @16 vs 36 min
# @4 (~22% faster, N=1 each so partly variance). PEAK NFS throughput is unchanged
# at ~270 MB/s either way — the single Longhorn RWX NFS share-manager caps it, and
# nconnect=16 hits the same wall. The time win comes from fewer pipeline STALLS
# (more reads in flight -> higher AVERAGE throughput), not a higher ceiling.
# Kept at 16 (helps or is harmless; 32 worth trying). The real cold-boot fix is a
# Longhorn RWO *block* volume (bypasses the share-manager). Warm boots ~5 min.
###############################################################################
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v19-vllm7ea567a-b12xc7dc733-fi801d57a-cu132-20260719}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8443}"
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v19-lmcache}"
L2_HOST_DIR="${L2_HOST_DIR:-/var/lib/lmcache-l2}"
PATCH_DIR=/root/glm52-vllm/patch
# LMCache DCP patches: vendored in patch/lmcache-v14 (self-contained); falls back
# to an external myshytf clone if present. See patch/lmcache-v14/SOURCE.md.
# Use the committed vendored patches (source of truth). The external myshytf clone
# at /root/glm-5.2-v11-lmcache/patches USED to override this, which silently ignored
# vendored edits (bit us with the KV-xfer crash fix, 2026-07-14). Dirs are now
# byte-identical; override only via an explicit LMC_PATCHES=... if ever needed.
LMC_PATCHES="${LMC_PATCHES:-$PATCH_DIR/lmcache-v14}"

MODEL_HOST="${MODEL_HOST:-$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/ )}"
MODEL_HOST="${MODEL_HOST%/}"
[[ -f "${MODEL_HOST}/config.json" ]] || { echo "ERROR: model snapshot not found" >&2; exit 1; }

mkdir -p "$CACHE_ROOT/cache" "$CACHE_ROOT/tmp" "$L2_HOST_DIR/l2" "$L2_HOST_DIR/tmp"

# Mount every community patch the serve script applies from /opt/.
PATCH_MOUNTS=()
for p in "$LMC_PATCHES"/*.py; do
  PATCH_MOUNTS+=( -v "$p:/opt/$(basename "$p"):ro" )
done

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --network host --ipc host --init --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  -e GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  -e MODEL="$MODEL_HOST" \
  -e SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-GLM-5.2 GLM-5.2-NVFP4}" \
  -e PORT="$PORT" \
  -e TP="${TP:-8}" \
  -e DCP="${DCP:-8}" \
  -e DCP_BACKEND="${DCP_BACKEND:-a2a}" \
  -e MTP="${MTP:-3}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
  -e GRAPH="${GRAPH:-64}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024000}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}" \
  -e MOE_MODE="${MOE_MODE:-a16}" \
  -e F8_DMA="${F8_DMA:-0}" \
  -e LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}" \
  -e INSTANTTENSOR_CONCURRENCY="${INSTANTTENSOR_CONCURRENCY:-16}" \
  ${INSTANTTENSOR_IO_DEPTH:+-e INSTANTTENSOR_IO_DEPTH=$INSTANTTENSOR_IO_DEPTH} \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}" \
  -e VLLM_DCP_QUERY_SPLIT="${VLLM_DCP_QUERY_SPLIT:-1}" \
  -e VLLM_B12X_MLA_CKV_GATHER="${VLLM_B12X_MLA_CKV_GATHER:-1}" \
  -e DCP_PREFILL_WORKSPACE="${DCP_PREFILL_WORKSPACE:-auto}" \
  -e FUSE_ALLREDUCE="${FUSE_ALLREDUCE:-0}" \
  -e GLM52_ENABLE_LMCACHE=1 \
  -e GLM52_LMCACHE_L1_GB="${LMCACHE_L1_GB:-768}" \
  -e GLM52_LMCACHE_L1_INIT_GB="${LMCACHE_L1_INIT_GB:-64}" \
  -e GLM52_LMCACHE_L2_GB="${LMCACHE_L2_GB:-0}" \
  -e GLM52_LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}" \
  -e TLS_ENABLE="${TLS_ENABLE:-0}" \
  -e TLS_CERT="${TLS_CERT:-/certs/cert.pem}" \
  -e TLS_KEY="${TLS_KEY:-/certs/key.pem}" \
  -v "$PATCH_DIR/serve_glm52_v14_lmcache.sh:/opt/serve_glm52_v14_lmcache.sh:ro" \
  "${PATCH_MOUNTS[@]}" \
  -v /root/glm52-vllm/certs:/certs:ro \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/cache:/cache" \
  -v "$CACHE_ROOT/tmp:/container-tmp" \
  -v "$L2_HOST_DIR/l2:/lmcache/l2" \
  -v "$L2_HOST_DIR/tmp:/lmcache/tmp" \
  "$IMAGE" \
  bash /opt/serve_glm52_v14_lmcache.sh

echo "Launched '$NAME' on port $PORT (v14+LMCache, DCP${DCP:-4} L1=${LMCACHE_L1_GB:-768}GB RAM-only)"
echo "Follow logs: docker logs -f $NAME"
