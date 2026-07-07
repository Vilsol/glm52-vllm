#!/usr/bin/env bash
# GLM-5.2 v14 clean Luke NVFP4 — DCP=4 + A16 + AG + LMCache MP connector.
# Base launcher adapted from /usr/local/bin/run-glm52-v14-server in the v14 image.
# LMCache/DCP patch set and MP server path mirror /home/g0san/glm52-v12/serve_glm52_v12.sh.
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_PREFIX_CACHE_RETENTION_INTERVAL 2>/dev/null || true
# Do NOT use expandable_segments with LMCache MP connector.
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

PY=/opt/venv/bin/python
PIP=/opt/venv/bin/pip

# ── LMCache install (one-time per container start) ──────────────────────
LMCACHE_PIN="${GLM52_LMCACHE_VERSION:-0.4.6}"
if ! ${PIP} show lmcache 2>/dev/null | grep -q "^Version: ${LMCACHE_PIN}$"; then
  echo "[glm52-v14] Installing LMCache==${LMCACHE_PIN} (one-time per container start)..."
  ${PIP} install "lmcache==${LMCACHE_PIN}" --quiet 2>&1 | tail -5
  echo "[glm52-v14] LMCache installed"
fi

# ── Apply v12 LMCache/DCP patches verbatim ──────────────────────────────
for patch in \
  /opt/patch_kv_xfer_assert_v10.py \
  /opt/patch_fs_l2_adapter.py \
  /opt/patch_l1_evict_flush_to_l2.py \
  /opt/patch_native_sync_store.py \
  /opt/patch_fs_native_startup_scan.py \
  /opt/patch_preempt_safe_restore.py \
  /opt/patch_dcp_lmcache.py \
  /opt/patch_dcp_mla_store.py \
  /opt/patch_odirect_aligned.py \
  /opt/patch_empty_tools.py \
  /opt/patch_free_locks_salt.py \
  /opt/patch_session_flush_to_l2.py; do
  if [[ -f "${patch}" ]]; then
    echo "[glm52-v14] Applying $(basename "${patch}")"
    "${PY}" "${patch}"
  fi
done

# ── Config ──────────────────────────────────────────────────────────────
MODEL="${MODEL:-/models/GLM-5.2-NVFP4}"
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-${SERVED_MODEL_NAME:-GLM-5.2-v13-lmcache}}"
read -r -a SERVED_MODEL_NAME_ARGS <<< "${SERVED_MODEL_NAMES}"
PORT="${PORT:-5317}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
DCP="${DCP:-4}"
DCP_BACKEND="${DCP_BACKEND:-ag_rs}"
MTP="${MTP:-3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GRAPH="${GRAPH:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024000}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MOE_MODE="${MOE_MODE:-a16}"
ONLINE_MXFP8="${ONLINE_MXFP8:-0}"
F8_DMA="${F8_DMA:-ag}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
QUANTIZATION_CONFIG_JSON="${QUANTIZATION_CONFIG_JSON:-{\"linear\":{\"weight\":\"mxfp8\"}}}"
GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"

case "${MOE_MODE}" in
  a4|native|default)
    B12X_MOE_FORCE_A8=0
    B12X_MOE_FORCE_A16=0
    ;;
  a16|force-a16)
    B12X_MOE_FORCE_A8=0
    B12X_MOE_FORCE_A16=1
    ;;
  force-a8-experimental|a8-experimental|a8)
    B12X_MOE_FORCE_A8=1
    B12X_MOE_FORCE_A16=0
    ;;
  *)
    die "MOE_MODE must be a4, a16, or force-a8-experimental"
    ;;
esac

case "${F8_DMA}" in
  0|ag|ring) ;;
  *) die "F8_DMA must be 0, ag, or ring" ;;
esac

[[ "${ONLINE_MXFP8}" =~ ^(0|1)$ ]] || die "ONLINE_MXFP8 must be 0 or 1"
[[ "${MTP}" =~ ^[0-9]+$ ]] || die "MTP must be an integer token count"
[[ "${MAX_NUM_SEQS}" =~ ^[0-9]+$ ]] || die "MAX_NUM_SEQS must be an integer"
[[ "${GRAPH}" =~ ^[0-9]+$ ]] || die "GRAPH must be an integer"
[[ "${#GLM52_INDEX_TOPK_PATTERN}" -eq 78 ]] || die "GLM52_INDEX_TOPK_PATTERN must be exactly 78 characters, got ${#GLM52_INDEX_TOPK_PATTERN}"

# ── v14 runtime environment ─────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPUS}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUTE_DSL_ARCH=sm_120a
export TORCH_CUDA_ARCH_LIST=12.0a
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND}"
export VLLM_USE_AOT_COMPILE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_MEGA_AOT_ARTIFACT=1
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_WO_PROJECTION=1
export VLLM_USE_B12X_MHC=1
export VLLM_USE_B12X_FP8_GEMM=1
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1
export VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-1}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
export VLLM_PCIE_DMA_FP8="${F8_DMA}"
export B12X_PCIE_DMA_FP8="${F8_DMA}"
export VLLM_DCP_GLOBAL_TOPK=1
export VLLM_DCP_SHARD_DRAFT=1
export VLLM_DCP_GLOBAL_TOPK_PREFILL_ONLY=0
export VLLM_DCP_TOPK_FORCE_DEEPGEMM=0
export B12X_MLA_SM120_UNIFIED=1
export B12X_DENSE_SPLITK_TURBO=1
export B12X_W4A16_TC_DECODE=1
export B12X_W4A8_TINY_DECODE=1
export B12X_MOE_FORCE_A8="${B12X_MOE_FORCE_A8}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16}"
export NCCL_PROTO=LL,LL128,Simple
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
if [[ -e /opt/libnccl-local-inference.so.2.30.4 ]]; then
  export LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4
  export VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4
fi
export TMPDIR="${TMPDIR:-/container-tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-/cache/vllm}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/cache/tilelang}"
export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-/cache/tilelang/tmp}"
export TVM_CACHE_DIR="${TVM_CACHE_DIR:-/cache/tvm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/cache/torch_extensions}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/cache/flashinfer}"

mkdir -p \
  "${TMPDIR}" \
  "${VLLM_CACHE_DIR}" \
  "${TILELANG_CACHE_DIR}" \
  "${TILELANG_TMP_DIR}" \
  "${TVM_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}"

# ── vLLM serve args ─────────────────────────────────────────────────────
hf_overrides="$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "${GLM52_INDEX_TOPK_PATTERN}")"
args=(
  -m vllm.entrypoints.cli.main serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME_ARGS[@]}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size "${DCP}"
  --quantization "${QUANTIZATION}"
  --kv-cache-dtype fp8
  --attention-backend B12X_MLA_SPARSE
  --moe-backend b12x
  --load-format "${LOAD_FORMAT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
  --max-cudagraph-capture-size "${GRAPH}"
  --async-scheduling
  --enable-chunked-prefill
  --enable-prefix-caching
  --enable-flashinfer-autotune
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
  --default-chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":false}'
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-request-id-headers
  --hf-overrides "${hf_overrides}"
)

if [[ "${DCP}" != "1" ]]; then
  args+=( --dcp-comm-backend "${DCP_BACKEND}" --dcp-kv-cache-interleave-size 1 )
fi

# Clean Luke baseline means ONLINE_MXFP8=0 → do not pass quantization-config.
if [[ "${ONLINE_MXFP8}" == "1" ]]; then
  args+=( --quantization-config "${QUANTIZATION_CONFIG_JSON}" )
fi

if [[ "${MTP}" != "0" ]]; then
  spec_json="$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"b12x","draft_sample_method":"probabilistic"}' "${MODEL}" "${MTP}")"
  args+=( --speculative-config "${spec_json}" )
fi

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=( --api-key "${VLLM_API_KEY}" )
fi
if [[ "${FUSE_ALLREDUCE:-0}" == "1" ]]; then
  args+=( -cc.pass_config.fuse_allreduce_rms=True )
fi

# ── LMCache MP server ───────────────────────────────────────────────────
if [[ "${GLM52_ENABLE_LMCACHE:-1}" == "1" ]]; then
  LMCACHE_MP_HOST="${GLM52_LMCACHE_MP_HOST:-localhost}"
  LMCACHE_MP_PORT="${GLM52_LMCACHE_MP_PORT:-5555}"
  LMCACHE_HTTP_PORT="${GLM52_LMCACHE_HTTP_PORT:-8088}"
  LMCACHE_CHUNK_SIZE="${GLM52_LMCACHE_CHUNK_SIZE:-256}"
  LMCACHE_L1_GB="${GLM52_LMCACHE_L1_GB:-48}"
  LMCACHE_L1_INIT_GB="${GLM52_LMCACHE_L1_INIT_GB:-48}"
  LMCACHE_L2_GB="${GLM52_LMCACHE_L2_GB:-2800}"
  LMCACHE_DISK_PATH="${GLM52_LMCACHE_DISK_PATH:-/lmcache/l2}"
  LMCACHE_LOG="${GLM52_LMCACHE_LOG:-/tmp/lmcache_mp_server.log}"

  mkdir -p "${LMCACHE_DISK_PATH}"
  echo "[glm52-v14] Starting LMCache MP server: tcp://${LMCACHE_MP_HOST}:${LMCACHE_MP_PORT}, L1=${LMCACHE_L1_GB}GB init=${LMCACHE_L1_INIT_GB}GB, L2=${LMCACHE_L2_GB}GB, disk=${LMCACHE_DISK_PATH}, chunk=${LMCACHE_CHUNK_SIZE}"
  rm -f "${LMCACHE_LOG}"
  L2_ARGS=()
  if [[ "${LMCACHE_L2_GB}" != "0" ]]; then
    L2_ARGS+=( --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"${LMCACHE_DISK_PATH}\",\"relative_tmp_dir\":\"tmp\",\"max_capacity_gb\":${LMCACHE_L2_GB},\"use_odirect\":true,\"num_workers\":8,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.1}}" )
  fi
  lmcache server \
    --host "${LMCACHE_MP_HOST}" \
    --port "${LMCACHE_MP_PORT}" \
    --chunk-size "${LMCACHE_CHUNK_SIZE}" \
    --l1-size-gb "${LMCACHE_L1_GB}" \
    --l1-init-size-gb "${LMCACHE_L1_INIT_GB}" \
    --l1-write-ttl-seconds 86400 \
    --l1-read-ttl-seconds 86400 \
    --eviction-policy LRU \
    "${L2_ARGS[@]}" \
    --http-port "${LMCACHE_HTTP_PORT}" \
    >"${LMCACHE_LOG}" 2>&1 &
  LMCACHE_MP_PID=$!
  trap 'kill ${LMCACHE_MP_PID:-} 2>/dev/null || true' EXIT

  _lmcache_ready=0
  for _i in $(seq 1 120); do
    if ! kill -0 "${LMCACHE_MP_PID}" 2>/dev/null; then
      echo "[glm52-v14] LMCache MP server exited during startup; log follows:" >&2
      sed -n '1,220p' "${LMCACHE_LOG}" >&2 || true
      exit 1
    fi
    if [[ -f "${LMCACHE_LOG}" ]] && grep -q "ZMQ cache server is running" "${LMCACHE_LOG}"; then
      _lmcache_ready=1
      break
    fi
    sleep 1
  done
  if [[ "${_lmcache_ready}" != "1" ]]; then
    echo "[glm52-v14] LMCache MP server did not become ready; log follows:" >&2
    sed -n '1,220p' "${LMCACHE_LOG}" >&2 || true
    exit 1
  fi
  echo "[glm52-v14] LMCache MP server ready"

  _kv_transfer_config=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://%s","lmcache.mp.port":%s,"lmcache.mp.mq_timeout":%s,"lmcache.mp.heartbeat_interval":5}}' "${LMCACHE_MP_HOST}" "${LMCACHE_MP_PORT}" "${GLM52_LMCACHE_MQ_TIMEOUT:-300}")
  args+=( --disable-hybrid-kv-cache-manager --kv-transfer-config "${_kv_transfer_config}" )
fi

echo "[glm52-v14] mode=clean-luke DCP=${DCP} DCP_BACKEND=${DCP_BACKEND} TP=${TP} MTP=${MTP} max-num-seqs=${MAX_NUM_SEQS} max-model-len=${MAX_MODEL_LEN} graph=${GRAPH} gpu-mem=${GPU_MEMORY_UTILIZATION} LMCACHE=${GLM52_ENABLE_LMCACHE:-1} MOE_MODE=${MOE_MODE} ONLINE_MXFP8=${ONLINE_MXFP8} F8_DMA=${F8_DMA}"
echo "[glm52-v14] served-models=${SERVED_MODEL_NAMES} HF_OVERRIDES=${hf_overrides}"
echo "[glm52-v14] exec: ${PY} -m vllm.entrypoints.cli.main serve ${MODEL} --served-model-name ${SERVED_MODEL_NAMES} --port ${PORT} --decode-context-parallel-size ${DCP} --kv-transfer-config <lmcache> ${VLLM_API_KEY:+--api-key <redacted>}"
exec "${PY}" "${args[@]}"
