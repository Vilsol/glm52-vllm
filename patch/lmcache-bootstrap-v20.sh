#!/usr/bin/env bash
###############################################################################
# lmcache-bootstrap-v20.sh — add LMCache warm-restore to the v20 image WITHOUT
# replacing the image's own launcher chain.
#
# WHY THIS SHAPE: our old v19 path used patch/serve_glm52_v14_lmcache.sh, which
# builds the whole `vllm serve` command itself with v14-era flags. Reusing it on
# v20 would throw away everything v20 added — the DCP comm rewrite, the exact
# owner top-k merge, and the pre-model PCIe calibration probe. So instead this
# script does ONLY the LMCache-specific setup and then hands off to the image's
# real entrypoint (serve-gilded-gnosis.sh -> v19 -> v16), which still chooses
# every DCP/calibration knob itself.
#
# The one thing the image cannot do for us is add --kv-transfer-config: v20's
# serve-glm52-v16.sh builds a fixed `cmd=(vllm serve ...)` array and does NOT
# pass "$@" through (verified 2026-07-27), so a connector flag cannot be
# injected as an argument. The launcher therefore also mounts a v16 patched to
# honor $GLM52_KV_TRANSFER_CONFIG, which this script exports below.
#
# Order matters: LMCache must be installed and patched, and its ZMQ server must
# be accepting connections, BEFORE vLLM starts and tries to register with it.
###############################################################################
set -euo pipefail

PY=/opt/venv/bin/python
PIP=/opt/venv/bin/pip
LMCACHE_PIN="${GLM52_LMCACHE_VERSION:-0.4.6}"

# ── 1. LMCache install (pip-pinned; independent of the vLLM build) ──────────
if ! ${PIP} show lmcache 2>/dev/null | grep -q "^Version: ${LMCACHE_PIN}$"; then
  echo "[lmcache-v20] Installing LMCache==${LMCACHE_PIN} ..."
  ${PIP} install "lmcache==${LMCACHE_PIN}" --quiet 2>&1 | tail -5
fi

# ── 2. Apply our 12 patches ────────────────────────────────────────────────
# 10 target the lmcache package (version-pinned, so v20-safe by construction);
# only patch_kv_xfer_assert_v10 (scheduler.py) and patch_preempt_safe_restore
# (lmcache_mp_connector.py) touch vLLM. Both were verified to apply cleanly to
# the 2026-07-26 v20 build before this launcher was written.
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
    echo "[lmcache-v20] Applying $(basename "${patch}")"
    "${PY}" "${patch}" || { echo "[lmcache-v20] FATAL: $(basename "${patch}") failed" >&2; exit 1; }
  fi
done

# ── 3. Start the LMCache MP (ZMQ) server ───────────────────────────────────
LMCACHE_MP_HOST="${GLM52_LMCACHE_MP_HOST:-localhost}"
LMCACHE_MP_PORT="${GLM52_LMCACHE_MP_PORT:-5555}"
LMCACHE_HTTP_PORT="${GLM52_LMCACHE_HTTP_PORT:-8088}"
LMCACHE_CHUNK_SIZE="${GLM52_LMCACHE_CHUNK_SIZE:-256}"
LMCACHE_L1_GB="${GLM52_LMCACHE_L1_GB:-768}"
LMCACHE_L1_INIT_GB="${GLM52_LMCACHE_L1_INIT_GB:-64}"
LMCACHE_L2_GB="${GLM52_LMCACHE_L2_GB:-0}"          # RAM-only: node-local disk is FORBIDDEN
LMCACHE_DISK_PATH="${GLM52_LMCACHE_DISK_PATH:-/lmcache/l2}"
LMCACHE_LOG="${GLM52_LMCACHE_LOG:-/tmp/lmcache_mp_server.log}"

L2_ARGS=()
if [[ "${LMCACHE_L2_GB}" != "0" ]]; then
  mkdir -p "${LMCACHE_DISK_PATH}"
  L2_ARGS+=( --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"${LMCACHE_DISK_PATH}\",\"relative_tmp_dir\":\"tmp\",\"max_capacity_gb\":${LMCACHE_L2_GB},\"use_odirect\":true,\"num_workers\":8,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.1}}" )
fi

echo "[lmcache-v20] Starting LMCache MP server: tcp://${LMCACHE_MP_HOST}:${LMCACHE_MP_PORT} L1=${LMCACHE_L1_GB}GB init=${LMCACHE_L1_INIT_GB}GB L2=${LMCACHE_L2_GB}GB chunk=${LMCACHE_CHUNK_SIZE}"
rm -f "${LMCACHE_LOG}"
# NOTE: flag list copied verbatim from patch/serve_glm52_v14_lmcache.sh — do not
# "tidy" it. --eviction-policy is REQUIRED by the 0.4.6 CLI (omitting it makes
# `lmcache server` exit immediately with a usage error), and the two TTLs are
# what keep entries alive for a day rather than the much shorter default.
lmcache server \
  --host "${LMCACHE_MP_HOST}" \
  --port "${LMCACHE_MP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_GB}" \
  --l1-init-size-gb "${LMCACHE_L1_INIT_GB}" \
  --l1-write-ttl-seconds "${GLM52_LMCACHE_WRITE_TTL:-86400}" \
  --l1-read-ttl-seconds "${GLM52_LMCACHE_READ_TTL:-86400}" \
  --eviction-policy "${GLM52_LMCACHE_EVICTION_POLICY:-LRU}" \
  "${L2_ARGS[@]}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  >"${LMCACHE_LOG}" 2>&1 &
LMCACHE_MP_PID=$!
trap 'kill ${LMCACHE_MP_PID:-} 2>/dev/null || true' EXIT

# Pinning 768 GB takes ~75s; wait for the explicit ready marker, not a sleep.
_ready=0
for _ in $(seq 1 "${GLM52_LMCACHE_READY_TIMEOUT:-600}"); do
  if ! kill -0 "${LMCACHE_MP_PID}" 2>/dev/null; then
    echo "[lmcache-v20] FATAL: MP server died during startup; log:" >&2
    sed -n '1,220p' "${LMCACHE_LOG}" >&2 || true
    exit 1
  fi
  if [[ -f "${LMCACHE_LOG}" ]] && grep -q "ZMQ cache server is running" "${LMCACHE_LOG}"; then
    _ready=1; break
  fi
  sleep 1
done
[[ "${_ready}" == "1" ]] || {
  echo "[lmcache-v20] FATAL: MP server not ready in time; log:" >&2
  sed -n '1,220p' "${LMCACHE_LOG}" >&2 || true
  exit 1
}
echo "[lmcache-v20] LMCache MP server ready"

# ── 4. Hand the connector config to the patched v16 and start the image ────
export GLM52_KV_TRANSFER_CONFIG
GLM52_KV_TRANSFER_CONFIG=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://%s","lmcache.mp.port":%s,"lmcache.mp.mq_timeout":%s,"lmcache.mp.heartbeat_interval":5}}' \
  "${LMCACHE_MP_HOST}" "${LMCACHE_MP_PORT}" "${GLM52_LMCACHE_MQ_TIMEOUT:-300}")

# The MP connector is INCOMPATIBLE with expandable_segments:True — vLLM refuses
# to start ("KV connector LMCacheMPConnector is incompatible with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"), because the CUDA VMM
# allocator can remap KV virtual addresses onto different physical pages and
# invalidate pinned KV memory.
#
# MUST be exported EMPTY, not unset. serve-glm52-v16.sh:47 reads
#   PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
# with `-` (not `:-`), so the default applies when the var is UNSET but an empty
# value is preserved. `unset` therefore hands us expandable_segments:True — the
# opposite of the intent. (This is why the community configs write the bare
# `PYTORCH_CUDA_ALLOC_CONF=` with nothing after the `=`.)
export PYTORCH_CUDA_ALLOC_CONF=""

echo "[lmcache-v20] handing off to the image launcher (DCP/calibration decided by v20)"
exec /usr/local/bin/serve-gilded-gnosis.sh "$@"
