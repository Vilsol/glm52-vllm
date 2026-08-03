#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v20 "Gilded Gnosis" (SparkInfer), TP8/DCP2. A/B CANDIDATE.
#
# Purpose: v20 image + LMCache warm-restore (RESTORES the 18.6x warm-restore we
# lost when moving off v19 on 07-24; prefill was being re-computed after eviction).
#
# STATUS 2026-07-27: v20 IS PRODUCTION. The 0723 build replaced v19 on 07-24 and
# ran 2d18h with ZERO crashes — v19 was dying at ~19.5h/22h MTBF with
# cudaErrorIllegalAddress, so the Xid31/FAULT_PDE fix is real on this host. Perf
# vs v19 was a wash (c1 106.1 vs 104.5; c8/c16 ~4-9% lower, N=1); we kept it for
# stability. This 0726 build adds the DCP comm rewrite + PCIe auto-calibration.
#
# v20 fixed the two release blockers that had kept us on v19:
#   * DCP2/DCP4 "invalid tensor layout" workspace crash — gone; use
#     DCP_PREFILL_WORKSPACE=auto (forcing 0 was the old workaround, now wrong).
#   * Xid31/FAULT_PDE cudaErrorIllegalAddress on long-context TP8/DCP2/MTP3 —
#     claimed fixed via tail-safe persistent DCP/DMA workspaces + safe MLA
#     query-BMM. This is the SAME signature as our 2026-07-23 768k-prefill crash
#     (BENCHMARKS.md Phase 14 §5), so this image is the prime suspect-fix.
# Community reports +45% prefill / +35% decode vs v17 at DCP2 (mcd6192).
#
# Settings match the old v19 production profile (DCP2/TP8/MTP3/A16/fp8-KV,
# 917504 ctx, gpu_mem 0.94) so benches stay comparable across images.
#
# VERIFIED AGAINST r4 (2026-07-27), all three anchors re-checked on the exact
# image before this launcher was pointed at it:
#   1. the hardcoded PCIe export is still in glm52-pcie-runtime-env.sh (line
#      moved 9 -> 11; our check is a regex, so line drift is harmless);
#   2. serve-glm52-v16.sh still has exactly one `cmd=(vllm serve ` and one
#      `"${spec_arg[@]}")` terminator, which the connector injection needs;
#   3. both vLLM-side patches apply (patch_kv_xfer_assert_v10 -> 3 guards,
#      patch_preempt_safe_restore -> SAFE-PREEMPT-RESTORE v3).
# The other 10 patches target the pip-pinned lmcache 0.4.6 package, so they are
# image-independent by construction.
#
# DIFFERENCES vs v19 production — READ:
#   * LMCache IS enabled here (L1=768GB RAM-only, L2=0). Expect NO throughput
#     change vs vllm-v20.sh — Phase 13 measured ~0; the entire point is
#     warm-restore (18.6x on a 90k re-send) so an evicted prefix is restored
#     from host RAM instead of re-prefilled.
#   * COLD BOOT IS SLOWER (~25-30 min): pinning 768GB of L1 reclaims physical
#     pages and evicts the ~436GB weight page cache, so weights reload from NFS.
#   * SERVED_MODEL_NAME single ("GLM-5.2"); the image launcher takes one name.
#   * SPEC_EXTEND fast path is auto-enabled in v20 (no env needed) — the win we
#     hand-set on v19 (Phase 14 §3) is default here.
#
# HOST FIX preserved (the reason this can't be a plain `docker run`):
#   The image HARDCODES  export VLLM_ENABLE_PCIE_ALLREDUCE=1  which collapses
#   batch-1 decode on THIS 2-NUMA PCIe host (c1 ~40 tok/s; documented v13/v14).
#   A docker -e can't override a script `export`, so we extract the file, relax
#   that line to honor the env, mount it back, default =0 (NCCL all-reduce).
#   Set VLLM_ENABLE_PCIE_ALLREDUCE=1 to A/B.
#
#   *** THE HARDCODE MOVES BETWEEN IMAGES — 2026-07-27 ***
#   v19/v20-0723: serve-glm52-v16.sh:267   `export VLLM_...=1`   (column 0)
#   v20-0726:     glm52-pcie-runtime-env.sh:9  `  export VLLM_...=1` (indented,
#                 inside configure_glm52_pcie_runtime_env()). The old sed
#                 silently matched NOTHING on the new image, which would have
#                 left PCIe all-reduce ON and tanked decode while looking fine.
#   => PATCH_TARGET is now explicit and a miss is FATAL, not a warning.
#   That function is shared by the pre-model PCIe calibration probe and the vLLM
#   process, so patching it keeps both consistent (which is what upstream wants).
###############################################################################
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sic3828fd-fi801d57a-cu132-20260727-r4}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8443}"
TP="${TP:-8}"
DCP="${DCP:-2}"
MTP="${MTP:-3}"
MOE_MODE="${MOE_MODE:-a16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-917504}"     # match v19 prod; 256-aligned (block64*dcp2*2)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"           # match v19 prod
# 0.96, not the 0.94 we inherited from v19: with the LMCache connector +
# --disable-hybrid-kv-cache-manager the KV budget is slightly tighter, and 0.94
# left only 23.27 GiB where 917504 ctx needs 23.38 GiB -> engine init aborted
# (2026-07-27). rtx6kpro validated 0.96 for TP8 the same day (DCP1 603,456 KV
# tokens / DCP4 2,285,824, 240k prompt + decode passed). Do NOT use 0.98: it
# boots but is unsafe for long-prefill runtime allocations.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
# DCP workspace: v20 fix REQUIRES auto (0 reintroduces the read-fault). auto.
DCP_PREFILL_WORKSPACE="${DCP_PREFILL_WORKSPACE:-auto}"
# query-split/CKV-gather: at TP8/DCP2 the helper auto-enables both (8:2 is in the
# gate). Matches v19. festr found query-split is the DCP4 cross-socket collapse
# culprit — but our DCP2 showed NO collapse to 256k (Phase 14 §1), so leave auto.
# Override DCP_QUERY_SPLIT=0 to test query-split-off (keeps CKV, loses ~4% prefill).
DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT:-auto}"
DCP_CKV_GATHER="${DCP_CKV_GATHER:-auto}"
# --- knobs added by the 2026-07-25/26 DCP rewrite --------------------------
# DCP_CKV_PREFETCH_DEPTH: depth-1 CKV prefetch overlaps the next layer's gather
#   with current-layer compute. GREAT on switched fabrics, but on multi-root-
#   complex / NUMA hosts (our class) it CAUSES the long-context prefill collapse:
#   procr2372 measured 128k prefill 2,907 -> 5,591 tok/s (+92%) by setting 0,
#   timricese confirmed on dual-Turin direct-attach. 'auto' lets the 0726 image's
#   pre-model PCIe probe decide (it correctly picked 0 for the NUMA host in
#   testing) — but VERIFY against explicit 0 on this box before trusting it.
DCP_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-auto}"
# DCP_TOPK_OWNER_MERGE: exact 2D owner merge (+7.7% 64k upstream). Helps some
#   topologies, hurts others (timricese: 0 was faster at DCP8; procr2: 0 slower).
DCP_TOPK_OWNER_MERGE="${DCP_TOPK_OWNER_MERGE:-auto}"
# PCIE_CALIBRATION: 0726 runs a lossless pre-model probe of the real fabric and
#   derives DMA crossover / query-split crossover / prefetch depth. 'force'
#   re-probes and ignores the cache; 'auto' reuses the fingerprinted cache.
PCIE_CALIBRATION="${PCIE_CALIBRATION:-auto}"
# F8_DMA: leave 0 (lossless). i8_ring is ~+6-8% prefill but upstream explicitly
#   says it is NOT quality-certified (top-20 logprob proxy only, no full KLD).
F8_DMA="${F8_DMA:-0}"
PCIE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"       # 0 = NCCL all-reduce (host fix); 1 to A/B
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v20-lmcache}"
# The hardcoded PCIe export lives HERE in the 0726 image (see header note).
PATCH_SRC="${PATCH_SRC:-/usr/local/bin/glm52-pcie-runtime-env.sh}"
PATCH_OUT="$CACHE_ROOT/glm52-pcie-runtime-env.patched.sh"

MODEL_HOST="${MODEL_HOST:-$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/ )}"
MODEL_HOST="${MODEL_HOST%/}"
[[ -f "${MODEL_HOST}/config.json" ]] || { echo "ERROR: model snapshot not found" >&2; exit 1; }

mkdir -p "$CACHE_ROOT/cache" "$CACHE_ROOT/tmp"

# --- extract the image's real launcher and relax the hardcoded PCIe export ---
docker rm -f v20extract >/dev/null 2>&1 || true
docker create --name v20extract "$IMAGE" >/dev/null
docker cp "v20extract:$PATCH_SRC" "$PATCH_OUT.orig" >/dev/null
docker rm -f v20extract >/dev/null
# FATAL on miss: a silent no-op here leaves PCIe all-reduce ON and collapses
# decode on this host while everything still *looks* healthy. The line moved
# once already (v16 -> pcie-runtime-env, col0 -> indented), so never assume.
if ! grep -qE '^[[:space:]]*export VLLM_ENABLE_PCIE_ALLREDUCE=1$' "$PATCH_OUT.orig"; then
  echo "FATAL: hardcoded 'export VLLM_ENABLE_PCIE_ALLREDUCE=1' not found in $PATCH_SRC." >&2
  echo "       The image layout changed again. Locate it with:" >&2
  echo "       docker run --rm --entrypoint bash $IMAGE -c 'grep -rn VLLM_ENABLE_PCIE_ALLREDUCE /usr/local/bin/'" >&2
  echo "       then set PATCH_SRC=<file>. Refusing to launch (decode would silently collapse)." >&2
  exit 1
fi
sed -E 's/^([[:space:]]*)export VLLM_ENABLE_PCIE_ALLREDUCE=1$/\1export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"/' \
  "$PATCH_OUT.orig" > "$PATCH_OUT"
grep -qE 'VLLM_ENABLE_PCIE_ALLREDUCE:-1' "$PATCH_OUT" || { echo "FATAL: patch did not apply" >&2; exit 1; }
chmod +x "$PATCH_OUT"

# --- patch #2: v20's serve-glm52-v16.sh builds a FIXED `cmd=(vllm serve ...)`
# and does NOT pass "$@" through, so a connector flag cannot be supplied as an
# argument. Teach it to honor $GLM52_KV_TRANSFER_CONFIG instead.
V16_SRC=/usr/local/bin/serve-glm52-v16.sh
V16_OUT="$CACHE_ROOT/serve-glm52-v16.lmcache.sh"
docker rm -f v20extract2 >/dev/null 2>&1 || true
docker create --name v20extract2 "$IMAGE" >/dev/null
docker cp "v20extract2:$V16_SRC" "$V16_OUT.orig" >/dev/null
docker rm -f v20extract2 >/dev/null
grep -q '"${spec_arg\[@\]}")' "$V16_OUT.orig" || {
  echo "FATAL: could not find the spec_arg cmd terminator in $V16_SRC — layout changed." >&2
  echo "       Inspect: docker run --rm --entrypoint bash $IMAGE -c 'grep -n \"cmd=(vllm serve\" -A40 $V16_SRC'" >&2
  exit 1; }
python3 - "$V16_OUT.orig" "$V16_OUT" <<'PYEOF'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
decl = ('glm52_lmcache_arg=()\n'
        'if [[ -n "${GLM52_KV_TRANSFER_CONFIG:-}" ]]; then\n'
        '  glm52_lmcache_arg=(--disable-hybrid-kv-cache-manager '
        '--kv-transfer-config "${GLM52_KV_TRANSFER_CONFIG}")\n'
        'fi\n'
        'cmd=(vllm serve ')
assert s.count('cmd=(vllm serve ') == 1, "cmd array not found exactly once"
s = s.replace('cmd=(vllm serve ', decl, 1)
old = '  "${spec_arg[@]}")'
assert s.count(old) == 1, "spec_arg terminator not found exactly once"
s = s.replace(old, '  "${spec_arg[@]}" \\\n  ${glm52_lmcache_arg[@]+"${glm52_lmcache_arg[@]}"})', 1)
open(dst, 'w').write(s)
print("v16 patched for --kv-transfer-config")
PYEOF
chmod +x "$V16_OUT"

PATCH_DIR=/root/glm52-vllm/patch
LMC_PATCHES="${LMC_PATCHES:-$PATCH_DIR/lmcache-v14}"
LMC_PATCH_MOUNTS=()
for p in "$LMC_PATCHES"/*.py; do LMC_PATCH_MOUNTS+=( -v "$p:/opt/$(basename "$p"):ro" ); done
[[ ${#LMC_PATCH_MOUNTS[@]} -eq 24 ]] || echo "NOTE: mounting $(( ${#LMC_PATCH_MOUNTS[@]} / 2 )) LMCache patches (expected 12)" >&2

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --network host --ipc host --init --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  --entrypoint /opt/lmcache-bootstrap-v20.sh \
  -e MODEL_FAMILY=glm52 \
  -e GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  -e MODEL="$MODEL_HOST" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2}" \
  -e PORT="$PORT" \
  -e TP="$TP" \
  -e DCP="$DCP" \
  -e DCP_QUERY_SPLIT="$DCP_QUERY_SPLIT" \
  -e DCP_CKV_GATHER="$DCP_CKV_GATHER" \
  -e DCP_PREFILL_WORKSPACE="$DCP_PREFILL_WORKSPACE" \
  -e DCP_CKV_PREFETCH_DEPTH="$DCP_CKV_PREFETCH_DEPTH" \
  -e DCP_TOPK_OWNER_MERGE="$DCP_TOPK_OWNER_MERGE" \
  -e PCIE_CALIBRATION="$PCIE_CALIBRATION" \
  -e MTP="$MTP" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e MOE_MODE="$MOE_MODE" \
  -e QUANTIZATION="${QUANTIZATION:-modelopt_fp4}" \
  -e ONLINE_QUANT="${ONLINE_QUANT:-none}" \
  -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
  -e LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}" \
  -e F8_DMA="$F8_DMA" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="$PCIE" \
  -v "$PATCH_OUT:$PATCH_SRC:ro" \
  -v "$V16_OUT:$V16_SRC:ro" \
  -v "$PATCH_DIR/lmcache-bootstrap-v20.sh:/opt/lmcache-bootstrap-v20.sh:ro" \
  "${LMC_PATCH_MOUNTS[@]}" \
  -e GLM52_LMCACHE_L1_GB="${LMCACHE_L1_GB:-768}" \
  -e GLM52_LMCACHE_L1_INIT_GB="${LMCACHE_L1_INIT_GB:-64}" \
  -e GLM52_LMCACHE_L2_GB="${LMCACHE_L2_GB:-0}" \
  -e GLM52_LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}" \
  -e GLM52_LMCACHE_VERSION="${LMCACHE_VERSION:-0.4.6}" \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/cache:/cache" \
  -v "$CACHE_ROOT/tmp:/container-tmp" \
  "$IMAGE"

echo "Launched '$NAME' — v20 DCP$DCP TP$TP MTP$MTP $MOE_MODE, PCIe-allreduce=$PCIE, ctx=$MAX_MODEL_LEN gpu_mem=$GPU_MEMORY_UTILIZATION, port $PORT (LMCache L1=${LMCACHE_L1_GB:-768}GB RAM-only)"
echo "Follow: docker logs -f $NAME   | expect VLLM_B12X_MLA_CKV_GATHER=1 and VLLM_DCP_QUERY_SPLIT=1 at 8:$DCP"
