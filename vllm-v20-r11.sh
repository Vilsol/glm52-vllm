#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v20 "Gilded Gnosis" r11, TP8/DCP2 + NATIVE LMCache.
#
# SUPERSEDES vllm-v20-lmcache.sh (r4 + our hand-rolled LMCache stack).
#
# WHY r11 AND NOT r9 (this launcher was written for r9 first, then repointed):
#   r9 ships a LMCache MP-server bug that hangs requests permanently. In
#   lmcache/v1/multiprocess/mq.py, _call_blocking_handler._notify_response, the
#   failure path was:
#       except Exception:
#           logger.exception("Error in blocking handler")
#   — it logs and sends NOTHING. Every success path puts frames on output_queue,
#   so on a handler exception the client future stays pending forever and the
#   vLLM request is pinned in WAITING_FOR_REMOTE_KVS until the engine restarts.
#   Verified present in r9 (mq.py:574) and fixed in r11 (mq.py:633-635, now
#   calls self._queue_error_response(prefix_frames, exc)). lmcache 0.5.2+
#   glm52dcp.3 -> .4. Reported by flobernd, observed live on TP8/DCP4.
#
#   This matters more than a normal bug for us specifically: OUR WATCHDOG
#   CANNOT SEE IT. glm52-watchdog.sh probes /health, which keeps returning 200
#   while an individual request hangs forever. We run LMCache continuously
#   against real users, so this is a silent-stall risk, not a theoretical one.
#
#   r11 also lands the rest of the LMCache #7-#17 durability/recovery stack
#   (safe MP fallback, native-FS restart accounting, durable L1 writeback,
#   recoverable late KV-transfer completion). Upstream validation shows perf
#   parity with r9 (DCP1 decode 87.6 vs 87.8; 64k prefill 5,898 vs 5,895), so
#   this is a reliability upgrade at no measured throughput cost.
#
#   r11's other headline change — DCP auto-policy corrections — targets DCP4/
#   DCP8 and is inert for our DCP2. We also pin the policy knobs (see TRAP 2),
#   so it cannot move under us either way.
#
#   CAVEAT: r11 is labelled a release candidate and was announced to named
#   testers, where r9 was a full release. To fall back, one env var:
#     IMAGE=voipmonitor/vllm:gilded-gnosis-v20-vllm34f26c2-side7739a-fi801d57a-cu132-20260728-r9
#   Every anchor this launcher patches was re-verified byte-identical on r11
#   (PCIe export at pcie-runtime-env.sh:11, wrapper TTLs at :72-73, v16
#   cmd+=("$@") at :391, the 140000 gather default, the all-explicit skip path).
#
# WHAT THIS DELETES, AND WHY IT IS SAFE (each verified against the r9 image
# on 2026-07-29 before this launcher was written):
#
#   1. patch/lmcache-bootstrap-v20.sh + the 12 lmcache patches — GONE.
#      r9 ships lmcache 0.5.2+glm52dcp.3 (the merged local-inference-lab DCP
#      fork) and /usr/local/bin/glm52-lmcache-wrapper.sh, which starts the MP
#      server, waits on /healthcheck, and supervises it for the model's whole
#      lifetime. Enable with LMCACHE_MODE=ram. Our pip-install + 12-patch
#      bootstrap is entirely redundant.
#
#   2. The serve-glm52-v16.sh connector-injection patch — GONE.
#      v16 now ends with  cmd+=("$@")  (line 386), so args reach `vllm serve`.
#      The wrapper appends --kv-transfer-config itself (wrapper line 198).
#      On r4 v16 built a fixed cmd array and dropped "$@", which is the only
#      reason we ever had to rewrite it.
#
#   3. --disable-hybrid-kv-cache-manager — GONE.
#      vllm/config/vllm.py:1632 now auto-disables HMA when the selected KV
#      connector does not implement SupportsHMA. Passing it by hand is a no-op
#      at best; letting the engine decide is what upstream tests.
#
#   4. export PYTORCH_CUDA_ALLOC_CONF="" and -e PYTORCH_CUDA_ALLOC_CONF= — GONE.
#      The wrapper (lines 143-150) now forces expandable_segments:False itself
#      while preserving any unrelated allocator settings. Our empty-string
#      trick existed only to defeat v16's ${VAR-default} single-dash default.
#
# WHAT IS STILL PATCHED (one file, down from three):
#   glm52-pcie-runtime-env.sh:11 STILL hardcodes
#       export VLLM_ENABLE_PCIE_ALLREDUCE=1
#   which collapses batch-1 decode on this 2-NUMA PCIe host (c1 ~40 tok/s,
#   documented v13/v14). A docker -e cannot override a script `export`, so we
#   extract, relax the line to honor the env, and mount it back. Default 0.
#   The line has moved three times across images (v16:267 col-0 ->
#   pcie-runtime-env:9 indented -> :11), so a miss is FATAL, never a warning.
#
# ---------------------------------------------------------------------------
# TWO r9 TRAPS THIS LAUNCHER DEFUSES — read before changing anything.
#
# TRAP 1: DCP_CKV_GATHER_MAX_TOKENS defaults to 140000 in r9 (v19:18-19).
#   Our r4 resolves the transient full-CKV gather to 524288 logical tokens
#   (observed in the running container's log). Taking the r9 default would
#   SHRINK that to 140k and push every prefill above ~140k onto the slow
#   fallback path — a silent regression on exactly the 256k/512k prefills we
#   benchmark and serve. We therefore pin 524288 to preserve current behavior.
#   The gather workspace costs VRAM, so if the KV budget ever fails to fit,
#   lower this BEFORE lowering GPU_MEMORY_UTILIZATION.
#
# TRAP 2: r9's PCIe calibration is reported broken on OUR EXACT TOPOLOGY.
#   misterfix (Discord, 2026-07-29) hit a calibration failure on a dual-Turin
#   2-NUMA box whose `nvidia-smi topo -m` is identical to ours (GPU0-3 NUMA0,
#   GPU4-7 NUMA1, SYS across the halves). Worse, the calibration cache is keyed
#   by vLLM build hash, so our warm r4 cache (vllm0c79e41...) will NOT be reused
#   by r9 (vllm34f26c2...) — r9 would re-probe from cold on first boot.
#
#   Rather than gamble a ~25min cold boot on it, we pin all four
#   calibration-driven knobs to the values r4's calibration actually derived on
#   this host. v19:141 then reports  calibration_status=skipped:all-explicit
#   and the probe never runs. This is upstream's own supported escape hatch
#   (festr: "just specify those values, it will override it").
#
#   The pinned values come from verify-r4-lmcache-20260727.txt:
#       VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1
#       VLLM_DCP_QUERY_SPLIT=1
#       VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=65536
#       DMA-min=25165824
#   To let r9 probe instead, set CALIBRATE=1 (see below). Do that only with the
#   GPUs idle and a log capture ready.
#
# ---------------------------------------------------------------------------
# DELIBERATE BEHAVIOR CHANGES vs r4 (everything else is held constant so any
# regression is attributable to the image, not to our config drift):
#
#   * r9's adaptive EXACT sparse-indexer folding (SparkInfer #87) is active.
#     CAREFUL — this is NOT the DCP_INDEXER_SHARDS knob, which is what the
#     release note wording suggests. Two separate things:
#       - DCP_INDEXER_SHARDS=auto resolves via glm52-dcp-prefill-policy.sh:149
#         ONLY for 8:4 (->2) and 8:8 (->4). Our 8:2 falls through to 0, which
#         is byte-identical to what r4 resolved. Setting it changes NOTHING
#         for us; it is left at auto purely so DCP4 experiments behave.
#       - The folding itself is SPARKINFER_INDEXER_TWO_LEVEL_FOLD (default
#         "auto", sparkinfer/attention/nsa_indexer/paged.py:97) with a 256 MiB
#         candidate-buffer cap. It is per-request and independent of DCP
#         sharding, so we DO get it at DCP2 without setting anything.
#         Verified absent from r4 entirely, so it is genuinely new here.
#     Upstream measured +136,704 KV tokens (+5.60%) — but at TP8/DCP4, so do
#     NOT expect that exact figure at DCP2. r4 gave us 990,592 KV tokens; treat
#     anything above that as the win and record the real number.
#     DISABLE: SPARKINFER_INDEXER_TWO_LEVEL_FOLD=off
#
#   * LMCache chunk size: 256 -> 512 (image default for power-of-two DCP).
#     The wrapper requires chunks to align to every effective DCP paged-cache
#     block; 512 is what the DCP-aware fork is validated against at DCP2.
#     917504 / 512 = 1792 exactly, so max_model_len stays aligned.
#
#   * LMCache TTLs: the wrapper HARDCODES 600s write / 300s read (lines 72-73).
#     That is far too aggressive for us — a 300s read TTL drops a user's
#     conversation prefix five minutes after their last turn even with 700+ GB
#     of L1 sitting idle, which defeats the entire point of warm-restore. We
#     patch the wrapper to honor LMCACHE_WRITE_TTL / LMCACHE_READ_TTL and keep
#     our 86400s. Capacity is still reclaimed by LRU at the 0.90 watermark.
#
# NOT CHANGED, ON PURPOSE: TP8/DCP2/MTP3/a16/fp8-KV, 917504 ctx, MAX_NUM_SEQS
# 16, GMU 0.96, F8_DMA=0 (i8_ring is NOT quality-certified upstream),
# VLLM_ENABLE_PCIE_ALLREDUCE=0, LOAD_FORMAT=instanttensor.
#
# KNOWN UPGRADE RISK NOT YET HIT BY US: jtazz reports instanttensor failing to
# load on a recent image while fastsafetensors works. If the model fails during
# weight load, try LOAD_FORMAT=fastsafetensors before suspecting anything else.
###############################################################################
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm9502cc7-side7739a-fi801d57a-cu132-20260729-r11}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8443}"
TP="${TP:-8}"
DCP="${DCP:-2}"
MTP="${MTP:-3}"
MOE_MODE="${MOE_MODE:-a16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-917504}"     # 256-aligned (block64*dcp2*2)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# 0.94, NOT the 0.96 we ran on r4. First r11 boot at 0.96 came up healthy and
# then died on the first real batch (2026-07-29 22:53) with:
#   torch.OutOfMemoryError: Tried to allocate 96.00 MiB. GPU 6 has 94.97 GiB
#   total of which 89.25 MiB is free ... Process 65 has 698.00 MiB in use
# in pynccl all_reduce -> torch.empty_like(in_tensor). Not a leak: two r11
# changes each ate headroom that 0.96 used to leave spare.
#   1. The exact indexer folding frees workspace, and vLLM hands the freed bytes
#      straight to KV cache — 1,113,600 tokens vs r4's 990,592 at the SAME 0.96.
#      Our safety margin silently became KV cache.
#   2. r11's LMCache wrapper starts --max-gpu-workers ${TP} = 8 GPU clients
#      (wrapper:59, absent from our old 0.4.6 bootstrap), costing ~0.7 GiB/GPU.
# 0.96 was calibrated against r4's memory profile and does not transfer.
# 0.94 gives back ~1.9 GiB/GPU for transient all-reduce//graph allocations and
# should still land ~1.03M KV tokens — i.e. STILL above r4's 990,592 at 0.96,
# so we keep the capacity win and get the headroom back. Verify the printed
# "GPU KV cache size" is >= 917,504 (the max_model_len floor) on every boot.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
F8_DMA="${F8_DMA:-0}"
PCIE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"      # 0 = NCCL all-reduce (host fix)
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v20-r9}"

# --- DCP / prefill policy --------------------------------------------------
DCP_PREFILL_WORKSPACE="${DCP_PREFILL_WORKSPACE:-auto}"
DCP_CKV_GATHER="${DCP_CKV_GATHER:-1}"
DCP_CKV_GATHER_MAX_TOKENS="${DCP_CKV_GATHER_MAX_TOKENS:-524288}"   # TRAP 1
DCP_TOPK_OWNER_MERGE="${DCP_TOPK_OWNER_MERGE:-1}"
DCP_INDEXER_SHARDS="${DCP_INDEXER_SHARDS:-auto}"   # resolves to 0 at 8:2 (= r4)
DCP_CKV_PREFETCH_WORKSPACE_MIB="${DCP_CKV_PREFETCH_WORKSPACE_MIB:-1024}"
# r9's exact two-level indexer folding. Image default is already "auto"; we name
# it explicitly so the knob is discoverable and so an A/B is a one-word change.
SPARKINFER_INDEXER_TWO_LEVEL_FOLD="${SPARKINFER_INDEXER_TWO_LEVEL_FOLD:-auto}"

# --- calibration: pinned by default (TRAP 2) --------------------------------
# CALIBRATE=1 releases all four knobs back to auto and lets r9 probe the fabric.
CALIBRATE="${CALIBRATE:-0}"
if [[ "$CALIBRATE" == "1" ]]; then
  DCP_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-auto}"
  DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT:-auto}"
  DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:-auto}"
  PCIE_DMA_MIN_BYTES="${PCIE_DMA_MIN_BYTES:-auto}"
  PCIE_CALIBRATION="${PCIE_CALIBRATION:-force}"
else
  DCP_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-1}"
  DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT:-1}"
  DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:-65536}"
  PCIE_DMA_MIN_BYTES="${PCIE_DMA_MIN_BYTES:-25165824}"
  PCIE_CALIBRATION="${PCIE_CALIBRATION:-auto}"   # moot: all-explicit -> skipped
fi

# --- LMCache (native, r9) ---------------------------------------------------
# Node-local storage is FORBIDDEN on this host: RAM only, never mode=disk.
LMCACHE_MODE="${LMCACHE_MODE:-ram}"
LMCACHE_L1_GB="${LMCACHE_L1_GB:-768}"
LMCACHE_L1_INIT_GB="${LMCACHE_L1_INIT_GB:-64}"   # wrapper would default to L1_GB
LMCACHE_START_TIMEOUT="${LMCACHE_START_TIMEOUT:-600}"   # wrapper default 120 is
                                                        # too tight for a big pin
LMCACHE_WRITE_TTL="${LMCACHE_WRITE_TTL:-86400}"
LMCACHE_READ_TTL="${LMCACHE_READ_TTL:-86400}"
# Pin the ports. The wrapper otherwise derives them from PORT-8000, so a PORT
# change would silently move them out from under the verify/bench scripts.
LMCACHE_PORT="${LMCACHE_PORT:-5998}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8532}"
LMCACHE_PROMETHEUS_PORT="${LMCACHE_PROMETHEUS_PORT:-9533}"

# VLLM_SERVER_DEV_MODE=1 exposes POST /reset_prefix_cache, which together with
# LMCache's POST :$LMCACHE_HTTP_PORT/cache/clear is the only way to measure
# cold-vs-warm restore deterministically (r_omk, Discord 2026-07-28). It is OFF
# by default on purpose: this box serves real users over the network, and the
# dev endpoints are unauthenticated — anyone who can reach the port could wipe
# the prefix cache. Turn it on only for a measurement window.
VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-0}"

if [[ "${LMCACHE_MODE,,}" == "disk" ]]; then
  echo "FATAL: LMCACHE_MODE=disk is forbidden on this host (no node-local storage)." >&2
  exit 1
fi

MODEL_HOST="${MODEL_HOST:-$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/ )}"
MODEL_HOST="${MODEL_HOST%/}"
[[ -f "${MODEL_HOST}/config.json" ]] || { echo "ERROR: model snapshot not found" >&2; exit 1; }

mkdir -p "$CACHE_ROOT/cache" "$CACHE_ROOT/tmp"

# --- extract + patch the image's own scripts --------------------------------
extract() {  # extract <path-in-image> <dest>
  local src="$1" dst="$2" cid
  cid="glm52x$$"
  docker rm -f "$cid" >/dev/null 2>&1 || true
  docker create --name "$cid" "$IMAGE" >/dev/null
  docker cp "$cid:$src" "$dst" >/dev/null
  docker rm -f "$cid" >/dev/null
}

# patch 1/2 — relax the hardcoded PCIe all-reduce export (host fix)
PATCH_SRC="${PATCH_SRC:-/usr/local/bin/glm52-pcie-runtime-env.sh}"
PATCH_OUT="$CACHE_ROOT/glm52-pcie-runtime-env.patched.sh"
extract "$PATCH_SRC" "$PATCH_OUT.orig"
if ! grep -qE '^[[:space:]]*export VLLM_ENABLE_PCIE_ALLREDUCE=1$' "$PATCH_OUT.orig"; then
  echo "FATAL: hardcoded 'export VLLM_ENABLE_PCIE_ALLREDUCE=1' not found in $PATCH_SRC." >&2
  echo "       The image layout changed again. Locate it with:" >&2
  echo "       docker run --rm --entrypoint bash $IMAGE -c 'grep -rn VLLM_ENABLE_PCIE_ALLREDUCE /usr/local/bin/'" >&2
  echo "       Refusing to launch (batch-1 decode would silently collapse)." >&2
  exit 1
fi
sed -E 's/^([[:space:]]*)export VLLM_ENABLE_PCIE_ALLREDUCE=1$/\1export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"/' \
  "$PATCH_OUT.orig" > "$PATCH_OUT"
grep -qE 'VLLM_ENABLE_PCIE_ALLREDUCE:-1' "$PATCH_OUT" || { echo "FATAL: PCIe patch did not apply" >&2; exit 1; }
chmod +x "$PATCH_OUT"

# patch 2/2 — make the wrapper's hardcoded LMCache TTLs env-overridable
WRAP_SRC="${WRAP_SRC:-/usr/local/bin/glm52-lmcache-wrapper.sh}"
WRAP_OUT="$CACHE_ROOT/glm52-lmcache-wrapper.patched.sh"
extract "$WRAP_SRC" "$WRAP_OUT.orig"
for lit in '--l1-write-ttl-seconds 600' '--l1-read-ttl-seconds 300'; do
  grep -qF -- "$lit" "$WRAP_OUT.orig" || {
    echo "FATAL: '$lit' not found in $WRAP_SRC — upstream changed the TTL defaults." >&2
    echo "       Re-read the wrapper before assuming our 86400s TTLs still apply." >&2
    exit 1; }
done
sed -E \
  -e 's/^([[:space:]]*)--l1-write-ttl-seconds 600$/\1--l1-write-ttl-seconds "${LMCACHE_WRITE_TTL:-600}"/' \
  -e 's/^([[:space:]]*)--l1-read-ttl-seconds 300$/\1--l1-read-ttl-seconds "${LMCACHE_READ_TTL:-300}"/' \
  "$WRAP_OUT.orig" > "$WRAP_OUT"
grep -qF 'LMCACHE_WRITE_TTL:-600' "$WRAP_OUT" || { echo "FATAL: write-TTL patch did not apply" >&2; exit 1; }
grep -qF 'LMCACHE_READ_TTL:-300'  "$WRAP_OUT" || { echo "FATAL: read-TTL patch did not apply" >&2; exit 1; }
chmod +x "$WRAP_OUT"

# Tearing down the previous container is NOT instant: with ~768 GB of pinned
# LMCache L1, the kill lands (exit 137) well before the name is released, and a
# bare `docker rm -f` followed immediately by `docker run` loses the race with
# "Conflict. The container name /glm52 is already in use" (hit 2026-07-29).
# That would also break the watchdog, which calls this script to restart. Wait
# for the name to actually disappear.
docker rm -f "$NAME" >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  docker inspect "$NAME" >/dev/null 2>&1 || break
  sleep 2
  docker rm -f "$NAME" >/dev/null 2>&1 || true
done
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "FATAL: container '$NAME' still exists after 120s of removal attempts." >&2
  docker ps -a --filter "name=$NAME" >&2
  exit 1
fi

# The launcher is named as the container COMMAND, not via --entrypoint: the
# image keeps NVIDIA's /opt/nvidia/nvidia_entrypoint.sh (which sets up the CUDA
# env) and its CMD defaults to run-kimi26-vllm, which just prints "Kimi launcher
# is not available in this vLLM ref" and exits 2. On r4 our bootstrap masked
# this by replacing the entrypoint outright; here we override only the CMD.
docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --network host --ipc host --init --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
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
  -e LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}" \
  -e F8_DMA="$F8_DMA" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="$PCIE" \
  -e DCP_PREFILL_WORKSPACE="$DCP_PREFILL_WORKSPACE" \
  -e DCP_QUERY_SPLIT="$DCP_QUERY_SPLIT" \
  -e DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="$DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS" \
  -e DCP_CKV_GATHER="$DCP_CKV_GATHER" \
  -e DCP_CKV_GATHER_MAX_TOKENS="$DCP_CKV_GATHER_MAX_TOKENS" \
  -e DCP_CKV_PREFETCH_DEPTH="$DCP_CKV_PREFETCH_DEPTH" \
  -e DCP_CKV_PREFETCH_WORKSPACE_MIB="$DCP_CKV_PREFETCH_WORKSPACE_MIB" \
  -e DCP_TOPK_OWNER_MERGE="$DCP_TOPK_OWNER_MERGE" \
  -e DCP_INDEXER_SHARDS="$DCP_INDEXER_SHARDS" \
  -e SPARKINFER_INDEXER_TWO_LEVEL_FOLD="$SPARKINFER_INDEXER_TWO_LEVEL_FOLD" \
  -e PCIE_CALIBRATION="$PCIE_CALIBRATION" \
  -e PCIE_DMA_MIN_BYTES="$PCIE_DMA_MIN_BYTES" \
  -e LMCACHE_MODE="$LMCACHE_MODE" \
  -e LMCACHE_L1_GB="$LMCACHE_L1_GB" \
  -e LMCACHE_L1_INIT_GB="$LMCACHE_L1_INIT_GB" \
  -e LMCACHE_START_TIMEOUT="$LMCACHE_START_TIMEOUT" \
  -e LMCACHE_WRITE_TTL="$LMCACHE_WRITE_TTL" \
  -e LMCACHE_READ_TTL="$LMCACHE_READ_TTL" \
  -e LMCACHE_PORT="$LMCACHE_PORT" \
  -e LMCACHE_HTTP_PORT="$LMCACHE_HTTP_PORT" \
  -e LMCACHE_PROMETHEUS_PORT="$LMCACHE_PROMETHEUS_PORT" \
  -e VLLM_SERVER_DEV_MODE="$VLLM_SERVER_DEV_MODE" \
  ${LMCACHE_CHUNK_SIZE:+-e LMCACHE_CHUNK_SIZE="$LMCACHE_CHUNK_SIZE"} \
  ${DRY_RUN:+-e DRY_RUN="$DRY_RUN"} \
  -v "$PATCH_OUT:$PATCH_SRC:ro" \
  -v "$WRAP_OUT:$WRAP_SRC:ro" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/cache:/cache" \
  -v "$CACHE_ROOT/tmp:/container-tmp" \
  "$IMAGE" /usr/local/bin/serve-gilded-gnosis.sh

echo "Launched '$NAME' — r11 TP$TP DCP$DCP MTP$MTP $MOE_MODE, PCIe-allreduce=$PCIE,"
echo "  ctx=$MAX_MODEL_LEN gpu_mem=$GPU_MEMORY_UTILIZATION ckv_gather_max=$DCP_CKV_GATHER_MAX_TOKENS"
echo "  indexer_shards=$DCP_INDEXER_SHARDS calibration=$([[ $CALIBRATE == 1 ]] && echo probing || echo pinned)"
echo "  LMCache native: mode=$LMCACHE_MODE L1=${LMCACHE_L1_GB}GB ttl=${LMCACHE_READ_TTL}s"
echo "                  zmq=$LMCACHE_PORT http=$LMCACHE_HTTP_PORT prom=$LMCACHE_PROMETHEUS_PORT"
echo
echo "Follow:  docker logs -f $NAME"
echo "Expect:  'PCIE_CALIBRATION_STATUS=skipped:all-explicit'  (pinned mode)"
echo "         'LMCache ready: mode=ram L1=768GB chunk=512'"
echo "         'GPU KV cache size: N tokens'  (r4 gave 990,592; expect >= that)"
echo "Health:  curl -s http://127.0.0.1:$LMCACHE_HTTP_PORT/healthcheck"
