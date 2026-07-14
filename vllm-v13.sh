#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v13 "eldritch" image, ~1M-context production profile.
#
# v13 (eldritch) is the official successor to v12 (dark-devotion). The reason to
# move: v12/dark-devotion is reported UNSTABLE under GLM-5.2 concurrency; v13
# merges the b12x + lucifer backends and fixes it (source: rtx6kpro glm5.2_v13.md).
# The pinned tag is the 20260629 "enlightenment" bugfix build — the canonical v13.
# Do NOT go back to the 20260625 launch build (GPU-CPU sync regressions, random
# OOMs) or the 20260627 vbfaa36b build (topk_scores_buffer crash on DCP).
#
# HOST CONSTRAINT (important): this is a 2-socket pod with no working cross-socket
# GPU P2P, so one thing from v13's *canonical* recipe is deliberately OMITTED:
#   * -cc.pass_config.fuse_allreduce_rms=True — its fused *custom* all-reduce
#     faults at init here (CUDA illegal memory access). On a single-socket host
#     you WOULD add it (~+8% on the FlashInfer path).
# ATTENTION BACKEND (measured on this host, 2026-07-02, this image, DCP2 MTP3):
#   FLASHINFER_MLA_SPARSE_SM120 (fuse-less) beats B12X_MLA_SPARSE across the board:
#   decode c1 86.9 vs 78.1 (+11%), 128k c1 78.2 vs 71.7 (+9%), 128k c2 +16%,
#   prefill ~+2%, KV 1,271,726 vs 1,240,368 (+2.5%), Estonia 6/8 vs 3/3 (both pass;
#   temp-0 nondeterminism). MTP acceptance 69.3% vs 74.9% — net still a clear win.
#   So SM120 is the DEFAULT here for DCP>1. It is upstream-unvalidated with MTP
#   (we validated it ourselves; watch production for long-context misses).
#   *** DCP1 + SM120 CRASHES on this host: CUDA_ERROR_ILLEGAL_ADDRESS in the
#   sparse_mla_sm120_decode autotune warmup. DCP1 therefore falls back to B12X. ***
#
# Measured on this host (v13 DCP2, no fuse): KV ~1.24-1.27M tokens, Estonia
# correct. The gain over v12: +11% decode c1 (SM120), concurrency STABILITY, and
# the 20260629 bugfixes. First boot on this tag is a cold compile (~12-15 min);
# later boots ~5-6 min warm.
#
# clear_thinking:false keeps CoT across turns (prefix-cache friendly + better
# quality); reasoning_effort pinned "high" (default "max" over-thinks/loops).
#
# Cache-isolated (cache-v13-0629) since AOT/compile artifacts are build-specific.
#
# A/B on a different port:  NAME=glm52-v13 PORT=8081 bash vllm-v13.sh
###############################################################################
set -euo pipefail

export IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x8ce61f9-cu132-20260629}"
export CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v13-0629}"   # build-specific AOT cache
export NAME="${NAME:-glm52}"
export PORT="${PORT:-8443}"

# Context / parallelism. DCP2 fits ~1M at gpu_mem 0.95 (measured KV ~1.24M). If boot
# ValueErrors "estimated maximum model length < max_model_len", lower MAX_MODEL_LEN.
export DCP_SIZE="${DCP_SIZE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"   # cudagraph capture = 32*(1+spec)=128

# MTP k=3 sweet spot (needs SHARD_DRAFT=1). Override to 2 for a more conservative
# boot, or 0 to disable.
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
export GLM51_DISABLE_MTP="${GLM51_DISABLE_MTP:-0}"

# DCP+MTP fixes (default-on in v13, set explicitly for clarity).
export VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}"
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}"

# Correctness + host fixes (A16 MoE decode + DSA index cache required for long-gen
# correctness). NOTE: no fuse_allreduce_rms here — see header (crashes on this host).
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
# Allocator: canonical-compose flag; speed-neutral here (measured 2026-07-02) but
# resists fragmentation on long-running servers (the v13 random-OOM mitigation).
# Flag-trial results (same date, all speed-neutral or worse — left at defaults):
# ONESHOT_ALLREDUCE 32K/256K/1MB flat/-4%; NCCL_BUFFSIZE 16M flat; NCHANNELS 16
# -5% c1; numactl --interleave=all not reproducible; BATCHED_TOKENS 16384 crashes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 2026-07-03 P2P fix (see entrypoint.sh header): with real GPU P2P working, NCCL
# beats the b12x custom PCIe all-reduce — its ONESHOT path is a batch-1 latency
# disaster over cross-socket P2P (c1 30 tok/s!). NCCL everywhere: c1 100.5 (+16%),
# c8 419.8 (+21%), prefill 3079 (+48%), 128k c1 96.9 (+24%), Estonia 3/3.
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"
export HF_OVERRIDES="${HF_OVERRIDES:-{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}}"

# Keep thinking across turns + cap reasoning effort. JSON stays space-free so it
# survives EXTRA_ARGS word-splitting (passed unquoted, like the other launchers).
export CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"reasoning_effort\":\"high\",\"clear_thinking\":false}}"

# cudagraph capture = seqs*(1+spec); --max-model-len override (serve hardcodes 256000).
if [[ "${GLM51_DISABLE_MTP}" == "1" ]]; then _m=1; else _m=$(( 1 + NUM_SPECULATIVE_TOKENS )); fi
export EXTRA_ARGS="--max-cudagraph-capture-size $(( MAX_NUM_SEQS * _m )) --linear-backend auto --max-model-len ${MAX_MODEL_LEN} --default-chat-template-kwargs ${CHAT_TEMPLATE_KWARGS}"
# Attention backend: SM120 default for DCP>1 (+11% decode, see header); B12X for
# DCP1 (SM120's DCP1 decode kernel faults on this host). ATTN_BACKEND overrides.
if [[ -z "${ATTN_BACKEND:-}" && "${DCP_SIZE}" -gt 1 ]]; then
  ATTN_BACKEND="FLASHINFER_MLA_SPARSE_SM120"
fi
if [[ -n "${ATTN_BACKEND:-}" ]]; then EXTRA_ARGS+=" --attention-backend ${ATTN_BACKEND}"; fi

# TLS: serve HTTPS with the self-signed cert (mounted at /certs by launch.sh).
# Default off (plain HTTP). TLS_ENABLE=1 serves HTTPS; clients then use https:// + -k (self-signed).
export TLS_ENABLE="${TLS_ENABLE:-0}"
if [[ "${TLS_ENABLE}" == "1" ]]; then
  EXTRA_ARGS+=" --ssl-keyfile ${TLS_KEY:-/certs/key.pem} --ssl-certfile ${TLS_CERT:-/certs/cert.pem}"
fi

# Reuse the shared experiment launcher (host NCCL/P2P fixes via entrypoint.sh).
exec bash /root/glm52-vllm/launch.sh
