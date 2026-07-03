#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v12 image, ~1M-context HIGH-PERF profile.
#
# This is the upgrade target for our v11 DCP2 server. The v12 build chain landed
# three wins over v11 (source: rtx6kpro glm5.2_v12.md + community benches 06-21):
#   1. DCP global-top-k (VLLM_DCP_GLOBAL_TOPK) — selects top-k across ALL DCP
#      ranks, not just rank-local. This *removes most of the DCP tax*: on v12,
#      DCP2 decode now TIES DCP1 at short ctx and BEATS it at 64k+ (our v11 had
#      DCP2 ~ -33%). It also fixes the low MTP acceptance under DCP.
#   2. MTP draft-sharding (VLLM_DCP_SHARD_DRAFT) — default 0 is a footgun: with
#      it OFF, MTP=3 was WORSE than MTP off under DCP. ON -> 60-70% acceptance,
#      and MTP k=3 is now safe (the v11 "k>=3 illegal memory" limit is lifted).
#   3. A16 scale-fix — stray scales were left resident in VRAM; the fix frees
#      ~5 GB/GPU on tp8, so DCP2 fits more KV than it did on v11.
#
# Community v12 DCP sweep (timricese, mtp x3, 8x dual-Turin), for reference:
#   KV pool tokens: DCP1 427,520 | DCP2 873,344 | DCP4 1,746,688 | DCP8 3,478,528
#   decode tok/s short c1: DCP1 93.7 | DCP2 93.9 | DCP4 85.1 | DCP8 83.8
#   decode tok/s  64k  c1: DCP1 78.5 | DCP2 88.9 | DCP4 72.7 | DCP8 37.4
#   prefill tok/s 8k:      DCP1 3881 | DCP2 3479 | DCP4 2708 | DCP8 1780
# -> DCP2 is the sweet spot for ~1M ctx: near-DCP1 decode, only ~10% less prefill.
#
# Also enables `clear_thinking:false` — GLM-5.2 otherwise clears its CoT after
# every turn, which tanks model quality AND destroys prefix-cache hit rates.
# reasoning_effort defaults to "max" (over-thinks/loops); we pin "high" for coding.
#
# This launcher is cache-isolated from v11 (cache-v12) because AOT/compile
# artifacts are build-specific. First boot is a cold compile (~12-15 min); be
# patient through the post-compile memory-profiling spin (NOT a hang).
#
# A/B against the running v11+LMCache server by giving it a different NAME/PORT,
# e.g.:  NAME=glm52-v12 PORT=8081 bash vllm-v12.sh
###############################################################################
set -euo pipefail

export IMAGE="${IMAGE:-voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622}"
export CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v12}"   # build-specific AOT cache
export NAME="${NAME:-glm52}"
export PORT="${PORT:-8080}"

# Context / parallelism. DCP2 fits ~1M on v12 at gpu_mem 0.95 (v11 fit 1.0M, and
# v12's A16 scale-fix frees ~5 GB/GPU more). If boot ValueErrors with
# "estimated maximum model length < max_model_len", lower MAX_MODEL_LEN to the
# engine's reported estimate (this is a clean config error, not a crash).
export DCP_SIZE="${DCP_SIZE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"   # match canonical v12 recipe; unlocks high-concurrency aggregate throughput (cudagraph capture = 32*(1+spec)=128)

# MTP: k=3 is the v12 sweet spot (needs SHARD_DRAFT=1, set below). Override to
# 2 for a more conservative first boot, or 0 to disable.
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
export GLM51_DISABLE_MTP="${GLM51_DISABLE_MTP:-0}"

# v12 DCP/MTP fixes — both default-OFF in the image, both wanted here.
export VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}"
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}"

# Correctness + host fixes (unchanged from v11: A16 MoE decode + DSA index cache
# are still required for long-gen correctness; v12 only made A16 cheaper in VRAM).
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
export HF_OVERRIDES="${HF_OVERRIDES:-{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}}"

# Chat-template kwargs: keep thinking across turns (prefix-cache friendly) and
# cap reasoning effort. JSON must stay space-free so it survives EXTRA_ARGS word
# splitting (it is passed unquoted, like the other launchers).
export CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"reasoning_effort\":\"high\",\"clear_thinking\":false}}"

# cudagraph capture = seqs*(1+spec); --max-model-len override (serve hardcodes
# 256000, argparse last-wins).
if [[ "${GLM51_DISABLE_MTP}" == "1" ]]; then _m=1; else _m=$(( 1 + NUM_SPECULATIVE_TOKENS )); fi
export EXTRA_ARGS="--max-cudagraph-capture-size $(( MAX_NUM_SEQS * _m )) --linear-backend auto --max-model-len ${MAX_MODEL_LEN} --default-chat-template-kwargs ${CHAT_TEMPLATE_KWARGS}"

# Reuse the shared experiment launcher (forwards all the above env, mounts the
# tool-call parser patch + caches, applies the NCCL host fixes via entrypoint.sh).
exec bash /root/glm52-vllm/launch.sh
