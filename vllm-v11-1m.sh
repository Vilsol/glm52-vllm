#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 LONG-CONTEXT profile — up to 1,000,000 tokens/request.
#
# Uses the newer v11 image (per-layer DCP-LSE fix lets DCP run with MTP) and
# DCP4 to multiply KV capacity. This is the *long-context* profile; it is
# ~2x slower than the default 256k config (vllm.sh) on both prefill and decode
# because context-parallel adds per-layer communication. Use vllm.sh for normal
# (<=256k) workloads, this for >256k up to 1M.
#
# Measured (this host): KV budget ~1.93M tokens; 1M req concurrency 1.93x;
# decode c16/ctx0 ~267 tok/s (vs 552 at DCP1); prefill ~1.5-1.6k tok/s (vs ~3k);
# 316k-token needle retrieval correct; Estonia 3/3. MTP accepts ~1.9/2 at DCP4.
#
# Boot is slow (~12-15 min): cold AOT compile + 1M memory profiling + 3 cudagraph
# capture phases. The post-compile "no broadcast block" idle spin is NORMAL
# memory profiling, NOT a hang — be patient (do not kill before ~20 min).
###############################################################################
set -euo pipefail

export IMAGE="${IMAGE:-voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618}"
export CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v11}"   # separate cache: different build
export NAME="${NAME:-glm52}"
export PORT="${PORT:-8080}"

# Long-context knobs
export DCP_SIZE="${DCP_SIZE:-4}"                  # 4x KV capacity (needed to fit 1M)
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"  # per-request context cap
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"
export GLM51_DISABLE_MTP="${GLM51_DISABLE_MTP:-0}"  # MTP works at DCP4 on v11 (modest gain). Set 1 for a simpler/robuster boot.

# Correctness + host fixes (same as vllm.sh)
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
export HF_OVERRIDES="${HF_OVERRIDES:-{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}}"

# cudagraph capture = seqs*(1+spec); --max-model-len override (serve hardcodes 256000, argparse last-wins)
if [[ "${GLM51_DISABLE_MTP}" == "1" ]]; then _m=1; else _m=$(( 1 + NUM_SPECULATIVE_TOKENS )); fi
export EXTRA_ARGS="--max-cudagraph-capture-size $(( MAX_NUM_SEQS * _m )) --linear-backend auto --max-model-len ${MAX_MODEL_LEN}"

# Reuse the shared experiment launcher (forwards all the above env, mounts the
# tool-call parser patch + caches, applies the NCCL host fixes via entrypoint.sh).
exec bash /root/glm52-vllm/launch.sh
