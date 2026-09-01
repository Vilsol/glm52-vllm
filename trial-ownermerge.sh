#!/usr/bin/env bash
# A/B: DCP_TOPK_OWNER_MERGE=0 (+ pinned indexer shards) at our production DCP4.
# ONE-OFF — not a default. The watchdog calls vllm-v20-r20.sh, which keeps
# owner-merge=1, so it will NOT adopt this.
#
# WHY (2026-09-01, from flobernd in #glm-53):
# He runs DCP8 on 8x RTX PRO 6000 and measures only a -8%..-20% penalty vs DCP4:
#     prefill  8k -16.8% · 32k -18.2% · 64k -19.0% · 128k -19.8%
#     decode   c1@32k -8.7% · c1@256k -10.5% · c1@512k -7.5% · c4@32k -17.8%
# We measured DCP8 at -81% on c8 (26.1 vs 140.0 steps/s) and called it a
# topology wall. He saw our HOST-TOPOLOGY.md and said it matches his ASUS
# ESC800A-E13P. vilsol in-channel: "I am seeing total collapse, TO 20%, not BY
# 20%" — so two similar boxes, opposite outcomes, and now we have his config.
#
# THE DIFFERENCES (his DCP8 compose vs our r20 launcher):
#     DCP_TOPK_OWNER_MERGE     0   <- ours 1        *** prime suspect ***
#     DCP_INDEXER_SHARDS       0   <- ours auto (->2 at 8:4, ->4 at 8:8)
#     VLLM_B12X_ABSORB_BMM     1   <- ours unset
#     VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD 0 <- ours unset
#     MAX_BATCHED_TOKENS    4096   <- ours 8192
#     GPU_MEMORY_UTILIZATION 0.80 + explicit --kv-cache-memory-bytes
#     image  gilded-gnosis r34     <- ours r20
#
# Owner-merge performs an owner EXCHANGE — exactly the all-to-all collective our
# Phase 13/19 analysis blamed for the DCP8 decode collapse. We ran DCP8 with it
# ON. DCP_TOPK_OWNER_MERGE=0 has been an untested lead in our notes since Phase
# 16; this is the first outside evidence it matters.
#
# WHY DCP4 FIRST, NOT DCP8: DCP4 is our live production topology, so this is a
# low-risk A/B against numbers measured yesterday on this exact box. If
# owner-merge=0 is neutral-or-better here, THEN re-test DCP8 with it (the real
# prize: a 3.36M pool for ~20% instead of ~81%). If it REGRESSES DCP4, we have
# learned that cheaply and DCP8 is not worth another boot.
#
# ONE VARIABLE AT A TIME: this script changes owner-merge and indexer-shards
# only. It deliberately does NOT touch ABSORB_BMM, the shared-experts threshold,
# MAX_BATCHED_TOKENS, GMU or the image — those are separate experiments, and
# bundling them would make a regression unattributable.
#
# BASELINE TO BEAT (DCP4, owner-merge=1, measured 2026-08-31, clean cells):
#     KV pool 1,706,495 @ ctx 851,968 (2.00x) / 1,708,544 @ 1,048,576 (1.63x)
#     decode steps/s  c1 37.2 · c4 100.2 · c8 140.0     accept ~2.7
#     prefill tok/s   8k 5,227 · 128k 5,487 · 256k 5,244 · 512k 4,697
#     preemptions 0
#
# AFTER: bash bench-battery.sh, then compare with agg-clean-cells.py.
# Return to production with: bash vllm-v20-r20.sh
set -euo pipefail
cd "$(dirname "$0")"
echo "=== A/B: DCP4 with DCP_TOPK_OWNER_MERGE=0, DCP_INDEXER_SHARDS=0"
echo "    baseline (owner-merge=1): c1 37.2 / c4 100.2 / c8 140.0 steps/s"
echo "    watch the boot line: 'owner-merge=0 indexer-shards=0'"
DCP_TOPK_OWNER_MERGE=0 \
DCP_INDEXER_SHARDS=0 \
bash vllm-v20-r20.sh
