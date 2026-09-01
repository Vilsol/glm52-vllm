#!/usr/bin/env bash
# DCP4 TRIAL — one-off, NOT a default. Run explicitly; the watchdog will NOT
# adopt this (it calls vllm-v20-r20.sh, which stays at DCP=2).
#
# WHY DCP4: our KV pool is 868,352 tokens against an 851,968 context = 1.02x.
# That is below the community's 1.5-3x rule of thumb and is what produced the
# 2026-08-29 preemption burst (6 preemptions at only 5 concurrent, pool 100%).
# DCP shards the KV cache across ranks, so DCP2 -> DCP4 should roughly double
# the logical pool. alitaj1230 measured a clean 3.6x going DCP1 -> DCP4 on Flash
# (1.25M -> 4.55M, TP4, same image).
#
# WHAT WE EXPECT TO PAY (our own Phase 8 data, GLM-5.2 on the v11 image, and the
# channel's consensus):
#   * PREFILL: the main cost. entropi, 06-27: "Higher DCP allows you bigger
#     aggregate KV budget, but costs you in prefill speed mostly (MTP buys back
#     the loss on decode)". Our old DCP4-vs-DCP1 numbers: prefill ~1516-1626
#     tok/s vs DCP1's ~3000, i.e. roughly HALF. Today at DCP2 we measure
#     5,186-5,598 tok/s, so watch for a large drop. Our workload is
#     prefill-heavy (71% of prompts are 100-200k), so this is the metric that
#     decides the trial.
#   * DECODE UNDER CONCURRENCY: entropi, 06-27: "At higher batch levels, you
#     still get better decode with lower DCP, even with MTP."
#   * MTP ACCEPTANCE: our Phase 8 note recorded accept dropping to ~1.9/2 (~45%)
#     at DCP4 vs DCP1's ~2.8. We currently sit at ~2.7/3. A drop here cuts
#     decode tok/s directly, so compare steps/s AND accept, not just tok/s.
#   * INDEXER SHARDING CHANGES: DCP_INDEXER_SHARDS=auto resolves to 0 at our 8:2
#     but to 2 at 8:4 (glm52-dcp-prefill-policy.sh:149). So DCP4 silently turns
#     ON indexer sharding — a second variable. koush, 08-06: "at the cost of
#     ~25% KV space, can get +~20% decode at dcp8 using replicated indexer".
#     Set DCP_INDEXER_SHARDS=0 to isolate DCP from the indexer change.
#
# ALIGNMENT: MAX_MODEL_LEN must be a multiple of block64 * dcp * 2 = 512 at DCP4.
# 851,968 / 512 = 1664 exactly, so it is already valid. LMCache chunk 512 also
# aligns. Nothing else needs to move.
#
# ---- SEQUENCING (2026-08-31 plan) ----------------------------------------
# STEP 1  this script: DCP4 at the CURRENT 851,968 context. Do not change two
#         things at once — if it fails we need to know it was DCP, not context.
#         GATE: read "GPU KV cache size". Expect ~1.7M (vs 868,352 at DCP2).
# STEP 2  bench-battery.sh — prefill is the metric that decides DCP4.
#         PASS if prefill >= ~4,000 tok/s (vs 5,186-5,598 at DCP2, i.e. lose no
#         more than ~25%) AND c1 decode steps/s >= ~34 (vs 38.6) AND MTP accept
#         >= ~2.4 (vs 2.72). FAIL on a prefill halving (our old DCP4 data) or an
#         accept collapse to ~1.9 (also our old data).
# STEP 3  only if STEP 2 passes: raise context to 1M.
#           MAX_MODEL_LEN=1048576 bash trial-dcp4.sh
#         1,048,576 / 512 = 2048, so it is DCP4-aligned. At an expected ~1.7M
#         pool that is a 1.63x pool/ctx ratio — inside the 1.5-3x band, and the
#         first time this box could hold a 1M session without preemption.
#         Re-run the battery: a longer context changes the KV budget, so the
#         STEP 2 gates must be re-checked, not assumed.
# STEP 4  DCP8 — SEE THE WARNING BELOW BEFORE SPENDING A BOOT ON IT.
#
# ---- DCP8: WE ALREADY MEASURED IT AND IT FAILED BADLY --------------------
# 2026-07-21, v19 image, THIS host (BENCHMARKS Phase 13):
#     DCP8 c1 54.2 / 29.5   c8 76 / 71   c16 95    KV 3.69M
#     DCP2 today            c8 426.9     c16 614.9 KV 868k
#   -> c16 decode collapses 614.9 -> 95, a 6.5x REGRESSION, for a 4.2x KV gain.
# Verbatim from that phase: "DCP8 is unusable on this host. Decode collapses ~7x
# INDEPENDENT of PCIe all-reduce backend - CKV engages fine; the wall is DCP8's
# 8-way cross-NUMA all-reduce per decode token on our PCIe-only 2-NUMA fabric
# (GPU0-3 / 4-7). koush's ~2x were on non-NUMA PCIe-x8 boxes."
# That is a TOPOLOGY wall, not a software one, so r20/CKV-gather is unlikely to
# have moved it. What HAS changed since: koush's replicated indexer (08-06,
# "~25% KV space for +~20% decode at dcp8") and DCP_INDEXER_SHARDS=auto -> 4 at
# 8:8. Those could soften it but will not plausibly recover 6.5x.
# If you still want the datapoint (ctx must be 1024-aligned; 1,048,576 is):
#     DCP=8 MAX_MODEL_LEN=1048576 bash vllm-v20-r20.sh
# Budget ~30 min boot + ~15 min bench, and expect to roll back.
#
# AFTER ANY TRIAL, return to production with:  bash vllm-v20-r20.sh
set -euo pipefail
cd "$(dirname "$0")"
echo "=== DCP4 trial: LIL NVFP4, TP8/DCP4/MTP3, ctx 851,968"
echo "    watch: 'GPU KV cache size' (expect ~1.7M vs 868,352 at DCP2)"
echo "    then bench prefill FIRST — it is the metric that decides this."
DCP=4 \
DCP_INDEXER_SHARDS="${DCP_INDEXER_SHARDS:-auto}" \
NAME="${NAME:-glm52}" \
bash vllm-v20-r20.sh
