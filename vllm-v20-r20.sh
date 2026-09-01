#!/usr/bin/env bash
###############################################################################
# GLM-5.2-NVFP4 — v20 "Gilded Gnosis" r20, TP8/DCP2 + NATIVE LMCache.
#
# SUPERSEDES vllm-v20-r11.sh (which remains the rollback target, still on disk).
#   Rollback is one env var:
#     IMAGE=voipmonitor/vllm:gilded-gnosis-v20-vllm9502cc7-side7739a-fi801d57a-cu132-20260729-r11
#   ...or just run vllm-v20-r11.sh, which is unchanged and still correct.
#
# WHY r20 — AND AN HONEST STATEMENT OF HOW WEAK THE CASE IS.
#   r11 -> r20 is nine releases (r12..r20) in four days. Read the release notes
#   and essentially ALL of it is EXL3/Trellis quantization work benchmarked at
#   TP4 (DCP1 or DCP4). There is NO NVFP4 TP8/DCP2 validation anywhere in
#   r12..r20. r18 says it "retains standard GLM/NVFP4 defaults" and r20 says
#   "Standard NVFP4/NF3 profiles are unaffected" — our path is ASSERTED
#   unchanged, never measured. Nobody in the channel runs our config on r20.
#
#   So we are upgrading to stay current, not to chase a number. Expect PARITY.
#   If the bench comes back materially better, be suspicious and re-run before
#   believing it; if it comes back worse, roll back to r11 without agonising.
#
#   The three changes that could plausibly touch us, all unquantified upstream:
#     * r17 "compressed MLA physical-stride fixes" — MLA is our attention path.
#     * r17 "corrected W4A16 planning/tails/capture resolution" — we run A16.
#     * r17 "a SparkInfer wheel packaging gate that prevents silent fallback
#       when runtime-JIT local headers are missing" — i.e. r11..r16 could
#       SILENTLY drop off the SparkInfer path. If r20 measures faster than r11,
#       this is the first thing to suspect as the cause.
#   Plus r13's LMCache "separated deadline exhaustion from real L2 persistence
#   failures, avoiding false circuit-breaker backoff" — the L2 half is
#   irrelevant to us (RAM only), but the deadline misclassification is not.
#
#   r17 also lists "semantic PCIe graph-channel isolation". That is the only
#   r12..r20 change touching the PCIe collective machinery we override, and it
#   comes with no before/after numbers. TREAT VLLM_ENABLE_PCIE_ALLREDUCE=0 AS
#   UNVALIDATED ON THIS IMAGE until the batch-1 decode cell is re-measured —
#   c1 is where the regression showed up historically (~40 tok/s when wrong).
#
# VERIFIED AGAINST THE PULLED r20 IMAGE ON 2026-08-03 (not assumed):
#   digest    sha256:40c891fd3fd573a92708e8a4bfa028ec91127a92491504c59006cf9735b20560
#             (matches the announced r20 digest exactly)
#   vllm      0.11.2.dev280+gilded.gnosis.v20.vllm72c35f1.si2b9bf2a...r20
#   lmcache   0.5.2+glm52dcp.4        <- SAME as r11, not a downgrade
#   entrypoint /opt/nvidia/nvidia_entrypoint.sh, CMD run-kimi26-vllm — so the
#             "name the launcher as COMMAND" trick below is still required.
#
#   Anchor re-verification (line numbers MOVED; our patches are content-regex,
#   so this is informational — but never assume, re-check on every bump):
#     glm52-pcie-runtime-env.sh   export VLLM_ENABLE_PCIE_ALLREDUCE=1  :11 -> :12
#     glm52-lmcache-wrapper.sh    TTL literals 600/300                 :72-73 (same)
#     serve-glm52-v16.sh          cmd+=("$@")                          :391 -> :420
#     serve-glm52-v19.sh          calibration_status=skipped:all-explicit :143
#     serve-glm52-v19.sh          DCP_CKV_GATHER_MAX_TOKENS default 140000 :18-19
#
#   glm52-lmcache-wrapper.sh is BYTE-IDENTICAL between r11 and r20. Our TTL
#   patch therefore carries over unchanged, and no LMCache RAM-path behaviour
#   moved. (It still emits --eviction-policy/-trigger-watermark/-ratio as MP
#   server flags at :74-76 while the generated JSON at :104 carries only
#   max_capacity_gb — the L2-cap bug procr2 diagnosed 2026-08-02. r20 was
#   published hours before his patch, so r20 does NOT contain it. Moot for us:
#   LMCACHE_MODE=ram, and mode=disk is FATAL below.)
#
#   glm52-pcie-runtime-env.sh differs from r11 ONLY in LD_PRELOAD assembly
#   (it now builds a local var and exports once, instead of exporting inside
#   the branch). Our patched line is untouched by that change.
#
#   glm52-dcp-prefill-policy.sh gained a 9th parameter, owner_merge_topology_safe:
#     "The exact owner exchange wins on a local PCIe fabric but loses its
#      launch/transport advantage when every rank crosses a NUMA boundary."
#   That reads like it was written for a host like ours — but it is gated on
#   tp:dcp == 8:8 ONLY, so at our 8:2 it cannot fire. The auto indexer-shard map
#   is UNCHANGED (8:4->2, 8:8->4, everything else ->0, so our 8:2 -> 0).
#   We also pin DCP_TOPK_OWNER_MERGE=1 explicitly, so this is inert twice over.
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
#   4. export PYTORCH_CUDA_ALLOC_CONF="" and -e PYTORCH_CUDA_ALLOC_CONF= — GONE
#      *ONLY WHILE LMCACHE IS ON*. The wrapper (lines 143-150) forces
#      expandable_segments:False, so with LMCACHE_MODE=ram we need nothing.
#      BUT with LMCACHE_MODE=off the wrapper `exec "$@"` at line 13 and NEVER
#      REACHES that logic, so v16:51's ${VAR-default} single-dash default
#      (expandable_segments:True) survives. vLLM then refuses to start the
#      native OffloadingConnector:
#        "KV connector OffloadingConnector is incompatible with
#         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
#      A passthrough is therefore wired below, applied ONLY when the caller sets
#      the variable, so default behaviour is unchanged. Hit 2026-08-03.
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
# TWO TRAPS THIS LAUNCHER DEFUSES — both RE-VERIFIED STILL LIVE IN r20.
# Read before changing anything.
#
# TRAP 1: DCP_CKV_GATHER_MAX_TOKENS defaults to 140000 (r20 v19:18-19, same as
#   r9/r11 — the default was NOT raised).
#   Our r4 resolves the transient full-CKV gather to 524288 logical tokens
#   (observed in the running container's log). Taking the r9 default would
#   SHRINK that to 140k and push every prefill above ~140k onto the slow
#   fallback path — a silent regression on exactly the 256k/512k prefills we
#   benchmark and serve. We therefore pin 524288 to preserve current behavior.
#   The gather workspace costs VRAM, so if the KV budget ever fails to fit,
#   lower this BEFORE lowering GPU_MEMORY_UTILIZATION.
#
# TRAP 2: the PCIe calibration is reported broken on OUR EXACT TOPOLOGY.
#   misterfix (Discord, 2026-07-29) hit a calibration failure on a dual-Turin
#   2-NUMA box whose `nvidia-smi topo -m` is identical to ours (GPU0-3 NUMA0,
#   GPU4-7 NUMA1, SYS across the halves). drock01057 diagnosed it as inherent to
#   that topology ("theres no cross-gpu cuda in that topo for some pairs") and
#   raised blackwell-llm-docker PR#9; NO r12..r20 release note mentions merging
#   it, so assume it is still broken. Worse, the calibration cache is keyed by
#   vLLM build hash, so our warm r11 cache (vllm9502cc7...) will NOT be reused
#   by r20 (vllm72c35f1...) — r20 would re-probe from cold on first boot.
#
#   NOTE: PCIE_CALIBRATION=off is the escape hatch the channel converged on for
#   this topology. We do NOT use it, because pinning all four knobs already
#   yields skipped:all-explicit, which is strictly more explicit — it records
#   the actual values in the log rather than silently taking image defaults.
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
# DELIBERATE BEHAVIOR CHANGES vs r11: NONE.
#
# Every knob below is byte-identical to vllm-v20-r11.sh except IMAGE and
# CACHE_ROOT. That is the whole point: this is a single-variable change so any
# delta in the bench is attributable to the image and nothing else. Do NOT
# bundle the clear_thinking fix, a GMU change, or a DCP_CKV_PREFETCH_DEPTH A/B
# into this cutover — land r20 at parity first, then change one thing at a time.
#
# The historical notes below are retained from the r11 launcher because they
# still describe why these values are what they are.
#
#   * The adaptive EXACT sparse-indexer folding (SparkInfer #87) is active.
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
# 16, GMU 0.94, F8_DMA=0 (i8_ring is NOT quality-certified upstream; flobernd
# confirmed 2026-07-30 that F8_DMA=ring raises KLD, magnitude unquantified),
# VLLM_ENABLE_PCIE_ALLREDUCE=0, LOAD_FORMAT=instanttensor.
#
# KNOWN UPGRADE RISK NOT YET HIT BY US: jtazz reports instanttensor failing to
# load on a recent image while fastsafetensors works. If the model fails during
# weight load, try LOAD_FORMAT=fastsafetensors before suspecting anything else.
###############################################################################
# ---------------------------------------------------------------------------
# PCIE BLOCK — the 2026-08-03 experiment campaign result. READ BEFORE CHANGING.
#
# Production now runs the b12x PCIe collective SIZE-GATED, with a COMPRESSED
# wire, instead of sending everything through PyNCCL.
#
# vLLM dispatches TP all-reduce three ways (custom_all_reduce.py):
#     payload <= VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE  -> b12x ONESHOT
#     payload >= VLLM_PCIE_DMA_MIN_BYTES               -> b12x DMA
#     everything else                                  -> PyNCCL
# Our traffic is BIMODAL with nothing in between:
#     decode  1-4 rows   = 12-48 KiB -> PyNCCL  (oneshot disabled, see below)
#     prefill 8192 rows  = 96 MiB    -> b12x DMA
#
# ONESHOT IS DISABLED ON PURPOSE (PCIE_ONESHOT_MAX=0). The historical batch-1
# decode collapse to ~40 tok/s (v13/v14) was the ONESHOT path, not DMA: a 12 KiB
# decode payload sits under the image's 64 KB oneshot limit, so DMA never even
# engaged there. Disabling the whole PCIe path to fix decode ALSO threw away the
# prefill DMA win. Gating by size keeps both.
#
# MEASURED WARM (medians, 2026-08-03; cold single runs are NOT comparable --
# see the warm-up note in run-r20-verify-bench.sh):
#     prefill  8k/64k/128k/256k/512k   +32.5/+35.6/+34.7/+32.4/+29.2 %
#     decode   c1 -4.2%, c4 -2.4%, c8 +0.2%, c16 -1.0%
# For long-context coding the trade is ~35:1 in time saved (a 256k prefill goes
# 63s -> 48s; a 1000-token reply loses 0.4s).
#
# F8_DMA=i8_ring CARRIES ALMOST ALL OF IT. DMA with F8_DMA=0 measured only
# +2.7% mean. Compression halves the bytes on a PCIe-bound box, which dominates
# the DMA-vs-NCCL algorithmic difference. Do NOT "play it safe" by setting
# F8_DMA=0 -- that keeps the quality cost and discards the speed.
#
# QUALITY COST, measured (measure-kld.sh, 27,495 token positions vs the PyNCCL
# baseline): 15 confident-token changes (>1 nat margin) = 0.055%. Uncompressed
# DMA costs 13, so compression adds ~2 -- inside Poisson noise. ~36% of all
# flips are at near-ties (<0.1 nat). i8_ring beats ring (15 vs 33 confident
# flips at identical speed), so ring is dominated; do not use it.
# CAVEAT: greedy generations DIVERGE from the old config within 6-41 tokens
# (5-22% token overlap). Each config is self-consistent going forward, but
# nothing captured before this cutover will reproduce.
#
# REJECTED, with warm numbers: DCP4 (-2% prefill, -5..11% decode for +92% KV we
# do not need -- LMCache L1 already backstops KV pressure); TWO_LEVEL_FOLD=off
# (neutral-to-worse, keep auto); MAX_BATCHED_TOKENS=16384 (probe showed the DMA
# gain plateaus ~20.5% past 8192, only +2.4pp).
#
# ROLLBACK: VLLM_ENABLE_PCIE_ALLREDUCE=0 GPU_MEMORY_UTILIZATION=0.94 F8_DMA=0 \
#           DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=65536 bash vllm-v20-r20.sh
#           (or use vllm-v20-r20.sh.pre-dma, saved at cutover)
# ---------------------------------------------------------------------------
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm72c35f1-si2b9bf2a-fi801d57a-cu132-20260802-r20}"
NAME="${NAME:-glm52}"
PORT="${PORT:-8443}"
TP="${TP:-8}"
# DCP 2 -> 4 (2026-08-31, MEASURED — the full ladder was benched on this host):
#
#   config   KV pool     pool/ctx   c1 steps/s  c4     c8     prefill 128k
#   DCP2       868,352     1.02x        38.6   107.6  153.0        5,576
#   DCP4     1,706,495     2.00x        37.2   100.2  140.0        5,487
#   DCP8     3,363,328     3.95x        19.7    24.5   26.1            -
#
# DCP4 costs ~3-8% decode and ~2% prefill to DOUBLE the pool. It took
# preemptions from 15 (accumulated on DCP2) to ZERO across a full bench battery,
# and preemption is a total loss — scheduler.py frees every KV block and sets
# num_computed_tokens=0, and LMCache refuses PREEMPTED requests outright. One
# avoided preemption of a 150k prompt is worth far more than 5% decode.
#
# My Phase 8 note predicted DCP4 would HALVE prefill. It did not (5,487 vs
# 5,576). That note was DCP4-vs-DCP1 on the June v11 image, before r20's CKV
# gather. Old topology numbers on this box go stale — re-measure, don't assume.
#
# DCP8 IS A TRAP and this is now measured twice (v19 2026-07-21, r20 2026-08-31).
# Decode collapses ~5.2x at c8 and barely scales with concurrency (19.7 -> 24.5
# -> 26.1 steps/s for 1 -> 4 -> 8 requests): every decoded token needs an 8-way
# all-reduce across our PCIe-only 2-NUMA fabric (GPU0-3 / 4-7) and that dominates.
# MTP acceptance was FINE at DCP8 (2.77-2.81), which rules out speculation and
# points squarely at the collective. r20's CKV gather + 4-way indexer sharding
# did NOT move it, though they clearly fixed DCP4. Do not retry without a
# topology change (NVLink / single NUMA).
#
# DCP must DIVIDE TP, so at TP8 the only legal values are 1, 2, 4, 8. DCP=6 is
# not available here — the Discord "dcp6" reports are all TP6/DCP6 (six GPUs
# total), which idles 2 of our 8 and raises per-GPU weight bytes; their measured
# pools were 774k and 999k, BELOW our DCP4.
DCP="${DCP:-4}"
MTP="${MTP:-3}"
MOE_MODE="${MOE_MODE:-a16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# GLM-5.3 CUT THIS FROM 917504 -> 851968 (2026-08-28). Measured, not estimated:
# 5.3's BF16 MTP layer costs +2.82 GiB/GPU (not the +1.62 that pure /8 sharding
# predicts — part of layer 78 replicates), so at GMU 0.93 available KV fell
# 25.43 -> 22.61 GiB. KV costs 26.72 KiB/token here, so 22.61 GiB tops out at
# 887,286 tokens and vLLM refused to boot at 917,504 (needs 23.38 GiB):
#   ValueError: ... 23.38 GiB KV cache is needed ... available 22.61 GiB
# 851,968 leaves ~4% margin. To get 917,504 back you would need GMU 0.94, which
# yields 924,555 — only 0.8% over the requirement AND sits exactly on the OOM
# ceiling documented above (768GB L1). Not worth it: a boot-time refusal is
# cheap, an OOM under load is not. Other lever: DCP_CKV_GATHER_MAX_TOKENS.
# 851,968 -> 1,048,576 (2026-08-31), restored to 1M by the DCP4 pool doubling.
# Alignment at DCP4 is block64*dcp4*2 = 512; 1,048,576 / 512 = 2048 exactly.
# At the measured 1,706,495-token DCP4 pool that is 1.63x pool/ctx — inside the
# 1.5-3x band (daring_hare, 06-30) and up from the 1.02x that was preempting.
# GATE ON BOOT: "GPU KV cache size" must exceed 1,048,576 or the engine refuses
# to start. If it lands short, drop back to 917,504 rather than raising GMU —
# 0.93 OOM'd twice on 5.3 (see the GPU_MEMORY_UTILIZATION block).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"    # 512-aligned (block64*dcp4*2)
# 16 -> 8 on 2026-08-31: the scheduler was ADMITTING more long-context sessions
# than the KV pool can hold, so it preempted them mid-generation.
#
# MEASURED 2026-08-29 06:44 (54h window, GLM-5.3, GMU 0.925):
#   vllm:num_preemptions_total   0 -> 6   in a single burst
#   max vllm:kv_cache_usage_perc 1.00     (pool 100% full)
#   max vllm:num_requests_running   5     <- only FIVE concurrent did it
#
# Preemption is not a stall, it is a total loss. scheduler.py:1235-1248:
#   self._free_request_blocks(request)     # every KV block freed
#   request.num_computed_tokens = 0        # everything computed is discarded
#   self.waiting.prepend_request(request)
# The request restarts from zero, which presents as "one agent suddenly hit
# prefill mid-run" while others keep decoding. LMCache CANNOT rescue it: the
# connector returns (0, False) for PREEMPTED before it even looks up
# (lmcache_mp_connector.py:1029-1031, "TODO: support loading KV for preempted
# requests in the future"). Upstream fix is LMCache PR #4612, still OPEN and
# blocked on store-ordering safety (PR #4836, opened by yatesdr/D-Rock
# 2026-08-31) -- taking #4612 alone risks cache poisoning, so we wait.
#
# Pool arithmetic at 868,352 tokens: 2 sessions -> 434k each, 3 -> 289k,
# 4 -> 217k, 5 -> 174k, 6 -> 145k. Five agentic sessions at ~170k exhausts it
# exactly, which is what the counters recorded. 8 is a compromise: it still
# admits more than the pool comfortably holds at 170k/session, but queueing
# costs latency whereas preemption costs the ENTIRE prefill.
# If preemptions persist, go to 6, then 5. Watch vllm:num_preemptions_total.
#
# NOTE the community does not hit this: their pasted logs show Waiting: 0 and
# GPU KV usage median 9.6%/max 26.3% (n=10), and "preempt" appears 0 times in
# 11,159 Discord messages. Our pool/ctx is 1.02x; the folk rule is 1.5-3x.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# CHAT TEMPLATE KWARGS — overrides the image preset at serve-glm52-v16.sh:411,
# which hardcodes '{"reasoning_effort":"high"}'. argparse keeps the LAST
# occurrence and REPLACES (does not merge), so passing this drops the preset's
# value entirely. Two effects, both wanted:
#
#   1. reasoning_effort disappears -> chat_template.jinja:2 falls through to
#      'max'. The preset was pinning us to High; Max is the template default.
#        {%- set effective_reasoning_effort =
#             'high' if reasoning_effort is defined and reasoning_effort == 'high'
#             else 'max' -%}
#
#   2. clear_thinking=false keeps <think> blocks for ALL prior assistant turns.
#      DEFAULT BEHAVIOUR IS A PREFILL DISASTER FOR AGENTIC CODING: the template
#      retains reasoning only while a turn is AFTER the last user message
#      (chat_template.jinja:70). During a tool-calling chain last_user_index is
#      pinned at the original prompt, so every assistant turn keeps its <think>
#      and gets prefilled+cached that way. The moment the user replies,
#      last_user_index jumps to the end and EVERY turn in the chain collapses to
#      <think></think> at once -> the rendered prefix diverges at the FIRST
#      assistant turn of the chain -> total cache miss -> full re-prefill.
#
#      Measured 2026-08-07 on this box, same conversation, same cache:
#        prime chain X            27,475 prompt / 27,264 cached (99.2%)
#        X + user reply, DEFAULT   9,331 prompt /      0 cached ( 0.0%)  <-- !!
#        X + user reply, CT=false 27,487 prompt / 27,264 cached (99.2%)
#      The DEFAULT prompt is SMALLER because 18,144 tokens of reasoning were
#      deleted; that deletion is exactly why it no longer matches.
#
#      Symptom this fixes: "agent finishes and asks a question, I answer, then
#      it takes minutes before the response starts" — the delay is the whole
#      chain re-prefilling at ~5,300 tok/s, and it scales with chain length.
#      It is NOT LMCache eviction; the blocks are resident but unreachable.
#
# COST: prompts carry their reasoning, so contexts grow faster against
# MAX_MODEL_LEN and L1 entries are bigger. Left LMCACHE_L1_GB at 768 on purpose
# (2026-08-07) — revisit if L1 usage_ratio sits above the 0.90 evict watermark.
# Set CHAT_TEMPLATE_KWARGS='' to fall back to the image preset.
#
# ON GLM-5.3 (weights-only successor, SAME arch — see MODEL_HOST below):
#   zai-org/GLM-5.3 rewrote the template header and fixed effect 2 upstream:
#     {%- set clear_thinking = clear_thinking if clear_thinking is defined
#           else false -%}
#   so prior-turn reasoning is RETAINED by default and this override becomes
#   redundant for that purpose. Its ONLY remaining effect on 5.3 is dropping
#   the preset's reasoning_effort -> Max. 5.3 also accepts 'low' (5.2 accepted
#   only 'high'; anything else -> 'max') and no longer honours enable_thinking.
#   Verified 2026-08-28 by rendering the real 5.3 template. NOTE the 5.3
#   template uses {% break %} and needs jinja2.ext.loopcontrols — a bare
#   jinja2.Environment() raises "unknown tag 'break'". Transformers supplies
#   it, so serving is unaffected; our own render-the-template tooling is not.
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS-{\"clear_thinking\":false\}}"
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
#
# *** 0.94 IS PROVISIONAL ON r20 — IT IS r11's NUMBER, NOT r20's. ***
# r12's release note says it "replaced one-time scratch observation with
# repeatable post-warmup target + draft peak accounting before KV allocation".
# That is EXACTLY the accounting that decides how much is left over for KV, so
# the r11->r20 KV figure can move at an unchanged GMU — in either direction.
# The r11 OOM above was precisely this failure mode: healthy boot, death on the
# first real batch, because freed memory had silently become KV cache.
# So on the first r20 boot:
#   1. read the printed "GPU KV cache size: N tokens";
#   2. if N is MATERIALLY ABOVE r11's 1,027,968, that is a WARNING, not a win —
#      the margin became KV. Drop GMU by 0.01 and reboot before serving.
#   3. if N < 917,504 the engine cannot hold one max-length request; raise GMU
#      or lower DCP_CKV_GATHER_MAX_TOKENS (in that order of preference).
#   4. check free VRAM under load, not just at idle.
# Community corroboration for 0.94 at large L1: timricese independently landed
# on exactly 0.94 with a 512GB L1; sininspira OOM'd at 0.95 with ~90 MB of
# margin, specifically during a 128k prefill. We run a LARGER L1 (768GB) than
# either, so 0.94 is a CEILING here, not a floor.
# 2026-08-29: CUT 0.93 -> 0.925 AFTER A PRODUCTION OOM ON GLM-5.3.
# At 23:52:42 on 2026-08-28, after 3h40m healthy at GMU 0.93, all 8 workers died:
#   MemoryError: CUDA out of memory. Tried to allocate 1.48 GiB.
#   GPU N has a total capacity of 94.97 GiB of which 1.41 GiB is free.
#   ... this process has 91.35 GiB memory in use.
# Allocation site was speculator.propose() (the MTP=3 draft path), 5 reqs running.
# Missed by 70 MiB. The API server caught EngineDeadError and exited 0, so docker
# reported a CLEAN exit -- this does NOT look like a crash from outside.
#
# Why 0.93 was not enough on 5.3 (it ran 3 weeks on 5.2): 5.3's BF16 MTP layer
# raised PROFILED peak activation 2.25 -> 3.49 GiB/GPU, and the profile is taken
# under a synthetic boot run. Real concurrency + MTP3 overshoots it: the process
# reached 91.35 GiB against an 88.32 GiB budget, i.e. 3.03 GiB of transients
# living OUTSIDE the KV budget. Non-budget headroom is (1-GMU)*94.97, so it is
# the ONLY knob here that trades KV for transient safety.
#   0.930 -> 6.65 GiB headroom, KV 886,912 tok (ctx margin 4.1%)  <- OOM'd
#   0.925 -> 7.12 GiB headroom, KV 868,285 tok (ctx margin 1.9%)
# 0.925 would have left ~1.88 GiB free at the OOM instant vs the 1.48 GiB ask.
#
# WATCH ON NEXT BOOT: "GPU KV cache size" must exceed 851,968. Predicted ~868k,
# margin only 1.9% -- if it lands short the engine refuses to start (loudly).
# If that happens, drop MAX_MODEL_LEN to 819200 rather than raising GMU back.
# If it OOMs again at 0.925, prefer MAX_BATCHED_TOKENS 8192->4096 (targets the
# prefill activation peak directly, costs no context) over cutting GMU further.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.925}"
F8_DMA="${F8_DMA:-i8_ring}"   # compressed DMA wire: carries ~all of the +33%
PCIE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"      # 1 = b12x, size-gated (see PCIE BLOCK)
# b12x oneshot dispatch limits (patch 4). Image defaults 64KB/84KB are kept;
# set both to 0 to route small payloads to NCCL while leaving the >=24MB
# prefill DMA path active. Only meaningful when PCIE=1.
PCIE_ONESHOT_MAX="${PCIE_ONESHOT_MAX:-0}"        # 0 = oneshot OFF -> decode stays on NCCL
PCIE_ONESHOT_FUSED_MAX="${PCIE_ONESHOT_FUSED_MAX:-0}"
# Separate cache root from r11's on purpose: this holds the patched copies of
# the image's scripts and the build-hash-keyed calibration cache. Sharing it
# would let an r20 artefact leak into an r11 rollback.
CACHE_ROOT="${CACHE_ROOT:-/root/glm52-vllm/cache-v20-r20}"

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
  DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS="${DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS:-49152}"
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
# MP request timeout. The image hardcodes 60s, which the 8th worker misses by
# ~2s during register_kv_caches (see patch 3 for the measured timeline).
LMCACHE_MQ_TIMEOUT="${LMCACHE_MQ_TIMEOUT:-600}"
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

# GLM-5.3 IS A DROP-IN HERE. It is a weights-only successor: config.json is
# dimension-for-dimension identical to GLM-5.2 (GlmMoeDsaForCausalLM, 78 layers,
# hidden 6144, 64 heads, moe_intermediate 2048, 256 routed experts, 8 per token,
# vocab 154880, 1,048,576 ctx, num_nextn_predict_layers 1 so MTP is present).
# Verified 2026-08-28. No new image is needed — override MODEL_HOST and reboot.
#   MODEL_HOST=/root/.cache/huggingface/hub/models--<repo>/snapshots/<rev> \
#     bash vllm-v20-r20.sh
# Re-derive GPU_MEMORY_UTILIZATION anyway if the quant recipe differs from
# lukealonso's 5.2 (weights bytes/GPU drive the KV residual), and re-read the
# CHAT_TEMPLATE_KWARGS block above — 5.3 changed the template defaults.
# Do NOT grab local-inference-lab/GLM-5.3-NVFP4: despite the name it is
# Glm5NextForConditionalGeneration (GLM-5.3-*Flash*, 185.7 GiB, vision tower),
# not this architecture. Community big-5.3 NVFP4s are ~432.9 GiB.
# --- ACTIVE: GLM-5.3 (RadixArk NVFP4) since 2026-08-28 -----------------------
# Chosen over incoai (quantized from the zai-org/GLM-5.3 *FP8* release, i.e.
# NVFP4-on-FP8) and Inferact (no published accuracy, 99.99-pct activation-scale
# clipping). RadixArk is the only one publishing parity vs its own BF16 source
# on an identical build (GSM8K 1285/1319 exact match; AIME 2026 94.17%), and its
# quantization_config has the same shape as lukealonso 5.2 (config_groups +
# wildcard ignore + producer modelopt). Verified on disk: 47/47 shards,
# 464,795,267,072 tensor bytes == index total_size, layer 78 (MTP) fully BF16.
#
# WEIGHT DELTA vs 5.2 — the ONLY category that changed is the MTP layer:
#   routed experts 379.69 GiB, attention 24.34, embed/lm_head 3.54 : IDENTICAL
#   MTP layer 78:  5.60 GiB (luke quantized it) -> 18.54 GiB (RadixArk BF16)
#   => +12.94 GiB total; MTP experts shard across TP, so +12.94/8 = +1.62 GiB/GPU
#   => weights/GPU 58.82 -> ~60.44 GiB.  GMU left at 0.93: the +1.62 is absorbed
#      by KV, projecting ~965k tokens (was 997,631) — still over the 917,504
#      floor. 0.94 would restore ~1.00M but spends 0.95 GiB/GPU of OOM headroom
#      at a value documented above as a CEILING at our 768GB L1. A boot that
#      lands under the floor fails LOUDLY; an OOM at 0.94 fails under load.
#      If the printed "GPU KV cache size" is < 917,504, raise to 0.94 and reboot.
# ACTIVE 2026-08-31: local-inference-lab (Luke's) GLM-5.3-NVFP4.
# Switched from RadixArk after drock's 64-window KLD matrix (2026-08-30) measured
# LIL NVFP4 + FP8 KV at 0.02071 overall — "recommended start", second only to
# official FP8 (0.01968), and TP8-tagged. RadixArk was never in that matrix; we
# had picked it on GSM8K/AIME parity, which is far coarser. Luke's producer is
# modelopt 0.39.0.dev290, the SAME version as his GLM-5.2 quant that ran here for
# weeks. Verified on disk: 85 index shards + 2 amax files, on-disk bytes ==
# index total_size exactly (note LIL counts safetensors headers in total_size
# where RadixArk did not; tensor payload is identical to the byte), layer 78
# (MTP) fully BF16, routed experts U8 + F8_E4M3 + F32.
MODEL_HOST="${MODEL_HOST:-/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-NVFP4/snapshots/cca10d1586255195d3279785fc85577bfc1e9227}"
# ROLLBACK to RadixArk 5.3 (still on disk, 432.9 GiB, ran 08-29..31 without fault):
#   MODEL_HOST=/root/.cache/huggingface/hub/models--RadixArk--GLM-5.3-NVFP4/snapshots/363e8f086905afd83db356a620f9aa401c23800a bash vllm-v20-r20.sh
# GLM-5.2 was DELETED 2026-08-31 to make room; re-download if ever needed.
# ROLLBACK to GLM-5.2 (unchanged, still on disk, 436 GiB):
#   MODEL_HOST="$(echo /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/*/)" bash vllm-v20-r20.sh
# NOTE: SERVED_MODEL_NAME stays "GLM-5.2" on purpose so existing client configs
# keep working; the API will report id "GLM-5.2" while serving 5.3 weights.
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

# patch 4/4 — make the b12x ONESHOT size limits env-overridable.
#
# WHY: vLLM dispatches TP all-reduce three ways (custom_all_reduce.py):
#     payload <= VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE   -> b12x ONESHOT
#     payload >= VLLM_PCIE_DMA_MIN_BYTES                -> b12x DMA
#     everything else                                   -> PyNCCL
#   Both b12x paths require VLLM_ENABLE_PCIE_ALLREDUCE=1 AND backend=b12x
#   (custom_all_reduce.py:52), so today, with our =0 host fix, we get PyNCCL
#   for EVERYTHING — including the large prefill all-reduces where the
#   2026-08-03 probe measured b12x DMA at +19.5% over NCCL:
#     rows  bytes        nccl_ms  dma_ms  gain
#        1      12288      0.104   0.709  -585.0%   <- decode
#      512    6291456      0.413   0.493   -19.1%
#     2048   25165824      1.462   1.446    +1.1%   <- crossover
#     8192  100663296      5.718   4.605   +19.5%   <- full prefill chunk
#   Note 8192 rows == our MAX_BATCHED_TOKENS, i.e. a full prefill chunk.
#
#   The historical batch-1 decode collapse (~40 tok/s, v13/v14) is almost
#   certainly the ONESHOT path, not DMA: a 1-row decode payload is 12 KB, far
#   below the image's 64 KB oneshot limit, so DMA never even engages there.
#   Disabling the whole PCIe allreduce to fix decode therefore ALSO threw away
#   the prefill DMA win. upstream anticipates exactly this tuning at
#   custom_all_reduce.py:531: "A deployment preflight can tune its crossover or
#   disable it when lossless DMA never beats NCCL on the selected PCIe topology."
#
#   Setting both oneshot limits to 0 makes the oneshot pool dispatch nothing
#   (buffer_size falls back to max(...,16), so no zero-size buffer) while
#   VLLM_PCIE_DMA_MIN_BYTES still routes >=24 MB to DMA.
#
#   DEFAULTS ARE UNCHANGED (64KB/84KB) — this patch only makes the knobs
#   reachable. The experiment is opt-in:
#     VLLM_ENABLE_PCIE_ALLREDUCE=1 PCIE_ONESHOT_MAX=0 PCIE_ONESHOT_FUSED_MAX=0 \
#       bash vllm-v20-r20.sh
#   ABORT CRITERION: if concurrency-1 decode falls toward ~40 tok/s, the gating
#   is not working — revert VLLM_ENABLE_PCIE_ALLREDUCE to 0 and reboot.
for lit in 'export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB' \
           'export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB'; do
  grep -qF -- "$lit" "$PATCH_OUT" || {
    echo "FATAL: '$lit' not found in $PATCH_SRC — oneshot limits moved." >&2
    exit 1; }
done
sed -E \
  -e 's/^([[:space:]]*)export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB$/\1export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${PCIE_ONESHOT_MAX:-64KB}"/' \
  -e 's/^([[:space:]]*)export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB$/\1export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${PCIE_ONESHOT_FUSED_MAX:-84KB}"/' \
  "$PATCH_OUT" > "$PATCH_OUT.tmp" && mv "$PATCH_OUT.tmp" "$PATCH_OUT"
grep -qF 'PCIE_ONESHOT_MAX:-64KB' "$PATCH_OUT" || { echo "FATAL: oneshot patch did not apply" >&2; exit 1; }
grep -qF 'PCIE_ONESHOT_FUSED_MAX:-84KB' "$PATCH_OUT" || { echo "FATAL: oneshot-fused patch did not apply" >&2; exit 1; }
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

# patch 3/3 — make the LMCache MP request timeout env-overridable.
#
# WHY (measured on the failed first r20 boot, 2026-08-03 12:36:39):
#   EngineCore died with
#     ConnectionError: LMCache server did not respond to register_kv_caches
#     within 60.0s. Is the server running?
#   The server had NOT failed. It registers workers SERIALLY at ~8.6s each:
#     12:35:47.880 / 12:35:56.202 / 12:36:04.769 / 12:36:13.635 /
#     12:36:13.672 / 12:36:22.778 / 12:36:31.864 / 12:36:41.431
#   Worker TP4 was last in that queue. It issued its request at 12:35:39 and
#   gave up at 12:36:39 on the hardcoded 60s timeout; the server finished its
#   registration at 12:36:41.431 — 2.4 SECONDS TOO LATE.
#
#   8 workers x ~8.6s ~= 62s against a 60s client timeout. This is a race we
#   lose by a hair, not an r20 defect: glm52-lmcache-wrapper.sh is BYTE-
#   IDENTICAL on r11, so r11 carries the same 2-second margin and has simply
#   been winning the coin flip. Registration cost scales with layer count and
#   worker count, so anything that slows a boot (cold page cache, a busy host,
#   more TP ranks) tips it over.
#
#   60 -> 600 costs nothing on a healthy boot: it is a ceiling, not a delay.
WRAP_MQ_LIT='                "lmcache.mp.mq_timeout": 60,'
grep -qF -- "$WRAP_MQ_LIT" "$WRAP_OUT" || {
  echo "FATAL: '\"lmcache.mp.mq_timeout\": 60,' not found in $WRAP_SRC." >&2
  echo "       Upstream changed the MP timeout plumbing — re-read the wrapper." >&2
  echo "       Without this patch the 8th worker can miss registration by ~2s." >&2
  exit 1; }
python3 - "$WRAP_OUT" <<'PYPATCH'
import sys
p = sys.argv[1]
s = open(p).read()
old = '                "lmcache.mp.mq_timeout": 60,\n'
new = '                "lmcache.mp.mq_timeout": int(os.getenv("LMCACHE_MQ_TIMEOUT", "60")),\n'
assert s.count(old) == 1, f"expected exactly 1 mq_timeout line, found {s.count(old)}"
open(p, "w").write(s.replace(old, new))
PYPATCH
grep -qF 'LMCACHE_MQ_TIMEOUT' "$WRAP_OUT" || { echo "FATAL: mq_timeout patch did not apply" >&2; exit 1; }

# --- L1 EVICTION BLAST RADIUS (patch 6, added 2026-08-31) -------------------
# The wrapper hardcodes the eviction knobs, so a `docker -e` cannot reach them.
#
# MEASURED over the 2026-08-29..31 weekend (54h, GLM-5.3, L1=768GB):
#   l1_evicted_chunks_total          10,594
#   l1_eviction_loop_triggered_total      2      <- only TWICE
# i.e. each trigger dumped 5,298 chunks = 2.71M tokens = ~82 GB IN ONE EVENT,
# because eviction_ratio 0.10 discards 10% of the whole cache at once, by LRU.
# LRU victims are exactly the PAUSED agentic sessions whose owner is about to
# come back -- so a resume that should have been a warm restore becomes a full
# re-prefill. Confirmed by LMCache's own histogram:
#   l1_chunk_evict_reuse_gap_seconds: count=110, all <=500s, mean 300s
# i.e. 110 chunks were evicted and needed again within ~5 minutes.
# Event #2 landed 2026-08-31 08:54, ~6 min before a user reported exactly this.
#
# Between events L1 idles at 82-86%, which LOOKS like headroom -- that is why
# this reads as "full prefill even though LMCache is not full". It is full: it
# reached the 0.90 watermark on 08-29 15:07 and has sat at the ceiling since.
#
# 0.10 -> 0.02 cuts the blast radius 5x (~82 GB -> ~16 GB per event); evictions
# become more frequent and much gentler. 0.90 -> 0.94 reclaims ~33 GB of usable
# cache before the first trigger. Neither enlarges L1: the working set genuinely
# exceeds 768 GB (L1 went 10% -> 90% in twelve hours), so if re-prefills persist
# the next lever is LMCACHE_L1_GB (host shows ~1,043 GB available), not these.
WRAP_EV_WM='  --eviction-trigger-watermark 0.90'
WRAP_EV_RATIO='  --eviction-ratio 0.10'
for lit in "$WRAP_EV_WM" "$WRAP_EV_RATIO"; do
  grep -qF -- "$lit" "$WRAP_OUT" || {
    echo "FATAL: '$lit' not found in $WRAP_SRC — upstream changed the eviction knobs." >&2
    echo "       Re-read the wrapper before assuming the blast-radius fix applied." >&2
    exit 1; }
done
python3 - "$WRAP_OUT" <<'PYPATCH'
import sys
p = sys.argv[1]
s = open(p).read()
for old, new in (
    ('  --eviction-trigger-watermark 0.90\n',
     '  --eviction-trigger-watermark "${LMCACHE_EVICT_WATERMARK:-0.94}"\n'),
    ('  --eviction-ratio 0.10\n',
     '  --eviction-ratio "${LMCACHE_EVICT_RATIO:-0.02}"\n'),
):
    assert s.count(old) == 1, f"expected exactly 1 of {old!r}, found {s.count(old)}"
    s = s.replace(old, new)
open(p, "w").write(s)
PYPATCH
grep -qF 'LMCACHE_EVICT_WATERMARK' "$WRAP_OUT" || { echo "FATAL: evict watermark patch did not apply" >&2; exit 1; }
grep -qF 'LMCACHE_EVICT_RATIO'     "$WRAP_OUT" || { echo "FATAL: evict ratio patch did not apply" >&2; exit 1; }

chmod +x "$WRAP_OUT"

# Tearing down the previous container is NOT instant: with ~768 GB of pinned
# LMCache L1, the kill lands (exit 137) well before the name is released, and a
# bare `docker rm -f` followed immediately by `docker run` loses the race with
# "Conflict. The container name /glm52 is already in use" (hit 2026-07-29).
# That would also break the watchdog, which calls this script to restart. Wait
# for the name to actually disappear.
# --- PRESERVE THE OUTGOING CONTAINER'S LOGS BEFORE DESTROYING IT -----------
# `docker rm -f` deletes /var/lib/docker/containers/<id>/*-json.log along with
# the container, so every restart used to destroy the evidence for the very
# failure that triggered it. That bit us on 2026-08-28: the engine OOM'd in
# speculator.propose(), the watchdog restarted, and the log had to be salvaged
# by hand before it was gone. Capture first. Capture must NEVER block or fail a
# restart, hence timeouts and `|| true` throughout.
CRASHLOG_DIR="${CRASHLOG_DIR:-/root/glm52-vllm/crashlogs}"
CRASHLOG_KEEP="${CRASHLOG_KEEP:-30}"
if [[ "${CRASHLOG_CAPTURE:-1}" == "1" ]] && docker inspect "$NAME" >/dev/null 2>&1; then
  mkdir -p "$CRASHLOG_DIR"
  _ts=$(date -u +%Y%m%dT%H%M%SZ)
  _cid=$(docker inspect -f "{{.Id}}" "$NAME" 2>/dev/null | cut -c1-12)
  _ec=$(docker inspect -f "{{.State.ExitCode}}" "$NAME" 2>/dev/null)
  _st=$(docker inspect -f "{{.State.Status}}" "$NAME" 2>/dev/null)
  _oom=$(docker inspect -f "{{.State.OOMKilled}}" "$NAME" 2>/dev/null)
  _base="$CRASHLOG_DIR/${_ts}-${_cid:-unknown}-${_st:-unknown}-exit${_ec:-x}"
  timeout 60  docker inspect "$NAME" >"${_base}.inspect.json" 2>/dev/null || true
  timeout 300 docker logs    "$NAME" >"${_base}.log"          2>&1 || true
  # the LMCache MP server logs inside the container /tmp, which is not mounted
  timeout 60  docker cp "$NAME:/tmp/lmcache-mp-${PORT}.log" "${_base}.lmcache.log" >/dev/null 2>&1 || true
  echo "  saved logs -> ${_base}.log ($(du -h "${_base}.log" 2>/dev/null | cut -f1)) status=$_st exit=$_ec oomkilled=$_oom"
  # retention: keep the newest $CRASHLOG_KEEP of each kind
  for _pat in "*.log" "*.inspect.json" "*.lmcache.log"; do
    ls -1t "$CRASHLOG_DIR"/$_pat 2>/dev/null | tail -n +$((CRASHLOG_KEEP+1)) | xargs -r rm -f
  done
fi

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
# Bound the container's json-file log. It was previously UNCAPPED (driver
# json-file, no opts), so a pathological log storm could grow it without limit.
# These live on /var/lib/docker, i.e. the EPHEMERAL pod overlay, not the /root
# NFS PVC -- so they vanish on pod restart regardless; the durable copy is the
# crashlogs/ capture above. At the observed ~16 MB/day this never rotates; the
# cap is a runaway guard, and 256m x 4 still holds ~2 months of normal history.
# `docker logs` reads across rotated files, so capture stays complete.
docker run -d --name "$NAME" \
  --log-driver json-file --log-opt max-size=256m --log-opt max-file=4 \
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
  -e PCIE_ONESHOT_MAX="$PCIE_ONESHOT_MAX" \
  -e PCIE_ONESHOT_FUSED_MAX="$PCIE_ONESHOT_FUSED_MAX" \
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
  -e LMCACHE_MQ_TIMEOUT="$LMCACHE_MQ_TIMEOUT" \
  -e LMCACHE_PORT="$LMCACHE_PORT" \
  -e LMCACHE_HTTP_PORT="$LMCACHE_HTTP_PORT" \
  -e LMCACHE_PROMETHEUS_PORT="$LMCACHE_PROMETHEUS_PORT" \
  -e VLLM_SERVER_DEV_MODE="$VLLM_SERVER_DEV_MODE" \
  ${PYTORCH_CUDA_ALLOC_CONF+-e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"} \
  ${LMCACHE_CHUNK_SIZE:+-e LMCACHE_CHUNK_SIZE="$LMCACHE_CHUNK_SIZE"} \
  ${DRY_RUN:+-e DRY_RUN="$DRY_RUN"} \
  -v "$PATCH_OUT:$PATCH_SRC:ro" \
  -v "$WRAP_OUT:$WRAP_SRC:ro" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/cache:/cache" \
  -v "$CACHE_ROOT/tmp:/container-tmp" \
  "$IMAGE" /usr/local/bin/serve-gilded-gnosis.sh \
  ${CHAT_TEMPLATE_KWARGS:+--default-chat-template-kwargs $CHAT_TEMPLATE_KWARGS} \
  ${EXTRA_VLLM_ARGS:-}
# EXTRA_VLLM_ARGS is appended to the container COMMAND and reaches `vllm serve`
# through the whole chain: serve-gilded-gnosis.sh -> glm52-lmcache-wrapper.sh
# (exec "$@" when LMCACHE_MODE=off) -> serve-glm52-v19.sh -> serve-glm52-v16.sh
# (cmd+=("$@") at :420). Used to test vLLM's NATIVE CPU KV offload as an
# alternative to LMCache, e.g.:
#   LMCACHE_MODE=off EXTRA_VLLM_ARGS='--kv-transfer-config {...}' ...
# NOTE: intentionally unquoted so multiple args split; keep the value simple.

echo "Launched '$NAME' — r20 TP$TP DCP$DCP MTP$MTP $MOE_MODE, PCIe-allreduce=$PCIE,"
echo "  ctx=$MAX_MODEL_LEN gpu_mem=$GPU_MEMORY_UTILIZATION ckv_gather_max=$DCP_CKV_GATHER_MAX_TOKENS"
echo "  indexer_shards=$DCP_INDEXER_SHARDS calibration=$([[ $CALIBRATE == 1 ]] && echo probing || echo pinned)"
echo "  chat_template_kwargs=${CHAT_TEMPLATE_KWARGS:-<image preset: reasoning_effort=high>}"
echo "  LMCache native: mode=$LMCACHE_MODE L1=${LMCACHE_L1_GB}GB ttl=${LMCACHE_READ_TTL}s"
echo "                  zmq=$LMCACHE_PORT http=$LMCACHE_HTTP_PORT prom=$LMCACHE_PROMETHEUS_PORT"
echo
echo "Follow:  docker logs -f $NAME"
echo "Expect:  'PCIE_CALIBRATION_STATUS=skipped:all-explicit'  (pinned mode)"
echo "         'LMCache ready: mode=ram L1=768GB chunk=512'"
echo "         'GPU KV cache size: N tokens'  — r11 gave 1,027,968 at this GMU."
echo "           N well ABOVE that is a WARNING (margin became KV): drop GMU 0.01."
echo "           N below 917,504 cannot hold one max-length request: raise GMU."
echo "Health:  curl -s http://127.0.0.1:$LMCACHE_HTTP_PORT/healthcheck"
echo
echo "NOT YET VALIDATED ON r20: VLLM_ENABLE_PCIE_ALLREDUCE=0 (r17 touched the"
echo "  PCIe collective path). Re-measure the concurrency-1 decode cell first —"
echo "  that is where this regressed historically (~40 tok/s when wrong)."
