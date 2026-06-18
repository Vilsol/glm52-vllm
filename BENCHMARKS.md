# GLM-5.2-NVFP4 Benchmark Journal — 8x RTX PRO 6000 Blackwell

Running log of every serving config tried, what changed, and how it moved
performance. Newest findings at the bottom of each section.

## Environment
- **Hardware:** 8x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each,
  PCIe (no NVLink), 2 NUMA nodes (GPU 0–3 / 4–7). Driver 580.119.02.
- **Model:** `lukealonso/GLM-5.2-NVFP4` — GlmMoeDsaForCausalLM, 744B-MoE /
  40B-active, 256 experts (8/token) + 1 shared, DSA sparse attn (index_topk=2048),
  first_k_dense_replace=3, MTP module, 1M max ctx. NVFP4 on expert MLPs (467 GB).
- **Image:** `voipmonitor/vllm:dark-devotion-39ae3ed-b12x5b2e018-cu132-20260617`
  (vLLM 0.11.2.dev279 b12x, CUDA 13.2). Canonical launcher `serve-glm52.sh`.
- **Serving constants:** TP=8, fp8 KV cache, `modelopt_fp4`, `b12x` MoE backend,
  `B12X_MLA_SPARSE` attention. KV budget ≈ **635,008 tokens**.
- **Benchmark:** `llm-inference-bench` (`llm_decode_bench.py`), sustained decode,
  15 s/cell, metric = aggregate decode tok/s (server stream usage). Per-user =
  1/ITL p50.

## Bring-up fixes (required before it would even start)
The stock `serve-glm52.sh` targets a different host; three fixes were needed:
| Symptom | Root cause | Fix |
|---|---|---|
| `Python interpreter not found .../.venv/bin/python` | script's default venv path absent | `PYTHON_BIN=/opt/venv/bin/python` |
| NCCL `unhandled system error`, `Could not open XML graph file ""` | image bakes `NCCL_GRAPH_FILE=""`; "-noxml" NCCL opens empty path | `unset NCCL_GRAPH_FILE; USE_NCCL_XML=0` |
| NCCL `Cuda failure 101 'invalid device ordinal'` | P2P over PCIe across 2 NUMA nodes unsupported | `NCCL_P2P_DISABLE=1` (b12x PCIe all-reduce still carries TP) |

Also: host GPUs are `0–7` (image defaults `CUDA_VISIBLE_DEVICES=2–9`).
Cold start ≈ 25–40 min (NFS weight read dominates); warm restart ≈ 4–6 min
because the 2 TB host RAM page-caches the 467 GB of weights.

Correctness sanity: "capital of France" → Paris, 17×23 → 391, clean reasoning. OK.

---

## Experiments (aggregate decode tok/s; [per-user tok/s p50])

### A — Baseline (image defaults): `max_num_seqs=16`, cudagraph capture=16, MTP on
`bench-baseline.json`
| ctx | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---|---|---|---|---|
| 0k | 83 | 146 | 250 | **74** | **148** |
| 16k | 77 | 123 | 205 | **79** | 159 |
| 64k | 70 | 116 | 196 | **80** | – |

**Finding:** throughput *collapses* at concurrency ≥ 8 (per-user ITL 59→9.6 tok/s).
Root cause: with MTP each decode step runs `num_seqs*(1+3)` tokens, but cudagraph
capture maxes at 16 tokens → only ~4 concurrent seqs stay on the graph; above
that it drops to the slow path. (Defaults are tuned for low-latency, c≤4.)
MTP itself is healthy — server logs show mean acceptance length 2.7–3.7.

### B — Fix cudagraph: capture=64 (= 16×4), `max_num_seqs=16`, MTP on
`bench-cg64.json` — change vs A: **only** `--max-cudagraph-capture-size 64`
| ctx | c=4 | c=8 | c=16 |
|---|---|---|---|
| 0k | 240 | **366** (+393%) | **559** (+278%) |
| 16k | 211 | 315 (+300%) | 485 (+205%) |
| 64k | 197 | 286 (+255%) | – |

**Finding:** cliff eliminated; throughput now scales monotonically. Per-user at
c=16: 9.7→37 tok/s. Low-concurrency unchanged (±4%, noise). **This is the single
biggest win.** Throughput still climbing at c=16 → seqs is now the limiter.

### C — Raise batch: `max_num_seqs=32`, capture=128 (=32×4), MTP on  ← **CHOSEN**
`bench-seqs32.json`
| ctx | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| 0k | 83[87] | 248[64] | 375[47] | 572[37] | **631[21]** |
| 16k | 74[76] | 217[55] | 320[41] | 490[31] | **593[19]** |
| 64k | 73[75] | 199[51] | 284[37] | – (KV cap) | – (KV cap) |

**Finding:** matches B at c≤16, extends curve to c=32. c=16→32 = +10–21% aggregate
for 2× concurrency (diminishing) while per-user halves. Knee ≈ c=16–32. Long ctx
is KV-capacity-limited (635k/64k ≈ 9 seqs). Picked `max_num_seqs=32` as the
throughput/latency sweet spot.

### D — MTP OFF control: `max_num_seqs=32`, capture=32
`bench-nomtp.json` — vs C (same batch, MTP on):
| ctx | c=1 | c=8 | c=16 | c=32 |
|---|---|---|---|---|
| 0k off | 50 | 228 | 344 | 458 |
| 0k on  | 83 (+64%) | 375 (+64%) | 572 (+66%) | 631 (+38%) |
| 16k on vs off | +53% | +32% | +30% | +14% |

**Finding:** MTP ON is faster at **every** concurrency, including peak. The ~2.8
acceptance more than offsets the 4× compute even when batched. **Keep MTP on.**

---

## Conclusion / chosen production config (`vllm.sh`)
`TP=8, max_num_seqs=32, cudagraph_capture=128 (=seqs×(1+spec)), MTP on (spec=3),
fp8 KV, b12x MoE, B12X_MLA_SPARSE`, plus the three bring-up fixes.

Ranked impact: **(1) cudagraph capture sizing — up to +390%**, (2) MTP on —
+14–66%, (3) max_num_seqs 16→32 — +10–21% tail. Single-stream ≈ 83–89 tok/s;
prefill ≈ 3.0–3.2k tok/s (8k–128k).

Knobs for other goals: pure offline batch → `MAX_NUM_SEQS=64` (capture auto→256);
lowest latency / interactive → `MAX_NUM_SEQS=8` keeps per-user ≈ 47 tok/s.

---

# Phase 2 — repo-inspired tuning (github.com/local-inference-lab/rtx6kpro)

Cloned to `./rtx6kpro`. The repo's own `models/glm5.2_v11.md` (dated 2026-06-18)
documents this exact model. Key deltas vs the image's stock `serve-glm52.sh`:

| Lever | Repo guidance | In our stock run? |
|---|---|---|
| `B12X_MOE_FORCE_A16=1` | v11 runtime forces A16 MoE decode; KLD shows "B12X A16" is the reference-accurate path (17/17 token match) | **NO** — not set by serve-glm52.sh |
| `use_index_cache:true` (hf-overrides) | v11 adds it to the DSA index_topk override | **NO** |
| `NCCL_MIN_NCHANNELS=8` | nccl-tuning: "major bandwidth gain" vs default 2 | NO |
| MTP tokens = **2** | spec-decode doc: MTP2 = stable sweet spot (+50-55%); MTP3 "risky"; >3 crashes | we use 3 |
| custom/PCIe all-reduce | pcie doc: "does NOT help on 8-GPU cross-socket; use `--disable-custom-all-reduce` (NCCL faster)" | we leave it on… |
| KV dtype fp8 | v11 KLD: fp8 fine for GLM-5.2 (KL 0.00009, 17/17). "bf16 mandatory" warning was older GLM-5/SGLang | fp8 ✓ |
| DCP4 | huge KV capacity (2.3M tok, 9.1x@256k) for long ctx; decode slower; needs v11 image for DCP4+MTP | DCP1 |

**Observed in our logs:** `b12x PCIe oneshot allreduce initialization failed:
failed to open CUDA IPC handle for peer rank 1. Falling back to PyNCCL` — this
k8s pod blocks CUDA IPC/P2P between GPUs (same root cause as the NCCL P2P fix),
so the custom all-reduce is already dead weight and silently falls back to NCCL.
This matches the repo's "8-GPU cross-socket → use NCCL" recommendation.

**Correctness (current config C: MTP3, fp8 KV, no A16):** CJK/garble watchdog at
40k ctx → **0 CJK chars**, coherent output, 89 tok/s single-stream. fp8 KV is
safe for GLM-5.2. ✓

### Planned experiments (attribution-focused)
- **E** = C + `B12X_MOE_FORCE_A16=1` + `use_index_cache:true` + `NCCL_MIN_NCHANNELS=8`
  (the "v11 alignment" bundle). Measure **decode AND prefill** vs C.
- **F** = best-of-E with **MTP=2** (attribute MTP2 vs MTP3).
- **G** = best config with `MAX_NUM_SEQS=64` (throughput tail).
- (later) DCP4 for the long-context capacity regime.

Results below as they land.

### E — C + `B12X_MOE_FORCE_A16=1` + `use_index_cache:true` + `NCCL_MIN_NCHANNELS=8`
`bench-E-a16idx.json`. Confirmed active: log `B12X MoE force-A16 enabled:
quant_mode=w4a16`, `use_index_cache: True`. (So config C ran the default
non-A16 / w4a4 MoE decode.)

Decode aggregate tok/s (C → E):
| ctx | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| 0k | 83→84 | 248→264 (+6%) | 375→413 (+10%) | 572→526 (−8%) | 631→**737 (+17%)** |
| 16k | 74→76 | 217→230 (+6%) | 320→354 (+11%) | 490→474 (−3%) | — |

Prefill tok/s (C → E): 8k 3122→2899, 64k 3054→2851, 128k 2869→2644 — **−7 to −8% everywhere.**

**Finding:** the bundle is a **decode-vs-prefill tradeoff**. Batched decode up to
+17% (and A16 is the KLD-accurate path), but prefill consistently −7%. Suspected
cause: A16 = w4a16 (16-bit activations) speeds memory-bound decode but slows
compute-bound prefill (prefill prefers 4-bit activations). `use_index_cache`
helps decode (reuses DSA indices) but not cold single-prefill. Ablating next.

### ⚠️ CORRECTNESS IS THE GATE (Discord field report, corroborated)
A community debugger documented that **default w4a4 MoE decode accumulates error
→ coherent short output but token-salad past ~3K *generated* tokens** on long
context, and that `config.json` ships `index_topk_pattern: null` so the DSA
indexer mis-selects tokens. **Fixes (now mandatory, not just speed):**
`B12X_MOE_FORCE_A16=1` (w4a16 decode) **and** `--hf-overrides
'{"use_index_cache":true,"index_topk_pattern":"FFFSSS…"}'`. Their result:
Estonia 0→10/10, LAVD garbage→5/5 EXACT. Also: **MTP k=2** (k≥3 → illegal
memory on long ctx) with **`--linear-backend auto`** (b12x has no NVFP4 linear
kernel for the nextn eh_proj). bf16 KV breaks the b12x sparse path (fp8 required,
NOT garbling); TRITON/FLASHINFER MLA unsupported on sm120. Our 256-expert
checkpoint is unpruned and boots fine (the 256→156 OOM they hit doesn't apply).

**New policy: run the Estonia correctness profile BEFORE any throughput bench;
only bench configs that pass.**

### E2 (ablation) — index_cache ON, A16 **OFF**, MTP k=3 — Estonia gate
`Estonia, 4 runs, max_tokens 8000`: **4/4 correct** (avg 4,828 completion tokens;
1 run hit the 8k cap but still scored correct). So `index_topk_pattern` +
`use_index_cache` alone already passes Estonia without A16. A16 remains the
default for the harder long-ctx robustness the field report documented (LAVD) and
for its decode speedup; the only cost is −7% prefill.

### F — CORRECTED FINAL: A16 + index_cache + MTP **k=2** + `--linear-backend auto`
(`max_num_seqs=32`, capture=96 (=32×3), `NCCL_MIN_NCHANNELS=8`)
Confirmed: `num_spec_tokens=2`, `quant_mode=w4a16`, `linear_backend=auto`,
`use_index_cache=True`.

**Estonia gate (4 runs, max 8000):** `correct_rate = 1.0 (4/4)`,
**hit_max_tokens = 0** — all terminated cleanly (vs E2's 1 cap-hit; cleaner than
k=3). avg 3,882 / p90 5,809 completion tokens. ✅ PASS → benched.

Decode aggregate tok/s (ctx0): c1 **77**, c4 257, c8 406, c16 552, c32 **763** (new peak).
Per-user p50: c1 81, c8 51, c32 25 tok/s.
Prefill tok/s: 8k 2936, 64k 2851, 128k 2429 (≈ −6 to −7% vs no-A16 C — the A16
correctness cost).

vs alternatives at ctx0: F(MTP2) beats C(MTP3,noA16) and E(MTP3,A16) at c=32
(763 vs 631/737), trades ~7% single-stream (77 vs 83) for stability + clean
termination + correctness. **MTP k=2 chosen** (also the field-report/repo
consensus; k≥3 risks illegal-memory on long ctx).

## CHOSEN PRODUCTION CONFIG = F  (`vllm.sh`)
`TP8, max_num_seqs=32, cudagraph_capture=96 (=seqs×(1+2)), MTP k=2,
draft_sample_method=probabilistic, --linear-backend auto, B12X_MOE_FORCE_A16=1,
hf-overrides use_index_cache+index_topk_pattern, NCCL_MIN_NCHANNELS=8, fp8 KV,
b12x MoE, B12X_MLA_SPARSE`, + the 3 bring-up fixes (PYTHON_BIN, unset
NCCL_GRAPH_FILE/USE_NCCL_XML=0, NCCL_P2P_DISABLE=1).

Ranked impact across the whole exercise:
1. **cudagraph capture = seqs×(1+spec)** — fixes the decode cliff, up to +390%.
2. **A16 + index_cache + index_topk_pattern** — REQUIRED for long-gen correctness
   (token-salad past ~3k tokens otherwise); A16 also +6–17% batched decode, −7% prefill.
3. **MTP on, k=2** — +~50% throughput, stable; vs k=3 (riskier) and off.
4. **max_num_seqs 16→32** — +10–21% throughput tail (c=32 peak 763 tok/s).

Prefill note: the only thing that *costs* prefill is A16 (−7%), and A16 is a
correctness requirement → not optional. If a pure-prefill/offline workload can
tolerate the long-gen risk, `B12X_MOE_FORCE_A16=0` recovers ~7% prefill.

---

# Phase 3 — deeper exploration + online research

**Research (web + repo `inference-engines/vllm.md`, vllm issues #37113/#43357, Sarathi
chunked-prefill paper):**
- **Power is NOT the bottleneck here.** GPUs are at the 600W max limit with **zero
  throttling**, yet draw only ~220–330W under load → decode is **comm/memory-bound**
  (8-GPU cross-NUMA all-reduce over NCCL-SHM). TP8 is forced (467GB needs all 8
  cards), so this comm floor can't be dodged. Repo confirms "500W ≈ 600W; 300W
  loses up to 30%" — we're already optimal on power.
- **`--enable-expert-parallel` is a trap on PCIe** ("kills batch throughput via
  inter-card traffic") — we correctly use TP-only. ✓
- **Chunked-prefill starves MoE expert reuse** (Sarathi/vllm): the per-iter token
  cap limits how many experts are reused per weight-load → prefill becomes memory-
  bound. Implication: **larger `max-num-batched-tokens` should help MoE prefill.** → test G.
- Undocumented b12x knobs exist (`B12X_MLA_SM120_UNIFIED`, `VLLM_USE_B12X_MHC`,
  `VLLM_USE_B12X_WO_PROJECTION`, `B12X_ATTN`) but are used only by *other* models'
  serve scripts (minimax/ds4), NOT the authoritative GLM-5.2 v11 config → likely
  incompatible with the DSA sparse-MLA path; not chasing blindly.
- A newer authoritative image exists: `voipmonitor/vllm:glm52-v11-darkdevotion-
  vllma86f74e-...20260618` (adds the DCP4+MTP per-layer-LSE fix + TP16 padding fix).
  Recommended if we want **DCP4+MTP for long context** (2.3M-token KV, 9.1x@256k).

### G — F + `MAX_NUM_BATCHED_TOKENS=16384` (prefill-focused) — ❌ FAILED
At `gpu_mem=0.96` the 16k-token prefill activation buffer **OOMs at runtime**
(94.9/95 GiB already used) → 500 errors → Estonia 0/3 (empty answers, not garble).
**The gate earned its keep.**

### G2 — G + `gpu_mem=0.93` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — ❌ not worth it
No OOM now (expandable_segments + lower util fixed it), KV dropped 635k→**462k**
tokens (−27% concurrency capacity). Estonia **2/3** (1 run didn't terminate).
Prefill vs F: 8k +0%, 16k +1%, 64k −1%, **128k +11%**. So bigger MoE prefill
chunks give ~nothing here except very-long (128k) context — GLM-5.2 prefill is
bound by the DSA indexer + A16 MoE compute, not chunk size. **Rejected**: tiny
prefill gain for a big KV/correctness cost. Keep `MAX_NUM_BATCHED_TOKENS=8192`.
(Useful byproduct: `expandable_segments:True` is a safe, v11-endorsed
fragmentation guard — documented as an optional knob.)

## Phase-3 conclusion
No new lever beat Config F within this image. Decode is at the **comm-bound floor**
(8-GPU cross-NUMA all-reduce; TP8 forced by the 467GB weights), and prefill is
compute-bound but not chunk-limited. The only remaining upside is a **different
image**: `glm52-v11-darkdevotion-...20260618` for DCP4+MTP long-context (2.3M-tok
KV) — recommended as a follow-up if >154k context with MTP is needed.
**Config F remains the production config.**

---

# Phase 4 — tool-calling bug (streaming) + fix

**Symptom (user, via agent client):** model "thinks", says what it'll do, then
stops — **no tool calls executed**.

**Diagnosis (deterministic):**
- Non-streaming tool calls: **work** (10/10). Streaming: **fail** (0/15).
- temp=0, seed=42, identical prompt: non-stream → `finish=tool_calls` (2 calls);
  stream → identical content preamble but `finish=stop`, **0 tool calls**. So it's
  a true streaming-parser bug, NOT MTP (fails with MTP off too), NOT sampling,
  NOT the reasoning parser (fails with it removed), NOT image version (the
  `glm4_moe`/`glm47` parsers are **byte-identical** in the v11 image).
- **Root cause:** `Glm4MoeModelToolParser._extract_tool_name_from_region` (in
  `vllm/tool_parsers/glm4_moe_tool_parser.py`) returns `None` for a
  **zero-argument** tool call — the name is terminated by `</tool_call>` with no
  `\n`/`<arg_key>`, so the streaming loop `break`s and emits nothing. Confirmed:
  arg-tool streams fine (`tool_calls`), zero-arg tool streams broken (0). Agent
  tools are often zero-arg (`codegraph_status`, `list_files` w/ defaults…), and a
  zero-arg call first in a batch kills the whole turn.

**Fix:** patched `_extract_tool_name_from_region` to accept `is_complete` and
treat a complete delimiter-less region as the full name (`patch/
glm4_moe_tool_parser.py`, mounted read-only over the original by `vllm.sh`).

**Validation (patched, MTP on, :8080):**
- single zero-arg streaming: `tool_calls [get_status]` ✅ (was 0)
- the temp0/seed42 failing batch: `tool_calls [codegraph_status, codegraph_files]` ✅
- streaming reliability @temp0.7: **10/10** (was 0/15) ✅
- **opencode end-to-end** (`opencode run`, streams + tools): ran `ls`, read result,
  replied "DONE: 2 entries" ✅ — full agent tool-loop works.

This patch is host/model-agnostic and should be upstreamed to vLLM.

---

# Phase 5 — v11 image + long context (target 1M)

Image: `voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-...20260618` (adds the
per-layer DCP-LSE fix for DCP+MTP). Parsers byte-identical → tool-call patch still
mounted. Host fixes (NCCL_P2P_DISABLE etc.) still required.

### DCP4, MTP **off**, `--max-model-len 1000000`, gpu_mem 0.94 — ✅ 1M CONTEXT WORKS
- KV cache: **2,167,391 tokens**; max concurrency for 1,000,000-tok request: **2.17×**.
- Boot quirk: after `torch.compile`, ~4 min of "no broadcast block" / idle-GPU
  spinning during **memory profiling** before the KV milestone — looks like a
  deadlock but is NOT. (Be patient; don't kill early.)
- **Needle-in-haystack @ 316,024 prompt tokens → retrieved `ZX9-QUASAR-7741`
  correctly.** A ~700k-token prompt also prefilled + generated.
- **Cost: DCP4 prefill ~1.5k tok/s (~½ of DCP1's ~3k)** — context-parallel adds
  per-layer comm. So 1M capacity trades prompt-processing speed.

### DCP4 + MTP — fragile
First DCP4+MTP+1M attempt sat in the post-compile memory-profiling spin for 8 min
and was killed (possibly premature, given DCP4-no-MTP also spun ~4 min). Retrying
with patience. v11 notes: DCP4+MTP was only "smoke tested", full sweep pending.

### DCP4 + MTP k=2 + 1M (v11) — ✅ WORKS (boot ~12-15 min, was killed early before)
KV cache **1,931,149 tokens**; 1M-req concurrency **1.93×**. Estonia **3/3** clean.
The first attempt's "deadlock" was a premature kill during a COLD compile +
3-phase cudagraph capture (PIECEWISE ~20s/iter at 1M); with warm cache it reached
KV in ~3 min. MTP accepts ~1.9/2 (~45%) at DCP4 (lower than DCP1's ~2.8).

Decode/prefill (our host):
| config | ctx/req | KV tok | decode c16 ctx0 | prefill 8k |
|---|---|---|---|---|
| DCP1 MTP k2 (vllm.sh, dark-dev) | 256k | 635,008 | **552** | **~2936** |
| DCP4 MTP k2 (vllm-v11-1m.sh) | 1,000k | 1,931,149 | 267 | ~1626 |
| DCP4 no-MTP (v11) | 1,000k | 2,167,391 | ~343* | ~1516 |

*cross-host ref. **DCP4 ≈ 2× slower than DCP1 on both prefill and decode — the
cost of long-context capacity.** Use DCP1 (vllm.sh) for ≤256k; DCP4 (vllm-v11-1m.sh)
only when you need >256k up to 1M.

### Tool calling on v11
- `tool_choice:"auto"` (agent default): streaming **5/5**, zero-arg + batch — patch works. ✅
- `tool_choice:"required"`/named: **500** xgrammar FSM error (`grammar rejected
  </tool_call>`); glm47 sets `supports_required_and_named=False`. Separate bug;
  workaround = use `auto` (which agents do).

## Final deliverables
- `vllm.sh` — production daily driver: dark-devotion image, DCP1, 256k, MTP k2, A16,
  tool-call patch. Fastest. On :8080.
- `vllm-v11-1m.sh` — long-context profile: v11 image, DCP4, up to 1M, MTP k2.
  ~2× slower; use when context > 256k.
- `patch/glm4_moe_tool_parser.py` — zero-arg streaming tool-call fix (mounted by both).

---

# Phase 6 — LMCache (prefill accelerator via KV reuse)

Recipe from github.com/myshytf/glm-5.2-v11-lmcache, adapted to this host
(`vllm-lmcache.sh`): v11 image + LMCacheMPConnector + 6 patches (4 stability,
2 DCP-aware) + our host NCCL fixes + our tool-call patch. LMCache installed at
runtime (`lmcache==0.4.6`). Config: DCP4, MTP k2, **L1=1024GB RAM, L2 disk
DISABLED (RAM-only)**, chunk_size 512 (= block 128 × dcp 4).

### Result — repeated long-context prefill
| request | prompt tokens | wall (prefill+8 tok) |
|---|---|---|
| COLD (1st) | 88,072 | **57.1 s** (full DCP4 prefill ~1.5k tok/s) |
| WARM (2nd, identical) | 88,072 | **0.4 s** |
**→ ~154× faster prefill on a cache hit.** LMCache log: `688/688 prefix hits
(688 L1, 0 L2) in 1.0 ms` — 100% hit from RAM, DCP-aware patches working
(all shards hit, not 1/688).

### Bring-up issues solved (this host)
1. Worker init `invalid device ordinal` — repo serve.sh omits the b12x env →
   vLLM used its CUDA-IPC custom all-reduce (blocked in pod). Fix: export the
   `VLLM_USE_B12X_*` + `VLLM_PCIE_ALLREDUCE_BACKEND=b12x` set in the entrypoint
   (routes to b12x PCIe → graceful NCCL fallback). lmcache's own KV transfer is
   TCP/ZMQ, so it sidesteps the IPC restriction.
2. Weight-load CUDA OOM — lmcache GPU overhead; fix `GPU_MEMORY_UTILIZATION=0.85`.
3. `register_kv_caches` 30s timeout — server slow under memory pressure; fix
   `lmcache.mp.mq_timeout=300`.

### Cost of L1=1024GB
The 1TB L1 evicts the OS page cache holding the 467GB weights → **cold NFS weight
reload (~16 min, ~88s/shard) every boot** (vs ~1 min warm). Boots take ~20-25 min.
The KV working set is tiny (MLA ≈ 0.5GB per 1M-token context), so L1=256GB already
holds ~500 cached contexts and keeps boots fast — recommended unless you truly
need 1TB of cached KV. Use case for LMCache: any repeated long prefix (agent/RAG
system prompt, shared document, multi-turn history) — turns minutes of prefill
into milliseconds.

---

# Phase 7 — DCP2: highest-perf ~1M context

Insight: DCP4 was the slowest option but we only need >256k (not the extra KV
headroom). DCP2 has HALF the context-parallel comm of DCP4 yet still fits 1M.

DCP2 + MTP k2 + A16, gpu_mem 0.95, max-model-len 1,000,000 (no lmcache):
**GPU KV = 1,002,943 tokens → fits a full 1M-token request (1.00x).** Estonia 3/3.

DCP2 vs DCP4 (ctx0, both ~1M-capable):
| metric | DCP4 | DCP2 | gain |
|---|---|---|---|
| decode c1 | 51 | 73 | +43% |
| decode c8 | 203 | 319 | +57% |
| decode c16 | 267 | 368 | +38% |
| prefill 8k | 1626 | 2159 | +33% |
| prefill 64k | 1495 | 2202 | +47% |
| prefill 128k | 1540 | 2153 | +40% |

**=> For ~1M context at highest perf, use DCP2, not DCP4 (+40-57%).** DCP ladder
(ctx0 decode c16 / prefill 8k / context): DCP1 552/2936/256k · **DCP2 368/2159/1M**
· DCP4 267/1626/1M+. DCP2 is the long-context sweet spot. Adding lmcache on top
gives the ~150x warm-prefill win for repeated context (chunk_size=256 at DCP2).

---

# Phase 8 — DCP2 + LMCache (best ~1M config) + capacity math

Target: ~1M context at highest perf WITH lmcache warm-prefill reuse.
Config: `vllm-lmcache.sh` with `DCP_SIZE=2 CHUNK=256 MTP=1 GPU_MEMORY_UTILIZATION=0.86
MAX_MODEL_LEN=900000 L1_GB=256 L2_GB=0`. DCP2 confirmed + lmcache healthy (chunk 256).

### LMCache L1 capacity (GLM-5.2 MLA, fp8)
MLA KV/token = kv_lora_rank(512)+qk_rope_head_dim(64) = 576 elems/layer × 78 layers
× 1 byte (fp8) = **~44 KB/token** (≈56 KB measured on-GPU incl. MTP + block padding).
| L1 RAM | ≈ tokens | ≈ full 1M-ctx | ≈ 128k docs | ≈ 8k prefixes |
|---|---|---|---|---|
| 256 GB | ~5–6 M | 5–6 | 45–55 | 700–900 |
| 1024 GB | ~20–24 M | ~20 | ~180 | ~2800 |
=> **256 GB is ample**; 1 TB is overkill and breaks boots (page-cache eviction).

### Boot reliability note
DCP2+lmcache + 1 TB L1 = pathological boot (weight load ~200 s/shard from cold NFS,
lmcache queue waits >150 s) — effectively unbootable in reasonable time. With
**L1=256 GB** the page cache survives and boots are normal. Repeated rapid relaunches
also thrash NFS/page-cache; let one boot finish (~15 min) before judging.

## FINAL recommendation
- **≤256k, fastest:** `vllm.sh` (DCP1, MTP k2, A16, max_num_seqs 32). Decode c16 552.
- **~1M context, highest perf:** **DCP2** (MTP k2, A16, gpu_mem 0.95) — fits 1.00M
  KV, +40–57% vs DCP4. Decode c16 368, prefill 8k ~2160.
- **+ repeated long prompts:** add LMCache (RAM-only L1=256 GB, chunk=block×dcp) for
  ~150× warm prefill. Proven at DCP4 (688/688 hits, 57 s→0.4 s); DCP2 uses chunk 256.

### DCP2 + LMCache — context-ceiling tradeoff (measured)
At `gpu_mem 0.86` (needed to avoid weight-load OOM with LMCache), DCP2 KV pool =
**18.55 GiB → estimated max model length 727,680**. So `MAX_MODEL_LEN=900000`
ValueErrors at boot ("KV cache needed 22.94 GiB > available 18.55 GiB"); set
`MAX_MODEL_LEN<=720000`. Net: LMCache's GPU buffer costs ~280k of context ceiling
(1.00M without LMCache @ gpu_mem 0.95 → ~720k with LMCache @ 0.86). Clean config
error, not a crash. Choose: full 1M cold-only (DCP2 no-LMCache) OR ~720k + ~150×
warm-prefill reuse (DCP2+LMCache) OR full 1M + LMCache via DCP4 (slower).
