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
| NCCL `Cuda failure 101 'invalid device ordinal'` | ~~P2P across 2 NUMA nodes unsupported~~ **wrong — see Phase 10**: forced-P2P driver keys + ACS redirect | ~~`NCCL_P2P_DISABLE=1`~~ superseded 2026-07-03: P2P fixed, now `NCCL_P2P_DISABLE=0` |

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

---

# Phase 9 — v13 "eldritch" 20260629 + FlashInfer SM120 (2026-07-01..02)

Image bump to the canonical v13 bugfix build
`voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x8ce61f9-cu132-20260629`
(20260625 launch build had GPU-CPU sync regressions/OOMs; 20260627 had a DCP
topk_scores_buffer crash). Speed ≈ v12 on B12X; the win is stability + bugfixes.
New launcher `vllm-v13.sh`; v12 launcher retained for reference.

**Attention backend A/B (DCP2 MTP3, 15s cells):** `FLASHINFER_MLA_SPARSE_SM120`
(fuse-less — the fuse pass crashes on this host) beats `B12X_MLA_SPARSE`:
decode c1 86.9 vs 78.1 (+11%), 128k c2 +16%, KV 1,271,726 vs 1,240,368 (+2.5%),
prefill +2%. MTP acceptance 69.3% vs 74.9% — net win regardless. Estonia 6/8
(temp-0 nondeterminism; passes). SM120+MTP was upstream-unvalidated; validated here.
**SM120 at DCP1 crashes** (illegal memory access in sparse_mla_sm120_decode
autotune) → launcher defaults SM120 only for DCP>1.

**Flag trials (8 one-variable boots, all on SM120/DCP2/MTP3):** oneshot-allreduce
32K/256K/1MB flat/−4%; NCCL_BUFFSIZE 16M flat; NCCL_MIN_NCHANNELS 16 −5% c1;
numactl interleave not reproducible (90.6 c1 @10s, 82.0 @15s confirm); 
MAX_NUM_BATCHED_TOKENS 16384 crashes at boot (b12x chunk ceiling / AOT ranges).
Adopted only `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (speed-neutral,
fragmentation insurance; costs ~1% KV).

---

# Phase 10 — GPU P2P fixed (2026-07-03): the big one

P2P was broken by config, not hardware. Two stacked root causes:
1. **Forced-P2P driver keys** (`ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2`
   in `/etc/modprobe.d/nvidia-p2p-override.conf`, from an early experiment) made
   the driver ADVERTISE P2P everywhere while every mapping failed
   (`CUDA invalid device ordinal` — the error that originally forced
   `NCCL_P2P_DISABLE=1`). Community-confirmed cause+fix (06-29 daily summary).
   Keys removed (backup: `grid_results/nvidia-p2p-override.conf.bak`) + module
   reload (runbook: `p2p-module-reload.sh` — evicts gpu-operator pods via node
   labels, stops persistenced, rmmod/modprobe via nsenter).
2. **ACS redirect on the PCIe switches** (4x Broadcom PEX89000 Gen5, 2 GPUs each):
   downstream ports had ACSCtl=0x001d (ReqRedir+CmpltRedir+UpstreamFwd) hairpinning
   ALL peer traffic through the root complex/IOMMU. Cleared to 0x0001 (SrcValid
   only) via setpci; persisted by host systemd unit `gpu-acs-p2p.service`
   (raw config-space write at offset 0x176, before kubelet).

**Measured P2P after fix (p2pmark):** intra-socket ~54 GB/s, cross-socket ~49 GB/s
(was ~42 flat host-staged; direct mappings failed entirely). Concurrent all-cross-
socket saturates ~155 GB/s aggregate (UPI ceiling — the real structural limit).

**Serving config:** `NCCL_P2P_DISABLE=0` + `VLLM_ENABLE_PCIE_ALLREDUCE=0`.
Critical interaction: with real P2P, the b12x PCIe oneshot all-reduce COLLAPSES
batch-1 decode (30 tok/s, reproducible) — tiny-payload cross-socket MMIO latency.
NCCL everywhere is strictly better.

**Production result (v13-0629 + SM120 + DCP2 + MTP3 + P2P, 15s cells, Estonia 3/3):**
| metric | v12 prod (07-01) | now | Δ |
|---|---|---|---|
| decode 0k c1 | 77.1 | **100.5** | +30% |
| decode 128k c1 | 65.4 | **96.9** | +48% |
| decode 0k c8 | 344.6 | **419.8** | +22% |
| prefill 8k | 1,988 | **3,079** | +55% |
| prefill 128k | 2,049 | **3,122** | +52% |
Single-stream decode now exceeds the community bare-metal reference (97.1).
NOTE: all pre-07-03 numbers in this journal are on the broken-P2P stack.
Host reboot: driver keys persist; ACS persists via the systemd unit; if the unit
is ever missing, ACS reverts to redirect (slow mode, still functional).

**luke-test image** (`eldritch-enlightenment-luke-test-v3f65c52-...-20260703`,
Luke's b12x prefill counteroffensive): on nvfp4 checkpoint NOT worth it yet —
`fold_values` buffer = max_model_len × 10,240 × fp32/GPU (38 GiB @1M → OOM);
int32 overflow caps ctx at ~200k (262144×10240 > 2^31); at 192k: Estonia 3/3,
prefill unchanged (~3.1-3.3k — fast path doesn't engage for modelopt_fp4),
decode +4-15%. The 5-6k prefill numbers are mxfp4-a8 only. Reported findings;
retest when the nvfp4 port lands.

**Unlocked for retest by working P2P:** `fuse_allreduce_rms` and SM120@DCP1
(both crashed under broken P2P), DCP4 (comm tax repriced), overnight grid sweep
(all axes stale).

---

# Phase 11 — v14 "eldritch v7" + LMCache (2026-07-07)

Image `voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707`
(recipe glm5.2_v14.md). v14 = Luke's b12x rework (beats FlashInfer SM120 — "everything
b12x now"), online MXFP8 dense-linear conversion, fp8 DMA, hybrid DCP, InstantTensor.

**Host-specific fix (same class as v13):** the image launcher hardcodes the b12x
PCIe all-reduce, which COLLAPSES batch-1 decode here (c1 30-45 tok/s, reproducible
across every DCP/MTP/F8 combo). We mount a patched launcher (`patch/run-glm52-v14-server`)
defaulting `VLLM_ENABLE_PCIE_ALLREDUCE=0` -> NCCL all-reduce. With that: v14 DCP2/1M/A16
c1 ~90, c8 ~445, prefill ~3.1k, Estonia 3/3 (SM120-free B12X path is also free of the
fold_values OOM, so 1M context boots fine — KV ~1.15M @ gpu0.93). f8=ring adds prefill
but rides the same PCIe-DMA path we disable, so it's off here.

## LMCache WORKS on v14 (production: `vllm-v14-lmcache.sh`, DCP2 + 768GB RAM)

Community MP-connector port (myshytf), vendored in `patch/lmcache-v14/` + patched
`patch/serve_glm52_v14_lmcache.sh` (lmcache 0.4.6 pip-installed at container start,
12 patches applied). How it's wired:
- `--kv-transfer-config {kv_connector: LMCacheMPConnector, kv_role: kv_both}` + a
  local ZMQ MP server holding the RAM cache; `--disable-hybrid-kv-cache-manager`.
- **RAM-only** (`L2_GB=0` drops the disk adapter): L1 = 768 GB pinned = ~15M tokens
  (53.4 KB/token measured from LMCache's KVLayerGroupInfo). ~16 users' 300k contexts.
- DCP2 (1.09M GPU pool) for speed; LMCache restores evicted contexts from RAM.
- Host overrides: `VLLM_ENABLE_PCIE_ALLREDUCE=0` (as above); large-L1 boot needs
  `GLM52_LMCACHE_MQ_TIMEOUT>=pinning-time` (768GB pins ~75s > the stock 30s register
  timeout -> we default 300s).

**Validated:** Estonia 5/5; bench within ~3% of plain v14 DCP2 (connector tax); 1M
context fits (KV 1.06M). **Warm restore proof (DCP2, 768GB):** cold 297k-token prefill
164.6s; after evicting from the GPU pool, re-send restored from RAM in **3.3s = 49x**
faster (lookup_hit_tokens confirmed the hit). Observability: `:8088/metrics`
(RAM-only, so no l2_* metrics; use `l1_write_chunks_total`/`l1_evicted_chunks_total`
names, not the newer doc names). Hit rate =
`lookup_hit_tokens_total / lookup_requested_tokens_total`.

Note: rides community patches on a day-old image; revisit when festr bakes LMCache in
or the nvfp4-KV image lands (both will likely need a patch refresh).

---

# Phase 12 — v15/v17 fathomless-firmament, TLS, LMCache crash fix (2026-07-13..15)

## Image progression
- **v15** = `fathomless-firmament-v19-vllm0d1ad03-b12x90172a5-cu132-20260709`. Pure
  rebase of the v14 runtime onto `dev/fathomless-firmament` (CuTe compile fallback,
  restored DCP A2A token cap). Our launcher (`run-glm52-v*-server`) is **byte-identical**
  across v14/v15/v17, so image bumps are a one-line `IMAGE=` swap. MTP0 cc1 vs v14: noise
  (+0.5–2%). Rode it in on a restart, not a dedicated one.
- **v17 (PRODUCTION)** = `fathomless-firmament-v17-vllm6ccc3eb-b12x1377d5f-fi801d57a-cu132-20260714`
  (recipe glm5.2_v17.md). Adds generalized DCP prefill workspace (PR #94), NF3-hybrid
  format/NVFP4-KV (for the TP4 madeby561 checkpoint — not us), B12X NF3 tile-binding fix.
  For our TP8/DCP2 the only applicable gain is PR #94 prefill (+3.5–4.8% per the doc);
  wired via `DCP_PREFILL_WORKSPACE=auto` in `vllm-v14-lmcache.sh`. Decode/KV/quality
  unchanged. `vllm-v14-lmcache.sh` default IMAGE now points here.

## TLS + port (2026-07-13)
Self-signed cert (RSA-4096, 1000-yr expiry, in `certs/`, gitignored key) wired via
`--ssl-keyfile/--ssl-certfile` behind `TLS_ENABLE`. **Cilium Hubble DPI on port 8080
mangles TLS** (works as HTTP, breaks as HTTPS) — moved to **8443**. Later disabled TLS
(`TLS_ENABLE=0` default) at user request; service = plain HTTP on 8443. uvicorn can't
toggle TLS live, so flipping it = a process restart.

## LMCache KV-xfer crash + fix (2026-07-14) — IMPORTANT
Under real load (`Running: 10, GPU KV 88.6%`) the engine died:
`assert RequestStatus.is_finished(req.status)` in scheduler `_update_from_kv_xfer_finished`
→ EngineCore fatal → EngineDeadError. Trigger: KV-cache-pressure **preemption** +
LMCache `SafePreemptRestore` fires a recv-finished callback for a still-active
(preempted) request. `patch_kv_xfer_assert_v10.py` guarded only 2 of the 3 asserts in
that function. **Fix:** Patch 3 (guarded skip — free only if finished, else log & keep
scheduled). Present on both v15 and v17 bases; validated (applies + compiles) on both.
GOTCHA: the launcher applies the **external** clone `/root/glm-5.2-v11-lmcache/patches/`,
which overrode the vendored edit — synced both, then repointed the launcher at the
committed vendored `patch/lmcache-v14/` (dirs are now byte-identical) so the fix can't
drift. See memory `lmcache-patch-external-clone-overrides`.

## v17 bench (llm-inference-bench 0.4.24, live LMCache/DCP2/MTP3, ctx0)
| c1 | c4 | c8 | c16 | c32 |
|---:|---:|---:|---:|---:|
| 91.2 | 288.6 | 440.6 | **657.2** | 644.2 |

Prefill (client prompt/TTFT): 8k 3,047 · 64k 3,224 · 128k 3,180 tok/s (flat to 128k).
`c32≈c16` because `MAX_NUM_SEQS=16` caps running seqs (not a cliff). `c1=91` matches the
Phase-11 v14 ~90 → **no regression**. c1 ~19% under the LMCache-free/131k reference is the
connector + 1M-ctx tax, not a v17 issue. MTP accept ~2.7, VRAM 97.5%, GPUs 93% / 2.1 kW.

## Cold-boot: the LMCache pin self-evicts the weights
Every LMCache cold boot loads at NFS speed (~270 MB/s, ~20–30 min) even when weights were
just page-cache-warm: pinning the 768 GB L1 reclaims physical pages and **evicts the idle
436 GB weight cache** (weights live on GPU after load → first LRU victims). Confirmed:
post-load `vmtouch` shows 96.5% resident, but the next boot is cold anyway. InstantTensor
`BUFFERED` is NOT wonky — it populates cache correctly; the pin just evicts it.
**Fix (respects no-node-local-disk rule):** `vmtouch -dl <blobs>` to mlock the 436 GB
(1.2 TB locked of 2 TB) → future restarts stay ~5 min. Not yet applied.

## Checkpoint alternatives evaluated — stay on `lukealonso/GLM-5.2-NVFP4`
- **madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid** (v17 TP4): higher headline GPQA but the win
  is vs plain NVFP4, not our A16 (KLD 0.060). It mandates `nvfp4_ds_mla` KV (breaks our
  fp8-calibrated LMCache), NF3 tile geometry is TP4-only (untested/fragile at TP8), MTP-off.
  Built for memory-starved 4-GPU serving — the regime we're not in. **No.**
- **RedHatAI/GLM-5.2-NVFP4-FP8**: **compressed-tensors** (LLM Compressor), NOT modelopt_fp4.
  Routes through vLLM's generic compressed-tensors MoE path, **not B12X** → loses the whole
  B12X kernel stack (MoE + B12X_MLA_SPARSE). Attention is FP8-block weights; needs vLLM
  PR #47780; MTP undocumented; GPQA 89.1 (−2.1, ~plain NVFP4, below the hybrid). Engineered
  for stock vLLM TP4. **No** — B12X is exactly what makes us fast.

---

# Phase 13 — v18/v19 "Gilded Gnosis" CKV-gather; DCP8 vs DCP2 (2026-07-21)

koush's full-CKV DCP prefill (vLLM PR #111) shipped in the gilded-gnosis images
(v18 gated it to TP8/DCP4+DCP8; **v19 extends the gate to 8:2**). We swept it.
Image: `gilded-gnosis-v19-vllm7ea567a-b12xc7dc733-fi801d57a-cu132-20260719`.
Bench: llm-inference-bench @86cf05c, 15s/cell, idle. TP8/MTP3/A16/fp8-KV. Aggregate decode tok/s @ctx0.

| Config | c1 | c8 | c16 | pf8k | pf128k | KV | LMCache | warm-restore |
|---|--:|--:|--:|--:|--:|--:|:-:|:-:|
| v17 DCP2 (old prod) | 86.9 | 435 | 662 | 3,019 | 3,175 | 1.07M | y | y |
| **v19 DCP2 +LMCache (NEW PROD)** | **101.8** | **448** | **689** | **3,779** | **3,973** | 1.04M | y | 18.6× |
| v19 DCP2 no-LMCache | 97.0 | 447 | 670 | 3,939 | 3,989 | 1.07M | n | n |
| v19 DCP8 PCIE0 | 54.2 | 76 | 95 | 2,816 | ~2,861 | 3.69M | n | - |
| v19 DCP8 PCIE1 | 29.5 | 71 | 95 | 3,277 | ~2,861 | 3.69M | n | - |

**DCP8 is unusable on this host.** Decode collapses ~7× (c16 662→95) *independent of PCIe
all-reduce backend* — CKV engages fine; the wall is DCP8's 8-way cross-NUMA all-reduce per
decode token on our PCIe-only 2-NUMA fabric (GPU0-3 / 4-7). koush's ~2× were on non-NUMA
PCIe-x8 boxes. My PCIe-allreduce=0 host-fix (right for DCP2 batch-1) is irrelevant here;
both backends collapse. Not pursuing DCP8.

**v19/DCP2 + LMCache = new production.** vs v17: decode c1 +17% (87→102), c16 +4%; prefill
+25% at 8k/128k — CKV-gather delivering *at DCP2*. Warm-restore intact (18.6× on a 90k re-send),
same 1M KV, same topology. All 12 LMCache patches + 3 KV-xfer guards apply cleanly on
gilded-gnosis (no re-validation needed). LMCache adds no throughput penalty.
Launcher: `vllm-v19-lmcache.sh` (DCP2, gpu_mem 0.94). 64k prefill omitted from headline
(single-scout JIT-spike noise; 8k/128k stable). Cold-boot caveat unchanged (768GB pin evicts weights).

---

# Phase 14 — long-context prefill, SPEC_EXTEND A/B, GPU health (2026-07-23)

Image unchanged: `gilded-gnosis-v19-...-20260719`. TP8/DCP2/MTP3/A16/fp8-KV, LMCache on.
**We do NOT move to v20**: three independent DCP-regression reports on 2026-07-22
(`ValueError: DCP workspace projection received an invalid tensor layout` at DCP2 *and*
DCP8; `DCP_PREFILL_WORKSPACE=auto/1` worked on v19, so it is a v20 regression; setting it
to 0 trades the crash for startup read faults). niklasb1337 reverted to a patched v19.
v20 also crashed for johnblackwell6000 at DCP4/TP8/MTP3. Revisit when those are fixed.

## 1. Prefill past 128k — the DCP4 taper does NOT reproduce at DCP2

**First finding: `llm-inference-bench` silently caps prefill at 128k.**
`llm_decode_bench.py:12553` computes `max_prefill = min(131072, server_context_length-64)`
and *discards* any `--prefill-contexts` above it with no warning. Every DCP-taper number
circulating in the RTX6kPRO Discord is therefore measured inside a ≤128k window, while
people run 262k+ in production. Local patch adds `LLM_BENCH_MAX_PREFILL` and logs drops.
(Patch lives in the `llm-inference-bench` clone, not this repo — reapply after `git pull`.)

Prefill tok/s, client metric, N=1/point, idle, DCP2 + CKV gather:

| ctx | 8k | 64k | 128k | 256k | 512k | 768k |
|---|--:|--:|--:|--:|--:|--:|
| tok/s | 2,754 | 3,458 | 3,862 | **3,843** | 2,942 | 3,149 |
| TTFT s | 2.98 | 18.65 | 33.37 | 67.04 | 175.08 | 245.31 |

**Flat 128k → 256k**, then a soft ~20% step down (512k/768k are inverted, so at N=1 treat
them as one level, not a trend). Compare ufear on **DCP4**: 5k @ctx0 → 2k @128k, a 60%
collapse *inside the range where we are flat*; DCP8 flat ~3.8k. koush only ever tested the
gather at DCP8 and warns DCP≠TP adds comms. **Conclusion: the taper others see does not hit
our DCP2. Keep `VLLM_B12X_MLA_CKV_GATHER=1`.**

## 2. Byte-identical verification is INVALID on this stack

The correctness recipe circulating in Discord (temp-0 greedy; MTP3-flag-on vs flag-off vs
MTP0 must be byte-identical) **cannot work here**. Control experiment: same config, same
seed, `temperature=0`, verified-idle server, two captures back-to-back, nothing changed:

| prompt | count | code | prose | recall | cjk |
|---|:-:|:-:|:-:|:-:|:-:|
| A vs B | IDENTICAL | differs @102 | differs @50 | n/a | differs @40 |

4/5 diverge with *no* variable changed. Only the short, tightly-constrained `count` prompt
is stable — consistent with batch-composition nondeterminism (reduction order perturbs
logits ~1 ULP, flips a near-tie, diverges from there); fp8 KV and MTP timing likely add to
it. **Any byte-comparison here reports "not lossless" unconditionally.** Anyone using that
test to accept/reject a kernel flag is reading noise. Tool: `spec-extend-lossless-check.py`
(kept — the capture/compare harness is still the right shape if batch-invariant kernels
ever land). Note this server names the reasoning field `reasoning`, NOT `reasoning_content`;
some prompts spend the entire token budget there and return `content: null`.

## 3. `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` — A/B, two cold boots

Routes the MTP q_len=4 verify batch through the sparse *decode* kernel instead of
extend/prefill. Off by default in v19; the in-tree comment (`b12x_mla_sparse.py:1065`)
warns later verify rows must attend to earlier draft rows. festr measured 127→160 tok/s and
verified it lossless — but **on the v20 kernel**, so we re-verified on ours.
Byte-comparison being unusable (§2), correctness was judged on **MTP acceptance rate**:
a verifier scoring drafts wrongly must shift acceptance.

| | flag=0 | flag=1 | Δ |
|---|--:|--:|--:|
| decode c1 | 96.0 | **104.5** | **+8.9%** |
| decode c8 | 468.8 | 449.8 | −4.1% |
| decode c16 | 705.5 | 676.3 | −4.1% |
| prefill 8k / 128k | 3,905 / 3,981 | 3,874 / 3,916 | ~flat |
| **MTP accept** | **60.86%** (17,034/27,990) | **60.70%** (16,244/26,760) | **−0.16pp** |

**Acceptance is flat at ~0.4σ** (pooled SE ≈0.42pp; drafts are correlated within sequences
so true SE is larger still) — no evidence of verifier corruption. **Shipped: flag=1 is the
launcher default**; `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0` reverts with no image change.
Caveats: our +8.9% is well short of festr's +26%, c8/c16 came back ~4% *lower* (N=1, 15s
cells — I would not defend −4% as real, but there is no multi-stream win either), and the
flag=1 64k prefill point (3,041) is a single-sample outlier, discarded. Acceptance rate is
a proxy; the conclusive test is task accuracy (GSM8K/GPQA per config) — not yet run.

## 4. GPU health — all 8 cards clean

koush found a faulty SM botching the hardware fp8 E4M3→f16 conversion (`cvt.rn.f16x2.e4m3x2`),
after 2 days chasing nondeterministic output. His isolated repro, swept over our 8 GPUs
(`run-fp8-sweep.sh`, 10,000 iters each, ~12 s/GPU): **CLEAN on all 8, 0 glitches.**
Our VBIOS `98.02.81.00.01` is newer than his faulty card (`98.02.52.00.02`); consistent with
an early-unit issue. **This clears GPU 3 — where the §5 crash landed — so that fault is the
B12X kernel, not silicon.** "Clean at this exposure" ≠ proven good; his fault took 2 days to
surface. Two upstream bugs found reviewing his source, worth reporting back:
(a) **no CUDA error checking at all** — after an Xid every call fails silently and it prints
`RESULT: CLEAN`, i.e. a real fault reports success. We added hard `CUDA_CHECK` + device-count
validation before running; kernel math left byte-identical so results stay comparable.
(b) the warp reduction starts at `offset = LANES_PER_ROW` instead of `LANES_PER_ROW/2`, so
each output sums across two adjacent rows — harmless for the replica-vs-replica diagnostic,
but the printed floats are meaningless (as is hw-vs-sw, whose LUT is wrong for `exp==15`).
The `.cu` itself is NOT vendored here (third-party, from Discord); `run-fp8-sweep.sh`
expects a `fp8_scale_test` binary built from it.

## 5. New crash signature: deep prefill → decode warmup

Killed a server at 22 h uptime. After the 768k prefill sweep completed, the decode grid
errored out entirely and the container exited:

```
Worker_TP3_DCP1  WorkerProc hit an exception
  deepseek_v2.py:1555 forward → cuda_graph.py:259 → piecewise_backend.py:380
  cudaErrorIllegalAddress
```

Same family as the 2026-07-16 B12X paged-indexer illegal-memory crash. Likely trigger: the
bench's pre-decode warmup runs at "C=1 max-runnable context", which immediately after a 768k
prefill is an extremely deep context. **Not hardware** (§4 clears GPU 3). Watchdog detected
in ~60 s and restarted cleanly (1/3 in window). Not yet deliberately reproduced — worth
doing, as it is a sharper lead on the ~19.5 h-MTBF hangs than anything so far.
**Operational lesson: dump `docker logs` to a file the instant a container exits** — the
watchdog recycled the container while I was reading the trace and the full stack was lost.

## 6. Bench hygiene — gate on idle

Two runs today were wasted by real user traffic on the server (`running_reqs=2/1`, `9/8`;
at c16 the extra stream pushed past `MAX_NUM_SEQS=16` so a request queued, warmup never
stabilised, and the cell reported `∅ (16/16)` capacity-limited). The contaminated numbers
(c1 64.6 vs 96.0 clean) looked exactly like a 33% regression. **Always gate a bench on
`vllm:num_requests_running == 0` and re-check after** — a one-line guard that saves an hour.

---

# Phase 15 — r4 → r11, native LMCache (2026-07-27..29)

## Image: `gilded-gnosis-v20-...-20260729-r11`, launcher `vllm-v20-r11.sh`

r11 retires our entire hand-rolled LMCache stack. The image ships lmcache
`0.5.2+glm52dcp.4` plus `/usr/local/bin/glm52-lmcache-wrapper.sh`, so the
pip-install + 12-patch bootstrap, the v16 connector-injection patch, and
`--disable-hybrid-kv-cache-manager` all became dead weight. Patches went 3 -> 1.

**Why r11 and not r9:** r9 has a LMCache MP-server bug that hangs requests
forever. `lmcache/v1/multiprocess/mq.py`, `_call_blocking_handler._notify_response`
logged the exception and sent NOTHING, so the client future never completed and
the vLLM request pinned in `WAITING_FOR_REMOTE_KVS` until engine restart. Fixed
in r11 (`_queue_error_response`). **Our watchdog cannot see this failure mode** —
`/health` returns 200 while an individual request hangs.

## The GMU trap — booted healthy, died on the first real batch

First r11 boot at the inherited GMU 0.96 came up green and then died:

    torch.OutOfMemoryError: Tried to allocate 96.00 MiB.
    GPU 6 ... 94.97 GiB total of which 89.25 MiB is free
    (pynccl all_reduce -> torch.empty_like)

Two r11 changes each ate headroom that 0.96 used to leave spare:
1. exact indexer folding freed workspace, and **vLLM hands freed bytes straight
   to KV cache** — 1,113,600 tokens vs r4's 990,592 at the SAME GMU. Our safety
   margin silently became KV cache.
2. the wrapper starts `--max-gpu-workers 8` LMCache GPU clients, ~0.7 GiB/GPU,
   which our old 0.4.6 bootstrap never did.

GMU 0.96 -> **0.94** gave 1,027,968 KV tokens (still above r4's 990,592) plus
~1.9 GiB/GPU back. **A capacity gain at an unchanged GMU is the warning sign,
not the win.**

## First clean bench since 0724 (all cells CLEAN)

| conc | r11 + LMCache | 0724 clean baseline | delta |
|---|---|---|---|
| 1  | 110.5 | 106.1 | +4.1% |
| 4  | 312.3 | 314.8 | -0.8% |
| 8  | 431.5 | 430.9 | +0.1% |
| 16 | 634.7 | 617.4 | +2.8% |

This settles the open question from Phase 14: **LMCache costs ~nothing in
throughput.** The apparent "40% decode regression" on r4 was foreign traffic in
the batch, which is why the driver now prints a per-cell CLEAN/CONTAMINATED
verdict from `avg_running_reqs`.

MTP acceptance: 68.72% lifetime (r4: 68.10%) — no regression. The 64.5% -> 61.3%
"drift" seen in bench windows is an artefact of synthetic bench prompts.

## Two upgrade traps defused (both still live in r20)

- `DCP_CKV_GATHER_MAX_TOKENS` defaults to **140000**; we pin 524288 or every
  prefill above ~140k drops to the slow fallback path.
- The PCIe calibration is broken on our 4+4 SYS-split 2-NUMA topology
  (misterfix hit it on identical `nvidia-smi topo -m`). Pinning all four
  calibration knobs yields `calibration_status=skipped:all-explicit` and the
  probe never runs.

## Container-removal race

`docker rm -f` returns before the name is released when 768 GB is pinned, so
`docker run` lost the race with *"the container name /glm52 is already in use"*.
This would also have broken the **watchdog's** restart path. The launcher now
waits up to 120 s for the name to actually disappear.

---

# Phase 16 — r20 + the PCIe DMA campaign (2026-08-03)

Production config changed for the first time since Phase 13. Headline: **+33%
prefill at every context size**, decode flat, from size-gating the b12x PCIe
collective and compressing its wire.

## 0. THE COLD-RUN ARTIFACT — read this before trusting any older number

**The first `llm_decode_bench` run after a boot measures warm-up, not the
config.** Runs 2 and 3 of the same config agree to **<1%**.

| ctx | cold penalty (candidate) | cold penalty (baseline) |
|---|---|---|
| 8k   | **-77%**  | -1.3% |
| 64k  | **-37%**  | **-42%** |
| 128k | -0.4%     | -1.7% |
| 256k | -9.6%     | -7.4% |
| 512k | -0.06%    | -0.1% |

The first pass pays JIT compilation, CUDA-graph + FlashInfer autotune, and a
cold page cache, and the penalty is **wildly context-dependent** — so it does
not cancel when comparing two configs, it just adds a large variable error to
whichever cell is most cold-sensitive.

This retroactively explains the "64k anomaly" chased for an entire session (DMA
regressed 64k, a query-split change fixed it, `ring` collapsed it) — 64k is
simply the second most cold-sensitive cell. It also disproves the assumed "~7%
noise floor": real warm reproducibility is 0.1-0.9%.

**Every prefill number in this file recorded before 2026-08-03 is a cold single
sample and is NOT comparable to a warm one.** `run-r20-verify-bench.sh` now runs
a discarded warm-up pass first (`SKIP_WARMUP=1` defeats it).

## 1. The probe is a supported deployment preflight — use it

`sparkinfer.comm.pcie.overlap_probe` must run with the GPUs EMPTY. It measures
the collective directly, and upstream anticipates exactly this tuning
(`custom_all_reduce.py:531`: *"A deployment preflight can tune its crossover or
disable it when lossless DMA never beats NCCL on the selected PCIe topology"*).

    torchrun --standalone --nproc-per-node=8 \
      -m sparkinfer.comm.pcie.overlap_probe \
      --tp-size 8 --dcp-size 2 --indexer-shards 1 --active-iterations 15

(`--indexer-shards 0` is rejected — it must divide tp and dcp; pass 1.)

TP all-reduce, lossless BF16, fine sweep (payload = rows x 6144 x 2 bytes):

| rows | bytes | nccl_ms | dma_ms | gain |
|---|---|---|---|---|
| 1     | 12288     | 0.075 | 0.445 | **-490.9%** |
| 512   | 6291456   | 0.488 | 1.000 | -104.8% |
| 1536  | 18874368  | 1.097 | 1.143 | -4.2% |
| 2048  | 25165824  | 1.483 | 1.815 | -22.4% |
| 2560  | 31457280  | 1.820 | 1.679 | +7.7% |
| 4096  | 50331648  | 2.896 | 2.474 | +14.6% |
| 8192  | 100663296 | 5.702 | 4.614 | **+19.1%** |

Below ~2560 rows the band is **non-monotonic noise** (NCCL protocol switching);
the probe's own `dma_min_bytes` recommendation came back as 0, 25165824 and
31457280 on three runs. Do not trust a single recommendation. Above 2560 it is
clean and monotonic, and 8192 rows reproduced at +19.5%/+19.1%.

Larger payloads **plateau** — 12288 rows +20.8%, 16384 +20.7%, 32768 +20.2% —
which is why `MAX_BATCHED_TOKENS=16384` was rejected without spending a boot.

Also reconfirmed by probe: `DCP_CKV_PREFETCH_DEPTH=1` is correct (+4.0% to
+16.9% overlap benefit across 8 context sizes; depth 0 would COST that).

## 2. Production change — size-gate the collective, compress the wire

vLLM dispatches TP all-reduce three ways (`custom_all_reduce.py`):

    payload <= VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE  -> b12x ONESHOT
    payload >= VLLM_PCIE_DMA_MIN_BYTES               -> b12x DMA
    everything else                                  -> PyNCCL

Our traffic is **bimodal with nothing in between**: decode is 1-4 rows (12-48
KiB), prefill is always 8192 rows (96 MiB) regardless of prompt length — a long
prompt just means *more* chunks, not bigger ones.

**The historical batch-1 decode collapse to ~40 tok/s was the ONESHOT path, not
DMA.** A 12 KiB decode payload sits under the image's 64 KB oneshot limit, so DMA
never engaged there at all. Disabling the whole PCIe path to fix decode also
discarded the prefill DMA win for a year. Gating by size keeps both.

New production defaults in `vllm-v20-r20.sh`:

    VLLM_ENABLE_PCIE_ALLREDUCE=1
    PCIE_ONESHOT_MAX=0  PCIE_ONESHOT_FUSED_MAX=0   (oneshot OFF -> decode on NCCL)
    GPU_MEMORY_UTILIZATION=0.93
    DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=49152
    F8_DMA=i8_ring

The two oneshot limits are hardcoded at `glm52-pcie-runtime-env.sh:14-15`, so a
`docker -e` cannot reach them — patch 4 makes them env-overridable.

## 3. Warm results (medians of 2 warm runs)

| ctx | baseline PyNCCL | new config | delta |
|---|---|---|---|
| 8k   | 4,026 | 5,334 | **+32.5%** |
| 64k  | 4,184 | 5,676 | **+35.6%** |
| 128k | 4,168 | 5,613 | **+34.7%** |
| 256k | 4,053 | 5,366 | **+32.4%** |
| 512k | 3,774 | 4,875 | **+29.2%** |

| conc | baseline | new | delta |
|---|---|---|---|
| 1  | 109.6 | 105.0 | -4.2% |
| 4  | 308.8 | 301.4 | -2.4% |
| 8  | 430.4 | 431.2 | +0.2% |
| 16 | 627.9 | 621.6 | -1.0% |

Uniform +29..36% across every context is exactly what the mechanism predicts.
For long-context coding the trade is ~35:1 in time: a 256k prefill goes 63s ->
48s; a 1000-token reply loses 0.4s. The c1 decode cost is likely per-call
dispatch overhead from having the b12x path active (decode still routes to
PyNCCL in both configs).

**`F8_DMA=i8_ring` carries almost all of it.** DMA with `F8_DMA=0` measured only
**+2.7%** mean. Compression halves the bytes crossing PCIe on a PCIe-bound box,
which dominates the DMA-vs-NCCL algorithmic difference. Do NOT "play it safe"
with `F8_DMA=0` — that keeps the quality cost and discards the speed.

Note `F8_DMA` was a **no-op for us before this change**: `custom_all_reduce.py`
reads `VLLM_PCIE_DMA_FP8` only inside the DMA branch, which never executed while
`VLLM_ENABLE_PCIE_ALLREDUCE=0`.

## 4. Measuring quality without a BF16 reference — `measure-kld.sh`

The published KLD table is vs BF16, which we cannot run (~1.5 TB). But
`F8_DMA=0` and `i8_ring` are **identical weights and quantization** — only the
wire format differs — so the PyNCCL config is a valid LOCAL reference and the
delta is exactly the cost of the change. Uses `prompt_logprobs=20` over ~8k-token
prompts (**they must be >=2048 tokens or the DMA path never engages and the
harness reports a spurious zero**).

vs the PyNCCL baseline, 27,495 token positions:

| config | mean KL | top-1 agreement | **confident** flips (>1 nat) |
|---|---|---|---|
| DMA, no compression | 0.009878 | 95.006% | **13** (0.047%) |
| DMA + `i8_ring`     | 0.011147 | 94.315% | **15** (0.055%) |
| DMA + `ring`        | 0.015367 | 93.850% | **33** (0.120%) |

Raw flip counts (~6%) are **misleading**: median reference margin at a flipped
position is 0.125 nats (a coin toss) vs 3.750 nats where the choice held, and
~36% of flips are outright near-ties. Weighted by confidence the change is
0.055%. `i8_ring` adds ~2 confident flips over uncompressed DMA — inside Poisson
noise — while carrying all the speed. **`ring` is dominated** (33 vs 15 at
identical speed): do not use it. timricese's hedged guess that `i8_ring` is the
gentler variant is confirmed.

**Caveat — generation diverges.** Greedy outputs vs the old config split within
6-41 tokens and share only 5-22% of a 256-token sequence, consistent with the
per-token flip rate compounding autoregressively. Each config is self-consistent
going forward, but nothing captured before the cutover reproduces. Divergence is
not degradation (a flip at a near-tie is arbitrary), but we have **no task-level
eval** on this box and cannot claim quality is unchanged.

## 5. Rejected, with numbers

- **DCP4** — -1.2..-3.9% prefill, **-5.4..-10.9% decode** (worst at c16, exactly
  where subagent fan-out lives) for +92% KV capacity (1,912,064 vs 996,480). Not
  worth it: LMCache L1 already backstops KV pressure from host RAM. Confirms the
  Phase 13 DCP2 choice, this time warm.
- **`MAX_BATCHED_TOKENS=16384`** — probe showed the DMA gain plateaus at ~20.5%
  past 8192; only +2.4pp for real VRAM cost.
- **`SPARKINFER_INDEXER_TWO_LEVEL_FOLD=off`** — +30.5% vs +32.9% mean; folding is
  neutral-to-slightly-positive. Keep `auto`.
- **`/dev/shm` hugepages** — premise does not apply. Our L1 is anonymous mmap
  already backed by THP (`AnonHugePages: 841 GB`, LMCache RSS 807 GB); `/dev/shm`
  holds 1.25 GB. `VmLck: 0` — the L1 is resident, not pinned.
- **native `OffloadingConnector`** — never booted, 3 attempts. Needs
  `expandable_segments:False`; that costs allocator efficiency and OOMs when
  stacked with the b12x path at working GMU. **Not a drop-in fallback if LMCache
  misbehaves** — it needs its own memory tuning.

## 6. Traps found in our own tooling

- **`LMCACHE_MODE=off` silently disables the allocator fix.** The wrapper forces
  `expandable_segments:False` at lines 143-150, but with LMCache off it
  `exec "$@"` at **line 13** and never gets there, so `serve-glm52-v16.sh:51`'s
  single-dash `${VAR-default}` reasserts `expandable_segments:True`. The launcher
  now has a conditional `PYTORCH_CUDA_ALLOC_CONF` passthrough.
- **LMCache MP registration races a hardcoded 60 s timeout.** At TP8 the server
  registers workers serially at ~8.6 s each; the 8th finished at **62.4 s** and
  the client had already given up. The wrapper is byte-identical on r11 and r20,
  so **every boot has been winning this by ~2 s**. Patch 3 makes
  `lmcache.mp.mq_timeout` env-overridable; we set 600.
- **Never point `CACHE_ROOT` at a cold NFS directory.** An empty JIT cache makes
  all 8 workers Triton-compile at once and NFS returned
  `OSError: [Errno 121] Remote I/O error`, taking the boot down.
- Bench-driver gates needed fixing: the KV ceiling now scales with DCP (a healthy
  DCP4 boot was aborted as an "OOM signature"), and the free-VRAM floor accepts
  `ALLOW_WARM_VRAM=1` (PyTorch's caching allocator retains blocks, so a warm
  re-bench reads ~1 GiB where boot showed ~3.6 GiB).
- `GPU_MEMORY_UTILIZATION=0.90` is **below the floor**: 917,504 tokens needs
  23.38 GiB of KV and 0.90 leaves 22.66 GiB. Engine refuses to start.
