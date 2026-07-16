# GLM-5.2-NVFP4 on 8× RTX PRO 6000 — Setup & Findings (review doc)

Short, reviewable summary of what we ran, rough numbers, pitfalls, and dead ends.
Full blow-by-blow with per-config tables is in `BENCHMARKS.md`. Launchers:
`vllm.sh` (fast 256k), `vllm-v11-1m.sh` (1M context), `vllm-lmcache.sh` (KV reuse).

## Hardware / model
- 8× RTX PRO 6000 Blackwell (SM120), 96 GB, **PCIe, no NVLink, 2 NUMA nodes**.
  This k8s pod **blocks CUDA IPC/P2P between GPUs** — the single biggest source of
  bring-up pain.
- `lukealonso/GLM-5.2-NVFP4`: 744B-MoE/40B-active, 256 experts, **MLA + DeepSeek
  Sparse Attention (DSA)**, MTP draft layer, 1M max ctx, NVFP4 expert weights (467 GB).
- Images: `voipmonitor/vllm:dark-devotion-39ae3ed-...` (base) and
  `...glm52-v11-darkdevotion-vllma86f74e-...` (v11; adds per-layer DCP-LSE fix for DCP+MTP).

## TL;DR recommended configs
| Goal | Config | Launcher |
|---|---|---|
| General / agents (≤256k), **fastest** | DCP1, MTP k2, A16, max_num_seqs 32 | `vllm.sh` |
| **~1M context, highest perf** | **DCP2**, MTP k2, A16, gpu_mem 0.95 (full 1.00M KV) | `vllm-v11-1m.sh` w/ `DCP_SIZE=2` |
| Repeated long prompts (RAG/agent) | + LMCache (RAM L1); ~720k ctx ceiling at gpu_mem 0.86 | `vllm-lmcache.sh` |

**LMCache vs context-ceiling tradeoff:** LMCache reserves a GPU buffer + needs lower
`gpu_mem` (0.86) to avoid weight-load OOM, which shrinks the on-GPU KV pool. So
DCP2 fits **1.00M** ctx *without* LMCache (gpu_mem 0.95) but only **~720k** *with*
LMCache (gpu_mem 0.86, measured: 18.55 GiB KV → est. max len 727,680). Pick: full
1M cold-only (DCP2, no LMCache) **or** ~720k with ~150× warm-prefill reuse (DCP2 +
LMCache). For full 1M *and* LMCache you'd need DCP4 (more KV headroom, but slower).

## Rough performance (aggregate decode tok/s @ ctx0; prefill tok/s)
| config | ctx/req | decode c1 | c16 | c32 | prefill 8k | notes |
|---|---|---|---|---|---|---|
| DCP1 (vllm.sh) | 256k | 77 | **552** | 763 | **~2940** | fastest; MTP accept ~2.8 |
| **DCP2** | **1M** | 73 | **368** | – | **~2160** | best long-ctx; MTP accept ~2.3 |
| DCP4 | 1M+ | 51 | 267 | – | ~1620 | slowest; only if you need >1M KV |
- Single-stream ~77–89 tok/s. **DCP tax:** each DCP step doubles per-layer comm →
  DCP2 ≈ −33%, DCP4 ≈ −50% vs DCP1 on both decode AND prefill. Decode is
  **communication-bound** (cross-NUMA all-reduce over NCCL-SHM); GPUs run at
  ~100 W (not power-bound).
- **LMCache warm-prefill reuse: 88k-token prompt 57.1 s → 0.4 s = ~150×** on a
  cache hit (100% L1 RAM hit). Cold prefill unchanged.

## What moved performance (ranked)
1. **Cudagraph capture = `max_num_seqs × (1+spec)`** — up to **+390%**. The image
   default (16) only graphs ~4 MTP seqs; decode collapses at concurrency ≥8.
2. **A16 MoE decode (`B12X_MOE_FORCE_A16=1`)** + `use_index_cache` — **required for
   long-gen correctness** (default w4a4 → token-salad past ~3k generated tokens);
   also +6–17% batched decode, −7% prefill.
3. **MTP on, k=2** — +30–66% at DCP1; modest at DCP4 (accept drops with DCP).
4. **DCP2 over DCP4 for ~1M** — +40–57% decode AND prefill.
5. **max_num_seqs 16→32** — +10–21% throughput tail (short contexts only).
6. **LMCache** — ~150× prefill on *repeated* context.

## Required fixes for THIS host (non-obvious)
- `PYTHON_BIN=/opt/venv/bin/python` (image's serve script points at a missing venv).
- `unset NCCL_GRAPH_FILE; USE_NCCL_XML=0` (baked empty graph path crashes "-noxml" NCCL).
- **`NCCL_P2P_DISABLE=1`** (P2P over PCIe/2-NUMA → `invalid device ordinal`; NCCL
  runs over SHM). The b12x PCIe all-reduce also can't open IPC here → falls back to
  NCCL gracefully (which the repo says is faster for 8-GPU cross-socket anyway).
- Export the `VLLM_USE_B12X_*` + `VLLM_PCIE_ALLREDUCE_BACKEND=b12x` set, else vLLM
  uses its standard CUDA-IPC custom all-reduce → hard `invalid device ordinal`.
- **Tool-call patch** (`patch/glm4_moe_tool_parser.py`, mounted): the stock streaming
  parser drops **zero-argument** tool calls (name ends in `</tool_call>` with no
  `\n`/`<arg_key>`) → agent clients get NO tool calls. Non-streaming was fine, so
  curl tests passed but opencode "thought then stopped." Fixed → opencode works, 10/10.

## Pitfalls / DO-NOT-TRY (saves hours)
- **bf16 KV cache** → instant 4-token breakdown on the b12x sparse path. **fp8 KV is
  required** (and is accuracy-fine for GLM-5.2 per KLD; the "fp8 garbles GLM-5"
  warning was older GLM-5 on plain SGLang).
- **`tool_choice:"required"`** → 500, xgrammar `grammar rejected </tool_call>`
  (parser is `supports_required_and_named=False`). Use `"auto"` (agents do).
- **`max_num_batched_tokens=16384` at gpu_mem 0.96** → OOM; and it barely helps
  prefill (GLM-5.2 prefill is bound by DSA indexer + A16 MoE, not chunk size).
- **DCP4+MTP "deadlock"** was a PREMATURE KILL: cold compile + a 3-phase cudagraph
  capture (PIECEWISE ~20 s/iter at 1M) ≈ 6 min of idle-looking GPU spin + "no
  broadcast block" warnings. **Be patient — long-ctx boots take ~15 min.**
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with LMCache** → crash. Don't.
- **LMCache L1 = 1 TB** → evicts the OS page cache holding the 467 GB weights →
  cold-NFS weight reload ~200 s/shard → **40-min boots**, and starves the lmcache
  server (>100 s queue waits). **Use L1 = 256 GB** — that's already **~5–6M tokens**
  (~5–6 full 1M contexts; MLA KV ≈ 44 KB/token) and keeps boots fast.
- LMCache + DCP needs `chunk_size = block_size × dcp_size` (DCP4→512, DCP2→256) and
  the 2 DCP-aware patches, else only DCP rank 0 stores → "1/N prefix hits".
- **LMCache lowers the context ceiling**: its GPU buffer + the lower gpu_mem (0.86)
  it needs shrink the KV pool. Set `MAX_MODEL_LEN` to the engine's reported
  "estimated maximum model length" (DCP2+LMCache @ 0.86 ≈ 727k) or it ValueErrors
  on boot. This is a clean config error, not a crash — just lower max-model-len.
- Don't trust cross-host reference numbers; benchmark on-host (the repo's doc numbers
  are from a different machine).
- Dead ends others reported (we did not re-try): fp32 router, KV CPU-offload
  connector ("didn't help"), `VLLM_USE_B12X_SPARSE_INDEXER=0`, TRITON/FLASHINFER MLA
  (unsupported on SM120 — B12X_MLA_SPARSE is the only DSA backend).

## Correctness gating (do this before trusting throughput)
Run the **Estonia long-answer profile** first; only benchmark configs that pass:
`uv run --with requests python3 llm_decode_bench.py --port 8080 --model GLM-5.2 \
  --test-profile estonia --profile-concurrency 2 --profile-runs 3 --max-tokens 7000`
Expect `correct_rate 1.0`, `hit_max_tokens 0`. All shipped configs pass.

## Operational notes
- Cold start ~25–40 min (NFS weight read); warm restart ~5 min IF the page cache
  still holds the weights. NOTE (Phase 12): the 768 GB LMCache L1 pin **evicts**
  the 436 GB weight cache on every boot (weights are idle on-GPU → first LRU
  victims), so cold is the norm. Fix = `vmtouch -dl <blobs>` to mlock them (RAM,
  not disk); not yet applied.
- 2 TB host RAM page-caches the weights → keep restarts fast.
- One container `glm52` on port **8443** (plain HTTP, TLS off; 8080 is
  Cilium-Hubble-DPI'd and mangles TLS). Production image = v17 fathomless-firmament
  via `vllm-v14-lmcache.sh`.

## Determinism / LMCache correctness note
This stack is **not bit-deterministic at temp 0** (MoE expert routing + TP/DCP
atomic reductions + async scheduling + b12x kernels). Two identical requests give
coherent but slightly different outputs — normal for large MoE+TP serving.
Verified LMCache is NOT the cause: two consecutive cache-HIT runs (identical
restored KV, temp 0) also differ from each other, so the variation is in decode,
not the KV restore. **LMCache adds no correctness penalty** (hits are faithful,
20/20). Estonia on DCP2+LMCache = 5/6 — that's task difficulty × nondeterminism,
the same any config sees (earlier "3/3" were lucky 3-run samples). Don't expect
exact reproducibility from this stack regardless of LMCache.
