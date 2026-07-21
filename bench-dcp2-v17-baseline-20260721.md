# GLM-5.2 DCP2 baseline — for DCP8/v19 comparison  (2026-07-21)

Server: PRODUCTION as-is — v17 `fathomless-firmament-v17-...20260714`, TP8/**DCP2**,
MTP3, A16, fp8 KV, **LMCache** (768GB), port 8443, plain HTTP.
KV pool: 1,072,128 tokens. Bench: llm-inference-bench @86cf05c, 15s/cell, idle system.
Raw JSON: `bench-dcp2-v17-baseline-20260721.json`

## Decode — aggregate tok/s (ctx0)
| c1 | c4 | c8 | c16 | c32 |
|---:|---:|---:|---:|---:|
| 86.9 | 278.6 | 435.1 | **661.7** | 589.2 |
(c32<c16: MAX_NUM_SEQS=16 caps running seqs — not a cliff. Peak = c16.)

## Prefill — client prompt_tokens/TTFT
| 8k | 64k | 128k |
|---:|---:|---:|
| 3,019 | 3,214 | 3,175 |

## What DCP8/v19 must beat (the comparison thesis)
- PREFILL is the CKV-gather win: v18/v19 doc showed DCP8 64k **+85%** vs v17.
  So expect DCP8 64k prefill to jump from ~3,214 toward ~4,400+ tok/s.
- DECODE at DCP8 historically LOSES vs DCP2 (~66 vs ~87 c1); v19's sparse-gather
  decode work is meant to narrow that — watch c1.
- KV pool: DCP8 ≈ 4.4M tokens vs 1.07M here (the capacity prize).
- CAVEAT: DCP8 config has **no LMCache** (warm-restore lost) — not in this bench,
  but a real production tradeoff.
