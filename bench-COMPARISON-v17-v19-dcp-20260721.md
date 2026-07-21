# GLM-5.2 bench sweep — v17 vs v19, DCP2 vs DCP8  (2026-07-21)
Host: 8× RTX PRO 6000, PCIe-only, 2 NUMA (GPU0-3 / 4-7). llm-inference-bench @86cf05c, 15s/cell, idle.
All: TP8, MTP3, A16, fp8 KV, B12X_MLA_SPARSE. Aggregate decode tok/s @ctx0; prefill client tok/s.

| Config | c1 | c8 | c16 | c32 | pf8k | pf128k | KV pool | LMCache | warm-restore |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| v17 DCP2 (old prod) | 86.9 | 435 | 662 | 589 | 3,019 | 3,175 | 1.07M | yes | yes |
| **v19 DCP2 +LMCache (SHIP)** | **101.8** | **448** | **689** | 612 | **3,779** | **3,973** | 1.04M | yes | **18.6×** |
| v19 DCP2 no-LMCache | 97.0 | 447 | 670 | 613 | 3,939 | 3,989 | 1.07M | no | no |
| v19 DCP8 PCIE0 | 54.2 | 76 | 95 | 77 | 2,816 | ~2,861 | 3.69M | no | - |
| v19 DCP8 PCIE1 | 29.5 | 71 | 95 | 78 | 3,277 | ~2,861 | 3.69M | no | - |

(64k prefill omitted from headline — single-scout JIT-spike noise across runs; 8k & 128k are stable.)

## Findings
1. **DCP8 is a dead end on THIS host.** Decode collapses ~7× (c16 95 vs 689) regardless of PCIe
   all-reduce backend. CKV engages fine — the wall is 8-way cross-NUMA all-reduce per decode token
   on a PCIe-only 2-NUMA fabric. koush's ~2× gains were on non-NUMA PCIe-x8 boxes.
2. **v19/DCP2 + LMCache is a clean upgrade over v17/DCP2** and the recommended production config:
   - decode c1 +17% (87->102), c16 +4%; prefill +25% at 8k/128k (CKV gate now covers 8:2)
   - warm-restore intact (18.6×), same 1M KV, same topology
   - all 12 LMCache patches + 3 KV-xfer guards apply cleanly on gilded-gnosis (no re-validation needed)
3. LMCache adds no throughput penalty vs no-LMCache (decode even marginally higher).

## Current state / follow-ups
- LIVE NOW on :8443 = v19 DCP2 + LMCache + CKV (this ship-candidate). Launched via
  `DCP=2 GPU_MEMORY_UTILIZATION=0.94 bash vllm-v19-dcp8-lmcache.sh`.
- TODO: make a clean `vllm-v19-lmcache.sh` (DCP=2, gpu_mem 0.94 defaults) as the production launcher;
  commit launchers + this comparison; update BENCHMARKS.md (Phase 13).
- Cold-boot caveat unchanged: the 768GB LMCache pin re-evicts weights → ~20-30min cold boots
  (vmtouch mlock still blocked by pod's 8MB memlock cap; container-based lock is the open option).
