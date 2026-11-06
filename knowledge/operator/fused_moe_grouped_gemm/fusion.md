---
title: fused_moe_grouped_gemm — fusion
kind: technique
operator: fused_moe_grouped_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/tuned_fmoe.csv
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---
# fused_moe_grouped_gemm — fusion

The whole point of this operator is that it is **already a mega-fusion**: sort → grouped GEMM (gate+up) →
activation → grouped GEMM (down) → weighted combine in one `fused_moe` pipeline. The remaining levers are
*what else* you fold into the two GEMM stages.

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **gate+up (g1u1)** | two stage-1 GEMMs into one | `use_g1u1`, kernel `..._g1u1_...` | one GEMM, shared X load |
| **SwiGLU act_and_mul → stage-1 epilogue** | `act(G)⊙U` fused after gate+up | stage-1 asm kernel | no separate activation pass over `[M,inter]` |
| **routed-weight multiply → GEMM epilogue** | `topk_weights` multiply in stage-1 (`doweight_stage1`) or stage-2 (`MulRoutedWeight1`) | DB-selected | removes a weighted-combine kernel |
| **fp8 quant/dequant → epilogue** | dequant scales applied in the GEMM epilogue (block-scale) | CK `moe_ck2stages_*` | no separate dequant pass; [scaled_quant_gemm](../scaled_quant_gemm/README.md) |
| **token sort fused with histogram** | align&sort builds the grouped layout in one kernel | `moe_sorting` | 7× align&sort ([moe_routing_topk](../moe_routing_topk/README.md)) |
| **shared-expert into the grouped GEMM** | shared MLP run as synthetic routed experts in the same kernel | `fused_moe_dp_shared_expert` ([shared_expert_fusion](../shared_expert_fusion/README.md)) | no separate shared Linear + residual add |

## The two-stage kernel anatomy (real on-box names)
- stage-1 fp8: `_ZN5aiter48fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf3E` — hand-tuned **asm**, g1u1.
- stage-2 fp8: `moe_ck2stages_gemm2_256x64x128x256_1x4_MulABScaleExpertWeight_v3_Nswizzle0_Quant2_
  MulRoutedWeight1_F8_F8_B16` — **CK 2-stage**, `MulRoutedWeight1` folds the router weight, `Quant2`
  applies scales, output bf16.
So stage-1 = asm, stage-2 = CK, and the activation/weight/quant are **already** epilogue-fused.

## The dispatch ↔ GEMM seam (EP)
Under EP the grouped GEMM consumes MoRI-EP's **3D packed layout** (`packed_recv_x`, `packed_recv_count`,
…) directly — no re-materialization. This is the [moe_dispatch_combine](../moe_dispatch_combine/README.md) hand-off
(`ENABLE_STANDARD_MOE_ADAPT=ON`, `MoriAll2AllManager`).

## What is NOT yet fused on AMD
- **combine reduction → down-proj epilogue** (single-kernel MoE, FlashDMoE-style) — the north-star; MoRI-EP
  zero-copy buffers approach it but no single fused combine+GEMM kernel ships on AMD ([moe_dispatch_combine](../moe_dispatch_combine/README.md)).
- **stage-1 → stage-2 without a global round-trip of the intermediate** — the `[M,inter]` activation is
  materialized between stages (it's large); keeping it in LDS/registers across both GEMMs is not done.
- **sort → stage-1** — removing the sort node at decode did not pay off in the measured shapes (below).

## What recorded attempts found (mxfp4 fused MoE, decode)
Recorded runs on an mxfp4 fused MoE (257 experts, top-k 9, 16-row blocks, decode M=6..42) tried to fold more
into the two stages. Of 16 directions that followed this page, a quarter beat their parent, and on average they
trailed same-round siblings that did not use it. Size the work a fold removes before building it.
- **Dropping the sort node did not pay for itself.**
  - At M=18, a sort-free identity route cut graph nodes from 3 to 2 but slowed that decode case by about 10%.
    In the engineer's eager per-kernel split, gemm1 grew 8.1 µs and gemm2 5.4 µs, while only 4.7 µs of sort was saved.
  - At M=6/12, a leader-dedup variant was correct but its speed stayed within noise.
- **Size the folded work first.** The engineer's microbenchmark put the `moe_buf` zero-fill inside the sort at
  0.1–0.26 µs per sort node, capping the saving under ~0.5% decode-weighted; the engineer closed it as a standalone lever.
- **Warming w2 from extra gemm1 CTAs lost at bs1 and was flat at bs2.** At bs1, w2 plus its scales was about 44 MB,
  more than the 32 MB of L2; even a 2.6 MB scale-only warm cost 1.3 µs at bs1.
- **Cache policy:** in a separate fused MoE kernel, caching (instead of non-temporal) shared-expert gate/up weight
  loads and down-proj e8m0 scale loads was associated with +1.29% / +1.22% (forward / reversed A/B); caching the
  w13 expert weights instead lost 1.9% / 2.2%.

## Cross-links
[moe_routing_topk](../moe_routing_topk/README.md) (feeds sort) · [moe_dispatch_combine](../moe_dispatch_combine/README.md) (EP feed) · [shared_expert_fusion](../shared_expert_fusion/README.md) ·
[scaled_quant_gemm](../scaled_quant_gemm/README.md) · [grouped_gemm_moe](../grouped_gemm_moe/README.md) · [`backends/aiter/fmoe.md`](../grouped_gemm_moe/backend/aiter.md).

## Sources
- on-box kernel names + fusion flags: `ROCm/aiter@a6bb49937:aiter/fused_moe.py`, `aiter/configs/tuned_fmoe.csv`.
- shared-expert fusion + EP co-design: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/fused_moe_grouped_gemm/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
