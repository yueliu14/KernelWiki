---
title: shared_expert_fusion — fusion
kind: technique
operator: shared_expert_fusion
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe_dp_shared_expert.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# shared_expert_fusion — fusion

This operator **is** a fusion — folding the always-on shared MLP into the sparse routed MoE. The levers are
*how deep* the fold goes and how it overlaps.

## Fusion depth ladder (shallow → deep)
| level | what's merged | where | payoff |
|---|---|---|---|
| 0 (none) | separate shared Linear + activation + residual add | baseline | — |
| 1 overlap | shared dense GEMM on a **separate stream/HW queue**, atomic-add into routed result | `GPU_MAX_HW_QUEUES=2` + aiter atomic-add | hides shared under routed compute |
| 2 fused kernel | shared stage-1/stage-2 run in the **same fused-MoE pipeline**, atomic-add | `fused_moe_dp_share_expert` | one launch, MFMA stays busy across both |
| 3 fused dispatch (EP) | shared injected as a **synthetic routed expert** (weight 1) → one dispatch for shared+routed | Wide-EP `grouped_topk` trick | single fused dispatch ([moe_dispatch_combine](../moe_dispatch_combine/README.md)) |

## The atomic-add seam
aiter passes the **no-shared MoE result buffer** into `fused_moe_dp_share_expert`; the shared output is
**atomic-added** onto it. This is the fusion glue — the shared and routed parts can be computed in any order
/ concurrently and still sum correctly (modulo bf16 atomic-add order, [numerics.md](numerics.md)).

## EP: shared-as-synthetic-routed-expert
Under Wide-EP, the shared expert is given top-k slots via `grouped_topk` so it dispatches with the routed
experts — **one** all-to-all carries both. This is why `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is
**incompatible with MoRI**: under MoRI the fusion is done MoRI-side (the synthetic-expert injection), not
via the vLLM op. Pick one.

## Fusion neighbors
- builds on [fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md) (the routed grouped GEMM the shared expert joins).
- shared routing folded into [moe_routing_topk](../moe_routing_topk/README.md) (the gate kernel can emit the shared slot).
- shared dispatch folded into [moe_dispatch_combine](../moe_dispatch_combine/README.md) (EP, synthetic expert).
- shared expert can be fp8 block-scale → shares the dequant epilogue ([scaled_quant_gemm](../scaled_quant_gemm/README.md)).

## What is NOT yet fused
- A **single kernel** that computes shared + routed + combine with no global round-trip of either
  intermediate (the north-star single-kernel MoE) — not on AMD yet.
- Shared expert keeping its `[M,inter]` intermediate in LDS across stage-1→stage-2 (materialized today).

## Cross-links
[fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md) · [moe_routing_topk](../moe_routing_topk/README.md) · [moe_dispatch_combine](../moe_dispatch_combine/README.md) · [scaled_quant_gemm](../scaled_quant_gemm/README.md) ·
[`backends/aiter/fmoe.md`](../grouped_gemm_moe/backend/aiter.md) ·
[`backends/mori_rccl/mori_ep.md`](../../general/libraries/mori_rccl/mori_ep.md).

## Sources
- atomic-add fusion glue, token-range, EP: `ROCm/aiter@a6bb49937:aiter/fused_moe_dp_shared_expert.py`.
- shared-as-routed + MoRI incompatibility: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/shared_expert_fusion/fusion.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
