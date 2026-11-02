# fused_moe_grouped_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Fused expert execution and compute/communication overlap.

2 AITER implementation references; 0 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Fused MoE compute and communication — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Fused MoE compute and communication — AITER flydsl](backend/flydsl_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: fused_moe_grouped_gemm — overview
kind: operator_overview
operator: fused_moe_grouped_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/tuned_fmoe.csv
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
```

</details>

# fused_moe_grouped_gemm  (the fused MoE up/gate/down grouped-GEMM mega-kernel)

## TL;DR
This is the compute heart of an MoE layer: the two grouped GEMMs (stage-1 gate+up, stage-2 down) over the
sorted per-expert token tiles, with activation and routed-weight combine fused in. On AMD it is
**`aiter.fused_moe`** — DB-driven, auto-selecting bf16 asm / fp8 block-scale / int8 / A4W4 by quant method.
The single most important fact: it is a **grouped GEMM** (many small per-expert GEMMs batched by the
align&sort permutation), so the lever is **fitting per-expert M-tiles to the 304 CUs** (the same
`moe_align_block_size` padding) and choosing the **quant path** (fp8 block-scale unlocks ~3× over an
unfused stack). This is the MoE specialization of [grouped_gemm_moe](../grouped_gemm_moe/README.md) and [scaled_quant_gemm](../scaled_quant_gemm/README.md).

## Math contract
Per expert `e`, over its assigned token rows `X_e [m_e, H]`:
1. **stage-1 (g1u1)**: `G = X_e · W_gate_eᵀ`, `U = X_e · W_up_eᵀ`; `A = act(G) ⊙ U` (SwiGLU). `w1` is
   `[E, 2·inter, H]` (gate+up fused). dtype: bf16 in / fp32 acc / bf16 (or fp8) intermediate.
2. **stage-2 (down)**: `Y_e = A · W_down_eᵀ`; `w2` is `[E, H, inter]`. Routed weight multiplied here
   (`MulRoutedWeight1`) or in stage-1 (`doweight_stage1`).
- The "grouped" part: tokens are pre-sorted (align&sort) so each expert's rows are a **contiguous,
  BLOCK_M-padded** run → one kernel does all experts. Only the **M-axis is grouped**; N (=inter/H) and K
  (=H/inter) are fixed per stage (cf. DeepGEMM's M-grouped design).

## Shape regimes
- **prefill (large M total)**: per-expert `m_e` can be large; compute-bound grouped GEMM → tile to fill 304
  CUs. DeepSeek-V3: H=7168, inter=2048, E=256, top-8.
- **decode (small M total)**: each expert sees few tokens (often `m_e < BLOCK_M`) → **padding waste**
  dominates and the GEMM is skinny/memory-bound. Masked grouped GEMM (DeepGEMM-style) avoids launching
  empty-expert work; CUDA-graph-friendly because the CPU doesn't know per-expert counts.

## Where it matters (Amdahl)
On an MoE model the grouped GEMM is the **largest GPU-time term of the MoE layer** (the experts are the
bulk of the parameters). AMD reports fused MoE up to **3×** vs an unfused stack, and block-scale fp8 fused
MoE is the headline MI300X MoE optimization. A 1.2× here moves e2e materially on MoE-heavy models.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live `fused_moe`, DB-driven) | [backends/aiter.md](backend/aiter.md) |
| ck | 🟢 sota (stage-2 `moe_ck2stages_*`, block-scale) | [backends/ck.md](backend/ck.md) |
| hip | 🟢 (stage-1 hand-tuned asm `fmoe_stage1_*`; editable) | [backends/hip.md](backend/hip.md) |
| triton | 🟡 competitive (vLLM/sglang Triton fused-MoE; per-shape JSON tuned) | [backends/triton.md](backend/triton.md) |
| flydsl | 🟢 sota (fp4 a4w4 2-stage + fused act/requant, CDNA4) | [backends/flydsl.md](backend/flydsl.md) |

## Fusion neighbors
- **act_and_mul (SwiGLU) fused into stage-1** (`g1u1`); **routed-weight multiply** into stage-1/stage-2
  epilogue; **fp8 quant/dequant** in the epilogue ([scaled_quant_gemm](../scaled_quant_gemm/README.md)); **shared-expert** folded in
  ([shared_expert_fusion](../shared_expert_fusion/README.md)). Consumes [moe_routing_topk](../moe_routing_topk/README.md) + [moe_dispatch_combine](../moe_dispatch_combine/README.md) output. See
  [fusion.md](fusion.md).

## Numerics
fp32 accumulate; fp8/int8/fp4 block-scale introduces quant error (DB rows carry `err1`/`err2`, e.g.
stage-2 ~2.3%). fnuz vs OCP fp8 on gfx942 vs gfx950. Validate **end-to-end** accuracy, not just kernel
tolerance. See [numerics.md](numerics.md).

## How to bench
aiter MoE tuner over `untuned_fmoe.csv`; isolated per-shape grouped-GEMM timing; e2e MoE model tok/s with a
tuned `tuned_fmoe.csv` deployed. Oracle = greedy/temp=0 + a small eval for quant paths.

## Sources
- aiter `fused_moe` (two-stage grouped GEMM, quant routing, kernel names): `ROCm/aiter@a6bb49937:aiter/fused_moe.py`, `aiter/configs/tuned_fmoe.csv`.
- 3× fused MoE / block-scale (AMD-reported): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- M-grouped (contiguous/masked) design: https://github.com/deepseek-ai/DeepGEMM (NVIDIA ref for the layout concept).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/fused_moe_grouped_gemm/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
