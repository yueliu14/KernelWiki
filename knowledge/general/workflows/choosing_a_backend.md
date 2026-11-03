---
title: Choosing a backend — per-operator-family recommendation guide
kind: workflow
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, training, both]
status: sota
updated: 2026-06-08
sources:
  - GEAK/perf_knowledge/index/sota_matrix.md
  - GEAK/perf_knowledge/index/decision_trees.md
  - GEAK/e2e_workflow/roles/op_benchmarker.md
---

# Choosing a backend

## TL;DR
This is the one-page prior that summarizes the per-operator `overview` SOTA tables (and
[`../index/sota_matrix.md`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/index/sota_matrix.md)) into a "what to try first" guide.
These are **priors for ORDERING**, never a verdict — the operator's SOTA cards + a
same-session A/B on the immutable oracle are the judge
([`optimize_single_kernel.md`](optimize_single_kernel.md)). Past results never justify
*not* trying; they only order the attempts.

## GEMM family
| operator | try first → then | notes |
|---|---|---|
| **dense_gemm** | aiter DB tune (engages live path) → FlyDSL author → hipBLASLt → asm | Triton usually loses to tuned hipBLASLt; need to beat the library → hand-asm or FlyDSL. [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md) |
| **scaled_quant_gemm** (fp8/int8) | aiter/CK scaled-GEMM → FlyDSL author (preferred for fp8/A4W4/mxfp4) | gfx942 fp8 FNUZ; gfx950 block-scaled MXFP. accuracy-gate. [`../operators/scaled_quant_gemm/`](../../operator/scaled_quant_gemm/README.md) |
| **skinny_gemv_decode** (M≤256) | aiter skinny/wvSplitK → split-K Triton/FlyDSL (fill 304 CUs) → fuse epilogue | [`../operators/skinny_gemv_decode/`](../../operator/skinny_gemv_decode/README.md) |
| **grouped_gemm_moe** | aiter fused_moe / FlyDSL / CK / asm | auto-selects quant kernel. [`../operators/grouped_gemm_moe/`](../../operator/grouped_gemm_moe/README.md) |
| **batched_gemm / splitk_streamk_gemm** | aiter / hipBLASLt → split-K Triton/FlyDSL | [`../operators/splitk_streamk_gemm/`](../../operator/splitk_streamk_gemm/README.md) |

## Attention family
| operator | try first → then | notes |
|---|---|---|
| **attention_prefill_fmha** | CK-Tile FMHA (fastest general) → AITER FA → Triton FA (editable) → TileLang (≈1.5× Triton on CDNA3) → asm | `VLLM_USE_TRITON_FLASH_ATTN=0` selects CK. FlashInfer N/A on AMD. [`../operators/attention_prefill_fmha/`](../../operator/attention_prefill_fmha/README.md) |
| **attention_decode_paged** | AITER asm decode (`pa_fwd_asm`) → CK → Triton | memory-bound; shuffled KV layout. [`../operators/attention_decode_paged/`](../../operator/attention_decode_paged/README.md) |
| **mla_attention** | AITER MLA (asm decode) → TileLang (≈95% of asm) → Triton MLA | gfx942: AITER_TRITON_MLA ~2-3% better; gfx950: AITER_MLA. [`attention_backend_selection.md`](attention_backend_selection.md) |
| **gqa_mqa / sliding_window / chunked_prefill** | AITER FA (3-path) → Triton unified | unsupported KV head-dim → avoid ROCM_ATTN. [`../operators/gqa_mqa_attention/`](../../operator/gqa_mqa_attention/README.md) |
| **linear_attention_gated_delta** | **Triton** (SOTA, editable) → HIP | the gated-delta Qwen3.5 path; ensure varlen serving survives. [`../operators/linear_attention_gated_delta/`](../../operator/linear_attention_gated_delta/README.md) |

## Editable custom kernels (norms / rope / act / elementwise)
| operator family | try first | notes |
|---|---|---|
| **rmsnorm / fused_add_rmsnorm / layernorm** | aiter fused → Triton (editable) | fuse residual-add. small Amdahl mass each → **stack**. [`../operators/rmsnorm/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/rmsnorm/README.md) |
| **rope / mrope** | aiter fused rope+kvcache → Triton | [`../operators/rope/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/rope/README.md) |
| **act_and_mul (silu/gelu)** | aiter → Triton; fuse into GEMM epilogue | [`../operators/act_and_mul_silu_gelu/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/act_and_mul_silu_gelu/README.md), [`../operators/gemm_epilogue_fused/`](../../operator/gemm_epilogue_fused/README.md) |
| **softmax / sampling / argmax_topk** | Triton (editable) → aiter | [`../operators/sampling_topk_topp/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/sampling_topk_topp/README.md) |
| **quant/dequant (fp8/fp4/int8), kv_cache_quant** | aiter scaled-quant → fused_norm_quant → Triton | accuracy-gate. [`../quantization/`](../quantization/README.md) |

## Collectives / MoE dispatch
| operator | try first | notes |
|---|---|---|
| **allreduce** | QuickReduce (up to 3× on MI300X, no code change) → RCCL/aiter | [`../operators/allreduce/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/allreduce/README.md), [`../backends/mori_rccl/`](../libraries/mori_rccl/overview.md) |
| **moe_dispatch_combine / all_to_all** | aiter / MoRI | [`../operators/moe_dispatch_combine/`](../../operator/moe_dispatch_combine/README.md) |
| **allgather / reduce_scatter** | RCCL/aiter → MoRI | [`../operators/allgather/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/allgather/README.md) |

## The cross-family heuristics (from the decision trees)
- **Serving caller consumes an AITER database?** → consider tuning that database or changing the
  active implementation, then verify engagement and end-to-end behavior. TunableOp and library
  settings apply only if the selected call path consults them; inspect that path first.
- **Raw torch/F.linear (no aiter)?** → hipBLASLt offline tune or TunableOp.
- **Beat the library matmul itself?** → hand-asm or FlyDSL; pure Triton usually loses.
- **Editable custom kernels** → Triton first (fastest to iterate).
- **Small Amdahl mass (<0.5% each)?** → STACK the cluster, gate combined.
- **Quantized GEMM** → gfx942: fp8 FNUZ via aiter/CK; gfx950: MXFP8/MXFP6/MXFP4 block-scaled
  (FP6 at FP4 rate). Always accuracy-gate, never byte-parity.

## gfx942 vs gfx950 quick note
- **gfx942 (CDNA3, MI300X/MI325X)**: fp8 FNUZ; AITER_TRITON_MLA slightly ahead for MLA.
- **gfx950 (CDNA4, MI350X)**: block-scaled MXFP8/MXFP6/MXFP4 MFMA (FP6 at FP4 rate);
  AITER_MLA (asm prefill) leads. See [`../hardware/cdna3_mi300/`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/hardware/cdna3_mi300),
  [`../hardware/cdna4_mi350/`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/hardware/cdna4_mi350).

## How to use this guide
1. Find the operator family above → get the ordered candidate list.
2. Run the Tier-A bake-off ([`optimize_single_kernel.md`](optimize_single_kernel.md)) — the
   A/B on the immutable oracle is the judge.
3. For a head op, also tune AND author; let the e2e gate pick best of {tuned, authored}.

## Sources
- Status badges + per-family rankings: `GEAK/perf_knowledge/index/sota_matrix.md` (each cell links the evidence card).
- Routing heuristics: `GEAK/perf_knowledge/index/decision_trees.md`.
- "try every backend, order by prior, never skip authoring": `GEAK/e2e_workflow/roles/op_benchmarker.md`.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/workflows/choosing_a_backend.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
