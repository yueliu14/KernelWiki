---
title: RMSNorm + fp8-quant fusion — the e2e win from quantizing the norm output in-kernel
kind: case_study
operator: fused_norm_quant
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/18466
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py
---

# RMSNorm + fp8 dynamic-quant fusion

This case is filed with scaled GEMM because it concerns preparation of the GEMM's quantized input
and its consumer scale contract. The original `fused_norm_quant` metadata identifies the supporting
operation described by the source; it does not add a standalone operator to the current catalog.

> The headline e2e number here is **vendor-reported** (SGLang issue #18466), labelled inline. We
> have not re-measured this fusion in the `e2e_workflow` eval dirs; the *mechanism* and the
> *gate* are the validated, transferable parts.

## Context
`fused_norm_quant` fuses **(fused-add) RMSNorm/LayerNorm + dynamic fp8/int8 quant** into one
kernel, so the norm output is written **already quantized**. Two wins from one fusion:
1. it **removes a whole standalone quant pass** (the quant is "free" while `y` is still in fp32
   registers — just a per-row/per-group max-reduction), and
2. it **halves the downstream GEMM input traffic** (qkv / up-gate / down read fp8 = ½ bytes, or
   fp4 = ¼) — and that downstream GEMM is the Amdahl head.

So although the norm itself is only **1–4% of GPU time**, the fusion's real value lands on the
**GEMM that consumes the quantized output**. Operator overview:
[`../../operators/fused_norm_quant/overview.md`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/README.md).
This is the [`../../operators/rmsnorm/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/rmsnorm/README.md) +
[`../../operators/quant_dequant_fp8/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_dequant_fp8/README.md) seam.

## Baseline
- **Unfused path:** RMSNorm writes bf16 → a separate dynamic-quant kernel reads bf16, writes fp8
  + scale → the GEMM reads fp8. Two passes over the activation, plus the norm output written at
  full bf16 width.
- Reference for the gate: bf16-input GEMM (or the unfused norm→quant→GEMM chain), same model.

## What works (the fusion + the kernel entrypoints)
- **per-token dynamic fp8:** `scale[m] = max(|y[m,:]|)/fp8_max`, `y_q = round(y/scale)` — fnuz fp8
  on gfx942. aiter: `rmsnorm2d_fwd_with_dynamicquant` / `_with_add_dynamicquant`.
- **group fp8** (`gated_rmsnorm_fp8_group_quant`, HIP): scale per group of 128 (head_dim=128).
- **smoothquant int8:** per-channel smoothing × per-token scale (`_smoothquant`).
- **mxfp4:** block-32 e8m0 scale (gfx950).
- Norm math in fp32, scale in fp32, RNE rounding. Backends: aiter (CK/asm + HIP group-quant +
  flydsl), Triton (`_quant_rms_norm_kernel` family), HIP.

## What didn't / the traps (kept honestly)
- **The norm alone is tiny (1–4% GPU)** — fusing the *norm* for its own sake won't clear an e2e
  gate. The win only materializes because the **downstream GEMM** reads half the bytes. If the
  GEMM doesn't actually consume the quantized output (still bf16 GEMM), the fusion buys little.
- **fnuz fp8 off-by-2× trap on gfx942** — the group/per-token scale must match the consumer
  GEMM's dequant exactly, or you get silent magnitude error.
- **It's a task-accuracy decision, not byte parity** — gate on a small eval (gsm8k), not on
  bit-exactness. fp8 dynamic quant changes the numbers.
- This is the same fp8-accuracy caution that **rejected `--quantization fp8` on Qwen3.5-27B** —
  see [`../by_model/llama_fp8_serving.md`](../../../general/case_studies/by_model/llama_fp8_serving.md).

## Final result (numbers, vendor-reported)
| metric | value | source / label |
|---|---|---|
| RMSNorm + FP8 dynamic-quant fusion, e2e latency | **1–6%** | **vendor** — SGLang #18466 (Qwen3, MI300X) |
| same, throughput | **1–2%** | **vendor** — SGLang #18466 |

The fusion is the **sibling lever** to the `act_and_mul` output-quant on the MLP down-proj side,
and the QK-norm+RoPE+quant variant (`fused_qk_rmsnorm_group_quant`,
`fused_qk_norm_rope_cache_quant`) is the attention-entry form. See
[`../../operators/fused_norm_quant/fusion.md`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/fusion.md).

## Lessons
1. **Fuse the norm for the GEMM, not for the norm.** The 1–4% norm is not the prize; the halved
   GEMM input traffic is. Route the decision by where the quantized bytes get *consumed*.
2. **Match the scale granularity to the consumer GEMM's dequant** (per-token vs group-128) — a
   mismatch is a silent accuracy bug, worse on gfx942 fnuz fp8.
3. **Gate on task accuracy** (gsm8k), never byte parity — dynamic fp8 quant is lossy by design.
4. **The win is small and stacks** — like the editable-kernel cluster, treat it as a
   carry-forward lever that compounds with the GEMM-side fp8 path, not a standalone headliner.

## Cross-links
- Operator: [`../../operators/fused_norm_quant/overview.md`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/README.md) · numerics: [`../../operators/fused_norm_quant/numerics.md`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/numerics.md) · aiter card: [`../../operators/fused_norm_quant/backends/aiter.md`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/backend/aiter.md)
- Seam: [`../../operators/rmsnorm/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/rmsnorm/README.md) · [`../../operators/quant_dequant_fp8/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_dequant_fp8/README.md) · [`../../operators/fused_add_rmsnorm/`](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_add_rmsnorm/README.md)
- Fusion strategy: [`../../optimization/kernel_fusion_strategy.md`](../../../general/optimization/kernel_fusion_strategy.md)
- fp8 serving recipe: [`../by_model/llama_fp8_serving.md`](../../../general/case_studies/by_model/llama_fp8_serving.md) · quant: [`../../quantization/`](../../../general/quantization/README.md)
- The GEMM that consumes the output: [`gemm_aiter_db_tuning.md`](../../dense_gemm/case_studies/gemm_aiter_db_tuning.md) · [`../../operators/scaled_quant_gemm/`](../README.md)

## Sources
- 1–6% e2e latency / 1–2% throughput, RMSNorm+FP8 dynamic-quant fusion on Qwen3 MI300X (vendor): https://github.com/sgl-project/sglang/issues/18466.
- aiter norm+quant entrypoints: `/sgl-workspace/aiter/aiter/ops/rmsnorm.py`, `/sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py`.

<!-- MANIFEST: RMSNorm+fp8 dynamic-quant fusion — vendor 1–6% e2e latency / 1–2% throughput (sglang #18466, Qwen3 MI300X); win lands on halved downstream GEMM traffic, gated on task accuracy not byte parity. -->

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/case_studies/by_kernel/fused_norm_quant_win.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
