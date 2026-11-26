---
title: quantization — conceptual & strategy layer (index)
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp4_e2m1, fp6_e2m3, fp6_e3m2, mxfp4, mxfp6, int8, int4]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html
---

# quantization/ — the strategy layer

> **What this section is.** The *conceptual / strategy* layer of quantization for AMD MI GPUs:
> which format, which granularity, which scaling, which accuracy gate. It sits
> **above** the bit-exact deep numerics already written per-operator:
> - [operators/quant_dequant_fp8](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_dequant_fp8/README.md) — FP8 cast/dequant numerics (FNUZ vs OCP, 224 cap, amax/scale).
> - [operators/quant_fp4_mxfp](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_fp4_mxfp/README.md) — MXFP4/6 block-floating-point numerics (E8M0, group 32, shuffle).
> - [operators/quant_int8](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_int8/README.md), [operators/scaled_quant_gemm](../../operator/scaled_quant_gemm/README.md), [operators/fused_norm_quant](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/README.md),
>   [operators/kv_cache_quant](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/kv_cache_quant/README.md) — the kernels that consume what these strategies produce.
>
> **Do not duplicate** the per-operator numerics here. This layer answers *"what should I do and why,"*
> and cross-links down to *"exactly how the bits behave."*

## The decision spine

1. **Pick a format** → [`formats_overview.md`](formats_overview.md). bf16/fp16 baseline → FP8 (E4M3
   default) → MXFP4/MXFP6 on CDNA4 → INT8/INT4 (weight-only). OCP MX spec is the anchor.
2. **Match the FP8 dialect to the arch** → [`fnuz_vs_ocp.md`](fnuz_vs_ocp.md). The **2× wrong-dialect
   trap**: CDNA3 (gfx942) = FNUZ; CDNA4 (gfx950) = OCP. Never bit-copy a checkpoint across the split.
3. **For 4/6-bit, understand block scaling** → [`block_scaling_mxfp.md`](block_scaling_mxfp.md). E8M0
   32-element block scale, power-of-2 amax, scale shuffle, FP6 runs at the FP4 rate.
4. **Choose granularity & static-vs-dynamic** → [`scaling_strategies.md`](scaling_strategies.md).
   per-tensor / per-token / per-channel / per-block, and the 224.0 ROCm dynamic cap.
5. **Gate the result** → [`accuracy_evaluation.md`](accuracy_evaluation.md). `err_ratio<0.05` isolated,
   task accuracy (MMLU/GSM8K/perplexity) e2e — **never byte parity**.
6. **Know what the HW can actually do** → [`hardware_support_matrix.md`](hardware_support_matrix.md).
   CDNA1–4 × {fp16,bf16,fp8,fp6,fp4,int8} MFMA support + rate.
7. **KV cache is its own axis** → [`kv_cache_quantization.md`](kv_cache_quantization.md).

## Files
| file | one-liner |
|---|---|
| [`formats_overview.md`](formats_overview.md) | every dtype, bit layout, OCP MX framing |
| [`fnuz_vs_ocp.md`](fnuz_vs_ocp.md) | CDNA3 FNUZ vs CDNA4 OCP, the 2× trap, conversion |
| [`block_scaling_mxfp.md`](block_scaling_mxfp.md) | E8M0 block scale, shuffle, block-scaled MFMA |
| [`scaling_strategies.md`](scaling_strategies.md) | granularity × dynamic/static, 224.0 cap |
| [`accuracy_evaluation.md`](accuracy_evaluation.md) | err_ratio gate, task accuracy, SR |
| [`hardware_support_matrix.md`](hardware_support_matrix.md) | CDNA1–4 MFMA dtype support + rate |
| [`kv_cache_quantization.md`](kv_cache_quantization.md) | fp8/int8 KV, per-head scale, paged layout |

## Hardware anchors
[hardware/cdna3_mi300](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/hardware/cdna3_mi300) · [hardware/cdna4_mi350](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/hardware/cdna4_mi350) · [hardware/shared/dtype_numerics](../hardware/shared/dtype_numerics.md)

## Sources
- OCP Microscaling (MX) Formats spec v1.0: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Matrix Core Programming on CDNA3 & CDNA4: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- ROCm model quantization techniques: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/quantization/README.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
