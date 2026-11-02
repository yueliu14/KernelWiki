# gemm_epilogue_fused

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


GEMM with activation, residual, normalization/quantization and communication boundaries.

2 AITER implementation references; 3 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [GEMM fusion boundaries in AITER](backend/aiter_aiter.md) |
| triton | [Gemm epilogue fused — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: gemm_epilogue_fused — overview
kind: operator_overview
operator: gemm_epilogue_fused
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@HEAD:aiter/tuned_gemm.py
```

</details>

# gemm_epilogue_fused  (`D = act(α·A·Bᵀ + bias + β·residual)`, optionally → quant)

## TL;DR
Epilogue fusion folds bias / activation / residual / output-quant into the GEMM's C write-out, removing
a full [M,N] memory round-trip per fused op. The single most important fact: the GEMM is compute-bound
so the appended epilogue is **nearly free**, while the separate elementwise kernel it replaces is a
memory-bound [M,N] pass — fuse the MLP up/gate `act_and_mul` and the output-quant first. Use
`OPTIMIZE_EPILOGUE=1` to keep the fused C write off the 512B Tagram hotspot.

## Math contract
`D[M,N] = act( α · (A[M,K]·B[N,K]ᵀ) + bias[N] + β · residual[M,N] )`, then optional fp8/fp4 quant of D
with a computed/given scale. act ∈ {identity, relu, gelu, silu, silu·mul (gated)}. dtype: bf16/fp16 in,
fp32 accumulate; out bf16 or quantized fp8/fp4. The fused flags (bias/act/residual/quant) are part of
the tuning lookup key.

## Shape regimes
- **MLP up/gate (prefill)**: K=hidden, N=inter (large, up to ~34816), fuse `silu·mul` to skip a giant
  [M,inter] pass — the highest-value fusion.
- **attn out / down proj**: fuse `+residual` (skip-add) into the GEMM.
- **decode (skinny M)**: epilogue cost is tiny; fusion still removes the extra kernel launch + pass.

## Where it matters (Amdahl)
The GEMM mass itself is the dense_gemm 78–81%; epilogue fusion doesn't change the matmul FLOPs but
deletes the extra elementwise/quant passes (each a [M,N] read+write). On wide MLP N this is a real
memory-traffic win and removes kernel-launch overhead; typically a few % when the unfused passes were
non-trivial.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live path, bias/scale epilogue) | [backends/aiter.md](backend/aiter.md) |
| hipblaslt | 🟢 sota (executed fused kernels) | [backends/hipblaslt.md](backend/hipblaslt.md) |
| ck | 🟢 sota (CShuffle CDEElementwise — richest fusion) | [backends/ck.md](backend/ck.md) |
| triton | 🟡 competitive (easy custom epilogue) | [backends/triton.md](backend/triton.md) |
| hip | 🟤 legacy (raw fused kernel) | [backends/hip.md](backend/hip.md) |
| flydsl | 🟡 competitive (MFMA row + LDS CShuffle epilogue) | [backends/flydsl.md](backend/flydsl.md) |

## Fusion neighbors
This *is* the fusion operator. Producer-side: norm+quant before the GEMM
([operators/fused_norm_quant/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/README.md)); activation op
([operators/act_and_mul_silu_gelu/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/act_and_mul_silu_gelu/README.md)); output quant
([operators/scaled_quant_gemm/overview.md](../scaled_quant_gemm/README.md), [operators/quant_dequant_fp8/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_dequant_fp8/README.md)). See
[fusion.md](fusion.md).

## Numerics
fp32 accumulate; act applied in fp32 before down-cast; quant epilogue needs task-accuracy gating →
[numerics.md](numerics.md).

## How to bench
Isolated: fused GEMM vs (GEMM + separate elementwise) at the exact (M,N,K,flags). e2e: same-session A/B,
gate delta>0.5% + non-overlap + engagement.

## Sources
- CShuffle epilogue / MultipleD (bias/act/residual): ROCm "Optimizing with Composable Kernel".
- Tagram hotspot / OPTIMIZE_EPILOGUE: ROCm workload guide.
- bias/scale epilogue on live path: `ROCm/aiter@HEAD:aiter/tuned_gemm.py`.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/gemm_epilogue_fused/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
