# scaled_quant_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


FP8, INT8, FP4 and microscaled projections, including packing and scale contracts.

8 AITER implementation references; 13 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Scaled and mixed-precision GEMM — AITER aiter](backend/aiter_aiter.md) |
| asm | [Scaled and mixed-precision GEMM — AITER asm](backend/asm_aiter.md) |
| ck | [Scaled and mixed-precision GEMM — AITER ck](backend/ck_aiter.md) |
| flydsl | [Scaled and mixed-precision GEMM — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Scaled and mixed-precision GEMM — AITER gluon](backend/gluon_aiter.md) |
| hip | [Scaled and mixed-precision GEMM — AITER hip](backend/hip_aiter.md) |
| opus | [Scaled and mixed-precision GEMM — AITER opus](backend/opus_aiter.md) |
| triton | [Scaled and mixed-precision GEMM — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [FP4 GEMM input and output contracts](backend/asm_notes.md)
- [OPUS FP8 GEMM: scale and preshuffle families](backend/opus_notes.md)

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: scaled_quant_gemm — overview
kind: operator_overview
operator: scaled_quant_gemm
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1, fp6_e2m3, fp6_e3m2, int8]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
```

</details>

# scaled_quant_gemm

## TL;DR
> A GEMM whose low-precision (fp8/fp6/fp4) inputs carry **per-block scale factors** applied inside the
> MFMA — on CDNA4 (gfx950) the hardware does this natively (block-scaled MFMA), and the key fact is
> **FP6 runs at FP4 rate** and 32-element blocks share one E8M0 scale.

## Math contract
- `C = (A·sA) @ (B·sB) (+ bias)`, where A,B are fp8/fp6/fp4 and `sA,sB` are per-block scales. On CDNA4 the
  scaled MFMA applies scales **after the dot product, before accumulation** (per CDNA4 ISA / Matrix Core blog).
- **Block scaling (MXFP)**: 32 contiguous elements share one **E8M0** (8-bit exponent, no mantissa) scale.
  Formats: mxfp8 (e4m3/e5m2), mxfp6 (e3m2/e2m3), mxfp4 (e2m1) — OCP microscaling standard.
- **CDNA4 instructions**: `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4` and the 16x16x128 variant
  (Gluon: `gl.amd.cdna4.mfma_scaled` / `v_mfma_scale_f32_16x16x128_f8f6f4`). Output fp32.
- **CDNA3 (gfx942)**: fp8 **e4m3/e5m2 FNUZ** MFMA, with tensor-level (not native 32-elem block) scaling.
- Layout quirk: fp4 inputs to the 256-bit-wide intrinsic operand are packed into a 256-bit type with the
  upper 128 bits zero (32 fp4 = 128 bits).

## Shape regimes
- Same M/N/K regimes as dense GEMM; low precision is used to raise arithmetic intensity (prefill
  compute-bound) and to shrink weights (decode memory-bound). Decode skinny shapes →
  [../skinny_gemv_decode/overview.md](../skinny_gemv_decode/README.md).

## Where it matters (Amdahl)
- fp8/fp4 GEMM is the path to peak Matrix Core throughput: MI355X ≈ 5 PFLOPS FP8 and ≈ 10 PFLOPS
  MXFP4/MXFP6 (whitepaper, via Gluon tutorial). Moving major GEMMs to block-scaled low precision is the
  largest single compute lever on CDNA4.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota | [backends/triton.md](backend/triton.md) |
| ck | 🟡 competitive | [backends/ck.md](backend/ck.md) |
| hip | 🟡 competitive | [backends/hip.md](backend/hip.md) |
| asm | 🟡 competitive | [backends/asm.md](backend/asm.md) |
| aiter | 🟢 sota | [backends/aiter.md](backend/aiter.md) |
| hipblaslt | 🟡 competitive | [backends/hipblaslt.md](backend/hipblaslt.md) |
| flydsl | 🟢 sota (preshuffle fp8/int8 + gfx950 MXFP4) | [backends/flydsl.md](backend/flydsl.md) |

## Fusion neighbors
- Input quant (norm+quant) upstream and output quant downstream fuse with the GEMM →
  [../fused_norm_quant/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/fused_norm_quant/README.md),
  [../quant_fp4_mxfp/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_fp4_mxfp/README.md), [fusion.md](fusion.md).

## Numerics
- E8M0 block scale application order, fp8 dynamic range, accuracy gating → [numerics.md](numerics.md).

## How to bench
- Bench per target shape/dtype; median ≥3 warm reps; oracle = bf16 dense reference + an accuracy gate
  (err_ratio / task metric). Compare achieved TFLOPS to the dtype's Matrix Core peak (never present peak as
  achievable).

## Sources
- Matrix Core programming (CDNA3/CDNA4 scaled MFMA): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Gluon GEMM tutorial (scaled MFMA, BF8/MXFP4 perf): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- CDNA4 ISA reference: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/scaled_quant_gemm/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
