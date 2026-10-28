# splitk_streamk_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Split-K and Stream-K schedules for poorly occupied GEMMs.

4 AITER implementation references; 0 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| asm | [Splitk streamk gemm — AITER asm](backend/asm_aiter.md) |
| flydsl | [Splitk streamk gemm — AITER flydsl](backend/flydsl_aiter.md) |
| opus | [OPUS Split-K: workspace, cluster launch and fused reduction](backend/opus_aiter.md) |
| triton | [Splitk streamk gemm — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: splitk_streamk_gemm — overview
kind: operator_overview
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - https://arxiv.org/abs/2301.03598
  - https://github.com/ROCm/composable_kernel
```

</details>

# splitk_streamk_gemm

## TL;DR
> A GEMM **decomposition strategy** (not a new math op): partition the K reduction (split-K) or the whole
> tile space (stream-K) across more workgroups to fill all CUs when the natural tile count is too small to
> saturate the GPU — the key fact is **it only helps when you're CU-underutilized** (small M·N or huge K).

## Math contract
- Computes the same `C = A @ B (+ bias)` as dense GEMM. Decomposition changes *who computes which partial
  sums*, not the result.
- **Split-K**: split the K dimension into `SPLIT_K` chunks; each workgroup computes a partial `C`; combine
  by atomic add or a workspace + reduction kernel.
- **Stream-K**: flatten all (tile × k-iteration) work into a 1-D iteration space and distribute it evenly
  across a fixed number of persistent workgroups (≈ CU count), so partial tiles balance across CUs; fix-up
  partial tiles with a reduction. (Osama et al., "Stream-K", arXiv:2301.03598.)
- Accumulate fp32; output dtype as dense GEMM.

## Shape regimes
- **Small-M / skinny** (decode-ish, M small, N·K moderate): few output tiles → many idle CUs → split-K /
  stream-K recover utilization. See [../skinny_gemv_decode/overview.md](../skinny_gemv_decode/README.md).
- **Large-K, small M·N**: classic split-K territory (the reduction dominates).
- **Large M·N**: plenty of tiles already → split-K/stream-K usually unnecessary or harmful (extra
  reduction cost). Use plain dense GEMM ([../dense_gemm/overview.md](../dense_gemm/README.md)).

## Where it matters (Amdahl)
- Matters for the GEMM shapes that otherwise leave the 304 CUs (MI300X) / 256 CUs (MI350X) idle — small
  batch decode projections and tall-skinny problems. Stream-K's value is wave-quantization removal: it
  turns a "1.2 waves" launch into balanced full occupancy.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota | [backends/triton.md](backend/triton.md) |
| ck | 🟡 competitive | [backends/ck.md](backend/ck.md) |
| hipblaslt | 🟡 competitive | [backends/hipblaslt.md](backend/hipblaslt.md) |
| asm | 🟡 competitive | [backends/asm.md](backend/asm.md) |
| hip | 🟡 competitive | [backends/hip.md](backend/hip.md) |
| flydsl | 🟡 competitive | [backends/flydsl.md](backend/flydsl.md) |

## Fusion neighbors
- The reduction/fix-up epilogue can fold bias/activation; for fp8 it can fold dequant scale →
  [../scaled_quant_gemm/overview.md](../scaled_quant_gemm/README.md),
  [../gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/README.md).

## Numerics
- Atomic-add reduction order is non-deterministic; workspace reduction is deterministic →
  [numerics.md](numerics.md).

## How to bench
- Sweep `SPLIT_K`/stream-K on/off across the target small-M / large-K shapes; median ≥3 warm reps; oracle =
  dense fp32 reference. Always compare against plain dense GEMM to confirm the decomposition actually wins
  for that shape.

## Sources
- Triton persistent/stream-K matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- Stream-K paper (Osama et al.): https://arxiv.org/abs/2301.03598
- Composable Kernel (split-K device ops): https://github.com/ROCm/composable_kernel

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/splitk_streamk_gemm/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
