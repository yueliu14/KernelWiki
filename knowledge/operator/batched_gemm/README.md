# batched_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Batched matrix products and head-wise projection variants.

6 AITER implementation references; 0 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Batched matrix multiplication — AITER aiter](backend/aiter_aiter.md) |
| ck | [Batched matrix multiplication — AITER ck](backend/ck_aiter.md) |
| flydsl | [Batched matrix multiplication — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Batched matrix multiplication — AITER gluon](backend/gluon_aiter.md) |
| opus | [OPUS BMM: exact-id BF16 and MXFP8 variants](backend/opus_aiter.md) |
| triton | [Batched matrix multiplication — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: batched_gemm — overview
kind: operator_overview
operator: batched_gemm
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - ROCm/aiter@HEAD:aiter/tuned_gemm.py
```

</details>

# batched_gemm  (`C[b] = A[b] · B[b]ᵀ [+ bias]`, b = 1..B)

## TL;DR
Batched GEMM runs B independent same-shape (or grouped) matmuls in one launch — the workhorse of
attention QK/PV projections and any per-head/per-expert matmul. The key fact: for **many small**
batched matmuls the launch/occupancy overhead dominates, so the lever is **one kernel that fills all
CUs across the batch** (≥1024 workgroups *total* over B), not B separate GEMMs. For *variable* per-batch
shapes use grouped GEMM instead → [operators/grouped_gemm_moe/overview.md](../grouped_gemm_moe/README.md).

## Math contract
`C[b][M,N] = A[b][M,K] · B[b][N,K]ᵀ + bias[b]`, b ∈ [0,B). Uniform (M,N,K) across the batch (true
batched); fp16/bf16 in, fp32 accumulate, bf16 out. fp8 variants take per-batch scales. Strided-batched
(single tensor, fixed stride) is the common attention layout.

## Shape regimes
- **attention QKᵀ / PV (per-head)**: B = num_heads × batch, M = seqlen (prefill) or 1 (decode),
  small N/K = head_dim (64/128). Tiny matmuls, huge B → occupancy-bound.
- **prefill**: large M per batch, moderate B → compute-bound, behaves like many dense GEMMs.
- **decode**: M≈1 per batch (GEMV-like) → skinny / split-K, batch fills the machine.

## Where it matters (Amdahl)
Mostly subsumed by the attention kernels (FMHA fuses QKᵀ·softmax·PV so the batched matmul never
materializes) → see [operators/attention_prefill_fmha/overview.md](../attention_prefill_fmha/README.md) and
[operators/mla_attention/overview.md](../mla_attention/README.md). Standalone batched GEMM matters for non-fused attention,
adapters/LoRA, and per-head projections; a 1.1× there is usually <2% e2e.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live dispatch) | [backends/aiter.md](backend/aiter.md) |
| hipblaslt | 🟢 sota (strided-batched kernels) | [backends/hipblaslt.md](backend/hipblaslt.md) |
| ck | 🟡 competitive (batched/grouped templates) | [backends/ck.md](backend/ck.md) |
| triton | 🟡 competitive (batched matmul kernels) | [backends/triton.md](backend/triton.md) |
| hip | 🟤 legacy (raw strided-batched / loop) | [backends/hip.md](backend/hip.md) |
| asm | 🟢 (peak per-shape, authored) | [backends/asm.md](backend/asm.md) |

## Fusion neighbors
`+bias`, `+act`, `+scale` epilogue (same as dense) → [fusion.md](fusion.md); the dominant fusion is
into **attention** (QKᵀ→softmax→PV) rather than a standalone epilogue.

## Numerics
fp32 accumulate; per-batch fp8 scales for quant variants → [numerics.md](numerics.md).

## How to bench
Isolated: strided-batched GEMM at the exact (B,M,N,K,dtype) via `hipblaslt-bench`/ckProfiler. e2e:
prefer benchmarking the *attention* kernel that subsumes it. Gate on delta>0.5% + non-overlap.

## Sources
- Strided-batched GEMM + occupancy levers: ROCm workload guide.
- CK batched/grouped templates: ROCm "Optimizing with Composable Kernel".
- Live dispatch through aiter: `ROCm/aiter@HEAD:aiter/tuned_gemm.py`.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/batched_gemm/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
