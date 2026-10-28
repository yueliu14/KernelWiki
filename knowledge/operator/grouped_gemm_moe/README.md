# grouped_gemm_moe

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Expert grouped GEMMs, token imbalance and low-bit expert weights.

8 AITER implementation references; 37 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Grouped GEMM and MoE expert computation — AITER aiter](backend/aiter_aiter.md) |
| asm | [Grouped GEMM and MoE expert computation — AITER asm](backend/asm_aiter.md) |
| ck | [Grouped GEMM and MoE expert computation — AITER ck](backend/ck_aiter.md) |
| flydsl | [Grouped GEMM and MoE expert computation — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Grouped GEMM and MoE expert computation — AITER gluon](backend/gluon_aiter.md) |
| hip | [Grouped GEMM and MoE expert computation — AITER hip](backend/hip_aiter.md) |
| opus | [Grouped GEMM and MoE expert computation — AITER opus](backend/opus_aiter.md) |
| triton | [Grouped GEMM and MoE expert computation — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [AITER FlyDSL MoE: requested tiles versus executed tiles](backend/flydsl_notes.md)
- [OPUS MoE: stage-1 schedules and stage-2 output strategies](backend/opus_notes.md)

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: grouped_gemm_moe — overview
kind: operator_overview
operator: grouped_gemm_moe
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
```

</details>

# grouped_gemm_moe

## TL;DR
> A batch of independent GEMMs with variable per-group M (one GEMM per MoE expert), executed in a single
> fused kernel launch keyed by a routing/offset table — the single biggest win is **doing all experts in
> one launch over a sorted/aligned token layout** instead of one launch per expert.

## Math contract
- For each expert `e` in `[0..E)`: `Y[rows_e] = X[rows_e] @ W_e (+ bias_e)`, where `rows_e` are the
  tokens routed to `e`. Equivalent to a block-diagonal GEMM with per-group `M_e` and shared `N,K`.
- MoE FFN uses two grouped GEMMs: gate/up `(K=hidden, N=2*inter)` then down `(K=inter, N=hidden)`,
  with an act-and-mul (silu/gelu) between → see [fusion.md](fusion.md) and
  [../act_and_mul_silu_gelu/overview.md](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/act_and_mul_silu_gelu/README.md).
- Inputs are pre-sorted by expert and padded so each group's M aligns to the tile (MoE align & sort);
  an `expert_ids`/`num_tokens_post_pad` offset table drives tile→group mapping.
- dtype: bf16/fp16 weights, or fp8/fp4 block-scaled (→ [../scaled_quant_gemm/overview.md](../scaled_quant_gemm/README.md)).
  Accumulate fp32.

## Shape regimes
- **Prefill**: large total token count → most groups have `M_e ≫ tile`; behaves like many dense GEMMs,
  compute-bound, tile-efficiency dominated.
- **Decode**: few tokens, top-k routing → many groups with tiny `M_e` (1..few tiles), memory-bound and
  highly imbalanced; tile balancing and avoiding empty-tile waste dominate.

## Where it matters (Amdahl)
- MoE FFN grouped GEMMs are the dominant GPU cost in MoE models (e.g. DeepSeek/Qwen-MoE). AITER's asm
  fused_moe path is reported as the best on AMD, with ~3x fused-MoE acceleration attributed to the
  grouped-GEMM-of-different-shapes structure (Bruce Xu, AITER/SGLang integration blog) — so moving this
  op moves e2e materially.

## Backend landscape (link table → SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟡 competitive | [backends/triton.md](backend/triton.md) |
| ck | 🟡 competitive | [backends/ck.md](backend/ck.md) |
| hip | 🟡 competitive | [backends/hip.md](backend/hip.md) |
| aiter | 🟢 sota | [backends/aiter.md](backend/aiter.md) |
| tilelang | 🧪 experimental | [backends/tilelang.md](backend/tilelang.md) |
| flydsl | 🟢 sota (2-stage gate+up / down GEMM, fp8/bf16, JIT) | [backends/flydsl.md](backend/flydsl.md) |

## Fusion neighbors
- Upstream: [../moe_routing_topk/overview.md](../moe_routing_topk/README.md) (topk + sort/align),
  [../moe_dispatch_combine/overview.md](../moe_dispatch_combine/README.md).
- Inline: act-and-mul, per-token/per-block quant epilogue → [fusion.md](fusion.md).
- Downstream: weighted combine/scatter back to token order. See also
  [../shared_expert_fusion/overview.md](../shared_expert_fusion/README.md).

## Numerics
- fp8/fp4 expert weights with per-group scales; routing weight multiply in fp32 → [numerics.md](numerics.md).

## How to bench
- Isolate the two grouped GEMMs with a fixed routing table (real captured `topk_ids`), median of ≥3 warm
  reps; oracle = per-expert dense reference `X_e @ W_e` gathered back. Compare against
  [../dense_gemm/overview.md](../dense_gemm/README.md) summed over groups for a sanity ceiling.

## Sources
- AITER repo + DeepSeek-R1/SGLang integration blog (grouped GEMM = MoE win): https://github.com/ROCm/aiter ,
  https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
- MoE align & sort design: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/grouped_gemm_moe/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
