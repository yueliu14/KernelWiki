# moe_routing_topk

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Expert selection, routing weights, token sorting and alignment for MoE GEMMs.

5 AITER implementation references; 4 owned corpus decisions.

## Contract and applicability

Routing maps logits to selected expert IDs and weights, then may build counts, sorting or aligned slot maps. Top-k/group restrictions, tie handling, score normalization, sentinels and index dtype must match the expert consumer.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [MoE routing, sorting and token assignment — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [MoE routing, sorting and token assignment — AITER flydsl](backend/flydsl_aiter.md) |
| hip | [MoE routing, sorting and token assignment — AITER hip](backend/hip_aiter.md) |
| opus | [MoE routing, sorting and token assignment — AITER opus](backend/opus_aiter.md) |
| triton | [MoE routing, sorting and token assignment — AITER triton](backend/triton_aiter.md) |

**PR evidence correction (2026-11-04):** [PR #1909](https://github.com/ROCm/aiter/pull/1909)
is a closed, unmerged proposal for plain `topk_sigmoid`. Its reported average 1.66x
does not measure `biased_grouped_topk` and does not establish a shipped gfx9 dispatch.
The imported material below conflates these implementations; do not transfer that
performance or support claim to grouped routing. See the [reviewed report](corpus/decisions.md#moe-sigmoid-dpp-selection)
for its comparison conditions, inconsistent extrema and confidence.

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
