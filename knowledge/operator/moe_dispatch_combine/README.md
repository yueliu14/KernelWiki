# moe_dispatch_combine

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Expert-parallel dispatch, gather, reduction and combine.

3 AITER implementation references; 4 owned corpus decisions.

## Contract and applicability

Dispatch maps token/route slots into expert-owned inputs; combine restores token outputs with the specified route weights and reduction. Preserve duplicate routes, local/global expert identity, padding and rank/stream completion before reusing buffers.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [All to all dispatch combine — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [MoE dispatch and combine — AITER flydsl](backend/flydsl_aiter.md) |
| hip | [All to all dispatch combine — AITER hip](backend/hip_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
