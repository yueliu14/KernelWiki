# fused_moe_grouped_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Fused expert execution and compute/communication overlap.

2 AITER implementation references; 1 owned corpus decisions.

## Contract and applicability

Follow routing through gate/up, activation, down and combination. A fused callable can retain several launches and global intermediates; identify the actual shared kernel bodies and which route weights, normalization and communication remain outside them.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Fused MoE compute and communication — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Fused MoE compute and communication — AITER flydsl](backend/flydsl_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
