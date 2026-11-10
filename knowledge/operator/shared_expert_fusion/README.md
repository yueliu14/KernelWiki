# shared_expert_fusion

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Shared-expert computation and its overlap or fusion with routed experts.

1 AITER implementation references; 2 owned corpus decisions.

## Contract and applicability

Shared experts and routed experts can consume related inputs but have distinct ownership and weighting. Fusion or overlap must preserve both contributions, residual semantics and synchronization; their adjacency does not establish shared device execution.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| flydsl | [FHMoE: routed and shared experts in one execution path](backend/flydsl_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
