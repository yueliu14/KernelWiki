# sliding_window_attention

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Window-limited attention and local/global layer combinations.

1 AITER implementation references; 3 owned corpus decisions.

## Contract and applicability

Sliding-window attention restricts each query to its permitted local context. Left/right-window conventions, causal masking, absolute positions and cache/page mapping determine the result; a short context is not by itself evidence of an SWA route.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| triton | [Sliding-window attention: mask and tile-pruning variants](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
