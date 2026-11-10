# chunked_prefill

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Prompt chunks attending to cached prefixes.

1 AITER implementation references; 1 owned corpus decisions.

## Contract and applicability

Chunked prefill attends new query tokens against the permitted prefix and current chunk, potentially alongside decode requests. Preserve per-request offsets, causal alignment, cached lengths and cache writes across chunk boundaries.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| triton | [Chunked prefill over an existing paged prefix](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
