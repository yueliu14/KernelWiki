# context_parallel_attention

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Attention split across context-parallel ranks and output merging.

1 AITER implementation references; 1 owned corpus decisions.

## Contract and applicability

Context-parallel attention partitions sequence state across ranks and must combine the corresponding attention/softmax contributions. Partition ownership, masks, communication order and synchronization are part of correctness; local kernel timing excludes distributed completion.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Round-robin context-parallel MLA and index selection](backend/aiter_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
