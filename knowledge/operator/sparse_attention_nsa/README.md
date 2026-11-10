# sparse_attention_nsa

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Selected-block and sparse MLA/NSA attention execution.

5 AITER implementation references; 2 owned corpus decisions.

## Contract and applicability

Sparse attention consumes selected blocks or indices plus the query/KV representation. Selection order, invalid/padded indices, block size, masks and softmax normalization must agree with the indexer and cache producer.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Sparse attention — AITER aiter](backend/aiter_aiter.md) |
| gluon | [Sparse attention — AITER gluon](backend/gluon_aiter.md) |
| hip | [Sparse attention — AITER hip](backend/hip_aiter.md) |
| opus | [Sparse attention — AITER opus](backend/opus_aiter.md) |
| triton | [Sparse attention — AITER triton](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
