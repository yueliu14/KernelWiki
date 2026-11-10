# attention_decode_paged

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Decode attention over paged KV caches, split/reduce and mixed precision.

5 AITER implementation references; 7 owned corpus decisions.

## Contract and applicability

Decode attention reads KV through a page/block map. Sequence lengths, page size, head mapping, cache dtype/scales and masks must agree. Split attention routes materialize partial output/softmax state and require a correct merge.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| asm | [Paged and unified attention decode — AITER asm](backend/asm_aiter.md) |
| flydsl | [Paged and unified attention decode — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Paged and unified attention decode — AITER gluon](backend/gluon_aiter.md) |
| hip | [Paged and unified attention decode — AITER hip](backend/hip_aiter.md) |
| triton | [Paged and unified attention decode — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [KV-cache preparation belongs to the attention workload](backend/aiter_notes.md)

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
