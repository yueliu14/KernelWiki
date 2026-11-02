# attention_indexer

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


MQA logits, index scores, block selection and top-k for sparse attention.

6 AITER implementation references; 0 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Attention index scoring and selection — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Attention index scoring and selection — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Attention index scoring and selection — AITER gluon](backend/gluon_aiter.md) |
| hip | [Attention index scoring and selection — AITER hip](backend/hip_aiter.md) |
| opus | [Attention index scoring and selection — AITER opus](backend/opus_aiter.md) |
| triton | [Attention index scoring and selection — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [Scoring and selection in the attention indexer](backend/flydsl_notes.md)
- [Sparse block selection: scoring, top-k and fused CP selection](backend/hip_notes.md)
- [OPUS MXFP4 indexer: architecture layouts and dynamic schedules](backend/opus_notes.md)

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

These kernels compute or select scores used to choose sparse-attention tokens/pages. Paged and contiguous caches, FP8/MXFP4 storage and prefill/decode schedules are separate contracts.

## Validation

Check selected indices as well as logits, including ties and masked pages. Keep scoring and top-k costs visible when evaluating sparse attention.
