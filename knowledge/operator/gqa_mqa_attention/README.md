# gqa_mqa_attention

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Grouped-query and multi-query head sharing.

1 AITER implementation references; 3 owned corpus decisions.

## Contract and applicability

Several query heads share a KV head (MQA uses one KV head). Preserve the Q-to-KV head mapping, strides and masks. This is a head-sharing contract that can reuse prefill or decode kernels; it does not imply a separate launch.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| triton | [GQA and MQA in unified paged attention](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
