# attention_prefill_fmha

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Forward FMHA for prompt processing, including variable-length causal attention.

6 AITER implementation references; 3 owned corpus decisions.

## Contract and applicability

Prefill attention consumes Q/K/V with the selected sequence layout, head grouping and mask. Causal alignment, variable lengths, scale and output/softmax-state precision belong to the contract. Follow the wrapper to determine whether cache preparation is included.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Prefill and full attention — AITER aiter](backend/aiter_aiter.md) |
| asm | [Prefill and full attention — AITER asm](backend/asm_aiter.md) |
| flydsl | [Prefill and full attention — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Prefill and full attention — AITER gluon](backend/gluon_aiter.md) |
| hip | [Prefill and full attention — AITER hip](backend/hip_aiter.md) |
| triton | [Prefill and full attention — AITER triton](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
