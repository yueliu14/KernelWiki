# linear_attention_gated_delta

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Recurrent and chunked gated-delta attention, including its causal-conv front end.

5 AITER implementation references; 6 owned corpus decisions.

## Contract and applicability

Gated-delta/linear attention advances recurrent state with the specified decay and update rule. Chunked, decode and speculative verification paths must agree on state ownership, precision, accepted-token updates and rollback/snapshot behavior.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Linear attention and gated delta rules — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Linear attention and gated delta rules — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Linear attention and gated delta rules — AITER gluon](backend/gluon_aiter.md) |
| hip | [Linear attention and gated delta rules — AITER hip](backend/hip_aiter.md) |
| triton | [Linear attention and gated delta rules — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [Causal state mixing inside gated-delta attention](backend/aiter_notes.md)

## Historical case studies

- [Hybrid-model attention-backend selection: Qwen3.5-27B](case_studies/gated_delta_backend_swap.md)

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
