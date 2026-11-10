# dense_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Dense attention and FFN projections; prefill versus decode tiling.

8 AITER implementation references; 19 owned corpus decisions.

## Contract and applicability

Dense projection computes `D[M,N] = A[M,K] @ B[N,K].T`, with bias or activation only where the selected entry point implements it. Check transpose/stride, accumulation/output dtype and the bias dispatch key. Prefill and small-M decode can select different implementations.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Dense matrix multiplication — AITER aiter](backend/aiter_aiter.md) |
| asm | [Dense matrix multiplication — AITER asm](backend/asm_aiter.md) |
| ck | [Dense matrix multiplication — AITER ck](backend/ck_aiter.md) |
| flydsl | [Dense matrix multiplication — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Dense matrix multiplication — AITER gluon](backend/gluon_aiter.md) |
| hip | [Dense matrix multiplication — AITER hip](backend/hip_aiter.md) |
| opus | [Dense matrix multiplication — AITER opus](backend/opus_aiter.md) |
| triton | [Dense matrix multiplication — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [AITER FlyDSL A16W16: useful tiles and occupancy-aware policy](backend/flydsl_notes.md)
- [AITER OPUS: exact kernel ids and shape-driven dispatch](backend/opus_notes.md)

## Historical case studies

- [AITER DB tuning: Qwen3.5-27B and the bias-key mismatch](case_studies/gemm_aiter_db_tuning.md)

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
