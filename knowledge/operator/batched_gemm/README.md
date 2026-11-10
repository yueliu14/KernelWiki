# batched_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Batched matrix products and head-wise projection variants.

6 AITER implementation references; 2 owned corpus decisions.

## Contract and applicability

Compute a matrix product for each batch or head. Batch strides, broadcasting, transpose conventions and output layout are part of the contract; a flattened dense GEMM is equivalent only when those mappings agree.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Batched matrix multiplication — AITER aiter](backend/aiter_aiter.md) |
| ck | [Batched matrix multiplication — AITER ck](backend/ck_aiter.md) |
| flydsl | [Batched matrix multiplication — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Batched matrix multiplication — AITER gluon](backend/gluon_aiter.md) |
| opus | [OPUS BMM: exact-id BF16 and MXFP8 variants](backend/opus_aiter.md) |
| triton | [Batched matrix multiplication — AITER triton](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
