# skinny_gemv_decode

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Small-M decode projections and GEMV-like regimes.

1 AITER implementation references; 2 owned corpus decisions.

## Contract and applicability

Small-M projection computes `Y[M,N] = X[M,K] @ W[K,N]` under the selected weight layout. M=1 is GEMV; larger decode batches need separate dispatch checks. Splitting K changes partial-output ownership and reduction order, so compare the complete projection.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| hip | [Small-M decode GEMM: three HIP dispatch paths](backend/hip_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
