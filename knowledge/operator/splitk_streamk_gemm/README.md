# splitk_streamk_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Split-K and Stream-K schedules for poorly occupied GEMMs.

4 AITER implementation references; 3 owned corpus decisions.

## Contract and applicability

Split-K partitions the K reduction; Stream-K distributes tile work across workgroups. Both must combine each partial exactly once. Preserve initialization, scratch/atomic protocols and reduction order, and include combination work when comparing schedules.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| asm | [Splitk streamk gemm — AITER asm](backend/asm_aiter.md) |
| flydsl | [Splitk streamk gemm — AITER flydsl](backend/flydsl_aiter.md) |
| opus | [OPUS Split-K: workspace, cluster launch and fused reduction](backend/opus_aiter.md) |
| triton | [Splitk streamk gemm — AITER triton](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
