# scaled_quant_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

FP8, INT8, FP4 and microscaled projections, including packing and scale contracts.

8 AITER implementation references; 12 owned corpus decisions.

## Contract and applicability

Quantized operands carry scales whose granularity, dtype and storage layout are part of the product. Distinguish per-tensor, per-token/channel, block-scaled and microscaled routes. Packed K and preshuffled weights must match the selected kernel; scales inside the K reduction cannot generally be moved to the epilogue.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Scaled and mixed-precision GEMM — AITER aiter](backend/aiter_aiter.md) |
| asm | [Scaled and mixed-precision GEMM — AITER asm](backend/asm_aiter.md) |
| ck | [Scaled and mixed-precision GEMM — AITER ck](backend/ck_aiter.md) |
| flydsl | [Scaled and mixed-precision GEMM — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Scaled and mixed-precision GEMM — AITER gluon](backend/gluon_aiter.md) |
| hip | [Scaled and mixed-precision GEMM — AITER hip](backend/hip_aiter.md) |
| opus | [Scaled and mixed-precision GEMM — AITER opus](backend/opus_aiter.md) |
| triton | [Scaled and mixed-precision GEMM — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [FP4 GEMM input and output contracts](backend/asm_notes.md)
- [OPUS FP8 GEMM: scale and preshuffle families](backend/opus_notes.md)

## Historical case studies

- [Norm + quant preparation for scaled GEMM (upstream-reported)](case_studies/fused_norm_quant_win.md)

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
