# gemm_epilogue_fused

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

GEMM with activation, residual, normalization/quantization and communication boundaries.

2 AITER implementation references; 4 owned corpus decisions.

## Contract and applicability

The epilogue consumes the completed GEMM result and may apply supported bias, activation, residual, normalization or quantization. Nonlinear operations usually require all K partials first. Preserve rounding points, externally visible outputs and any remaining reduction or communication boundary.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [GEMM fusion boundaries in AITER](backend/aiter_aiter.md) |
| triton | [Gemm epilogue fused — AITER triton](backend/triton_aiter.md) |

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
