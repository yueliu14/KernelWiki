# grouped_gemm_moe

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Expert grouped GEMMs, token imbalance and low-bit expert weights.

8 AITER implementation references; 8 owned corpus decisions.

## Contract and applicability

Expert GEMMs consume routed token/slot mappings and expert-specific weights. Route counts, padding/sentinels, quantization layouts and output ownership must agree with dispatch and combine. Different expert populations are distinct workloads even at the same total token count.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Grouped GEMM and MoE expert computation — AITER aiter](backend/aiter_aiter.md) |
| asm | [Grouped GEMM and MoE expert computation — AITER asm](backend/asm_aiter.md) |
| ck | [Grouped GEMM and MoE expert computation — AITER ck](backend/ck_aiter.md) |
| flydsl | [Grouped GEMM and MoE expert computation — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Grouped GEMM and MoE expert computation — AITER gluon](backend/gluon_aiter.md) |
| hip | [Grouped GEMM and MoE expert computation — AITER hip](backend/hip_aiter.md) |
| opus | [Grouped GEMM and MoE expert computation — AITER opus](backend/opus_aiter.md) |
| triton | [Grouped GEMM and MoE expert computation — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [AITER FlyDSL MoE: requested tiles versus executed tiles](backend/flydsl_notes.md)
- [OPUS MoE: stage-1 schedules and stage-2 output strategies](backend/opus_notes.md)

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
