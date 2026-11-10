# mla_attention

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)

Multi-head latent attention, compressed KV and decode/prefill variants.

7 AITER implementation references; 2 owned corpus decisions.

## Contract and applicability

MLA uses compressed latent KV and positional components with projection/absorption choices. Match latent and rotary dimensions, head layout, cache representation and scaling before substituting a conventional attention or paged-decode path.

Use the linked corpus conditions and pinned entry point for exact supported shapes and GPUs.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Multi-head latent attention — AITER aiter](backend/aiter_aiter.md) |
| asm | [Multi-head latent attention — AITER asm](backend/asm_aiter.md) |
| flydsl | [Multi-head latent attention — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Multi-head latent attention — AITER gluon](backend/gluon_aiter.md) |
| hip | [Multi-head latent attention — AITER hip](backend/hip_aiter.md) |
| opus | [Multi-head latent attention — AITER opus](backend/opus_aiter.md) |
| triton | [Multi-head latent attention — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [AITER MLA: cache layout, causal queries and backend dispatch](backend/aiter_notes.md)
- [Native HK MLA: v3.2 and v4.0 decode pipelines](backend/hip_notes.md)

## Historical reference

[Imported GEAK overview](geak_reference.md) retains the original metadata, backend landscape, recipes and provenance. Its rankings and measurements belong to the original source context.
