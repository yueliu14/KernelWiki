# qk_norm_rope_cache

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Fused Q/K normalization, positional encoding, quantization and cache writes.

5 AITER implementation references; 0 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Fused Q/K normalization, RoPE and cache writes — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Fused Q/K normalization, RoPE and cache writes — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Fused Q/K normalization, RoPE and cache writes — AITER gluon](backend/gluon_aiter.md) |
| hip | [Fused Q/K normalization, RoPE and cache writes — AITER hip](backend/hip_aiter.md) |
| triton | [Fused Q/K normalization, RoPE and cache writes — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## Operator contract

This fusion couples normalization with rotary position encoding and sometimes quantization, splitting and paged cache writes. Position ids, Q/K head counts, rotary convention and cache slot mappings are part of the contract.

## Validation

Validate Q and K outputs plus modified cache locations. Include position tails, partial rotary dimensions and the quantization scale layout.
