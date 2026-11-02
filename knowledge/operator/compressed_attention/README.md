# compressed_attention

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Compression stages and fused attention over compressed representations.

1 AITER implementation references; 1 owned corpus decisions.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| flydsl | [Compressed attention — AITER flydsl](backend/flydsl_aiter.md) |

## Development decisions

[Corpus decisions](corpus/decisions.md) — source-grounded constraints and scoped experience summaries.

## Operator contract

Compression changes the representation consumed by attention; fused paths combine preparation and attention steps. Compression ratio, positional features and cache layout must match the source wrapper.

## Validation

Validate the compression result and final attention output. Include preparation cost when comparing against a decomposed reference.
