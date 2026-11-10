# Workload-level kernel knowledge

Workloads organize multi-operator subgraphs by their semantic inputs, transformations and outputs. They own cross-stage data/state contracts, scheduling, dispatch, communication, fusion and complete-subgraph measurement. A semantic chain can have several implementations and launch counts.

Start with a chain below, then its case and reusable decisions. The [case and decision index](CORPUS.md) owns model, phase, backend and launch-boundary listings. Use [operator knowledge](../operator/README.md) for constituent kernels, [top-model histories](../top_model/README.md) for dated serving comparisons, and [general knowledge](../general/README.md) for reusable methods.

## Semantic chains

| Semantic chain | Scope |
| --- | --- |
| [qkv_projection_norm_rope_cache](qkv_projection_norm_rope_cache/README.md) | Hidden-state projection → Q/K normalization and position encoding → attention inputs and KV state |
| [router_pooler_topk](router_pooler_topk/README.md) | Shared input → routing projection/top-k and pooler projection/norm → expert IDs, weights and input |
| [projection_norm_residual](projection_norm_residual/README.md) | Branch projection → normalization → residual update → next-input normalization |
| [moe_expert_pipeline](moe_expert_pipeline/README.md) | Routed tokens → preparation → expert gate/up, activation and down projection → combined output |
| [attention_output_moe](attention_output_moe/README.md) | Attention output + residual → output projection/TP completion → routing/experts → reduced MoE contribution + residual |
| [decoder_layer](decoder_layer/README.md) | Layer input/residual and KV state → attention + FFN → next-layer representation and state |
| [decoder_stack](decoder_stack/README.md) | Step inputs and committed state → groups of layers → hidden output, residuals and verify state |

## Reading a case

Each chain README states its boundary and links cases, topic notes and constituent operators. Cases own model-specific revisions, shapes, numerical rules, physical launches and evidence limits. Topic pages hold reusable contracts; `corpus/decisions.md` records conditioned choices and validation plans. Shared source descriptions and availability live in `_sources/`.

For example, [MoE dataflow](moe_expert_pipeline/dataflow.md) explains token/slot ownership; its [Muse Spark case](moe_expert_pipeline/cases/muse_spark_decode_dataflow.md) supplies concrete buffers and a worked storage calculation. [Fusion patterns](fusion_patterns.md) connect such cases to candidate boundaries and whole-chain measurement.

## Evidence vocabulary

| Description | What it establishes |
| --- | --- |
| Semantic chain | Required computation; a decomposed diagram is not a measured baseline. |
| Implemented contract | Source establishes representation, dependencies, ownership or dispatch. |
| Implemented fusion | Named operations share a kernel body or a producer writes the consumer's required representation; state which launches and buffers remain. |
| Joint measurement | Several kernels are timed as one callable; they may retain global intermediates. |
| Candidate / hypothesis | A proposed change with explicit conditions and a validation experiment. |

`fusion`, `persistent`, `megakernel` and `monokernel` are nonexclusive implementation labels. Source launch boundaries are not profiler counts, runtime engagement or measured eliminated launches. Implementation confidence is separate from effect and independent attribution. The current decision schema records source-only candidates with unknown effects.

Performance claims follow the [repository evidence contract](../README.md#performance-evidence). An already-fused baseline does not measure the benefit of introducing fusion. Use a comparable serving A/B before attributing an end-to-end gain.

## Adding knowledge

Use the [maintenance guide](../../tools/workload/README.md) for semantic ownership, naming, source/catalog schemas and checks; the [case template](_templates/workload_case.md) and optional [fusion supplement](_templates/fusion_case.md) define case content. Keep one owner per shared explanation and link it from other cases.
