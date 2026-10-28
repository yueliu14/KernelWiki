# Workload-level kernel knowledge

This collection organizes multi-operator execution chains by their **semantic inputs, transformations and outputs**. It records which boundaries are fused, which intermediates still reach device memory, and how dispatch changes with serving conditions. The same semantic chain can appear in several models and can be implemented with different numbers of GPU launches.

Start with a semantic chain below or the [fusion patterns](fusion_patterns.md). Backend implementation details remain in the [operator collection](../operator/README.md); reusable profiling and tuning methods remain in [general knowledge](../general/README.md).

## Semantic chains

| Semantic chain | Scope | Initial case |
| --- | --- | --- |
| [qkv_projection_norm_rope_cache](qkv_projection_norm_rope_cache/README.md) | Hidden-state projection → Q/K normalization and position encoding → attention inputs and KV state | [Muse Spark decode](qkv_projection_norm_rope_cache/cases/muse_spark_decode.md) |
| [router_pooler_topk](router_pooler_topk/README.md) | Shared input → routing projection/top-k and pooler projection/norm → expert IDs, weights and input | [Muse Spark decode](router_pooler_topk/cases/muse_spark_decode.md) |
| [projection_norm_residual](projection_norm_residual/README.md) | Branch projection → normalization → residual update → next-input normalization | [Muse Spark decode](projection_norm_residual/cases/muse_spark_decode.md) |
| [moe_expert_pipeline](moe_expert_pipeline/README.md) | Routed tokens → preparation → expert gate/up, activation and down projection → combined output | [Muse Spark decode](moe_expert_pipeline/cases/muse_spark_decode.md) |

The initial cases use reviewed extracted source and local metadata. They introduce no GPU measurements or isolated fusion-speedup claim.

## Directory structure and naming

```text
workload/
  README.md
  fusion_patterns.md
  _templates/fusion_case.md
  _sources/<source_set>.md
  _sources/<source_set>.yaml
  <semantic_chain>/
    README.md
    cases/<model_or_stack>_<regime>.md
```

Name the directory after the composed computation, keeping its external inputs and outputs explicit. Use an existing category when a new case has the same composition. Model names, GPU generations, dtypes, prefill/decode and backend names belong in case filenames or metadata. Split-K, kernel counts and particular fusion schedules are implementation choices within the category; changing them does not rename the semantic chain.

For example, `operator/qk_norm_rope_cache/` describes Q/K preparation implementations. `workload/qkv_projection_norm_rope_cache/` connects the preceding projection to that preparation and the cache side effects. Each workload README states its boundary and links the constituent operators. Whole-model serving reports can continue to live in [general case studies](../general/case_studies/README.md).

## What belongs here

- A producer/consumer chain with its tensors, ordering, state mutations and physical GPU launches.
- Fusion across GEMM, normalization, activation, quantization, routing or cache writes, including partial fusion that retains multiple launches.
- Shape- and phase-dependent dispatch, graph execution, distributed boundaries and fallback paths.
- Positive, negative and proposed fusion cases, with evidence sufficient to distinguish them.

Use these descriptions precisely:

| Description | What it establishes |
| --- | --- |
| Semantic chain | The operations required by the model; a decomposed diagram is not necessarily a measured implementation. |
| Implemented fusion | Source shows operations sharing a kernel body or a producer writing directly in the consumer's required representation. State which operations are fused and which launches remain. |
| Joint measurement | Multiple kernels are timed together as one callable. They may still communicate through global-memory tensors. |
| Fusion candidate | A dependency or repeated adjacency suggests a boundary to investigate; feasibility and benefit remain unverified. |

## Adding knowledge

Create `workload/<semantic_chain>/README.md` with the model-independent contract, fusion boundaries and constituent operator links. Add focused `cases/` pages using the [case template](_templates/fusion_case.md), and index each case in its chain README. Keep model-specific constraints and measured results in the case. Shared run context and source digests can live under `_sources/` so several cases can cite the same evidence without duplicating it. Link new semantic chains here and keep documents in English.

Each case should identify the model/stack revision, phase, shapes, dtypes, TP/EP, graph mode, caller and actual device entry points. Show the semantic chain and implementation boundaries separately. Include materialized intermediates, numerical rounding, mutation/aliasing and synchronization requirements, plus the dispatch branches the evidence does not cover.

Label source inspection, reported measurements and measurements reproduced for the entry separately. For local-only artifacts, record their location and content digest, and state that readers need access to those artifacts. Prefer pinned public source links when available. An optimization run against an already-fused baseline does not measure the benefit of introducing that fusion.

Performance claims require the [repository evidence contract](../README.md#performance-evidence). Report latency for the complete chain as well as any relevant kernel breakdown, and use an end-to-end A/B before claiming a serving gain. Track shared kernels by call site so the same time is not counted twice.
