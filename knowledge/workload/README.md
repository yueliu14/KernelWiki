# Workload-level kernel knowledge

This collection organizes multi-operator execution subgraphs by their **semantic inputs, transformations and outputs**. It records data contracts, scheduling, dispatch, memory/state management, communication, fusion and whole-subgraph measurement. A subgraph may include branches and joins. The same semantic computation can appear in several models and can be implemented with different numbers of GPU launches.

Start with a semantic chain below, the [non-fusion MoE dataflow example](moe_expert_pipeline/cases/muse_spark_decode_dataflow.md), or the [fusion patterns](fusion_patterns.md). Backend implementation details remain in the [operator collection](../operator/README.md); reusable profiling and tuning methods remain in [general knowledge](../general/README.md).

The [case and decision index](CORPUS.md) adds searchable phase/backend/execution-form metadata
and source-grounded fusion decisions. The initial public monokernel review covers DeepSeek-V4.1-Flash,
MiniMax-M3.1-Flash and Qwen3.8-Flash-Next at one fixed sgl-amd-mono commit. Implementation evidence,
effect confidence and independent attribution are kept separate; no new GPU results are claimed.

## Semantic chains

| Semantic chain | Scope | Initial case |
| --- | --- | --- |
| [qkv_projection_norm_rope_cache](qkv_projection_norm_rope_cache/README.md) | Hidden-state projection → Q/K normalization and position encoding → attention inputs and KV state | [Muse Spark decode](qkv_projection_norm_rope_cache/cases/muse_spark_decode.md) |
| [router_pooler_topk](router_pooler_topk/README.md) | Shared input → routing projection/top-k and pooler projection/norm → expert IDs, weights and input | [Muse Spark decode](router_pooler_topk/cases/muse_spark_decode.md) |
| [projection_norm_residual](projection_norm_residual/README.md) | Branch projection → normalization → residual update → next-input normalization | [Muse Spark decode](projection_norm_residual/cases/muse_spark_decode.md) |
| [moe_expert_pipeline](moe_expert_pipeline/README.md) | Routed tokens → preparation → expert gate/up, activation and down projection → combined output | [Fusion boundaries](moe_expert_pipeline/cases/muse_spark_decode.md) · [Token reuse and execution contracts](moe_expert_pipeline/cases/muse_spark_decode_dataflow.md) |
| [attention_output_moe](attention_output_moe/README.md) | Attention output + residual → output projection/TP completion → routing/experts → reduced MoE contribution + residual | [MiniMax-M3.1-Flash](attention_output_moe/cases/minimax_m31_decode.md) |
| [decoder_layer](decoder_layer/README.md) | Layer input/residual and KV state → attention + FFN → next-layer representation and state | [DeepSeek-V4.1-Flash](decoder_layer/cases/deepseek_v41_decode.md) |
| [decoder_stack](decoder_stack/README.md) | Step inputs and committed state → groups of layers → hidden output, residuals and verify state | [Qwen3.8-Flash-Next](decoder_stack/cases/qwen38_flash_next_verify.md) |

The initial cases use reviewed extracted source and local metadata. They introduce no GPU measurements or isolated fusion-speedup claim.

## Directory structure and naming

```text
workload/
  README.md
  catalog.yaml
  CORPUS.md                        # generated case/decision index
  fusion_patterns.md
  _templates/workload_case.md
  _templates/fusion_case.md
  _sources/<source_set>.md
  _sources/<source_set>.yaml
  <semantic_chain>/
    README.md
    dataflow.md
    scheduling.md
    dispatch.md
    measurement.md
    cases/<case>.md
    corpus/decisions.yaml           # where reusable decisions have been curated
    corpus/decisions.md             # generated
```

Name the directory after the composed computation, keeping its external inputs and outputs explicit. Use an existing category when a new case has the same composition. Model names, GPU generations, dtypes, prefill/decode and backend names belong in case filenames or metadata. Split-K, kernel counts and particular fusion schedules are implementation choices within the category; changing them does not rename the semantic chain.

Create topic files when there is substantive knowledge to record; a short section in the README is sufficient otherwise. `dataflow.md` can include memory/state lifetime, and `scheduling.md` can include communication. Separate `memory.md`, `communication.md` or `fusion.md` when those topics grow. Case names can include a topic suffix, such as `muse_spark_decode_dataflow.md`, and their `topics` metadata supports discovery across these concerns.

For example, `operator/qk_norm_rope_cache/` describes Q/K preparation implementations. `workload/qkv_projection_norm_rope_cache/` connects the preceding projection to that preparation and the cache side effects. Each workload README states its boundary and links the constituent operators. Whole-model serving reports can continue to live in [general case studies](../general/case_studies/README.md).

## Knowledge topics

| Topic | Questions to answer | Example |
| --- | --- | --- |
| Data contracts | Which layouts, dtypes, scales, indices and rounding points must agree across producers and consumers? | [MoE token versus slot representation](moe_expert_pipeline/dataflow.md) |
| Scheduling | Which stages depend on others, which branches can overlap, and which resources do they share? | [MoE preparation, scatter, expert and finalize dependencies](moe_expert_pipeline/scheduling.md) |
| Dispatch | How do phase, token count, layouts and routing distribution select a complete execution path? | [MoE eligibility and fallback accounting](moe_expert_pipeline/dispatch.md) |
| Memory and state | When is a buffer last consumed, when can storage be reused, and who owns mutated state? | [MoE buffer lifetimes and unwritten slots](moe_expert_pipeline/dataflow.md#memory-lifetime-and-state) |
| Communication | Is an output rank-local or globally complete, and where must synchronization occur? | [Local combine and distributed completion](moe_expert_pipeline/scheduling.md#communication-boundary) |
| Measurement | What is the timed subgraph boundary, its critical path, dispatch coverage and contribution to serving? | [MoE measurement protocol](moe_expert_pipeline/measurement.md) |
| Fusion | Which operations or materializations can be combined, and what parallelism or numerical constraints limit them? | [Fusion patterns](fusion_patterns.md) |

Useful knowledge includes implemented execution contracts, positive or negative measurements, and proposed optimizations. A change can improve layout reuse, path selection or overlap while preserving the number of launches. Retain correct execution contracts even when no optimization has been measured.

Use these descriptions precisely:

| Description | What it establishes |
| --- | --- |
| Semantic chain | The operations required by the model; a decomposed diagram is not necessarily a measured implementation. |
| Implemented execution contract | Source establishes representation, dependencies, ownership or dispatch. It does not establish a performance benefit. |
| Implemented fusion | Source shows operations sharing a kernel body or a producer writing directly in the consumer's required representation. State which operations are fused and which launches remain. |
| Joint measurement | Multiple kernels are timed together as one callable. They may still communicate through global-memory tensors. |
| Fusion candidate | A dependency or repeated adjacency suggests a boundary to investigate; feasibility and benefit remain unverified. |
| Optimization hypothesis | A proposed change to dataflow, scheduling, dispatch, memory, communication or fusion, with a stated validation experiment. |

## Adding knowledge

Create `workload/<semantic_chain>/README.md` with the model-independent contract, topic navigation and constituent operator links. Add focused `cases/` pages using the [workload case template](_templates/workload_case.md); use the [fusion supplement](_templates/fusion_case.md) when appropriate. Index each case in its chain README. Keep model-specific constraints and reviewed outcome summaries in the case. Shared public comparison context and evidence availability can live under `_sources/` so several cases can cite the same source IDs. Link new semantic chains here and keep documents in English.

Each case should identify the model/stack revision, phase, shapes, dtypes, TP/EP, graph mode, caller and actual device entry points. Show the semantic chain and implementation boundaries separately. Include materialized intermediates, numerical rounding, mutation/aliasing and synchronization requirements, plus the dispatch branches the evidence does not cover.

Register cases in `catalog.yaml`; keep its model/phase/owner/source metadata consistent with the
case front matter. Execution forms (`fusion`, `staged`, `persistent`, `megakernel`, `monokernel`)
are nonexclusive labels. Backend names describe actual implementations, not the upstream term
"monokernel". An empty backend list means it has not been classified in this index.

Distill reusable choices into the owning semantic chain's `corpus/decisions.yaml`. Schema 1 records
source-only decisions with explicit conditions, alternatives, validation plans and unknown effects.
It does not accept inferred performance metrics. Use [the workload maintenance guide](../../tools/workload/README.md)
for validation, generated pages and the optional pinned Git source-location audit.

Label source inspection, reported measurements and measurements reproduced for the entry separately.
For privately inspected sources, publish an evidence ID, its role, availability and verification status.
Keep internal paths, file digests, raw reports and the mapping from IDs to run artifacts in a private
ledger. A private-source ID attributes an inspection; it does not let public readers reproduce it.
Prefer pinned public source links when available. An optimization run against an already-fused baseline
does not measure the benefit of introducing that fusion.

The `_sources/*.yaml` index uses `schema_version: 1`, `reviewed_on`, `evidence_kind`, `access` and
`sources`. Each source has `id`, `kind`, `availability`, `verification` and `supports`; a public source
also has a `reference` URL pinned to a full Git commit. Use `availability: private` with
`verification: curator_inspected` for existing private inspections, or `public` with `source_linked`
for a public source citation. Neither status certifies a benchmark or a performance winner.

Performance claims require the [repository evidence contract](../README.md#performance-evidence). Report latency for the complete chain as well as any relevant kernel breakdown, and use an end-to-end A/B before claiming a serving gain. Track shared kernels by call site so the same time is not counted twice.
