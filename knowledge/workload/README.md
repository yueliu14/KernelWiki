# Workload-level kernel knowledge

This collection records how kernels work together within an LLM inference workload: the dataflow through a model block, which boundaries are fused, which intermediates still reach device memory, and how dispatch changes with serving conditions. A workload can be a model, a decoder block or a serving path such as speculative verification.

Start with the [fusion patterns](fusion_patterns.md), then follow a concrete workload. Backend implementation details remain in the [operator collection](../operator/README.md); reusable profiling and tuning methods remain in [general knowledge](../general/README.md).

## Workloads

| Workload | Scope | Evidence |
| --- | --- | --- |
| [Muse Spark](muse_spark/README.md) | Small-batch target verification: QKV preparation, router/pooler, unpool boundary, compact MXFP4 MoE and split-KV attention | Reviewed extracted source and local extraction metadata; no new GPU measurements or isolated fusion speedup claim |

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

Create `workload/<workload_name>/README.md` as the entry point and add focused case pages using the [case template](_templates/fusion_case.md). Link the workload here and link its constituent operator pages from the case. Keep new documents in English.

Each case should identify the model/stack revision, phase, shapes, dtypes, TP/EP, graph mode, caller and actual device entry points. Show the semantic chain and implementation boundaries separately. Include materialized intermediates, numerical rounding, mutation/aliasing and synchronization requirements, plus the dispatch branches the evidence does not cover.

Label source inspection, reported measurements and measurements reproduced for the entry separately. For local-only artifacts, record their location and content digest, and state that readers need access to those artifacts. Prefer pinned public source links when available. An optimization run against an already-fused baseline does not measure the benefit of introducing that fusion.

Performance claims require the [repository evidence contract](../README.md#performance-evidence). Report latency for the complete chain as well as any relevant kernel breakdown, and use an end-to-end A/B before claiming a serving gain. Track shared kernels by call site so the same time is not counted twice.
