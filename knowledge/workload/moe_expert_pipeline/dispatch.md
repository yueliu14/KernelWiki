# MoE workload dispatch and coverage

Dispatch selects a complete execution path whose preparation, expert backend and finalizer agree on their interfaces. A faster expert kernel is useful only for calls routed to that compatible path.

## Three contracts to record

| Contract | Evidence needed | Question |
| --- | --- | --- |
| Implementation eligibility | Source assertions/predicates | Can this implementation accept the tensor shape, dtype, weight/scale layout and expert ownership? |
| Caller dispatch | Actual model/backend caller and flags | Does the serving path choose it, and which fallback handles other inputs? |
| Observed workload coverage | Per-phase call/shape/routing records | What fraction of the measured workload actually takes that path? |

The [Muse Spark case](cases/muse_spark_decode_dataflow.md#dispatch-evidence-and-proposed-measurement) records these separately: the source predicate, extracted verification buckets and an unestablished caller fallback. Supported sizes do not establish observed coverage or a prefill route.

## Selecting and comparing paths

Include phase, token rows, top-k, expert ownership, activation/weight format and graph mode in the dispatch context. Record how the model turns request batch size into rows: speculative verification can process multiple tokens per request. Routing skew is another workload dimension even when a branch predicate uses only row count.

When choosing between expert backends, compare preparation/conversion, expert computation and finalization together. Include weight packing as setup cost or runtime cost according to where it actually occurs. Validate the producer/consumer contracts before comparing timings.

Report fallback cases and their workload weight. Give correctness-only envelope probes zero performance weight, and keep source-supported shapes separate from observed shapes. A UT's synthetic routing coverage is not evidence for every serving expert distribution.

Related: [scheduling](scheduling.md), [measurement](measurement.md), [grouped GEMM / MoE](../../operator/grouped_gemm_moe/README.md).
