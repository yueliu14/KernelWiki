# Measuring the complete MoE expert pipeline

Define the same external boundary for baseline and candidate: prepared hidden states plus IDs/route weights and expert parameters in, combined output out. State whether output is rank-local or includes distributed completion. A boundary change is part of the experiment and must be reported.

## Measurements that answer different questions

| Quantity | What it establishes |
| --- | --- |
| Logical allocation size | Representation footprint derived from shape and dtype; not an HBM-traffic or latency measurement |
| Stage/kernel timing | Where work is spent inside the pipeline; overlapping durations are not additive elapsed time |
| Complete local-pipeline latency | Cost through local finalization under the specified input, routing, cache and graph conditions |
| Distributed-pipeline elapsed time | Completion including the specified communication and rank skew |
| Serving throughput/latency | Workload effect after engagement, output correctness and comparable serving conditions are checked |

For the [Muse Spark token-reuse example](cases/muse_spark_decode_dataflow.md), the 17 KiB versus 136 KiB input-representation comparison is arithmetic against a hypothetical expanded buffer. It supplies no measured speedup. Gate/up workgroups can read the reused input repeatedly; actual cache/traffic behavior requires measurement.

## A concrete evaluation protocol

1. Freeze source/image revisions, graph mode, streams, layouts/scales, routing and numerical reference. Include local ownership, padding and unwritten-slot behavior in correctness checks.
2. Evaluate observed row-count buckets with their recorded weights, plus independent correctness probes for skew, padding and ranks with few or no local routes. Record the limits of synthetic inputs and routing.
3. Measure the entire local call, including route preparation, quantization, scatter and finalization. Record warmup, cache policy, repetitions/noise and the speedup aggregation definition. Treat conversion and allocation/setup costs consistently in eager and graph modes.
4. For a workspace or scheduling change, retain the same semantic outputs and timing boundary. Inspect peak live storage and state lifetimes; account for extra events, launches or retained scratch. Verify graph replay and concurrent-call isolation when they are in scope.
5. Inspect the caller and measure distributed completion before attributing any benefit to communication overlap. Recheck serving engagement, fallback coverage and end-to-end metrics before claiming a serving gain.

Changes to representation, backend choice or overlap may preserve all five launches in the original source. Judge their measured complete-pipeline effect. Label an unimplemented experiment as proposed, a source observation as inspected, and historical measurements with their own evidence and environment.

Related: [general measurement guidance](../fusion_patterns.md#measure-the-chain-and-its-workload-contribution), [profiling](../../general/profile_bottleneck/README.md), [repository evidence contract](../../README.md#performance-evidence).
