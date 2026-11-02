# Fusion case supplement

Start with the [workload case template](workload_case.md), include `fusion` in the case's `topics`, and add the details below. Cases about dataflow, scheduling, dispatch, memory or communication can use the general template directly.

## Fusion boundary

Show both the semantic operations and the physical launches. Name which operations share a body, which materializations disappear, and which tensors or synchronization boundaries remain. A decomposed diagram is an illustrative reference unless it names an actual measured baseline.

Explain ownership of the fused data and any tradeoff in parallelism, resource pressure, recomputation or backend choice. Preserve rounding points, side effects and companion kernels in the correctness and timing boundary.

## Comparison

When measurements exist, identify baseline/candidate revisions, complete-chain latency, observed launch counts, intermediate footprint/traffic, correctness, repetition/noise and workload weights. Label logical bytes separately from measured traffic. Separate introducing fusion from tuning an already-fused implementation, and report serving engagement and end-to-end results independently.

Record negative results and remaining limitations with the same evidence standard. Source inspection alone supports implementation structure, not a speedup claim.
