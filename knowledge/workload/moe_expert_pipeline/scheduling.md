# MoE scheduling and distributed completion

Build the dependency graph from tensor producers and consumers. Launch adjacency and individual kernel duration are insufficient to determine whether stages can overlap or whether an overlap improves the complete pipeline.

## Stage dependencies

Input quantization and route-count construction may be logically independent given their inputs. Route placement needs the resulting block offsets; expert gate/up computation needs both quantized input and route placement; down projection needs the expert activation; finalization needs per-slot outputs. A schedule that overlaps work must honor all of these edges.

The [Muse Spark dataflow case](cases/muse_spark_decode_dataflow.md) has five ordered launches. Its preparation launch computes both quantization and route/block metadata. Splitting or overlapping these tasks would be a new schedule to evaluate, including its extra launch cost, resource contention and synchronization. No overlap is inferred from this source.

## Routing distribution and useful work

For local expert route counts `c_e` and tile height `B`, the number of padded route rows is `sum_e(B * ceil(c_e / B))`. Record this alongside the number of useful local routes `sum_e(c_e)`. Different routing distributions with the same token count can produce different padding, expert weight reuse and load balance.

The Muse Spark implementation launches against a capacity bound and guards unused expert blocks. The count of active/padded blocks and the launched grid are therefore distinct quantities. Its gate/up and down stages use the same slot map. Changing sorting or tile allocation must preserve that mapping and finalizer ownership.

A more balanced assignment, less padding or more overlap is an optimization hypothesis until measured with representative routing and the same semantic outputs. Include skewed routing and nearly empty local ranks in the experiment.

## Communication boundary

Name the rank scope of the output. A local weighted sum over owned experts is still a contribution to the distributed result. The inspected Muse Spark `run_compact_a4` ends at that local result and describes it as preceding TP reduction. Its wrapper does not contain the collective; the source does not establish its algorithm, timing or overlap.

Finalization may include per-expert normalization before route-weighted accumulation. Moving reduction or normalization across this boundary can change nonlinear operations and floating-point rounding. A communication change requires the downstream caller, process-group/rank ownership, stream/event dependencies and a distributed correctness comparison.

When comparing schedules, separately name the local-compute boundary and the boundary that includes distributed completion. Include collective time and rank skew for the latter; report critical-path elapsed time when work overlaps.

Related: [dataflow](dataflow.md), [dispatch](dispatch.md), [measurement](measurement.md), [MoE dispatch/combine](../../operator/moe_dispatch_combine/README.md).
