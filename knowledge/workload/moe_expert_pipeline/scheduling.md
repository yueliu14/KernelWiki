# MoE scheduling and distributed completion

Build the dependency graph from tensor producers and consumers. Launch adjacency and individual kernel duration are insufficient to determine whether stages can overlap or whether an overlap improves the complete pipeline.

## Stage dependencies

Input quantization and route-count construction may be logically independent given their inputs. Route placement needs the resulting block offsets; expert gate/up computation needs both quantized input and route placement; down projection needs the expert activation; finalization needs per-slot outputs. A schedule that overlaps work must honor all of these edges.

See the [Muse Spark schedule](cases/muse_spark_decode_dataflow.md#scheduling-and-communication) for actual ordering. Splitting or overlapping preparation tasks would require evaluating extra launches, resource contention and synchronization.

## Routing distribution and useful work

For local expert route counts `c_e` and tile height `B`, the number of padded route rows is `sum_e(B * ceil(c_e / B))`. Record this alongside the number of useful local routes `sum_e(c_e)`. Different routing distributions with the same token count can produce different padding, expert weight reuse and load balance.

Active/padded blocks and capacity grids are distinct quantities; the case gives a concrete bound. Changing sorting or tile allocation must preserve the consumer slot mapping and finalizer ownership.

A more balanced assignment, less padding or more overlap is an optimization hypothesis until measured with representative routing and the same semantic outputs. Include skewed routing and nearly empty local ranks in the experiment.

## Communication boundary

Name the rank scope of the output. A local weighted sum over owned experts is still a contribution to the distributed result. A wrapper without the collective cannot establish its algorithm, timing or overlap; inspect the downstream caller.

Finalization may include per-expert normalization before route-weighted accumulation. Moving reduction or normalization across this boundary can change nonlinear operations and floating-point rounding. A communication change requires the downstream caller, process-group/rank ownership, stream/event dependencies and a distributed correctness comparison.

When comparing schedules, separately name the local-compute boundary and the boundary that includes distributed completion. Include collective time and rank skew for the latter; report critical-path elapsed time when work overlaps.

Related: [dataflow](dataflow.md), [dispatch](dispatch.md), [measurement](measurement.md), [MoE dispatch/combine](../../operator/moe_dispatch_combine/README.md).
