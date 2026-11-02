# Fusion patterns across a workload

These patterns connect the [general fusion strategy](../general/optimization/kernel_fusion_strategy.md) to concrete workload boundaries. They describe opportunities and tradeoffs; each implementation needs its own shape contract and evidence.

## Patterns and examples

| Pattern | Boundary to examine | Benefit to investigate | Constraint | Example / operator reference |
| --- | --- | --- | --- | --- |
| Consume GEMM partials in a fused epilogue | Split-K partials → reduction → normalization / residual / cache write | Remove a standalone reduction output and subsequent passes while retaining a parallel GEMM | Nonlinear operations generally need the fully reduced value; partial buffers and the producer/consumer launch boundary remain | [Muse QKV](qkv_projection_norm_rope_cache/cases/muse_spark_decode.md), [Muse unpool](projection_norm_residual/cases/muse_spark_decode.md); [GEMM epilogues](../operator/gemm_epilogue_fused/README.md) |
| Pack projections with a common input | Separate projections of X → joint packed-weight projection | Reduce dispatches and create an opportunity to reuse input data | Packing cost, precision, GEMM geometry and branch-specific backends; shared input does not guarantee one physical read | [Muse router/pooler](router_pooler_topk/cases/muse_spark_decode.md); [dense GEMM](../operator/dense_gemm/README.md) |
| Produce the consumer's quantized format | GEMM → activation → quantization → next GEMM | Avoid storing full-precision activation before converting it | Scale granularity, packing layout and rounding must match the consumer | [Muse compact MoE](moe_expert_pipeline/cases/muse_spark_decode.md); [quantized GEMM](../operator/scaled_quant_gemm/README.md) |
| Fuse state preparation | Q/K projection output → norm → RoPE → KV cache writes | Eliminate repeated passes over Q/K and redundant cache preparation | Cache slot ownership, padding, index dtype, full/SWA mapping and quantization scales | [Muse QKV](qkv_projection_norm_rope_cache/cases/muse_spark_decode.md); [Q/K preparation](../operator/qk_norm_rope_cache/README.md) |
| Fuse a decoder boundary | Unpool/projection → branch norm → residual update → next input norm | Keep row intermediates within the epilogue | Exact residual dtype and rounding; the next norm needs the complete row | [Muse unpool](projection_norm_residual/cases/muse_spark_decode.md) |
| Keep a split/reduce pipeline | Partitioned attention → merge partial state | Retain parallelism when one unsplit workgroup would underfill the GPU | Removing a launch may lose parallelism or require a costly global synchronization protocol | [Muse split-KV](#source-example-split-kv-and-adjacent-kernels); [paged attention](../operator/attention_decode_paged/README.md) |

## Reason about the actual boundary

Start from the live caller and trace, then inspect the producer and consumer source. Record which output is consumed, whether other consumers need it, and whether their grids can share ownership of the data. Same-stream adjacency identifies a candidate but does not prove a dependency, legal fusion or an eliminated launch. Graph replay reduces host submission overhead; it does not itself fuse device kernels or remove intermediate tensors.

For an intermediate with `n` elements of `b` bytes, eliminating one complete write and one complete read removes `2*n*b` bytes of **logical traffic**. This is not a measured HBM reduction: caching, spills, layout conversion, remaining consumers and recomputation can change physical traffic. A split-K pipeline must still account for its partial buffer. Count any final outputs that remain part of the API.

Fusion can increase register/LDS pressure, reduce occupancy, duplicate input reads or replace a faster library GEMM with a slower custom kernel. Row reductions, top-k and cross-rank collectives impose dependencies that an elementwise epilogue does not. Preserve collective placement and rank ownership unless the change explicitly implements and validates a distributed protocol.

## Source example: split-KV and adjacent kernels

**Evidence:** `swa_source`, `c8_report` in the [Muse Spark c8 source ledger](_sources/muse_spark_c8.yaml); see the [source context](_sources/muse_spark_c8.md). These are inspected-source and extraction-report observations, with no new measurement.

The SWA verification wrapper launches `_swa_verify_partials` and then `_swa_verify_reduce`. Partial output and softmax state are materialized between them. The source asserts eight query tokens per sequence, head dimension 64, a Q/KV head ratio of eight and page size 16. Partial workgroups compute attention with causal and left-window masks; the second kernel combines their states. This is an attention pipeline measured jointly, with the split/merge boundary retained. Collapsing it needs a separate parallelism, synchronization and performance argument.

The extraction also found norm/gate followed by output projection on the same stream. Its final task split records that these are **sequential operations**, not alternative dispatch branches. Adjacency is useful for proposing a fusion experiment; it does not prove an implemented fused norm/GEMM kernel. The same caution applies to the adjacent `_prepare_a4` and `_scatter_a4` launches in the [compact MoE case](moe_expert_pipeline/cases/muse_spark_decode.md).

Split-KV is an implementation strategy within [paged attention](../operator/attention_decode_paged/README.md). It does not define a separate workload semantic category.

## Measure the chain and its workload contribution

Measure each complete caller under the serving phase, shape distribution and graph mode. Preserve stream/event dependencies in the benchmark. Include cache and residual mutations in correctness checks, not only returned tensors. Use actual routing distributions for MoE and cover fallback dispatches; synthetic cases and median context lengths have narrower evidence scope.

For cases `j` with observed call weights `w_j`, an aggregate chain speedup can be defined as `sum(w_j * baseline_time_j) / sum(w_j * candidate_time_j)`. State the definition; averaging per-case speedup ratios answers a different question. Give correctness-only envelope cases zero performance weight.

If a chain accounts for a fraction `f` of **non-overlapping baseline elapsed time**, a speedup `s` gives the fixed-workload Amdahl estimate `1 / ((1-f) + f/s)`. Sums of overlapping GPU activity, shared kernels charged to several callers and changes to serving schedules do not satisfy that model. Report those estimates separately from measured end-to-end throughput, TTFT and inter-token latency. A paired, comparable serving run is still needed.

Related methods: [profiling](../general/profile_bottleneck/README.md), [numerical stability](../general/optimization/numerical_stability.md), [kernel integration](../general/workflows/integrating_a_new_kernel.md). The historical [end-to-end workflow](../general/workflows/optimize_e2e_model.md) has its own TP=1 experiment rules; preserve the actual workload's TP/EP when documenting or evaluating a distributed case.
