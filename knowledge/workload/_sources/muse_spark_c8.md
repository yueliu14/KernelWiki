# Muse Spark c8 source context

This source context is shared by the semantic-chain cases listed below. Model and run names identify evidence; the knowledge collection is organized by semantic chains.

## Scope

The first entry covers `Muse-Spark-Open-816B-A42B-Quark-MXFP4` on 8 MI355X GPUs (`gfx950`), SGLang TP8/EP8, concurrency 8, DFLASH with eight draft tokens and FP8 KV cache. Target verification is classified as decode in the extraction metadata and runs under HIP graphs; prefill runs eagerly. In the observed verification buckets, `M = 8 * batch_size`.

The source snapshot is the `muse-spark-open-1020` image with digest `sha256:7343002d0dc2c1e525c0563f0c5c0377b471f966b3d604343b93a6366c2f7935`. The extraction records simulated acceptance length `4.05` with real draft tokens; this setting is part of its workload context. See [evidence index](muse_spark_c8.yaml) (`c8_environment`) for availability and inspection status, and the [InferenceX reference run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428437135) for the already-public workload reference.

## Referencing cases

| Semantic chain | Source-specific case |
| --- | --- |
| QKV projection, normalization and cache preparation | [Muse Spark decode](../qkv_projection_norm_rope_cache/cases/muse_spark_decode.md) |
| Router, pooler and top-k | [Muse Spark decode](../router_pooler_topk/cases/muse_spark_decode.md) |
| Projection, normalization and residual update | [Muse Spark decode](../projection_norm_residual/cases/muse_spark_decode.md) |
| MoE expert pipeline | [Muse Spark decode](../moe_expert_pipeline/cases/muse_spark_decode.md) |
| MoE dataflow, memory and execution contracts | [Muse Spark token reuse](../moe_expert_pipeline/cases/muse_spark_decode_dataflow.md) |
| Retained split/merge and adjacent kernels | [Fusion boundary examples](../fusion_patterns.md#source-example-split-kv-and-adjacent-kernels) |

## Evidence boundaries

These are source observations checked on 2026-10-28, supported by privately inspected source and
extraction metadata. The [evidence index](muse_spark_c8.yaml) identifies each source's role and
availability. Internal locations, file digests and the mapping to raw artifacts are retained privately.
The source contents are not published or independently reproduced here. The CI link identifies the
reference workload and does not independently substantiate every implementation detail in this entry.

No GPU benchmark was run for this documentation change. The extraction report describes baseline parity on seeded synthetic inputs and reconstructed metadata; it does not establish correctness on captured serving activations. It also reports incomplete coverage for some caller branches. Those limits travel with the cases.

Results from other stacks and speedups over the already-fused implementation do not isolate the
benefit of introducing a particular fusion. Future fusion conclusions need their own explicit
baseline, candidate, conditions and validation summary; raw ablation runs remain private.

`_dense_partial` is shared by the QKV, router and unpool call sites. Attribute time using the caller/branch before summing workload savings; a kernel-name total cannot be independently charged to all three. The same rule applies to companion epilogues and partial/reduce pairs selected as a group.

Related: [workload index](../README.md), [fusion patterns](../fusion_patterns.md), [workload case template](../_templates/workload_case.md).
