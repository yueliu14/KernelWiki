# Muse Spark c8 source context

This source context is shared by the semantic-chain cases listed below. Model and run names identify evidence; the knowledge collection is organized by semantic chains.

## Scope

The first entry covers `Muse-Spark-Open-816B-A42B-Quark-MXFP4` on 8 MI355X GPUs (`gfx950`), SGLang TP8/EP8, concurrency 8, DFLASH with eight draft tokens and FP8 KV cache. Target verification is classified as decode in the extraction metadata and runs under HIP graphs; prefill runs eagerly. In the observed verification buckets, `M = 8 * batch_size`.

The source snapshot is the `muse-spark-open-1020` image with digest `sha256:7343002d0dc2c1e525c0563f0c5c0377b471f966b3d604343b93a6366c2f7935`. The extraction records simulated acceptance length `4.05` with real draft tokens; this setting is part of its workload context. See [source ledger](muse_spark_c8.yaml) (`c8_environment`) for the artifact location and hash, and the [InferenceX reference run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428437135) for the cited workload reference.

## Referencing cases

| Semantic chain | Source-specific case |
| --- | --- |
| QKV projection, normalization and cache preparation | [Muse Spark decode](../qkv_projection_norm_rope_cache/cases/muse_spark_decode.md) |
| Router, pooler and top-k | [Muse Spark decode](../router_pooler_topk/cases/muse_spark_decode.md) |
| Projection, normalization and residual update | [Muse Spark decode](../projection_norm_residual/cases/muse_spark_decode.md) |
| MoE expert pipeline | [Muse Spark decode](../moe_expert_pipeline/cases/muse_spark_decode.md) |
| Retained split/merge and adjacent kernels | [Fusion boundary examples](../fusion_patterns.md#source-example-split-kv-and-adjacent-kernels) |

## Evidence boundaries

These are source observations checked on 2026-10-28, supported by local extraction metadata. The [source ledger](muse_spark_c8.yaml) pins the inspected files by SHA-256. The extraction package was supplied by hongtaom and is available in the local `musespark_c8_opt_1023_hongtaom` workspace; its files are not bundled here. The CI link identifies the reference workload and does not independently substantiate every implementation detail in this entry.

No GPU benchmark was run for this documentation change. The extraction report describes baseline parity on seeded synthetic inputs and reconstructed metadata; it does not establish correctness on captured serving activations. It also reports incomplete coverage for some caller branches. Those limits travel with the cases.

The separate `muse_spark_opt_1021` handoff uses image digest `sha256:72065dfc0b87818ab415f96c0fc08848233d25f22b002ac8f28c33fad6ec82a4` (`earlier_handoff` in the ledger). Its kernel-tuning and serving results belong to that stack. Neither those results nor speedups over the already-fused c8 implementation isolate the benefit of introducing a particular fusion. Future fusion ablations should be added with their own baseline, candidate and measurements.

`_dense_partial` is shared by the QKV, router and unpool call sites. Attribute time using the caller/branch before summing workload savings; a kernel-name total cannot be independently charged to all three. The same rule applies to companion epilogues and partial/reduce pairs selected as a group.

Related: [workload index](../README.md), [fusion patterns](../fusion_patterns.md), [case template](../_templates/fusion_case.md).
