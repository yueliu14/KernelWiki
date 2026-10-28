# Muse Spark workload knowledge

Muse Spark illustrates why a workload-level view is useful: a callable may span several device launches while fusing projection reduction, normalization, routing or state updates inside one of them. The [decode fusion chains](decode_fusion_chains.md) document those boundaries from the extracted implementation.

## Scope

The first entry covers `Muse-Spark-Open-816B-A42B-Quark-MXFP4` on 8 MI355X GPUs (`gfx950`), SGLang TP8/EP8, concurrency 8, DFLASH with eight draft tokens and FP8 KV cache. Target verification is classified as decode in the extraction metadata and runs under HIP graphs; prefill runs eagerly. In the observed verification buckets, `M = 8 * batch_size`.

The source snapshot is the `muse-spark-open-1020` image with digest `sha256:7343002d0dc2c1e525c0563f0c5c0377b471f966b3d604343b93a6366c2f7935`. The extraction records simulated acceptance length `4.05` with real draft tokens; this setting is part of its workload context. See [sources.yaml](sources.yaml) (`c8_environment`) for the artifact location and hash, and the [InferenceX reference run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428437135) for the cited workload reference.

## Entries

| Execution chain | What is implemented |
| --- | --- |
| [QKV projection and cache preparation](decode_fusion_chains.md#qkv-projection-and-cache-preparation) | Split-K projection followed by one reduction/norm/RoPE/cache epilogue |
| [Joint router and pooler](decode_fusion_chains.md#joint-router-and-pooler) | Packed projection feeding a shared top-k and pooler-normalization epilogue |
| [Unpool projection and layer boundary](decode_fusion_chains.md#unpool-projection-and-layer-boundary) | GEMM partials consumed by branch norm, residual update and next-input norm |
| [Compact MXFP4 MoE](decode_fusion_chains.md#compact-mxfp4-moe) | Activation/requantization inside expert computation, with preparation, scatter, down projection and finalize still separate |
| [Split-KV and adjacent kernels](decode_fusion_chains.md#split-kv-and-adjacent-kernels) | A retained partial/merge pipeline and an example of adjacency without established fusion |

## Evidence boundaries

These are source observations checked on 2026-10-28, supported by local extraction metadata. The [source ledger](sources.yaml) pins the inspected files by SHA-256. The extraction package was supplied by hongtaom and is available in the local `musespark_c8_opt_1023_hongtaom` workspace; its files are not bundled here. The CI link identifies the reference workload and does not independently substantiate every implementation detail in this entry.

No GPU benchmark was run for this documentation change. The extraction report describes baseline parity on seeded synthetic inputs and reconstructed metadata; it does not establish correctness on captured serving activations. It also reports incomplete coverage for some caller branches. Those limits travel with the cases.

The separate `muse_spark_opt_1021` handoff uses image digest `sha256:72065dfc0b87818ab415f96c0fc08848233d25f22b002ac8f28c33fad6ec82a4` (`earlier_handoff` in the ledger). Its kernel-tuning and serving results belong to that stack. Neither those results nor speedups over the already-fused c8 implementation isolate the benefit of introducing a particular fusion. Future fusion ablations should be added with their own baseline, candidate and measurements.

Related: [workload index](../README.md), [fusion patterns](../fusion_patterns.md), [case template](../_templates/fusion_case.md).
