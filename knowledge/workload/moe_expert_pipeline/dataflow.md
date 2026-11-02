# MoE data contracts and intermediate memory

The useful unit of reuse changes across the expert pipeline. Before expert-dependent computation, several selected experts can consume the same token representation. After applying different expert weights, their activations and outputs generally need distinct storage. Identify those boundaries before deciding whether to copy, gather, pack or quantize data.

## Token rows and routing slots

For `M` tokens and top-k `K`, a routing slot identifies a `(token, selected expert)` pair. With token-major slots, `slot = token * K + choice` and `token = slot // K`. Expert-major scheduling can store a permutation of slot IDs while preserving this identity. Record sentinels, duplicate/tie behavior and padding as part of the contract.

A per-token quantized input can be referenced through that map when quantization is independent of the selected expert. This permits reuse without materializing one activation copy per route. It still allows repeated device reads by different workgroups. Per-expert scales, preprocessing or stochastic conversion can invalidate equivalence and must be checked separately.

Gate/up results depend on expert weights. Their post-activation quantized values and scales must retain the slot identity expected by the down projection and output combination. Equal tensor dimensions do not make a per-token buffer interchangeable with a per-slot buffer.

The [Muse Spark dataflow case](cases/muse_spark_decode_dataflow.md) shows this directly: input quantization uses `M` rows, expert activations use `8*M` slot rows, and the consumer selects its row with `slot // 8` or `slot` respectively.

## Memory lifetime and state

Track each buffer from its producer through its final consumer, including outstanding work on other streams. Reuse becomes legal only after that last use. A smaller logical representation, storage aliasing and reduced physical traffic are separate claims.

- Keep routing metadata alive through all expert stages that read it, and IDs/route weights through finalization.
- Preserve input quantization values and scales together until the gate/up stage completes.
- Keep expert activation values/scales until down projection completes, and per-slot output until finalization completes.
- Do not assume every row of an allocated slot buffer was written. Nonlocal and padded slots may be unspecified; downstream consumers must preserve the ownership/mask contract.
- Graph capture and concurrent invocations constrain scratch reuse. Record stable addresses, capture buckets and event dependencies before introducing a shared workspace. An `empty` allocation in source does not by itself prove a per-replay allocation cost.

In the Muse Spark source, the finalizer checks expert ownership before reading each per-slot row. A consumer that reads every allocated row would violate that contract. The existing case documents allocations and lifetimes; it does not claim that workspace reuse was implemented or measured.

Related: [scheduling](scheduling.md), [measurement](measurement.md), [quantized GEMM](../../operator/scaled_quant_gemm/README.md), [dispatch/combine](../../operator/moe_dispatch_combine/README.md).
