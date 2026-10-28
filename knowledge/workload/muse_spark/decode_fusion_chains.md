# Muse Spark decode fusion chains

These cases describe the [c8 source snapshot and workload](README.md). Source IDs below refer to the content-addressed records in [sources.yaml](sources.yaml). Semantic chains explain the computation; they are not measured unfused baselines. Launch counts describe the inspected source path, not a new profiler measurement.

## QKV projection and cache preparation

**Evidence:** `qkv_source`, `qkv_metadata`, `qkv_provenance`.

```text
Semantic chain:
  QKV + output-gate projection -> split reduction -> Q/K norm -> optional RoPE
                              -> Q, K, V, gate outputs + scaled FP8 KV cache writes

Implemented (two launches):
  [_dense_partial] -> FP32 partials[8, M, 2304]
                  -> [_qkv_epilogue: reduction + norm + RoPE + outputs/cache]
```

`qkv_proj_qk_norm_rope_cache` accepts contiguous BF16 input `[M,8192]` and weights `[2304,8192]`, with `0 < M <= 32`. The projection includes 16 Q heads, two KV heads and 16 output-gate heads, each of dimension 64. Its epilogue reduces slices directly from the partial buffer; it does not first materialize a completed packed projection for a separate norm/RoPE kernel.

The buffer of split-K partials remains, as do the returned BF16 Q/K/V/gate tensors. The fusion therefore removes an intermediate boundary without making the whole chain a single launch. Cache writes occur in the same epilogue. Only the owning Q-head program writes each KV head, and negative cache locations or negative full-to-SWA mappings suppress writes.

The implementation explicitly rounds projection reduction, normalization and RoPE products through BF16. FP8 cache conversion uses those rounded K/V values and their scales. Fusing the math while keeping everything in FP32 would change the contract. Correctness must check cache mutations as well as returned tensors, including padded slots and index widths.

The task metadata covers observed `M=8,16,24,32` buckets and a zero-weight negative-slot probe. It states that larger batches use a separate projection plus Q/K preparation route. Coverage of this callable is not coverage of every attention-preparation path.

**Reusable lesson:** a split-K reduction is a useful attachment point for a row/head epilogue. Keep its partial-buffer cost in the comparison and preserve the original rounding and state-write rules. Related operators: [Q/K preparation](../../operator/qk_norm_rope_cache/README.md), [Split-K GEMM](../../operator/splitk_streamk_gemm/README.md).

## Joint router and pooler

**Evidence:** `router_source`, `router_metadata`, `c8_report`.

```text
Semantic branches sharing X:
  router projection -> sigmoid -> bias-corrected top-8 selection -> route weights/IDs
  pooler projection -> RMSNorm -> expert input

Implemented for M=8 or 16 (two launches):
  [_dense_partial with packed weights] -> FP32 partials[8, M, 4608]
    -> [_joint_epilogue: routing programs + pooler-normalization programs]
```

`pack_joint_router_pooler_weight` packs `[pooler; router_hi; router_lo]` into BF16 `[4608,8192]` weights. The router is supplied in FP32; the high and residual low BF16 parts are projected separately and combined in the epilogue. This precision contract is part of the joint-projection design.

`_joint_epilogue` uses different programs in one grid for the two branches. Routing programs perform split reduction, sigmoid, bias-corrected expert selection and normalized route weights. Pooler programs reduce, round to BF16 and normalize the 4096-wide expert input. The fused launch does not imply that the two branches communicate through shared memory, and the packed projection does not guarantee a single physical read of X across its workgroups.

The callable accepts `M=8,16,24`. `M=24` uses AITER `gemm_a16w16(..., skip_reduce=True)` as the producer; inspect that dependency's dispatch before assigning a device-launch count. The metadata says batch size four and above use another caller, `router_pooler_topk`. The extraction reports 93.01% coverage across the decode routing distribution, leaving that fallback outside this task.

Tie behavior, exact expert IDs and the distinction between biased selection scores and the unbiased scores used for route weights are observable outputs. Weight packing/setup costs also need an explicit placement in any benchmark.

**Reusable lesson:** combine projections with a common input when their shape and precision contracts permit it, then combine compatible epilogues. Record every producer backend and fallback separately. Related operators: [routing/top-k](../../operator/moe_routing_topk/README.md), [dense GEMM](../../operator/dense_gemm/README.md).

## Unpool projection and layer boundary

**Evidence:** `unpool_source`, `unpool_metadata`, `c8_report`.

```text
Semantic chain:
  expert output @ W_unpool.T -> branch input-scale RMSNorm
    -> alpha * residual_stream + beta * normalized_branch -> next-input RMSNorm

Implemented (two launches):
  [_dense_partial] -> FP32 partials[split_k, M, 8192]
    -> [_unpool_reduce_boundary: reduction + branch norm + residual + next norm]
    -> BF16 next input + updated FP32 residual stream
```

The epilogue owns a complete 8192-wide row, so it can reduce the GEMM splits and compute both norms without storing each intermediate row. It still stores the updated residual stream and the next input because both are externally visible outputs.

The source keeps BF16 rounding after the projection sum, after branch input scaling and after branch normalization. `pre_bf16` controls another rounding point before the next-input norm. Moving scales across normalization or dropping these conversions changes the computation.

The standalone wrapper accepts `0 < M <= 64`, input width 4096 and output width 8192, with split count 2, 4 or 8. The **serving dispatch recorded in metadata is narrower**: a MoE layer with the boundary present, no post-projection bias, TP8/EP8, `sp is None`, the fusion flag enabled and `0 < rows <= 16`. The report says larger observed batches take hipBLASLt plus a separate norm/residual path. Its 75.378% decode coverage must not be generalized to the wrapper's larger input envelope.

**Reusable lesson:** decoder-boundary fusion can remove several full-row passes, but wrapper support, actual caller dispatch and measured coverage are different contracts. Related operators: [GEMM epilogues](../../operator/gemm_epilogue_fused/README.md), [grouped GEMM / MoE](../../operator/grouped_gemm_moe/README.md).

## Compact MXFP4 MoE

**Evidence:** `moe_source`, `moe_metadata`, `c8_report`.

```text
Implemented local expert path (five launches):
  [_prepare_a4: route/block metadata + input MXFP4 quantization]
    -> [_scatter_a4: route placement]
    -> [_expert_a4, ACT=True: gate/up GEMMs + SiLU*up + MXFP4 requantization]
    -> [_expert_a4, ACT=False: down GEMM]
    -> [muse_moe_finalize_vector_flydsl: per-expert norm + route-weighted sum]
    -> local result before TP reduction
```

The gate/up kernel computes `SiLU(gate) * up`, rounds the activation to BF16 and packs it into MXFP4 with scales inside the same kernel. The down kernel consumes that packed representation. Full BF16 activation does not need a standalone materialization followed by a quantization launch, but packed activation/scales remain in global memory.

Other boundaries remain too: route metadata between prepare/scatter/expert stages, BF16 per-token/per-slot down-projection output, and the final local result. The finalizer applies expert normalization and routing weights; the cross-rank reduction is outside `run_compact_a4`. Removing the local finalizer or collective requires analysis beyond the GEMM body.

The source predicate requires `M` to be a multiple of eight with `0 < M <= 512`, hidden width 4096, top-8 routing and 32 local experts, with native packed weight/scale layouts. These are source applicability limits; they do not imply that every row count was observed in this c8 workload. The extraction used reconstructed routing and synthetic MXFP4 weights. That evidence does not establish behavior under every real expert-imbalance or cache-residency pattern.

**Reusable lesson:** describe fusion at each boundary of the MoE pipeline. A multi-launch MoE callable can contain meaningful activation/quantization and finalizer fusion while retaining staging buffers and communication. Related operators: [grouped GEMM / MoE](../../operator/grouped_gemm_moe/README.md), [quantized GEMM](../../operator/scaled_quant_gemm/README.md), [dispatch/combine](../../operator/moe_dispatch_combine/README.md).

## Split-KV and adjacent kernels

**Evidence:** `swa_source`, `c8_report`.

The SWA verification wrapper launches `_swa_verify_partials` and then `_swa_verify_reduce`. Partial output and softmax state are materialized between them. The source asserts eight query tokens per sequence, head dimension 64, a Q/KV head ratio of eight and page size 16. Partial workgroups compute attention with causal and left-window masks; the second kernel combines their states. This is an attention pipeline measured jointly, with the split/merge boundary retained. Collapsing it needs a separate parallelism, synchronization and performance argument.

The extraction also found norm/gate followed by output projection on the same stream. Its final task split records that these are **sequential operations**, not alternative dispatch branches. Adjacency is useful for proposing a fusion experiment; it does not prove an implemented fused norm/GEMM kernel. The same caution applies to the adjacent `_prepare_a4` and `_scatter_a4` launches above.

## Outcome and remaining evidence

The cases establish implementation structure from the pinned local source. They do not establish a speedup over unfused code. The extraction report's correctness checks use the frozen installed implementation with seeded inputs; no new correctness or performance run was performed here.

`_dense_partial` is shared by the QKV, router and unpool call sites. Attribute time using the caller/branch before summing workload savings; a kernel-name total cannot be independently charged to all three. The same rule applies to companion epilogues and partial/reduce pairs selected as a group.

To add a performance conclusion, compare the complete chain against a precisely identified baseline on the same shapes and execution mode, retain mutation and numerical checks, and validate serving engagement and end-to-end metrics. Record a fusion ablation separately from tuning an already-fused baseline. Use the [case template](../_templates/fusion_case.md) and [measurement guidance](../fusion_patterns.md#measure-the-chain-and-its-workload-contribution).
