---
semantic_chain: qkv_projection_norm_rope_cache
model: Muse-Spark-Open-816B-A42B-Quark-MXFP4
phase: decode_target_verify
evidence: source_inspection_and_local_reports
source_set: ../../_sources/muse_spark_c8.yaml
---

# Muse Spark QKV projection, normalization and cache preparation

This case instantiates the [qkv_projection_norm_rope_cache semantic chain](../README.md) in the [Muse Spark c8 source snapshot](../../_sources/muse_spark_c8.md). Source IDs refer to the [shared source ledger](../../_sources/muse_spark_c8.yaml). Semantic diagrams are not measured unfused baselines; launch counts describe the inspected source path.

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

**Reusable lesson:** a split-K reduction is a useful attachment point for a row/head epilogue. Keep its partial-buffer cost in the comparison and preserve the original rounding and state-write rules. Related operators: [Q/K preparation](../../../operator/qk_norm_rope_cache/README.md), [Split-K GEMM](../../../operator/splitk_streamk_gemm/README.md).

## Outcome and evidence limits

The case establishes implementation structure from the pinned local source. It does not establish a speedup over unfused code. The extraction report's correctness checks use the frozen installed implementation with seeded inputs; no new correctness or performance run was performed here. Preserve the [shared coverage and attribution limits](../../_sources/muse_spark_c8.md#evidence-boundaries) when using this case.

For a new comparison, follow the [case template](../../_templates/fusion_case.md) and [measurement guidance](../../fusion_patterns.md#measure-the-chain-and-its-workload-contribution).
