---
semantic_chain: projection_norm_residual
model: Muse-Spark-Open-816B-A42B-Quark-MXFP4
phase: decode_target_verify
evidence: source_inspection_and_local_reports
source_set: ../../_sources/muse_spark_c8.yaml
---

# Muse Spark unpool projection and layer boundary

This case instantiates the [projection_norm_residual semantic chain](../README.md) in the [Muse Spark c8 source snapshot](../../_sources/muse_spark_c8.md). Source IDs refer to the [shared source ledger](../../_sources/muse_spark_c8.yaml). Semantic diagrams are not measured unfused baselines; launch counts describe the inspected source path.

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

**Reusable lesson:** decoder-boundary fusion can remove several full-row passes, but wrapper support, actual caller dispatch and measured coverage are different contracts. Related operators: [GEMM epilogues](../../../operator/gemm_epilogue_fused/README.md), [grouped GEMM / MoE](../../../operator/grouped_gemm_moe/README.md).

## Outcome and evidence limits

The case establishes implementation structure from the pinned local source. It does not establish a speedup over unfused code. The extraction report's correctness checks use the frozen installed implementation with seeded inputs; no new correctness or performance run was performed here. Preserve the [shared coverage and attribution limits](../../_sources/muse_spark_c8.md#evidence-boundaries) when using this case.

For a new comparison, follow the [case template](../../_templates/fusion_case.md) and [measurement guidance](../../fusion_patterns.md#measure-the-chain-and-its-workload-contribution).
