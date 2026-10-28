---
semantic_chain: moe_expert_pipeline
model: Muse-Spark-Open-816B-A42B-Quark-MXFP4
phase: decode_target_verify
evidence: source_inspection_and_local_reports
source_set: ../../_sources/muse_spark_c8.yaml
---

# Muse Spark compact MXFP4 MoE

This case instantiates the [moe_expert_pipeline semantic chain](../README.md) in the [Muse Spark c8 source snapshot](../../_sources/muse_spark_c8.md). Source IDs refer to the [shared source ledger](../../_sources/muse_spark_c8.yaml). Semantic diagrams are not measured unfused baselines; launch counts describe the inspected source path.

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

**Reusable lesson:** describe fusion at each boundary of the MoE pipeline. A multi-launch MoE callable can contain meaningful activation/quantization and finalizer fusion while retaining staging buffers and communication. Related operators: [grouped GEMM / MoE](../../../operator/grouped_gemm_moe/README.md), [quantized GEMM](../../../operator/scaled_quant_gemm/README.md), [dispatch/combine](../../../operator/moe_dispatch_combine/README.md).

## Outcome and evidence limits

The case establishes implementation structure from the pinned local source. It does not establish a speedup over unfused code. The extraction report's correctness checks use the frozen installed implementation with seeded inputs; no new correctness or performance run was performed here. Preserve the [shared coverage and attribution limits](../../_sources/muse_spark_c8.md#evidence-boundaries) when using this case.

For a new comparison, follow the [case template](../../_templates/fusion_case.md) and [measurement guidance](../../fusion_patterns.md#measure-the-chain-and-its-workload-contribution).
