---
semantic_chain: router_pooler_topk
model: Muse-Spark-Open-816B-A42B-Quark-MXFP4
phase: decode_target_verify
topics: [fusion, dataflow, dispatch]
evidence: source_inspection_and_local_reports
source_set: ../../_sources/muse_spark_c8.yaml
---

# Muse Spark joint router and pooler

This case instantiates the [router_pooler_topk semantic chain](../README.md) in the [Muse Spark c8 source snapshot](../../_sources/muse_spark_c8.md). Source IDs refer to the [shared source ledger](../../_sources/muse_spark_c8.yaml). Semantic diagrams are not measured unfused baselines; launch counts describe the inspected source path.

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

**Reusable lesson:** combine projections with a common input when their shape and precision contracts permit it, then combine compatible epilogues. Record every producer backend and fallback separately. Related operators: [routing/top-k](../../../operator/moe_routing_topk/README.md), [dense GEMM](../../../operator/dense_gemm/README.md).

## Outcome and evidence limits

The case establishes implementation structure from the pinned local source. It does not establish a speedup over unfused code. The extraction report's correctness checks use the frozen installed implementation with seeded inputs; no new correctness or performance run was performed here. Preserve the [shared coverage and attribution limits](../../_sources/muse_spark_c8.md#evidence-boundaries) when using this case.

For a new comparison, follow the [workload case template](../../_templates/workload_case.md) and [measurement guidance](../../fusion_patterns.md#measure-the-chain-and-its-workload-contribution).
