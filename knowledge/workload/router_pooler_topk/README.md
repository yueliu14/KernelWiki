# Router, pooler and top-k

This semantic chain produces both routing decisions and the projected input consumed by experts from the same hidden states. The pooler is an input projection for the expert branch; it is not assumed to exist in every MoE model. Expert FFN execution begins after this boundary.

```text
hidden states -> router projection -> scoring/top-k -> expert IDs and route weights
             -> pooler projection -> normalization -> expert input
```

## Contract and fusion boundaries

Record router and pooler weight formats, score transformation, bias semantics, top-k/tie rules, route-weight normalization and expert-input normalization. A joint projection must preserve the precision requirements of both branches.

Projections sharing the input may use packed weights. Their postprocessing may share a launch while remaining independent programs. Compare producer backends, packing cost, intermediate layouts and dispatch coverage for the complete chain. Different kernels or shape branches can implement the same semantic contract.

## Cases

| Case | Implementation | Evidence |
| --- | --- | --- |
| [Muse Spark decode](cases/muse_spark_decode.md) | Packed pooler/router projection with joint routing and normalization epilogue | Inspected source and local extraction metadata; no fusion ablation |

## Related operators

[Dense GEMM](../../operator/dense_gemm/README.md) and [routing/top-k](../../operator/moe_routing_topk/README.md) cover the constituent implementations. The downstream [MoE expert pipeline](../moe_expert_pipeline/README.md) consumes their outputs. Return to the [semantic-chain index](../README.md).
