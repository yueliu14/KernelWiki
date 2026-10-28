# MoE expert pipeline

This semantic chain takes routed token inputs, expert IDs/weights and expert parameters through token preparation, expert FFNs and output combination. Routing-score computation is upstream. Each case must say whether its output is a rank-local contribution or a fully combined distributed result.

```text
routed tokens -> token preparation -> expert gate/up projections
              -> activation -> down projection -> route-weighted output combination
```

## Contract and fusion boundaries

Record expert ownership, token/slot ordering, padding, top-k, activation, weight/activation formats and any expert normalization. Quantization, sorting and dispatch are implementation stages whose layouts must agree with the consuming GEMMs. State collective placement and the output's rank scope.

Potential fusion boundaries include preparation with quantization, gate/up computation with activation and requantization, and output postprocessing with combination. The semantic category includes staged and more extensively fused schedules. Retain intermediate buffers, load imbalance and cross-rank dependencies in the comparison.

## Cases

| Case | Implementation | Evidence |
| --- | --- | --- |
| [Muse Spark decode](cases/muse_spark_decode.md) | Compact MXFP4 experts with fused activation/requantization and separate finalizer | Inspected source and local extraction metadata; no fusion ablation |

## Related operators

[Grouped GEMM / MoE](../../operator/grouped_gemm_moe/README.md), [fused MoE](../../operator/fused_moe_grouped_gemm/README.md), [quantized GEMM](../../operator/scaled_quant_gemm/README.md) and [dispatch/combine](../../operator/moe_dispatch_combine/README.md) cover constituent implementations. Return to the [semantic-chain index](../README.md).
