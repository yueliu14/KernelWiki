# QKV projection, normalization, RoPE and cache preparation

This semantic chain starts with hidden states and projection weights and ends with attention-ready Q/K/V plus updated KV cache state. It connects the projection to Q/K preparation. Attention computation itself is downstream of this boundary.

```text
hidden states -> QKV projection -> Q/K normalization -> positional encoding
                               -> attention inputs + KV cache writes
```

## Contract and fusion boundaries

Record projection/head layout, normalization convention, RoPE configuration, cache format/scales and slot mapping. A case may also project an output gate; document that additional output explicitly. The model-specific order of scaling, normalization, position encoding and dtype conversion is part of the contract.

Candidate boundaries include projection output → normalization, normalization → RoPE and prepared K/V → cache storage. A fused epilogue can consume GEMM partials directly while retaining the producer's partial buffer. The category includes fused and decomposed implementations; each case records its actual launches, tensors and cache mutations.

## Cases

| Case | Implementation | Evidence |
| --- | --- | --- |
| [Muse Spark decode](cases/muse_spark_decode.md) | Split-K QKV/gate projection followed by a reduction/norm/RoPE/cache epilogue | Inspected source and local extraction metadata; no fusion ablation |

## Related operators

[Dense GEMM](../../operator/dense_gemm/README.md), [Split-K GEMM](../../operator/splitk_streamk_gemm/README.md) and [Q/K preparation](../../operator/qk_norm_rope_cache/README.md) describe the constituent implementations. Return to the [semantic-chain index](../README.md).
