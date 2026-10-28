# Projection, normalization and residual update

This semantic chain maps a branch output back into the residual width, normalizes it, updates the residual stream and prepares the normalized input for the next computation. Unpool projection is one instance. Attention or FFN output projection belongs here when it has this same composition; different normalization ordering must be documented as a distinct contract variant.

```text
branch output -> projection -> branch normalization
             -> residual-stream update -> next-input normalization
```

## Contract and fusion boundaries

Record normalization axes and scale placement, residual coefficients/dtype, output aliasing and which intermediate rounding points must be preserved. Both the updated stream and normalized next input may be externally visible outputs.

A row-owning epilogue may combine projection reduction, branch norm, residual update and next-input norm. Compare the cost of materialized projection/row intermediates against the register and reduction cost of that epilogue. Standalone wrapper support and the model caller's dispatch predicate can differ; preserve both in the case.

## Cases

| Case | Implementation | Evidence |
| --- | --- | --- |
| [Muse Spark decode](cases/muse_spark_decode.md) | Split-K unpool projection plus a fused layer-boundary epilogue | Inspected source and local extraction metadata; no fusion ablation |

## Related operators

[Dense GEMM](../../operator/dense_gemm/README.md) and [GEMM epilogues](../../operator/gemm_epilogue_fused/README.md) describe the projection and its local fusion options. Return to the [semantic-chain index](../README.md).
