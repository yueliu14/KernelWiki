# Decoder layer execution

A decoder layer transforms its incoming hidden/residual representation through attention and a
feed-forward block into the representation required by the following layer. KV updates, residual
branches, routing weights and rank completion are part of the contract. A layer can include sparse
selection, compression, shared experts or model-specific mixers.

This category includes both staged and fused implementations. `monokernel` and `megakernel` are
implementation labels; the real launch graph and memory handoffs define their boundaries.

## Dataflow and state

```text
incoming hidden/residual + positions + KV state
  -> attention input preparation -> attention -> output projection / TP completion
  -> residual / normalization / routing -> expert or dense FFN -> TP completion
  -> next-layer hidden/residual representation + updated KV state
```

Record the exact representation at both ends. A pending residual/mixer tuple can be a valid output
only when its consumer understands it. Include auxiliary hidden-state captures and any input that
must first be materialized for another branch. Source-level stages inside a kernel are not separate
launches, and sharing a launch does not imply register-only communication.

## Scheduling, dispatch and measurement

Identify which consumers need local values, all producer workgroups, or all ranks. Keep the cache
publication and residual-settling boundaries explicit when considering larger fusion. Separate
runtime admission from observed engagement: model/shape support in a spec is not a deployment trace.

Compare the same phase, rows, context, precision and TP configuration. Include conditional indexer,
compressor, copy and collective work in the timed layer boundary. A full-step comparison also includes
embedding, metadata, final normalization, logits and speculative glue where applicable. Preserve
fallbacks and their runtime frequency instead of averaging unlike paths.

## Reviewed case and decisions

- [DeepSeek-V4.1-Flash decode and target verify](cases/deepseek_v41_decode.md): K1/K2,
  conditional selection, pending residuals and surrounding SGLang work.
- [Workload decisions](corpus/decisions.md): cache publication, conditional launch accounting and
  deferred residual ownership. These are source precedents; effects have not been assessed.

## Constituent knowledge

Use [QKV preparation](../qkv_projection_norm_rope_cache/README.md),
[MoE execution](../moe_expert_pipeline/README.md), [attention indexers](../../operator/attention_indexer/README.md),
[sparse attention](../../operator/sparse_attention_nsa/README.md), and
[shared experts](../../operator/shared_expert_fusion/README.md) for smaller boundaries and implementations.
Return to the [workload index](../CORPUS.md).
