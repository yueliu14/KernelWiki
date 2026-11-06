# Attention output through MoE completion

This subgraph starts with the attention core's **unprojected, rank-local output** and the incoming
residual. It includes the attention output projection and TP completion, residual/normalization,
routing, shared and routed expert computation, and the final MoE collective. It returns the
rank-complete MoE contribution and the residual representation expected by the next consumer.
Attention itself and its KV mutation are upstream of this boundary.

```text
attention output + residual
  -> o_proj -> TP completion -> residual + norm -> router
  -> shared/routed experts -> weighted combine -> TP completion
  -> MoE output + residual after attention
```

## Contracts to preserve

- Define whether the final MoE residual addition happens here or in the following input norm.
- Preserve the router's bias, selection, normalization and weight-application order, including the
  shared expert's precision and contribution.
- Track padded rows independently from live routes. An unspecified padding residual is not a valid
  live result, even if the MoE output of that row is zero.
- Keep scratch, peer buffers and epoch ownership consistent across layers, widths and graph replays.

## Persistent execution and evidence

A persistent schedule needs a progress argument as well as compatible math. The compiled kernel's
residency matters: tile/shape/compiler changes can invalidate an assumed all-resident grid.
Cross-rank payload publication and storage reuse have separate dependencies. Record the actual
memory scopes and completion protocol before treating a launch reduction as a usable optimization.

Measure the complete projection-to-MoE boundary against the corresponding staged path. Keep the
preceding attention time separate, then use a serving comparison for whole-model claims. Test both
returned buffers and their next consumer, and retain the actual numerical envelope and route behavior.

## Reviewed case and decisions

- [MiniMax-M3.1-Flash decode and target verify](cases/minimax_m31_decode.md): framework attention
  followed by one fused post-attention launch per sparse layer/rank.
- [Workload decisions](corpus/decisions.md): residual/padding handoff, compiled residency gates and
  peer-buffer epoch ownership. No measured speedup is assigned in this source-only pass.

Related: [MoE pipeline](../moe_expert_pipeline/README.md),
[projection/norm/residual](../projection_norm_residual/README.md),
[MoE routing](../../operator/moe_routing_topk/README.md),
[dispatch/combine](../../operator/moe_dispatch_combine/README.md),
[shared experts](../../operator/shared_expert_fusion/README.md), and the [workload index](../CORPUS.md).
