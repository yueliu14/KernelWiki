# Decoder stack execution

A decoder stack composes several decoder layers, possibly of different attention kinds, while
preserving the model's hidden/residual and cache/state contracts. The case defines its outer boundary:
embedding or a hidden-state handoff at entry, and final hidden states or another layer handoff at exit.
Logits, sampling, speculative acceptance and draft execution are included only when explicitly stated.

## Dataflow and lifetime

```text
step inputs + positions + committed state
  -> prelude / first layer -> groups of decoder layers -> final mixer / hidden output
  -> downstream logits / accept / draft consumers
```

For hybrid recurrent/attention models, distinguish committed recurrent state from per-token verify
scratch and from immediately written KV/index caches. Register/LDS/global-scratch lifetimes span
stages differently; count real materializations even inside one persistent launch. Preserve positions,
request identity and the next consumer of every externally visible state.

## Scheduling and fusion depth

Use bounded groups when compilation size, specialization count, resources or synchronization make a
single larger launch unsuitable. The number of layers per launch is an implementation choice, not
the semantic category. A static persistent schedule with tagged handoffs needs compatible memory
visibility, tag lifetimes, producer progress and residency. Do not transfer a single-GPU mechanism
to a TP collective without its additional ownership and communication proof.

## Dispatch and measurement

Separate target verify, target decode and draft steps. Count persistent stack launches separately
from framework helpers and graph submissions. Measure cold compilation and warm execution separately,
including resource use, scratch traffic, dispatch coverage and correctness of the following accept/draft
step. A smaller launch count is a candidate benefit, not a measured speedup.

## Reviewed case and decisions

- [Qwen3.8-Flash-Next target verify](cases/qwen38_flash_next_verify.md): 48 layers in 13
  persistent stack launches on one GPU, with logits outside that count.
- [Workload decisions](corpus/decisions.md): tagged stage mailboxes, verify-state preservation and
  bounded fusion depth. Effects and independent attribution are unassessed.

Related: [decoder layers](../decoder_layer/README.md),
[gated-delta attention](../../operator/linear_attention_gated_delta/README.md),
[sparse attention](../../operator/sparse_attention_nsa/README.md),
[attention indexers](../../operator/attention_indexer/README.md),
[MoE pipeline](../moe_expert_pipeline/README.md), and the [workload index](../CORPUS.md).
