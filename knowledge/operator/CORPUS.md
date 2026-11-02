# Operator-local corpus

Corpus content lives beside each operator's overview and backend notes:

```text
knowledge/operator/<operator>/corpus/
  scope.yaml
  source_evidence.yaml
  decisions.yaml
  performance_axes.yaml
  source_evidence.md       # generated
  decisions.md             # generated
  tuned_configs.yaml       # where upstream seeds are available
  seeds.md                 # generated where applicable
```

The [operator catalog](catalog.yaml) and this generated navigation page live alongside the operator directories. The shared production and validation tools live under [tools/corpus](../../tools/corpus/README.md).

**23 operators · 107 canonical cards · 4481 snapshot-qualified source observations · 359 shipped groups.** Shared references are not counted as additional knowledge records.

## GEMM and projection fusions

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [batched_gemm](batched_gemm/README.md) | 2 | 7 | 178 | [Decisions](batched_gemm/corpus/decisions.md) · [Evidence](batched_gemm/corpus/source_evidence.md) |
| [dense_gemm](dense_gemm/README.md) | 18 | 14 | 303 | [Decisions](dense_gemm/corpus/decisions.md) · [Evidence](dense_gemm/corpus/source_evidence.md) |
| [gemm_epilogue_fused](gemm_epilogue_fused/README.md) | 4 | 26 | 304 | [Decisions](gemm_epilogue_fused/corpus/decisions.md) · [Evidence](gemm_epilogue_fused/corpus/source_evidence.md) |
| [scaled_quant_gemm](scaled_quant_gemm/README.md) | 12 | 15 | 577 | [Decisions](scaled_quant_gemm/corpus/decisions.md) · [Evidence](scaled_quant_gemm/corpus/source_evidence.md) |
| [skinny_gemv_decode](skinny_gemv_decode/README.md) | 2 | 18 | 143 | [Decisions](skinny_gemv_decode/corpus/decisions.md) · [Evidence](skinny_gemv_decode/corpus/source_evidence.md) |
| [splitk_streamk_gemm](splitk_streamk_gemm/README.md) | 3 | 15 | 76 | [Decisions](splitk_streamk_gemm/corpus/decisions.md) · [Evidence](splitk_streamk_gemm/corpus/source_evidence.md) |

## Attention, KV caches and preparation fusions

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [attention_decode_paged](attention_decode_paged/README.md) | 7 | 4 | 134 | [Decisions](attention_decode_paged/corpus/decisions.md) · [Evidence](attention_decode_paged/corpus/source_evidence.md) |
| [attention_prefill_fmha](attention_prefill_fmha/README.md) | 3 | 1 | 64 | [Decisions](attention_prefill_fmha/corpus/decisions.md) · [Evidence](attention_prefill_fmha/corpus/source_evidence.md) |
| [chunked_prefill](chunked_prefill/README.md) | 1 | 2 | 25 | [Decisions](chunked_prefill/corpus/decisions.md) · [Evidence](chunked_prefill/corpus/source_evidence.md) |
| [compressed_attention](compressed_attention/README.md) | 1 | 0 | 82 | [Decisions](compressed_attention/corpus/decisions.md) · [Evidence](compressed_attention/corpus/source_evidence.md) |
| [context_parallel_attention](context_parallel_attention/README.md) | 1 | 1 | 64 | [Decisions](context_parallel_attention/corpus/decisions.md) · [Evidence](context_parallel_attention/corpus/source_evidence.md) |
| [gqa_mqa_attention](gqa_mqa_attention/README.md) | 3 | 1 | 63 | [Decisions](gqa_mqa_attention/corpus/decisions.md) · [Evidence](gqa_mqa_attention/corpus/source_evidence.md) |
| [linear_attention_gated_delta](linear_attention_gated_delta/README.md) | 6 | 0 | 275 | [Decisions](linear_attention_gated_delta/corpus/decisions.md) · [Evidence](linear_attention_gated_delta/corpus/source_evidence.md) |
| [mla_attention](mla_attention/README.md) | 2 | 1 | 180 | [Decisions](mla_attention/corpus/decisions.md) · [Evidence](mla_attention/corpus/source_evidence.md) |
| [qk_norm_rope_cache](qk_norm_rope_cache/README.md) | 2 | 0 | 175 | [Decisions](qk_norm_rope_cache/corpus/decisions.md) · [Evidence](qk_norm_rope_cache/corpus/source_evidence.md) |
| [sliding_window_attention](sliding_window_attention/README.md) | 3 | 0 | 24 | [Decisions](sliding_window_attention/corpus/decisions.md) · [Evidence](sliding_window_attention/corpus/source_evidence.md) |
| [sparse_attention_nsa](sparse_attention_nsa/README.md) | 2 | 0 | 82 | [Decisions](sparse_attention_nsa/corpus/decisions.md) · [Evidence](sparse_attention_nsa/corpus/source_evidence.md) |

## MoE routing, expert GEMMs and combine

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [fused_moe_grouped_gemm](fused_moe_grouped_gemm/README.md) | 1 | 7 | 381 | [Decisions](fused_moe_grouped_gemm/corpus/decisions.md) · [Evidence](fused_moe_grouped_gemm/corpus/source_evidence.md) |
| [grouped_gemm_moe](grouped_gemm_moe/README.md) | 8 | 17 | 553 | [Decisions](grouped_gemm_moe/corpus/decisions.md) · [Evidence](grouped_gemm_moe/corpus/source_evidence.md) |
| [moe_dispatch_combine](moe_dispatch_combine/README.md) | 4 | 0 | 140 | [Decisions](moe_dispatch_combine/corpus/decisions.md) · [Evidence](moe_dispatch_combine/corpus/source_evidence.md) |
| [moe_routing_topk](moe_routing_topk/README.md) | 3 | 0 | 84 | [Decisions](moe_routing_topk/corpus/decisions.md) · [Evidence](moe_routing_topk/corpus/source_evidence.md) |
| [shared_expert_fusion](shared_expert_fusion/README.md) | 2 | 0 | 94 | [Decisions](shared_expert_fusion/corpus/decisions.md) · [Evidence](shared_expert_fusion/corpus/source_evidence.md) |

## Attention index scoring and selection

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [attention_indexer](attention_indexer/README.md) | 9 | 0 | 480 | [Decisions](attention_indexer/corpus/decisions.md) · [Evidence](attention_indexer/corpus/source_evidence.md) |

## General methods

[Reusable development decisions](../general/corpus/decisions.md) — 8 cards for measurement, integration and experiment design.

## Reference contract

- Each decision card and each source observation has one owning operator. Other operators use IDs and links; they do not copy the body.
- Source/config references include a snapshot: `aiter_b3ad6c46:src_…` or `aiter_a6bb4993:cfg_…`. Existing content-ID suffixes are preserved; a path move does not change identity.
- The catalog uses schema version 3 and repository-relative paths. Decisions can carry source observations and scoped historical experience summaries.
- Historical reports are not reproduced validation. Each experience entry states its verification status; private runs and raw measurement artifacts stay outside the public repository.
- Operator membership and source availability do not certify complete backend coverage, runtime engagement or a measured speedup.
- Correctness-test links are pinned sources, not test results. No new GPU performance measurements or disassembly results are included.

## Source snapshots

- `aiter_a6bb4993`: [a6bb499375849eec45d68c5ccaebc8865fd422c0](https://github.com/ROCm/aiter/tree/a6bb499375849eec45d68c5ccaebc8865fd422c0).
- `aiter_b3ad6c46`: [b3ad6c461703110dc36286265c16e453ef0e2dd2](https://github.com/ROCm/aiter/tree/b3ad6c461703110dc36286265c16e453ef0e2dd2).

## Maintenance

```bash
python3 tools/corpus/build.py --emit
python3 tools/corpus/build.py --check
python3 tools/corpus/validate.py
python3 tools/corpus/validate.py --verify-source /path/to/aiter
python3 tools/corpus/extract.py --operator attention_indexer --aiter /path/to/aiter --check
```

Build and validate all operators by default; `--operator` or `--family` limits the selected output/audit. Extraction reads the declared git objects, including multiple revisions in one operator. See the tool guide for refreshing evidence and adding operators.
