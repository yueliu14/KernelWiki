# Operator-local corpus

Corpus content lives beside each operator's overview, backend notes and learned knowledge:

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

This top-level directory holds only the catalog and this generated navigation page. The shared production and validation tools live under [tools/corpus](../tools/corpus/README.md).

**23 operators · 69 canonical cards · 4481 snapshot-qualified source observations · 359 shipped groups.** Shared references are not counted as additional knowledge records.

## GEMM and projection fusions

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [batched_gemm](../knowledge/operator/batched_gemm/README.md) | 2 | 7 | 178 | [Decisions](../knowledge/operator/batched_gemm/corpus/decisions.md) · [Evidence](../knowledge/operator/batched_gemm/corpus/source_evidence.md) |
| [dense_gemm](../knowledge/operator/dense_gemm/README.md) | 16 | 14 | 303 | [Decisions](../knowledge/operator/dense_gemm/corpus/decisions.md) · [Evidence](../knowledge/operator/dense_gemm/corpus/source_evidence.md) |
| [gemm_epilogue_fused](../knowledge/operator/gemm_epilogue_fused/README.md) | 3 | 26 | 304 | [Decisions](../knowledge/operator/gemm_epilogue_fused/corpus/decisions.md) · [Evidence](../knowledge/operator/gemm_epilogue_fused/corpus/source_evidence.md) |
| [scaled_quant_gemm](../knowledge/operator/scaled_quant_gemm/README.md) | 8 | 15 | 577 | [Decisions](../knowledge/operator/scaled_quant_gemm/corpus/decisions.md) · [Evidence](../knowledge/operator/scaled_quant_gemm/corpus/source_evidence.md) |
| [skinny_gemv_decode](../knowledge/operator/skinny_gemv_decode/README.md) | 2 | 18 | 143 | [Decisions](../knowledge/operator/skinny_gemv_decode/corpus/decisions.md) · [Evidence](../knowledge/operator/skinny_gemv_decode/corpus/source_evidence.md) |
| [splitk_streamk_gemm](../knowledge/operator/splitk_streamk_gemm/README.md) | 3 | 15 | 76 | [Decisions](../knowledge/operator/splitk_streamk_gemm/corpus/decisions.md) · [Evidence](../knowledge/operator/splitk_streamk_gemm/corpus/source_evidence.md) |

## Attention, KV caches and preparation fusions

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [attention_decode_paged](../knowledge/operator/attention_decode_paged/README.md) | 1 | 4 | 134 | [Decisions](../knowledge/operator/attention_decode_paged/corpus/decisions.md) · [Evidence](../knowledge/operator/attention_decode_paged/corpus/source_evidence.md) |
| [attention_prefill_fmha](../knowledge/operator/attention_prefill_fmha/README.md) | 2 | 1 | 64 | [Decisions](../knowledge/operator/attention_prefill_fmha/corpus/decisions.md) · [Evidence](../knowledge/operator/attention_prefill_fmha/corpus/source_evidence.md) |
| [chunked_prefill](../knowledge/operator/chunked_prefill/README.md) | 1 | 2 | 25 | [Decisions](../knowledge/operator/chunked_prefill/corpus/decisions.md) · [Evidence](../knowledge/operator/chunked_prefill/corpus/source_evidence.md) |
| [compressed_attention](../knowledge/operator/compressed_attention/README.md) | 1 | 0 | 82 | [Decisions](../knowledge/operator/compressed_attention/corpus/decisions.md) · [Evidence](../knowledge/operator/compressed_attention/corpus/source_evidence.md) |
| [context_parallel_attention](../knowledge/operator/context_parallel_attention/README.md) | 1 | 1 | 64 | [Decisions](../knowledge/operator/context_parallel_attention/corpus/decisions.md) · [Evidence](../knowledge/operator/context_parallel_attention/corpus/source_evidence.md) |
| [gqa_mqa_attention](../knowledge/operator/gqa_mqa_attention/README.md) | 1 | 1 | 63 | [Decisions](../knowledge/operator/gqa_mqa_attention/corpus/decisions.md) · [Evidence](../knowledge/operator/gqa_mqa_attention/corpus/source_evidence.md) |
| [linear_attention_gated_delta](../knowledge/operator/linear_attention_gated_delta/README.md) | 3 | 0 | 275 | [Decisions](../knowledge/operator/linear_attention_gated_delta/corpus/decisions.md) · [Evidence](../knowledge/operator/linear_attention_gated_delta/corpus/source_evidence.md) |
| [mla_attention](../knowledge/operator/mla_attention/README.md) | 2 | 1 | 180 | [Decisions](../knowledge/operator/mla_attention/corpus/decisions.md) · [Evidence](../knowledge/operator/mla_attention/corpus/source_evidence.md) |
| [qk_norm_rope_cache](../knowledge/operator/qk_norm_rope_cache/README.md) | 2 | 0 | 175 | [Decisions](../knowledge/operator/qk_norm_rope_cache/corpus/decisions.md) · [Evidence](../knowledge/operator/qk_norm_rope_cache/corpus/source_evidence.md) |
| [sliding_window_attention](../knowledge/operator/sliding_window_attention/README.md) | 2 | 0 | 24 | [Decisions](../knowledge/operator/sliding_window_attention/corpus/decisions.md) · [Evidence](../knowledge/operator/sliding_window_attention/corpus/source_evidence.md) |
| [sparse_attention_nsa](../knowledge/operator/sparse_attention_nsa/README.md) | 1 | 0 | 82 | [Decisions](../knowledge/operator/sparse_attention_nsa/corpus/decisions.md) · [Evidence](../knowledge/operator/sparse_attention_nsa/corpus/source_evidence.md) |

## MoE routing, expert GEMMs and combine

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [fused_moe_grouped_gemm](../knowledge/operator/fused_moe_grouped_gemm/README.md) | 1 | 7 | 381 | [Decisions](../knowledge/operator/fused_moe_grouped_gemm/corpus/decisions.md) · [Evidence](../knowledge/operator/fused_moe_grouped_gemm/corpus/source_evidence.md) |
| [grouped_gemm_moe](../knowledge/operator/grouped_gemm_moe/README.md) | 1 | 17 | 553 | [Decisions](../knowledge/operator/grouped_gemm_moe/corpus/decisions.md) · [Evidence](../knowledge/operator/grouped_gemm_moe/corpus/source_evidence.md) |
| [moe_dispatch_combine](../knowledge/operator/moe_dispatch_combine/README.md) | 4 | 0 | 140 | [Decisions](../knowledge/operator/moe_dispatch_combine/corpus/decisions.md) · [Evidence](../knowledge/operator/moe_dispatch_combine/corpus/source_evidence.md) |
| [moe_routing_topk](../knowledge/operator/moe_routing_topk/README.md) | 2 | 0 | 84 | [Decisions](../knowledge/operator/moe_routing_topk/corpus/decisions.md) · [Evidence](../knowledge/operator/moe_routing_topk/corpus/source_evidence.md) |
| [shared_expert_fusion](../knowledge/operator/shared_expert_fusion/README.md) | 2 | 0 | 94 | [Decisions](../knowledge/operator/shared_expert_fusion/corpus/decisions.md) · [Evidence](../knowledge/operator/shared_expert_fusion/corpus/source_evidence.md) |

## Attention index scoring and selection

| operator | owned cards | shared cards | owned observations | reading |
|---|---:|---:|---:|---|
| [attention_indexer](../knowledge/operator/attention_indexer/README.md) | 8 | 0 | 480 | [Decisions](../knowledge/operator/attention_indexer/corpus/decisions.md) · [Evidence](../knowledge/operator/attention_indexer/corpus/source_evidence.md) |

## Reference contract

- Each decision card and each source observation has one owning operator. Other operators use IDs and links; they do not copy the body.
- Source/config references include a snapshot: `aiter_b3ad6c46:src_…` or `aiter_a6bb4993:cfg_…`. Existing content-ID suffixes are preserved; a path move does not change identity.
- The catalog uses schema version 2 and repository-relative paths. This replaces the old family-level file layout.
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
