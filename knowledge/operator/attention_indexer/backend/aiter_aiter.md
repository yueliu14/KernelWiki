---
title: "Attention index scoring and selection: AITER aiter"
kind: implementation_knowledge
operator: attention_indexer
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Attention index scoring and selection — AITER aiter

These kernels compute or select scores used to choose sparse-attention tokens/pages. Paged and contiguous caches, FP8/MXFP4 storage and prefill/decode schedules are separate contracts.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/msa_block_select.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/msa_block_select.py) · **Recent**

Source documentation: Score every block of the index key cache against the query. The specialization for rows short enough that a request's whole query fits one MFMA tile, which is what decode and speculative decode look like. That leaves nothing to fold along the query axis, so the waves split the block range instead and prefetch pages into registers rather than staging them in LDS.

Symbols: `_score_decode_raw`, `_score_prefill_raw`, `_topk_raw`, `_topk_cp_raw`, `_ptr`, `_pow2_floor`, `_pow2_ceil`, `_topk_waves`, `_shard_topk_waves`, `_score_split`, `_tokens_per_tile`, `_resolve_q_tiles_prefill`, `_score_split_prefill`, `_check_score_tensors`, `_launch_score_decode`, `_launch_score_prefill`, `pa_sparse_block_score_decode`, `pa_sparse_block_score_prefill`, `pa_sparse_block_topk`, `_topk_slots`, `topk_cp_candidate_numel`, `pa_sparse_block_topk_cp`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_msa_block_select_cp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_msa_block_select_cp.py)
- [op_tests/test_flydsl_dcp_topk_merge.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_dcp_topk_merge.py)
- [op_tests/test_flydsl_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_mqa_logits.py)
- [op_tests/test_flydsl_fp8_paged_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_paged_mqa_logits.py)
- [op_tests/test_flydsl_minimax_m3_index_score.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_minimax_m3_index_score.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py)
- [op_tests/test_msa_block_select.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_msa_block_select.py)
- [op_tests/test_pa_mqa_logits_offset.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_offset.py)
- [op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py)
- [op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py)
- [op_tests/test_pa_mqa_logits_mxfp4_opus.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_mxfp4_opus.py)
- [op_tests/triton_tests/attention/test_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fp8_mqa_logits.py)
- [op_tests/triton_tests/attention/test_pa_mqa_logits_2d_context_lens.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_mqa_logits_2d_context_lens.py)
- [op_tests/triton_tests/attention/test_pa_mqa_logits_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_mqa_logits_mxfp4.py)
- [op_tests/triton_tests/utils/pa_mqa_logits_mxfp4_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/pa_mqa_logits_mxfp4_ref.py)

## Performance and validation

Check selected indices as well as logits, including ties and masked pages. Keep scoring and top-k costs visible when evaluating sparse attention.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
