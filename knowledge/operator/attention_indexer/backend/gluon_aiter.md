---
title: "Attention index scoring and selection: AITER gluon"
kind: implementation_knowledge
operator: attention_indexer
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Attention index scoring and selection — AITER gluon

These kernels compute or select scores used to choose sparse-attention tokens/pages. Paged and contiguous caches, FP8/MXFP4 storage and prefill/decode schedules are separate contracts.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/gluon/pa_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gluon/pa_mqa_logits.py) · **Recent**

Symbols: `_sum_combine`, `_load_persistent_mqa_tile`, `_gluon_deepgemm_fp8_paged_mqa_logits`, `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`, `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/fp8_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/fp8_mqa_logits.py) · **Recent**

Symbols: `elementwise_max_prop_nan`, `_relu_f32_dual_gfx1250`, `relu_f32`, `_load_kv_scales_block`, `_store_logits_block`, `MQATDMKVLoaderConfig`, `MQATDMKVLoader`, `_mqa_dot`, `mqa_logits_loop_double_buf`, `mqa_logits_loop_pipelined`, `_gluon_fp8_mqa_logits_kernel`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py) · **Recent**

Source documentation: CDNA4 async-copy loader. Shared holds K as [HEAD_SIZE, BLOCK_KV].

Symbols: `elementwise_max_prop_nan`, `relu_f32`, `_load_kv_scales_block`, `_store_logits_block`, `_offset_bases_to_blocked`, `_make_kv_load_layouts_cdna4`, `MQAAsyncKVLoaderConfig`, `MQAAsyncKVLoader`, `_mqa_dot`, `_make_head_reduction_plan`, `_split_leaf`, `_weighted_fma_fold_serial`, `_fold_head_axis`, `_fold_head_tail`, `_weighted_sum_fma_fold`, `_load_row_operands`, `_row_logits_split`, `_row_logits`, `_emit_row_logits`, `mqa_logits_loop_double_buf`, `_gluon_fp8_mqa_logits_kernel`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_M`, `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py) · **Recent**

Source documentation: Paged MXFP4 MQA-logits kernel for gfx950, plus its scheduler kernel. logits[b*next_n + n, t] = sum_h relu(q[b, n, h, :] . kv[t, :]) * w[b*next_n + n, h]

Symbols: `_axis_layout`, `_scale_group_layout`, `_offset_bases_to_blocked`, `_staging_layouts`, `_shuffled_offsets`, `_shuffled_scale_offsets`, `_token_major_offsets`, `_token_major_scale_offsets`, `_fold_plan`, `_prepare_candidates_kernel`, `_max_nan`, `_relu`, `_fma_unpacked`, `_split_leaf`, `_fold_heads`, `_fold_tail`, `Config`, `_load_q_chunk`, `_load_q_row`, `KVState`, `_load_scales_wide`, `RegLoader`, `LDSLoader`, `_dot`, `_row_logits`, `Program`, `_loop_with_reg`, `_loop_with_lds`, `_pa_mqa_logits_mxfp4_kernel`, `_pa_mqa_logits_mxfp4_sched_kernel`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_M`, `BLOCK_P`, `BLOCK_S`, `BLOCK_T`, `num_warps`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_msa_block_select_cp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_msa_block_select_cp.py)
- [op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py)
- [op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py)
- [op_tests/test_flydsl_dcp_topk_merge.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_dcp_topk_merge.py)
- [op_tests/test_flydsl_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_mqa_logits.py)
- [op_tests/test_flydsl_fp8_paged_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_paged_mqa_logits.py)
- [op_tests/test_flydsl_minimax_m3_index_score.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_minimax_m3_index_score.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py)
- [op_tests/test_msa_block_select.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_msa_block_select.py)
- [op_tests/test_pa_mqa_logits_mxfp4_opus.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_mxfp4_opus.py)
- [op_tests/test_pa_mqa_logits_offset.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_offset.py)
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
