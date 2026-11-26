---
title: "Sparse attention: AITER triton"
kind: implementation_knowledge
operator: sparse_attention_nsa
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Sparse attention — AITER triton

Sparse kernels consume selected pages, blocks or token indices. Selection granularity, duplicates, padding and causal masking affect the logical work independently of dense attention shapes.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/pa_decode_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode_sparse.py) · **Recent**

Source documentation: Sparse paged-decode attention over a unified or split KV pool with per-token paged indices. TODO: add details once API has settled

Symbols: `_check_out`, `pa_decode_sparse`, `_as_int32_contiguous_1d`, `_decode_num_splits_occ`, `_last_round_fills`, `_rounds_suit_32`, `_dsv4_block_m`, `_staged_block_m`, `_launch_splits`, `_pa_decode_sparse_gfx950_gluon`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `BLOCK_M`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/pa_prefill_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill_sparse.py) · **Recent**

Source documentation: Sparse paged-prefill attention over two KV sources (prefix + extend) with per-head sink bias — gfx1250 (gluon) only. Exposes pa_prefill_sparse — grid (T, cdiv(H, BLOCK_H)), one token and BLOCK_H heads per CTA. Same grid as the decode kernel. No split-K: prefill fills the GPU via the token dimension. Gfx950(mla_gluon) and others(Triton _sparse_attn_prefill_kernel, they take a 1-D (kv_indices, kv_indptr) pair over one pool.

Symbols: `_pinned_prefill_config`, `pa_prefill_sparse`, `_prep_single_source`, `_as_index_contiguous_1d`, `_as_int32_contiguous_1d`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/qsa.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/qsa.py) · **Recent**

Source documentation: QSA operators for index scoring, selection and sparse paged GQA.

Symbols: `_prev_pow2`, `_table_width_bucket`, `_kv_splits_heuristic`, `_kernel_config`, `_check_vector`, `qsa_paged_mqa_logits`, `qsa_expand_block_indices`, `qsa_sparse_paged_gqa`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_M`, `BLOCK_N`, `BLOCK_Q`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/sparse_mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla.py) · **Recent**

Source documentation: Sparse (top-k gathered) MLA attention. Everything after softmax_scale is keyword-only: kv_indptr and kv_indices are both flat int32, so a positional mix-up would type-check and return garbage. Supported KV cache formats. The format is inferred from kv_buffer's shape, dtype and kv_scale; each row gives what the caller has to pass. R is the QK width, kv_lora_rank + qk_rope_head_dim.

Symbols: `_check_out`, `_cache_pointers`, `_get_config`, `_check_packed_arch`, `_check_fp8_arch`, `_check_lds_budget`, `_mla_num_splits`, `_async_launch_config`, `_resolve_dot_precision`, `_dsv32_row`, `_classify_flat`, `_classify_cache`, `_check_geometry`, `_check_index_stream`, `_forward_paged`, `sparse_mla_fwd`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/sparse_mla_dsv4_train.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/sparse_mla_dsv4_train.py) · **Recent**

Symbols: `_build_inverted_topk`, `sparse_mla_fwd`, `SparseMLADSV4Function`, `sparse_mla_dsv4_train`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_G`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py) · **Recent**

Source documentation: Sparse paged-decode attention kernels (split-K + per-token paged indices). Two-kernel decomposition of a flash-decode whose K range for each token is a gathered subset of a unified KV pool: _pa_decode_sparse : split-K main kernel. Grid (T, ceil(H/BLOCK_H), KV_SPLITS). Each program owns one token, one head-block, and one slice of the token's sparse K range; writes pre-sink (m, l, acc) partials.

Symbols: `_pa_decode_sparse`, `_pa_decode_sparse_reduce`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/qsa.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/qsa.py) · **Recent**

Source documentation: QSA triton kernels for index scoring and sparse paged GQA.

Symbols: `_qsa_paged_mqa_logits_kernel`, `_qsa_paged_mqa_logits_mfma_kernel`, `_expand_selected_groups`, `_qsa_sparse_paged_gqa_kernel`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py) · **Recent**

Symbols: `_get_lds_limit`, `_pack_dense_prefix_to_ragged_kernel`, `_compute_topk_lens_kernel`, `_pack_global_topk_ragged_kernel`, `_compute_combined_lens_kernel`, `_combine_topk_swa_indices_ragged_kernel`, `_exp`, `_prefill_prune_configs`, `_get_prefill_autotune_configs`, `_sparse_attn_prefill_kernel`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `BLOCK_SIZE`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py) · **Recent**

Symbols: `_sparse_mla_fwd_kernel`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_G`, `BLOCK_H`, `BLOCK_K`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py) · **Recent**

Symbols: `cdiv_fn`, `apply_softcap`, `find_seq_idx`, `_kernel_unified_attention_sparse_mla_2d`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_Q`, `BLOCK_SIZE`.

### `aiter/ops/triton/attention/unified_attention_sparse_mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/unified_attention_sparse_mla.py)

Source documentation: This function computes the sparse attention. Note: topk_indices index the KV cache, not block_table.

Symbols: `unified_attention_sparse_mla`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_SIZE`, `num_stages`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx942/triton/attention/sparse_attention_dsv4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/sparse_attention_dsv4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/sparse_mla_dsv4_train/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/sparse_mla_dsv4_train/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/sparse_mla_dsv4_train/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/sparse_mla_dsv4_train/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py)
- [op_tests/triton_tests/attention/test_pa_decode_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_decode_sparse.py)
- [op_tests/triton_tests/attention/test_pa_prefill_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill_sparse.py)
- [op_tests/triton_tests/attention/test_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla.py)
- [op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py)
- [op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py)
- [op_tests/test_jit_cache_transaction.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_jit_cache_transaction.py)
- [op_tests/test_pa_sparse_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_sparse_prefill.py)
- [op_tests/test_vsa_sparse_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_vsa_sparse_attention.py)

## Performance and validation

Include index preparation when comparing complete operators and validate empty selections, tail blocks and causal boundaries.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
