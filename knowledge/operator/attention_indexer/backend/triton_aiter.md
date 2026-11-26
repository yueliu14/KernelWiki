---
title: "Attention index scoring and selection: AITER triton"
kind: implementation_knowledge
operator: attention_indexer
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Attention index scoring and selection — AITER triton

These kernels compute or select scores used to choose sparse-attention tokens/pages. Paged and contiguous caches, FP8/MXFP4 storage and prefill/decode schedules are separate contracts.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/fp8_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fp8_mqa_logits.py) · **Recent**

Source documentation: This function computes the logits to be used by a topk function for sparse attention. Q: [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8 KV: [seq_len_kv, HEAD_SIZE], dtype float8 kv_scales: [seq_len_kv], dtype float32 weights: [seq_len, NUM_HEADS], dtype float32 cu_starts: [seq_len], dtype int32, start indices cu_ends: [seq_len], dtype int32, end indices clean_logits: bool. If True, positions outside [cu_starts[i], cu_ends[i]) in row i are explicitly written as -inf.

Symbols: `_async_copy_accepts_distributed_layout`, `_permute_accepts_constexpr_tuple`, `_gfx950_kv_splits`, `_gfx942_tile_fits_lds`, `fp8_mqa_logits`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_M`, `matrix_instr_nonkdim`, `num_buffers`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/pa_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_mqa_logits.py) · **Recent**

Source documentation: Build the FP8 decode persistent grid entirely on the GPU. The returned int32 [NumCTAs, 4] table contains packed query row, first chunk, chunk count and context length. Pass it as PersistentSchedule to :func:deepgemm_fp8_paged_mqa_logits. Rebuild it when context lengths, next_n, ChunkK or max_model_len change. Both construction and execution can be captured in a CUDA graph; out allows storage reuse.

Symbols: `_context_lens_and_stride`, `deepgemm_fp8_paged_mqa_logits_ragged_k`, `deepgemm_fp8_paged_mqa_logits_stage1_ragged_k`, `deepgemm_fp8_paged_mqa_logits_stage1`, `_compile_deepgemm_fp8_paged_mqa_logits`, `deepgemm_fp8_paged_mqa_logits_schedule`, `deepgemm_fp8_paged_mqa_logits_persistent_schedule`, `deepgemm_fp8_paged_mqa_logits`.

Parameters referenced in this file: `BLOCK_B`, `BLOCK_S`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/pa_mqa_logits_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_mqa_logits_mxfp4.py) · **Recent**

Source documentation: Everything needed to write a preshuffled cache and to read it back.

Symbols: `mfma_nonk_dim`, `kv_tile`, `cache_format`, `_split_cache`, `_spec_block_m`, `plan_block_m`, `_kv_splits`, `_select_config`, `select_config`, `cache_strides`, `build_candidate_gather`, `build_schedule`, `_check_schedule`, `paged_mxfp4_mqa_logits`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_M`, `BLOCK_S`, `BLOCK_T`, `num_buffers`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/attention/block_lut.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/block_lut.py) · **Recent**

Source documentation: Triton kernel to build the block-sparse LUT (kv_block_indices) from a 4D block attention mask without using nonzero or argsort.

Symbols: `_block_attn_mask_to_lut_kernel`, `block_attn_mask_to_lut_kernel`.

Parameters referenced in this file: `BLOCK_KB`.

### `aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py) · **Recent**

Symbols: `_fp8_mqa_logits_kernel`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_Q`.

### `aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py) · **Recent**

Symbols: `_sum_combine`, `_deepgemm_fp8_paged_mqa_logits_stage1_ragged_k`, `_deepgemm_fp8_paged_mqa_logits_ragged_k`, `_deepgemm_fp8_paged_mqa_logits_stage1`, `_deepgemm_fp8_paged_mqa_logits_persistent_schedule`, `_deepgemm_fp8_paged_mqa_logits_varctx_schedule`, `_deepgemm_fp8_paged_mqa_logits`, `_gluon_deepgemm_fp8_paged_mqa_logits`, `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`, `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx`.

Parameters referenced in this file: `BLOCK_B`, `BLOCK_S`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py)
- [op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py)
- [op_tests/triton_tests/attention/test_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fp8_mqa_logits.py)
- [op_tests/triton_tests/attention/test_pa_mqa_logits_2d_context_lens.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_mqa_logits_2d_context_lens.py)
- [op_tests/triton_tests/attention/test_pa_mqa_logits_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_mqa_logits_mxfp4.py)
- [op_tests/triton_tests/utils/pa_mqa_logits_mxfp4_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/pa_mqa_logits_mxfp4_ref.py)
- [op_tests/multigpu_tests/test_msa_block_select_cp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_msa_block_select_cp.py)
- [op_tests/test_flydsl_dcp_topk_merge.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_dcp_topk_merge.py)
- [op_tests/test_flydsl_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_mqa_logits.py)
- [op_tests/test_flydsl_fp8_paged_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_paged_mqa_logits.py)
- [op_tests/test_flydsl_minimax_m3_index_score.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_minimax_m3_index_score.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py)
- [op_tests/test_msa_block_select.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_msa_block_select.py)
- [op_tests/test_pa_mqa_logits_mxfp4_opus.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_mxfp4_opus.py)
- [op_tests/test_pa_mqa_logits_offset.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_offset.py)

## Performance and validation

Check selected indices as well as logits, including ties and masked pages. Keep scoring and top-k costs visible when evaluating sparse attention.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
