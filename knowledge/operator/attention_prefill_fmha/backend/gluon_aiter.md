---
title: "Prefill and full attention: AITER gluon"
kind: implementation_knowledge
operator: attention_prefill_fmha
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Prefill and full attention — AITER gluon

Tiled attention interleaves QK products, online softmax and probability-value products. Inference paths require the correct causal or variable-length mask, head layout and KV-cache contract. Use forward entry points with dropout disabled.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/mha.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/mha.py) · **Recent**

Symbols: `_make_kv_shared_layouts`, `_make_load_layout`, `_buffer_load_2d`, `_load_k`, `_load_v`, `_store_k_smem`, `_load_k_smem`, `_attn_qk`, `_attn_softmax_pv`, `_attn_fwd_inner`, `_attn_fwd`, `_get_config`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_OUT`, `BLOCK_DMODEL_PE`, `BLOCK_DMODEL_POW2`, `BLOCK_M`, `BLOCK_N`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx950/gluon/attention/chunk_delta_attn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/attention/chunk_delta_attn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/attention/mha/mha.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/attention/mha/mha.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/cpp/mha/compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/mha/compile.py)
- [op_tests/op_benchmarks/flydsl/bench_mha_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_fwd.py)
- [op_tests/op_benchmarks/triton/bench_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_attn_res.py)
- [op_tests/op_benchmarks/triton/bench_deepgemm_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_deepgemm_attention.py)
- [op_tests/op_benchmarks/triton/bench_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_extend_attention.py)
- [op_tests/op_benchmarks/triton/bench_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mha.py)
- [op_tests/op_benchmarks/triton/bench_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_prefill.py)
- [op_tests/op_benchmarks/triton/compare_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/compare_mha.py)
- [op_tests/test_flydsl_fmha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fmha.py)
- [op_tests/test_fmha_fwd_mxfp8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_mxfp8_asm.py)
- [op_tests/test_fmha_fwd_with_sink_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_asm.py)
- [op_tests/test_fmha_fwd_with_sink_varlen_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_varlen_asm.py)
- [op_tests/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha.py)
- [op_tests/test_mha_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_flydsl.py)
- [op_tests/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_fp8.py)
- [op_tests/test_mha_native_splitkv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_native_splitkv.py)
- [op_tests/test_mha_optional_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_optional_flydsl.py)
- [op_tests/test_mha_v4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4.py)
- [op_tests/test_mha_v4_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4_sparse.py)
- [op_tests/test_mha_varlen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen.py)
- [op_tests/test_mha_varlen_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_fp8.py)
- [op_tests/test_mha_varlen_large_kv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_large_kv.py)
- [op_tests/triton_tests/attention/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/__init__.py)
- [op_tests/triton_tests/attention/mha_test_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/mha_test_utils.py)
- [op_tests/triton_tests/attention/test_chunked_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_chunked_pa_prefill.py)
- [op_tests/triton_tests/attention/test_dsv4_indexer.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_dsv4_indexer.py)
- [op_tests/triton_tests/attention/test_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_extend_attention.py)
- [op_tests/triton_tests/attention/test_fav3_sage.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage.py)
- [op_tests/triton_tests/attention/test_fav3_sage_compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage_compile.py)
- [op_tests/triton_tests/attention/test_la_paged.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_la_paged.py)
- [op_tests/triton_tests/attention/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha.py)
- [op_tests/triton_tests/attention/test_mha_dao_ai.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_dao_ai.py)
- [op_tests/triton_tests/attention/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_fp8.py)
- [op_tests/triton_tests/attention/test_mha_spatial_swizzle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_spatial_swizzle.py)
- [op_tests/triton_tests/attention/test_mha_v3.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_v3.py)
- [op_tests/triton_tests/attention/test_mha_with_pe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_pe.py)
- [op_tests/triton_tests/attention/test_mha_with_sink.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_sink.py)
- [op_tests/triton_tests/attention/test_na3d_flash.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_na3d_flash.py)
- [op_tests/triton_tests/attention/test_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill.py)
- [op_tests/triton_tests/attention/test_prefill_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_prefill_attention.py)
- [op_tests/triton_tests/attention/test_qsa.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_qsa.py)
- [op_tests/triton_tests/fusions/test_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_attn_res.py)

## Performance and validation

Record sequence lengths, masks, heads and dtypes. Use the matching forward reference with dropout disabled; do not compare kernel latency with an end-to-end serving metric.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
