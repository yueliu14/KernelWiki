---
title: "Sparse attention: AITER aiter"
kind: implementation_knowledge
operator: sparse_attention_nsa
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Sparse attention — AITER aiter

Sparse kernels consume selected pages, blocks or token indices. Selection granularity, duplicates, padding and causal masking affect the logical work independently of dense attention shapes.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/mla_sparse_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mla_sparse_prefill.py) · **Recent**

Source documentation: Sparse prefill attention with split fp8 NoPE and bf16 RoPE inputs (asm). Signature-compatible with :func:aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus. Args: q_nope: [T, H, 512] fp8 query without positional encoding. q_rope: [T, H, 64] bf16 query RoPE encoding part. unified_kv_nope: [total_pages, 512] fp8 prefix KV NoPE source. unified_kv_rope: [total_pages, 64] bf16 prefix KV RoPE source.

Symbols: `mla_sparse_prefill_fp8_asm_fwd`, `_mla_sparse_prefill_fp8_asm_fake`, `mla_sparse_prefill_fp8_asm`.

### `aiter/ops/msa_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/msa_attention.py) · **Recent**

Source documentation: Sparse block selection for index-head attention.

Symbols: `pa_sparse_block_score_decode`, `pa_sparse_block_score_prefill`, `pa_sparse_block_topk`, `pa_sparse_block_topk_cp`.

### `aiter/ops/qsa.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/qsa.py) · **Recent**

Source documentation: QSA top-level dispatch: selects the best backend (triton, FlyDSL, etc.) for each QSA operator based on the input shape and available backends. Consumers should import from here rather than from the backend-specific modules directly.

Symbols: `qsa_paged_mqa_logits`, `qsa_select_paged_tokens`.

### `aiter/ops/vsa_sparse_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/vsa_sparse_attention.py) · **Recent**

Source documentation: Run VSA block-sparse attention on contiguous BHSD tensors. block_lut stores one absolute K-block index followed by delta-encoded indices for each 128-token Q block. block_counts gives the number of active entries in each LUT row. The final LUT slot is reserved as a lookahead sentinel by the current CK pipeline.

Symbols: `_validate_vsa_inputs`, `_vsa_sparse_attention_fake`, `_vsa_sparse_attention_fwd`, `vsa_sparse_attention`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_jit_cache_transaction.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_jit_cache_transaction.py)
- [op_tests/test_pa_sparse_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_sparse_prefill.py)
- [op_tests/test_vsa_sparse_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_vsa_sparse_attention.py)
- [op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py)
- [op_tests/triton_tests/attention/test_pa_decode_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_decode_sparse.py)
- [op_tests/triton_tests/attention/test_pa_prefill_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill_sparse.py)
- [op_tests/triton_tests/attention/test_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla.py)
- [op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_sparse_mla_dsv4_train.py)
- [op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py)

## Performance and validation

Include index preparation when comparing complete operators and validate empty selections, tail blocks and causal boundaries.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
