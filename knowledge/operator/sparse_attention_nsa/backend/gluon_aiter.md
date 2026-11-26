---
title: "Sparse attention: AITER gluon"
kind: implementation_knowledge
operator: sparse_attention_nsa
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Sparse attention — AITER gluon

Sparse kernels consume selected pages, blocks or token indices. Selection granularity, duplicates, padding and causal masking affect the logical work independently of dense attention shapes.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_prefill_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_prefill_sparse.py) · **Recent**

Source documentation: Gluon (gfx1250) sparse prefill attention over two KV sources with sink. Grid: (T, cdiv(H, BLOCK_H)) — each CTA handles one query token and BLOCK_H query heads. Same grid as the decode kernel, but iterates two KV regions (prefix + extend) sequentially sharing the online softmax accumulator. No split-K: prefill has enough tokens to fill the GPU. Derived from pa_decode_sparse with KV_SPLITS=1, minus FP8/reduce, plus the second KV region.

Symbols: `_region_prologue`, `_pa_prefill_sparse`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py) · **Recent**

Source documentation: Gluon (gfx950) sparse decode for DeepSeek-V4 and MLA (GLM-5). One kernel serves two geometries, selected by the ROPE_SEPARATE constexpr: False: rope gets no LDS buffer of its own, either because it lives inside the pow-2 row (DSv4) or because there is none at all (GLM-5.3-Flash, ROPE_DIM=0). One LDS buffer, kv_smem [BLOCK_K, HEAD_SIZE], one QK MFMA chain, V = that same buffer. True (MLA): rope is appended (kv_lora_rank latent + rope = QK width; V = the latent only).

Symbols: `_max2`, `_rmax`, `_row_partials_layout`, `_row_reduce_lds`, `_cache_load`, `_fp8_to_f32`, `_fp8_to_bf16`, `_scale_load`, `_widen_scales`, `_split2`, `_split2_dim0`, `_split_ax`, `_scaled_upcast_asm`, `Cfg`, `Fmt`, `Seg`, `_deq_store`, `_deq_store_tile`, `_read_slots_at`, `_read_slots`, `_split_slot`, `_slots`, `_first_valid_slot`, `_with_target`, `_qk_scores`, `_tile_rows`, `_copy_lds_buffer`, `_copy_tile`, `_async_segment`, `_gather_full`, `_stage`, `_qkpv`, `_qkpv_lds`, `_decode_tile`, `_staged_segment`, `_dsv4_slot_ids`, `_split_slot_pow2`, `_dsv4_gather`, `_dsv4_scales`, `_dsv4_segment`, `_segment`, `_inv_rope`, `_mxfp8`, `_epilogue_store`, `_xcd_work`, `_sparse_mla`, `_next_pow2`, `_sparse_mla_reduce`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_SIZE`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py)

Source documentation: Gluon (gfx1250) port of _pa_decode_sparse with 2-stage software pipelining. Mirrors aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py (the merged split + fused variant). Slot tensor loads are synchronous (analogous to physical_block_idx loads in gluon/mla.py::_mla_decode_fwd_kernel); the KV cache itself is gathered via gfx1250.async_copy.global_to_shared into a 2-deep ring buffer so the next tile's gather is in flight while the math for the current tile runs.

Symbols: `_pa_decode_sparse`, `_pa_decode_sparse_reduce`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `BLOCK_K`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx942/gluon/attention/sparse_mla/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/gluon/attention/sparse_mla/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/attention/sparse_mla/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/attention/sparse_mla/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla.py)
- [op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sparse_mla_dsv4_train.py)
- [op_tests/test_jit_cache_transaction.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_jit_cache_transaction.py)
- [op_tests/test_pa_sparse_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_sparse_prefill.py)
- [op_tests/test_vsa_sparse_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_vsa_sparse_attention.py)
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
