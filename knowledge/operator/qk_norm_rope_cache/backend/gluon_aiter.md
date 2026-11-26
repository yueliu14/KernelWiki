---
title: "Fused Q/K normalization, RoPE and cache writes: AITER gluon"
kind: implementation_knowledge
operator: qk_norm_rope_cache
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused Q/K normalization, RoPE and cache writes — AITER gluon

This fusion couples normalization with rotary position encoding and sometimes quantization, splitting and paged cache writes. Position ids, Q/K head counts, rotary convention and cache slot mappings are part of the contract.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/fusions/fused_qk_norm_rope_group_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/fusions/fused_qk_norm_rope_group_quant.py) · **Recent**

Symbols: `get_fused_qk_norm_rope_group_quant_layouts`, `norm_quant_rope`, `store_rows`, `fused_qk_norm_rope_group_quant_gfx1250`.

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_qk_norm_rope_group_quant/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/fusions/fused_qk_norm_rope_group_quant/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_qknorm_ar.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_qknorm_ar.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_gate_fp8_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_gate_fp8_quant.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_group_quant.py)
- [op_tests/op_benchmarks/triton/bench_fused_reduce_qk_norm_rope_swa_write.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_reduce_qk_norm_rope_swa_write.py)
- [op_tests/test_flydsl_qk_norm_rope_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_qk_norm_rope_quant.py)
- [op_tests/test_fused_qk_norm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm.py)
- [op_tests/test_fused_qk_norm_mrope_cache_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_mrope_cache_quant.py)
- [op_tests/test_fused_qk_norm_rope_1way_perhead.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_1way_perhead.py)
- [op_tests/test_fused_qk_norm_rope_2way_perhead.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_2way_perhead.py)
- [op_tests/test_fused_qk_norm_rope_cache_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_cache_quant.py)
- [op_tests/test_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_group_quant.py)
- [op_tests/test_fused_qknorm_idxrqknorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qknorm_idxrqknorm.py)
- [op_tests/triton_tests/fusions/test_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_qk_norm_rope_group_quant.py)
- [op_tests/triton_tests/fusions/test_fused_reduce_qk_norm_rope_swa_write.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_reduce_qk_norm_rope_swa_write.py)
- [op_tests/triton_tests/rope/test_fused_qk_norm_rope_cached.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_fused_qk_norm_rope_cached.py)
- [op_tests/triton_tests/rope/test_fused_qk_norm_rope_gate_fp8_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_fused_qk_norm_rope_gate_fp8_quant.py)
- [op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py)

## Performance and validation

Validate Q and K outputs plus modified cache locations. Include position tails, partial rotary dimensions and the quantization scale layout.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
