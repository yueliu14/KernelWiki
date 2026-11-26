---
title: "Fused Q/K normalization, RoPE and cache writes: AITER aiter"
kind: implementation_knowledge
operator: qk_norm_rope_cache
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused Q/K normalization, RoPE and cache writes — AITER aiter

This fusion couples normalization with rotary position encoding and sometimes quantization, splitting and paged cache writes. Position ids, Q/K head counts, rotary convention and cache slot mappings are part of the contract.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/fused_qk_norm_mrope_cache_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_norm_mrope_cache_quant.py) · **Recent**

Symbols: `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`.

### `aiter/ops/fused_qk_norm_rope_cache_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qk_norm_rope_cache_quant.py) · **Recent**

Source documentation: MiniMax TP1 qkv split with full-vector q/k RMSNorm and RoPE. Unlike the generic qk-norm RoPE kernels, MiniMax normalizes Q across num_heads_q * head_dim and K across num_heads_k * head_dim.

Symbols: `_fused_qk_norm_rope_cache_quant_shuffle_hip`, `fused_qk_norm_rope_cache_quant_shuffle`, `_fused_qk_rmsnorm_kernel`, `_minimax_qk_norm_rope_kernel`, `minimax_qk_norm_rope`, `_fused_qk_rmsnorm`, `fused_qk_norm_rope_cache_block_quant_shuffle`, `fused_qk_norm_rope_cache_pts_quant_shuffle`, `fused_qk_norm_rope_2way`, `fused_qk_norm_rope_1way`, `_fused_qk_norm_rope_group_quant_kernel`, `fused_qk_norm_rope_group_quant`, `_fused_qk_norm_rope_2way_fp8_perhead_quant_kernel`, `fused_qk_norm_rope_2way_fp8_perhead_quant`, `_fused_qk_norm_rope_1way_fp8_perhead_quant_kernel`, `fused_qk_norm_rope_1way_fp8_perhead_quant`, `_fused_kv_norm_rope_group_quant_kernel`, `fused_kv_norm_rope_group_quant`, `_v_2way_per_head_fp8_quant_kernel`, `v_2way_per_head_fp8_quant`, `_v_1way_per_head_fp8_quant_kernel`, `v_1way_per_head_fp8_quant`.

### `aiter/ops/fused_qknorm_idxrqknorm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_qknorm_idxrqknorm.py) · **Recent**

Symbols: `_is_fp8_e4m3_tensor`, `_fused_qknorm_idxrqknorm_hip`, `fused_qknorm_idxrqknorm`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_fused_qknorm_ar.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_qknorm_ar.py)
- [op_tests/test_flydsl_qk_norm_rope_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_qk_norm_rope_quant.py)
- [op_tests/test_fused_qk_norm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm.py)
- [op_tests/test_fused_qk_norm_mrope_cache_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_mrope_cache_quant.py)
- [op_tests/test_fused_qk_norm_rope_1way_perhead.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_1way_perhead.py)
- [op_tests/test_fused_qk_norm_rope_2way_perhead.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_2way_perhead.py)
- [op_tests/test_fused_qk_norm_rope_cache_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_cache_quant.py)
- [op_tests/test_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_group_quant.py)
- [op_tests/test_fused_qknorm_idxrqknorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qknorm_idxrqknorm.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_gate_fp8_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_gate_fp8_quant.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_group_quant.py)
- [op_tests/op_benchmarks/triton/bench_fused_reduce_qk_norm_rope_swa_write.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_reduce_qk_norm_rope_swa_write.py)
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
