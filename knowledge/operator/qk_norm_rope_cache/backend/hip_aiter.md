---
title: "Fused Q/K normalization, RoPE and cache writes: AITER hip"
kind: implementation_knowledge
operator: qk_norm_rope_cache
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused Q/K normalization, RoPE and cache writes — AITER hip

This fusion couples normalization with rotary position encoding and sometimes quantization, splitting and paged cache writes. Position ids, Q/K head counts, rotary convention and cache slot mappings are part of the contract.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/fused_qk_norm_rope_cache_quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fused_qk_norm_rope_cache_quant.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/fused_qknorm_idxrqknorm.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fused_qknorm_idxrqknorm.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/fused_qk_norm.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_qk_norm.cu) · **Recent**

Symbols: `fused_qk_rmsnorm_kernel`.

### `csrc/kernels/fused_qk_norm_mrope_cache_quant.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_qk_norm_mrope_cache_quant.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/fused_qk_norm_rope_cache_quant.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_qk_norm_rope_cache_quant.cu) · **Recent**

Symbols: `warpReduceSum`, `divUp`, `abs`, `fusedQKNormRopeQuantCacheShuffleKernel`, `fusedQKNormRopeBlockQuantCacheShuffleKernel`, `fused_rope_rms_2way_kernel`, `fused_rope_rms_1way_kernel`, `fused_rope_rms_1way_quad_kernel`, `fused_rope_rms_2way_amax_kernel`, `qk_partial_amax_to_perhead_scale_kernel`, `static_fp8_quant_perhead_kernel`, `minimax_apply_neox_rope_pack`, `minimax_apply_gptj_rope_pack`, `minimax_qk_norm_rope_kernel`, `qk_1way_output_partial_amax_kernel`, `atomic_fmax_pos`, `v_amax_to_descale_kernel`, `__launch_bounds__`, `__launch_bounds__`, `__launch_bounds__`, `__launch_bounds__`, `swa_row_for_token`, `load_rope_cos_sin`, `k_wave_norm_rope_group_quant_impl`, `constexpr`, `fuse_qk_norm_rope_group_quant_cache_kernel_impl`, `constexpr`, `q_rope_gptj`, `fuse_qk_norm_rope_finegrained_impl`, `constexpr`, `__launch_bounds__`, `__launch_bounds__`, `fuse_kv_norm_rope_group_quant_cache_kernel_impl`, `__launch_bounds__`.

### `csrc/kernels/fused_qknorm_idxrqknorm.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_qknorm_idxrqknorm.cu) · **Recent**

Symbols: `warpReduceSum`, `warpReduceMax`, `fp8Max`, `loadElems`, `storeElems`, `copyRawElems`, `storeCacheElems`, `storeIndexQElems`, `copyOrQuantizeCacheElems`, `storeCacheElemsStrided`, `normAndRope`, `fusedQKNormIdxrQKNormKernel`.

### `csrc/include/fused_qk_norm_mrope_cache_quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fused_qk_norm_mrope_cache_quant.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

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

## Additional Q/K normalization and quantization fusion

The [fused Q/K RMSNorm wrapper](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/fused_qk_rmsnorm_group_quant.py) and [native kernel body](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/csrc/kernels/fused_qk_rmsnorm_group_quant.cu) expose preparation variants for quantized attention inputs. They were previously filed with standalone normalization rather than this attention-fusion family.

`fused_qk_rmsnorm_group_quant` supports a quantized-Q path with FP8 or packed FP4 output when the caller supplies output and scale buffers. Packed FP4 rejects transposed scales at this snapshot. Per-token quantization uses the separate `fused_qk_rmsnorm_per_token_quant` entry; passing group size zero through the group-quantized path is rejected. These routines do not by themselves imply RoPE or a paged cache write: compare the actual fused boundary required by the attention consumer.

Validate the Q/K normalization results, quantized Q payload and scales together using the [group-quant tests](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_fused_qk_rmsnorm_group_quant.py) and [per-token tests](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/op_tests/test_fused_qk_rmsnorm_per_token_quant.py). This addition is pinned to AITER `b3ad6c461703` and adds no GPU measurements.

## Performance and validation

Validate Q and K outputs plus modified cache locations. Include position tails, partial rotary dimensions and the quantization scale layout.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
