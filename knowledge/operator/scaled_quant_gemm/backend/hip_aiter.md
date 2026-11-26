---
title: "Scaled and mixed-precision GEMM: AITER hip"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Scaled and mixed-precision GEMM — AITER hip

The quantization format, scale granularity and scale layout are part of the operator contract. Follow the wrapper from logical K to packed storage K and from source weights to the kernel-specific preshuffle.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/kernels/quant_mxfp4_gemm.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_mxfp4_gemm.cu) · **Recent**

Symbols: `swap_adjacent_lane`, `swap_lane_distance_two`, `load_as_float`, `load_group_values`, `hadamard32`, `group_amax`, `quant_mxfp4_gemm_group`, `__launch_bounds__`.

### `csrc/kernels/quant_mxfp6_gemm.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/quant_mxfp6_gemm.cu) · **Recent**

Symbols: `swap_adjacent_lane`, `swap_lane_distance_two`, `mxfp6_to_float`, `to_bf16_dot_operand`, `load_group_values`, `hadamard32`, `group_amax`, `quant_mxfp6_extreme_group`, `quant_mxfp6_group`, `__launch_bounds__`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py)
- [op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py)
- [op_tests/test_gemm_a16w8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w8_asm.py)
- [op_tests/test_gemm_a4w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w4.py)
- [op_tests/test_gemm_a4w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w6.py)
- [op_tests/test_gemm_a6w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w4.py)
- [op_tests/test_gemm_a6w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w6.py)
- [op_tests/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8.py)
- [op_tests/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale.py)
- [op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py)
- [op_tests/test_gemm_a8w8_blockscale_fallback.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_fallback.py)
- [op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py)
- [op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py)
- [op_tests/test_mxfp8fp4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp8fp4gemm.py)
- [op_tests/test_quant_mxfp4_gemm_fused.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp4_gemm_fused.py)
- [op_tests/test_quant_mxfp6_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp6_gemm.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py)
- [op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py)

## Performance and validation

Compare identical quantized inputs and scale tensors. Count quantization, shuffle and reduction kernels when they execute on the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
