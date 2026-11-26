---
title: "Dense matrix multiplication: AITER hip"
kind: implementation_knowledge
operator: dense_gemm
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Dense matrix multiplication — AITER hip

Separate the arithmetic kernel from weight preparation and dispatch. Compare tiles at the actual M/N/K, strides, output type and weight layout. Small M often needs a different grid or split-K policy from prefill.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/kernels/iq2r/iq2r_gemm_gfx950.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu) · **Recent**

Symbols: `physical_tile`, `triplet_base`, `lane_record_offset`, `metadata_offset`, `load_u32`, `wave_min`, `__launch_bounds__`, `__launch_bounds__`, `__launch_bounds__`, `decode_fp8`, `iq2r_materialize_kernel`, `iq2r_wait_vmcnt`, `iq2r_triplet_mfma`, `iq2r_load_compact_triplet`, `iq2r_issue_quad`, `iq2r_read_quad`, `iq2r_scheduled_load_dwordx3`, `iq2r_scheduled_load_dwordx4`, `iq2r_scheduled_load_compact_uniform`, `iq2r_scheduled_buffer_load1`, `iq2r_scheduled_load_quad`, `glm53_quad_decode_mfma`, `__launch_bounds__`, `glm53_remap8`, `__launch_bounds__`, `glm53_row_offset`, `glm53_apply_signs`, `glm53_decode_packed_atom`, `glm53_down_tp4_atom`, `__launch_bounds__`, `glm53_ws_source_base`, `__launch_bounds__`, `__launch_bounds__`, `__launch_bounds__`, `__launch_bounds__`, `glm53_sign_word`, `glm53_quad_decode_mfma`, `__launch_bounds__`, `glm53_issue_codebook`, `glm53_quad_decode_mfma_cached`, `__launch_bounds__`, `glm53_cross_k_decode_mfma`, `__launch_bounds__`.

### `csrc/include/gemm_common.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/gemm_common.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/gemm_dispatch_utils.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/gemm_dispatch_utils.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_gemm_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py)
- [op_tests/test_deepgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_deepgemm.py)
- [op_tests/test_f4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_f4gemm.py)
- [op_tests/test_flydsl_hgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_hgemm.py)
- [op_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w16.py)
- [op_tests/test_gemm_codegen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_codegen.py)
- [op_tests/test_gemm_group32_interface.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_group32_interface.py)
- [op_tests/test_opus_a16w16_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_opus_a16w16_gemm.py)
- [op_tests/triton_tests/gemm/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/__init__.py)
- [op_tests/triton_tests/gemm/basic/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/__init__.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py)
- [op_tests/triton_tests/gemm/batched/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/__init__.py)
- [op_tests/triton_tests/gemm/fused/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/__init__.py)
- [op_tests/triton_tests/gemm/grouped/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/__init__.py)
- [op_tests/triton_tests/gemm/grouped/test_gmm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/test_gmm.py)
- [op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py)

## Performance and validation

Measure packing and split-K reduction separately, then include both in the deployed operator cost. Keep accumulation and epilogue semantics unchanged.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
