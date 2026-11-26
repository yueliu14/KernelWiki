---
title: "Dense matrix multiplication: AITER flydsl"
kind: implementation_knowledge
operator: dense_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Dense matrix multiplication — AITER flydsl

Separate the arithmetic kernel from weight preparation and dispatch. Compare tiles at the actual M/N/K, strides, output type and weight layout. Small M often needs a different grid or split-K policy from prefill.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/bpreshuffle_gemm_gfx1250.py) · **Recent**

Source documentation: gfx1250 (WMMA) backend for the FlyDSL a8w8 bpreshuffle GEMM.

Symbols: `_lazy_import`, `_as_1d_fp32`, `run_preshuffle_gemm_a8_gfx1250`, `parse_wmma_kernel_name`, `run_gemm_a8w8_bpreshuffle_gfx1250`.

Parameters referenced in this file: `BLOCK_K`, `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/gemm_a16w16_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_gfx1250.py) · **Recent**

Source documentation: gfx1250 FlyDSL backend for the A16W16 GEMM: torch-facing wrapper.

Symbols: `_lazy_import`, `_p`, `_cached_launcher`, `_split_k_counters`, `gemm_a16w16`.

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_a16w16_policy.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16w16_policy.py) · **Recent**

Source documentation: Shape-aware policy selection for the FlyDSL A16W16 GEMM.

Symbols: `GemmConfigPruner`, `gfx1250_lds_bytes`, `_gfx1250_configs`, `get_flydsl_a16w16_configs`.

Parameters referenced in this file: `split_k`.

### `aiter/ops/flydsl/gemm_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_kernels.py) · **Recent**

Source documentation: High-level FlyDSL GEMM APIs.

Symbols: `_normalize_dtype_name`, `flydsl_hgemm_kernel_name`, `get_flydsl_hgemm_kernel_params`, `flydsl_hgemm`, `_get_compile_fn`, `_get_preshuffle_split_buffers`, `_check_preshuffle_flat_buffer_capacity`, `_check_preshuffle_split_capacity`, `flydsl_preshuffle_gemm_a8`.

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gather_gemm_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_gemm_8wave.py) · **Recent**

Source documentation: 8-wave fused gather + kv_b_proj expansion for DeepSeek MLA on CDNA4 (gfx950).

Symbols: `_raw`, `lds_block_n`, `_gather_a_offsets`, `_WideG2SLoader`, `BlockScale`, `_pack_fp8`, `StoreKV`, `_RopeCopy`, `compile_gather_kv_b_proj_8w`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py) · **Recent**

Source documentation: Compute C[M, N] = A[M, K] @ B[K, N]. Each layout character controls only the corresponding tensor stride: N is row-major and T is column-major. Logical tensor shapes never change. Inputs that violate the selected layout or DMA alignment are rejected. Set user_kwargs["split_k"] above 1 to atomically reduce K partitions. Set user_kwargs["k_waves"] above 1 for full-tile workgroup-local slice-K. out_dtype may be the input dtype or torch.float32.

Symbols: `GemmA16W16Gfx950Param`, `GemmABLoadContext`, `AsyncLoadTile`, `make_gemm_a16w16_gfx950_param`, `make_gemm_a16w16_gfx950_kernel_name`, `make_gemm_ab_lds_layouts`, `make_gemm_ab_load_context`, `async_load_to_lds`, `write_cshuffle_vec_to_global`, `gemm_a16w16_gfx950_kernel`, `gemm_a16w16_hti_gfx950_kernel`, `gemm_a16w16_gfx950`, `make_gemm_a16w16_param_and_validate`, `assert_no_k_tail`, `get_split_k_buffers`, `_dynamic_tensor_arg`, `gemm_a16w16`.

Parameters referenced in this file: `split_k`.

### `aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_gfx950_utils.py) · **Recent**

Symbols: `wait_vmcnt_and_barrier`, `get_llvm_ptr`, `_global_sync_ptr`, `store_global_f32_vec`, `BlockSwizzle`, `SplitKProtocol`, `make_lds_layout`, `make_transposed_lds_layout`, `get_wave_lds_offset`, `swizzled_col_idx`, `transposed_contiguous_idx`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_THREADS`, `split_k`.

### `aiter/ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py) · **Recent**

Symbols: `_f32`, `apply_activation_scalar`, `compile_gemm_a16w16`.

Parameters referenced in this file: `num_buffers`, `num_warps`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_common_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py) · **Recent**

Source documentation: Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8).

Symbols: `make_lds_copy_ops`, `make_sgpr_opaque`, `_lds_ptr`, `lds_load_b128_raw`, `lds_load_b32_raw`, `lds_store_b128_raw`, `workgroup_barrier`, `pipeline_fence`, `pipeline_fence_signal`, `pipeline_fence_wait`, `fmin_f32`, `fclamp_f32`, `_fma_f32`, `fused_silu_swiglu_elem`, `_tanh_f32`, `situv2_consts`, `fused_situv2_elem`, `batched_situv2`, `batched_silu_swiglu`.

### `aiter/ops/flydsl/kernels/preshuffle_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/preshuffle_gemm.py) · **Recent**

Source documentation: Preshuffle GEMM (layout API): f16/bf16/fp8/int8, ping-pong scf.for loop with scheduler hints.

Symbols: `_get_preload`, `compile_preshuffle_gemm`.

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

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

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
