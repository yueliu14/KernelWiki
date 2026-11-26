---
title: "Scaled and mixed-precision GEMM: AITER flydsl"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Scaled and mixed-precision GEMM — AITER flydsl

The quantization format, scale granularity and scale layout are part of the operator contract. Follow the wrapper from logical K to packed storage K and from source weights to the kernel-specific preshuffle.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/gemm_a16wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a16wfp4.py) · **Recent**

Source documentation: High-level gfx950 dense BF16 x MXFP4 FlyDSL API.

Symbols: `DenseGemmConfig`, `PreshuffledA16WFP4Weight`, `_require_gfx950`, `a16wfp4_shape_supported`, `a16wfp4_config_legal`, `_select_tile_n`, `_select_k_wave`, `_decode_tile_n`, `_decode_config`, `_select_gemm_config`, `_validate_buffer_limits`, `prepare_gemm_a16wfp4_weight`, `flydsl_gemm_a16wfp4`.

Parameters referenced in this file: `tile_k`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py) · **Recent**

Source documentation: 8-wave CDNA4 (gfx950) backend for the FlyDSL a8w8 ptpc bpreshuffle GEMM.

Symbols: `_lazy_import`, `lds_bytes`, `_validate`, `compile_8wave_gemm`, `_as_1d_fp32`, `_as_i8`, `flydsl_8wave_gemm_a8`, `parse_8wave_kernel_name`, `run_gemm_a8w8_bpreshuffle_8wave`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_a8w8_mxfp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8.py) · **Recent**

Source documentation: FlyDSL mxfp8 GEMM on a (16, 16)-preshuffled weight, on whichever arch has a kernel for the operands' w_scale block and scale layouts -- the 2-D counterpart of batched_gemm_a8w8. XQ [M, K] fp8, WQ [N, K] fp8 preshuffled, x_scale / w_scale e8m0, Out [M, N] bf16 / fp16. Each scale's layout is read off its rank: a 2-D scale is the unshuffled row-major one ("row": x_scale [M, K / 32], w_scale [N / WN, K / WK]), a 1-D one a pre-shuffled flat buffer ("mfma_tile").

Symbols: `_runner`, `_scale_layout`, `gemm_a8w8_mxfp8_supported`, `run_gemm_a8w8_mxfp8`.

### `aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_a8w8_mxfp8_8wave.py) · **Recent**

Source documentation: 8-wave CDNA4 (gfx950) backend for the FlyDSL mxfp8 (1x32 e8m0) bpreshuffle GEMM.

Symbols: `_lazy_import`, `lds_bytes`, `_validate`, `compile_mxfp8_8wave_gemm`, `_as_u8`, `flydsl_mxfp8_8wave_gemm`, `make_mxfp8_8wave_kernel_name`, `parse_mxfp8_8wave_kernel_name`, `run_gemm_a8w8_mxfp8_8wave`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py) · **Recent**

Source documentation: Tune spaces for every FlyDSL a8w8 (ptpc) bpreshuffle pipeline on CDNA. One operator, two pipelines, swept together under the same flydsl libtype so a single --libtype flydsl run picks one winner per shape: * preshuffle -- 4-wave MFMA, gfx942 + gfx950, fp8 or int8 weights. * 8wave -- 8-wave CDNA4 MFMA_Scale, gfx950 only, fp8 only.

Symbols: `kernelInstance`, `_ki`, `_smem_align`, `_smem_finalize_size`, `preshuffle_gemm_estimated_lds_bytes`, `kernel_instance_estimated_lds_bytes`, `max_lds_bytes_for_tune`, `_padded_m`, `kernel_fits_shape`, `_vgpr_per_simd`, `_estimate_max_wpe`, `k_split_candidates`, `_build_kernels_list`, `EightWaveKernelInstance`, `kernel_instance_estimated_lds_bytes_8wave`, `kernel_fits_shape_8wave`, `is_8wave_enabled`, `_build_kernels_list_8wave`, `Pipeline`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_K_8WAVE`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_wmma_common.py) · **Recent**

Source documentation: Full-style gfx1250 WMMA candidates for a8w8 (ptpc) bpreshuffle GEMM.

Symbols: `WmmaKernelInstance`, `_align_up`, `_ceil_div`, `_tile_valid`, `_cluster_valid`, `kernel_instance_estimated_lds_bytes`, `_build_kernels_list`, `kernel_fits_shape`.

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common.py) · **Recent**

Source documentation: Full-style gfx1250 WMMA candidates for mxfp8_128 bpreshuffle GEMM.

Symbols: `WmmaKernelInstance`, `_align_up`, `_ceil_div`, `_tile_valid`, `_cluster_valid`, `_mxfp8_128_stage_scale_bytes`, `kernel_instance_estimated_lds_bytes`, `_build_kernels_list`, `kernel_fits_shape`, `compute_grid_ctas`, `is_compute_kernel`.

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gemm_tune/flydsl_gemm_mxscale_preshuffle_common.py) · **Recent**

Source documentation: Tune catalog for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA). Mirrors the aiter a8w8-bpreshuffle tune catalog style (kernelInstance + name + build_kernels_list), but the mxscale_preshuffle kernel exposes only tile_m/n/k + waves_per_eu as perf knobs (no async/lds/cshuffle/xcd/scheduler/split-K), so the candidate schema and kernelName grammar are correspondingly small.

Symbols: `_estimate_max_wpe`, `_a_row_bytes`, `make_kernel_name`, `kernelInstance`, `parse_kernel_name`, `row_scale_panel`, `col_scale_panel`, `estimated_lds_bytes`, `_max_lds_bytes`, `instance_valid`, `fits_shape`, `_build_kernels_list`, `candidates_for`, `EightWaveKernelInstance`, `fits_shape_8wave`, `Pipeline`, `_fits_mxpsh_row`, `_fits_mxpsh_row_w32x32`, `_fits_8wave_w32x32`, `_fits_mxpsh_row_w128x128`, `_fits_8wave_w128x128`.

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/mxfp4_gemm1_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm1_kernels.py) · **Recent**

Source documentation: Launch GEMM1; v2 output keeps payload rows in expert-sorted order.

Symbols: `_get_compiled_mxfp4_gemm1_port`, `_assert_supported`, `flydsl_mxfp4_gemm1`.

### `aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py) · **Recent**

Source documentation: gfx1250 FlyDSL backend for mxfp8_128 / mxfp8_32 bpreshuffle GEMM. mxfp8_128: fp8 e4m3 activations/weights with fp8_e8m0 block scales over a 128-element block (distinct from the fp32-scale blockscale GEMM). mxfp8_32: A scales are row-major (M, K//32) by default, optionally 32x4 shuffled; B scales are always 32x4 shuffled (N, K//32).

Symbols: `_fused_splitk_ok`, `_lazy_import`, `_require_e8m0_scale`, `check_persistent_n_tiles`, `_run_mxfp8_bpreshuffle_gemm_a8_gfx1250`, `_parse_name`, `parse_wmma_kernel_name`, `parse_mxfp8_32_wmma_kernel_name`, `compute_kernel_k_pair`, `cluster_m_grid_ok`, `resolve_cluster_m`, `cluster_m_fallback_values`, `is_compute_wmma_kernel_name`, `_resolved_wmma_config`, `run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250`, `run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250`.

Parameters referenced in this file: `BLOCK_K`, `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm/mxscale_preshuffle.py) · **Recent**

Source documentation: MXFP4/MXFP6/MXFP8 A x MXFP4/MXFP8 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded into a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).

Symbols: `_load_partial`, `_make_dma_layouts`, `_scale_mma_atoms`, `_bq_view`, `_launch_gemm_impl`, `launch_gemm`, `launch_gemm_fused`, `launch_splitk_reduce`.

Parameters referenced in this file: `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a16wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4.py) · **Recent**

Source documentation: gfx950 dense BF16 x MXFP4 GEMM kernel. The activation is consumed as BF16 without quantization. Preshuffled packed E2M1 weights and E8M0 block scales are loaded directly into registers, where v_cvt_scalef32_pk_bf16_fp4 produces transient BF16 fragments for mfma_f32_16x16x32_bf16. A dequantized weight matrix is never written to global memory. The K loop double-buffers A-LDS. Prefill (k_wave=1) still issues the next A/B tile before MFMA.

Symbols: `compile_gemm_a16wfp4`.

Parameters referenced in this file: `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a16wfp4_helpers.py) · **Recent**

Source documentation: Components for the gfx950 dense BF16 x MXFP4 GEMM.

Symbols: `DenseGemmTraits`, `make_dense_operand_loaders`, `make_dense_accumulators`, `load_b_tile`, `advance_k_tile`, `load_a_frags`, `_mma_order`, `compute_dense_tile`, `reduce_dense_k_wave`, `make_dense_store`, `store_dense_bf16`.

Parameters referenced in this file: `tile_k`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a8w8_256x256_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_256x256_gfx1250.py) · **Recent**

Source documentation: A8W8 (FP8 E4M3 activation x FP8 E4M3 weight) 256x256 / 128x256 MXscale GEMM for gfx1250.

Symbols: `launch_gemm_a8w8_256x256`.

Parameters referenced in this file: `num_buffers`, `num_warps`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py) · **Recent**

Source documentation: 8-wave FP8 matmul with row-wise scaling for AMD CDNA4 (gfx950 only).

Symbols: `make_fp8_buffer_tensor`, `swizzle_128`, `compute_global_swizzle`, `pack_i32x4_i32x8`, `wait_barrier`, `G2SLoader`, `S2RLoader`, `StoreC`, `Mfma16x16x128`, `_xcd_swizzle_any`, `compile_fp8_gemm_8w`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py) · **Recent**

Source documentation: FP8xFP8 preshuffle GEMM kernels for gfx1250.

Symbols: `launch_gemm_a8w8`.

Parameters referenced in this file: `num_buffers`, `num_warps`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mxfp4_gemm1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm1.py) · **Recent**

Source documentation: Pick the deepest K pipeline that still fits two workgroups per CU.

Symbols: `gemm1_grid`, `_gemm1_body`, `_bm_constants`, `default_k_stages`, `default_epi_splits`, `compile_gemm1_a4w4_port`.

### `aiter/ops/flydsl/kernels/mxfp4_gemm2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm2.py) · **Recent**

Symbols: `aq_bytes_for`, `saq_slot_bytes`, `tiling`, `_issue_a_load_lds`, `compile_gemm2_a4w4_port`, `_gemm2_body`, `_flat_bf16_epilog`, `_cshuffle_flat_bf16_epilog`, `_flat_mxfp4_epilog`, `_atomic_bf16_epilog`.

### `aiter/ops/flydsl/kernels/mxfp4_gemm_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_gemm_common.py) · **Recent**

Source documentation: XOR16 swizzle for an FP8 LDS row whose width is 128 or 256 bytes.

Symbols: `_raw`, `_udiv`, `_umod`, `_scale_mma_atoms`, `_lds_ptr3`, `_gep3`, `_global_base_ptr1`, `_gep1`, `_global_ptr1`, `_global_i32_at`, `_global_f32_at`, `_global_i32_load`, `_global_scalar_tiles`, `_scalar_store`, `_layout_idx`, `_buffer_rsrc`, `_lds_swizzle_mask`, `lds_swizzle_mask_f8`, `lds_dma_dst`, `global_typed_ptr`, `_global_i32_buffer_view`, `_global_i32_buffer_tiles`, `lds_typed_ptr`, `lds_vec_load`, `lds_dma_atom_128`, `flat_buffer_view`, `_fabs_f32`, `_e8m0_roundup`, `_e8m0_from_amax`, `_inline_e8m0`, `_pkmax_u16`, `_clamp_gate_up_batch`, `_swiglu_mul_batch`, `_situ_mul_batch`, `_activation_mul_batch`, `_umax_i32`, `_dpp_umax_step`, `_inline_dpp_quad_amax`, `_inline_dpp_pair_amax`, `k_half_for`, `k_tiles_total_for`, `kunroll_for`, `kas_c_k1_for`, `kbs_c_k1_for`, `num_n_blocks_for`, `kbs_c_n1_for`, `bq_bytes_for`, `bscale_bytes_for`, `kmchunks_for`, `lds_acc_bytes_for`, `fp8out_scale_blk`, `fp8out_row_bytes`.

### `aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp8_gemm_8wave.py) · **Recent**

Source documentation: 8-wave MXFP8 matmul (per-1x32 E8M0 scales) for AMD CDNA4 (gfx950 only). Ported from FlyDSL kernels/gemm/mxfp8_gemm_8wave.py (incl. ROCm/FlyDSL#1275, row-major scales). The K pipeline is the a8w8 ptpc 8-wave one (gemm_a8w8_8wave), whose loaders it shares; the MFMA carries the E8M0 scales, so the epilogue only converts. MxMfma issues one scaled atom per 16x16 tile with scale_a= / scale_b=, the form the pinned FlyDSL takes.

Symbols: `ScalePreshuffledS2R`, `ScaleRawLds`, `MxMfma`, `StoreBf16`, `compile_mxfp8_gemm_8w`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `waves_per_eu`.

### `aiter/ops/flydsl/mxfp4_gemm2_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mxfp4_gemm2_kernels.py)

Symbols: `_epilog_of`, `_get_compiled_mxfp4_gemm2_port`, `_dummy_out_scale`, `_assert_supported`, `flydsl_mxfp4_gemm2`.

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

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
