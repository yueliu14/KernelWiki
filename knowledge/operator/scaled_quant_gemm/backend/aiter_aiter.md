---
title: "Scaled and mixed-precision GEMM: AITER aiter"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Scaled and mixed-precision GEMM — AITER aiter

The quantization format, scale granularity and scale layout are part of the operator contract. Follow the wrapper from logical K to packed storage K and from source weights to the kernel-specific preshuffle.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/gemm_op_a16w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a16w8.py) · **Recent**

Source documentation: out = (A @ B.T) * B_scale with BF16 activations and FP8 weights, for decode on gfx950. The activations stay in BF16, so there is no activation quantization step and no activation quantization error. The weights are read as FP8, half the bytes of BF16 weights. A: [M, K] bf16, contiguous, with M from 1 to 8. B: [N, K] torch.float8_e4m3fn, contiguous. B_scale: [N] or [N, 1] fp32, one scale per row of B. out: [M, N] bf16, contiguous. A new tensor is returned when out is None.

Symbols: `_gemm_a16w8_asm`, `gemm_a16w8_asm`.

### `aiter/ops/gemm_op_a4w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a4w4.py) · **Recent**

Source documentation: A4W4 GEMM (4-bit quantized matmul) returning one plain-dtype tensor. On gfx1250 the call is dispatched to the dedicated F4GEMM asm path. MXFP4 vs NVFP4 is selected by the presence of global_A_scale/ global_B_scale (NVFP4 per-tensor global scales).

Symbols: `compute_gemm_SplitK`, `get_GEMM_config`, `_f4gemm_asm_dispatch`, `gemm_a4w4_fake`, `gemm_a4w4`, `_gemm_a4w4_asm`, `gemm_a4w4_asm`, `_mxfp4_gemm_asm`, `_nvfp4_gemm_asm`, `_is_mxfp8_out`, `_alloc_f4gemm_out`, `_alloc_f4gemm_out_scale`, `unpack_mxfp8_out_scale`, `gemm_mxfp4_asm`, `gemm_nvfp4_asm`, `_as_global_scale`, `gemm_a4w4o8_fake`, `gemm_a4w4o8`, `gen_gemm_a4w4_blockscale_fake_tensors`, `gen_gemm_a4w4_blockscale_tune_fake_tensors`, `_gemm_a4w4_blockscale`, `gemm_a4w4_blockscale`, `_gemm_a4w4_blockscale_tune`, `gemm_a4w4_blockscale_tune`.

Parameters referenced in this file: `splitK`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/gemm_op_a4w6.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a4w6.py) · **Recent**

Source documentation: gfx950 A4W6 GEMM.

Symbols: `_default_gemm_a4w6_kernel`, `_get_GEMM_A4W6_config_cached`, `get_GEMM_A4W6_config`, `clear_gemm_a4w6_config_cache`, `_compiled_gemm_a4w6_configs`, `_select_gemm_a4w6_kernel`, `_gemm_a4w6_asm`, `gemm_a4w6_asm`, `gemm_a4w6`.

### `aiter/ops/gemm_op_a6w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a6w4.py) · **Recent**

Source documentation: gfx950 A6W4 GEMM. Activations use the existing Hadamard-rotated MXFP6 packing contract. Weights use the same per-1x32 rotation and E8M0 scale layout, but store E2M1 values in a compact 16-byte-per-lane C0 tile.

Symbols: `quant_mxfp4_gemm_hip_out`, `_launch_quant_mxfp4_gemm_hip_out`, `quant_mxfp4_gemm_hip_out`, `_default_gemm_a6w4_kernel`, `_get_GEMM_A6W4_config_cached`, `get_GEMM_A6W4_config`, `clear_gemm_a6w4_config_cache`, `_compiled_gemm_a6w4_configs`, `_select_gemm_a6w4_kernel`, `mxfp4_gemm_pack_size`, `_pack_mxfp4_gemm_torch`, `_validate_mxfp4_gemm_input`, `_normalize_round_mode`, `_quant_mxfp4_gemm_torch_legacy`, `quant_mxfp4_gemm`, `_gemm_a6w4_asm`, `gemm_a6w4_asm`, `gemm_a6w4`.

### `aiter/ops/gemm_op_a6w6.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a6w6.py) · **Recent**

Source documentation: Quantize into caller-provided gfx950 GEMM-layout buffers.

Symbols: `_normalize_quant_backend`, `_is_gfx950_device_index`, `_compiled_is_gfx950_device`, `_is_gfx950_device`, `quant_mxfp6_gemm_hip_out`, `_launch_quant_mxfp6_gemm_hip_out`, `quant_mxfp6_gemm_hip_out`, `_e2m3_table`, `_hadamard32_np`, `_had32_t`, `_rotate_k32_torch`, `_rotate_k32_quant_work`, `_rotate_k32_np`, `quant_mxfp6`, `_pack32`, `pack_big`, `pack_scale`, `_tables`, `quant_mxfp6_torch`, `dequant_mxfp6_torch`, `_pack32_torch`, `pack_big_torch`, `pack_scale_torch`, `_ceil`, `_load_gemm_a6w6_configs`, `clear_gemm_a6w6_config_cache`, `_default_gemm_a6w6_kernel`, `get_GEMM_A6W6_config`, `_compiled_gemm_a6w6_configs`, `_select_gemm_a6w6_kernel`, `_compute_mxfp6_gemm_pack_layout`, `_cached_mxfp6_gemm_pack_layout`, `_mxfp6_gemm_pack_layout`, `mxfp6_gemm_pack_size`, `_launch_quant_mxfp6_gemm_triton`, `quant_mxfp6_gemm_out`, `quant_mxfp6_gemm`, `_gemm_a6w6_asm`, `gemm_a6w6_asm`, `gemm_a6w6`.

Parameters referenced in this file: `BLOCK_M`, `num_warps`, `splitK`.

### `aiter/ops/gemm_op_a8w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a8w4.py) · **Recent**

Source documentation: gfx1250 MXFP8 (activation) x MXFP4 (weight) GEMM (a8w4). D[M,N] bf16 = A @ B^T with e8m0 block scales. Select kernel and split count from b_intype=mxfp4 rows in the shared tuned CSV unless overridden. Try exact M first, then get_padded_m at granularity levels 0 and 1; all other key fields must match exactly. Reuse a row only if its saved and actual shapes satisfy the kernel guards. Lookup does not pad inputs or change output shape.

Symbols: `_mxfp8_mxfp4_gemm_asm`, `_gemm_a8w4_mxfp8_fake`, `gemm_a8w4_mxfp8`.

Parameters referenced in this file: `tile_m`, `tile_n`.

### `aiter/ops/gemm_op_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a8w8.py) · **Recent**

Source documentation: Notes for use gemm_a8w8_ASM: 1. WQ(weight) must be shuffle, you can use 'weightshuffle = shuffle_weight(weight,layout=(32,16))' 2. Use asm gemm must give bias, if not have bias, please give 'bias=torch.zeros(n,dtype=dtypes.fp32,device='cuda')'

Symbols: `_hip_blockscale_supported`, `_ck_a8w8_supported`, `gen_gemm_a8w8_ck_fake_tensors`, `gemm_a8w8_ck`, `gen_gemm_a8w8_bpreshuffle_ck_fake_tensors`, `gemm_a8w8_bpreshuffle_ck`, `gen_gemm_a8w8_bpreshuffle_cktile_fake_tensors`, `gemm_a8w8_bpreshuffle_cktile`, `_parse_flydsl_kernel_name`, `gemm_a8w8_bpreshuffle_flydsl`, `gemm_a8w8_mxfp8_128_bpreshuffle_flydsl`, `_gemm_a8w8_asm`, `gemm_a8w8_asm`, `gen_gemm_a8w8_blockscale_ck_fake_tensors`, `gemm_a8w8_blockscale_ck`, `gemm_a8w8_blockscale_cktile`, `gemm_a8w8_blockscale_bpreshuffle_ck`, `gemm_a8w8_blockscale_bpreshuffle_cktile`, `_flatmm_a8w8_blockscale_asm`, `flatmm_a8w8_blockscale_asm`, `_gemm_a8w8_blockscale_bpreshuffle_asm`, `get_zero_bias_buf_keyed`, `get_zero_bias_buf`, `gemm_a8w8_blockscale_bpreshuffle_asm`, `compute_gemm_SplitK`, `get_CKGEMM_config`, `get_GEMM_config_with_quant_type`, `gemm_a8w8_fake`, `gemm_a8w8`, `gemm_a8w8_ASM`, `gemm_a8w8_CK`, `gemm_a8w8_bpreshuffle_fake`, `gemm_a8w8_bpreshuffle`, `_blockscale_triton`, `gemm_a8w8_blockscale_fake`, `_group32_w_scale_block`, `_load_mxscale_bpreshuffle_tuned`, `get_mxscale_bpreshuffle_config`, `_gemm_mxscale_bpreshuffle`, `gemm_a8w8_blockscale`, `flatmm_a8w8_blockscale_ASM`, `_flydsl_mxfp8_fallback_kernel`, `gemm_a8w8_blockscale_bpreshuffle_fake`, `gemm_a8w8_blockscale_bpreshuffle`, `_abpreshuffle_config_from_bpreshuffle`, `gemm_a8w8_blockscale_abpreshuffle_fake`, `gemm_a8w8_blockscale_abpreshuffle`, `gfx950_a8w8_blockscale_ASM`, `gen_gemm_a8w8_tune_fake_tensors`, `gemm_a8w8_tune`, `gen_gemm_a8w8_blockscale_tune_fake_tensors`, `gemm_a8w8_blockscale_tune`, `gemm_a8w8_blockscale_cktile_tune`, `gemm_a8w8_bpreshuffle_tune`, `gemm_a8w8_blockscale_bpreshuffle_cktile_tune`, `gemm_a8w8_blockscale_bpreshuffle_tune`, `gemm_a8w8_bpreshuffle_cktile_tune`, `_mxfp8_mxfp8_gemm_asm`, `_mxfp8fp4_gemm_validate`, `_reduce_mxfp8_partials`, `_mxfp8_config_int`, `_load_mxfp8_gemm_configs`, `_mxfp8_kernel_configs`, `_normalize_mxfp8_splitk`, `_validate_mxfp8_splitk`, `_validate_mxfp8_tuned_config`, `get_mxfp8_gemm_config`, `_resolve_mxfp8_gemm_config`, `_gemm_a8w8_mxfp8_fake`, `gemm_a8w8_mxfp8`, `_mxfp8_bpreshuffle_tuned_nk`, `mxfp8_bpreshuffle_tuned`, `_mxfp8_32_fallback_kernel_name`, `_load_mxfp8_bpreshuffle_tuned`, `_get_mxfp8_bpreshuffle_config`, `gemm_a8w8_mxfp8_bpreshuffle_fake`, `gemm_a8w8_mxfp8_bpreshuffle`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `matrix_instr_nonkdim`, `num_buffers`, `num_stages`, `num_warps`, `splitK`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/gemm_op_mixed_mxfp.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_mixed_mxfp.py) · **Recent**

Source documentation: Shared runtime tuning helpers for mixed MXFP GEMMs.

Symbols: `_get_device_gfx_cu`, `_load_mixed_mxfp_configs`, `_find_mixed_mxfp_config`.

Parameters referenced in this file: `splitK`.

### `aiter/ops/mxfp8fp4gemm_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mxfp8fp4gemm_common.py) · **Recent**

Source documentation: MXFP8 ASM configuration, using the shared tuned/untuned CSV convention.

Symbols: `_Mxfp8Config`, `get_mxfp8_config_file`, `get_mxfp8_asm_dir`, `mxfp8_compile_guard`.

Parameters referenced in this file: `splitK`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/configs/a4w4_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a4w4_blockscale_tuned_gemm.csv)
- [aiter/configs/a4w4_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a4w4_blockscale_untuned_gemm.csv)
- [aiter/configs/a4w6_asm_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a4w6_asm_tuned_gemm.csv)
- [aiter/configs/a4w6_asm_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a4w6_asm_untuned_gemm.csv)
- [aiter/configs/a6w4_asm_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a6w4_asm_tuned_gemm.csv)
- [aiter/configs/a6w4_asm_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a6w4_asm_untuned_gemm.csv)
- [aiter/configs/a6w6_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a6w6_blockscale_tuned_gemm.csv)
- [aiter/configs/a6w6_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a6w6_blockscale_untuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_group32_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_group32_tuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_group32_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_group32_untuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_untuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_tuned_gemm.csv)
- [aiter/configs/a8w8_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_blockscale_untuned_gemm.csv)
- [aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv)
- [aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv)
- [aiter/configs/a8w8_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_tuned_gemm.csv)
- [aiter/configs/a8w8_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_untuned_gemm.csv)
- [aiter/configs/asm_a8w8_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/asm_a8w8_gemm.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv4.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv4.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_glm5.2.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_glm5.2.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_mm2.5.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_mm2.5.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen38_27b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen38_27b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_235b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_235b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_dsv3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_dsv3.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_mm2.5.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_mm2.5.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_qwen3.5_397b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_qwen3.5_397b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_qwen38_27b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_qwen38_27b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_qwen3_235b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_untuned_gemm_qwen3_235b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_gemma4_31b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_gemma4_31b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_glm5.2.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_glm5.2.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_glm5_1.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_glm5_1.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_glm5_3_flash.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_glm5_3_flash.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen36_27b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen36_27b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen38_27b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen38_27b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_14b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_14b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_235b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_235b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_5_122b_a10b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_5_122b_a10b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_5_397b_a13b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_5_397b_a13b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_vl_235b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_vl_235b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_vl_32b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_vl_32b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_ds_v3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_ds_v3.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_glm5_3_flash.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_glm5_3_flash.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen38_27b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen38_27b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen3_235b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen3_235b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen3_next_80b_a3b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen3_next_80b_a3b.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm47.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm47.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5.2.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5.2.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5_3_flash.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5_3_flash.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_kimik3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_kimik3.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_minimax_m3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_minimax_m3.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_qwen3_5_397b_a17b_mxfp4_attnfp8.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_qwen3_5_397b_a17b_mxfp4_attnfp8.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_glm47.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_glm47.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_glm5.2.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_glm5.2.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_glm5_3_flash.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_glm5_3_flash.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_kimik3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_kimik3.csv)
- [aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_minimax_m3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_bpreshuffle_untuned_gemm_minimax_m3.csv)
- [aiter/configs/model_configs/dsr1_a8w8_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsr1_a8w8_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv3_a4w4_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv3_a4w4_blockscale_untuned_gemm.csv)
- [aiter/configs/model_configs/dsv3_a8w8_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv3_a8w8_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv41_a8w8_blockscale_group32_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_a8w8_blockscale_group32_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv41_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_a8w8_blockscale_abpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_a8w8_blockscale_abpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_a8w8_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_a8w8_blockscale_untuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/gemma4_31b_a4w4_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gemma4_31b_a4w4_blockscale_tuned_gemm.csv)
- [aiter/configs/model_configs/gemma4_31b_a4w4_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gemma4_31b_a4w4_blockscale_untuned_gemm.csv)
- [aiter/configs/model_configs/glm53_flash_a8w8_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm53_flash_a8w8_blockscale_tuned_gemm.csv)
- [aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_untuned_gemm.csv)
- [aiter/configs/model_configs/glm5_a8w8_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_a8w8_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/glm5_a8w8_bpreshuffle_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_a8w8_bpreshuffle_untuned_gemm.csv)
- [aiter/configs/model_configs/kimik3_a4w4_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a4w4_blockscale_tuned_gemm.csv)
- [aiter/configs/model_configs/kimik3_a4w4_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a4w4_blockscale_untuned_gemm.csv)
- [aiter/configs/model_configs/kimik3_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/minimax_m3_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/minimax_m3_mxfp4_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_mxfp4_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/minimax_m3_mxfp4_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_mxfp4_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/q3vl_a4w4_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/q3vl_a4w4_blockscale_tuned_gemm.csv)
- [aiter/configs/model_configs/q3vl_a4w4_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/q3vl_a4w4_blockscale_untuned_gemm.csv)
- [aiter/configs/model_configs/qwen38_27b_a4w4_blockscale_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_27b_a4w4_blockscale_tuned_gemm.csv)
- [aiter/configs/model_configs/qwen38_27b_a4w4_blockscale_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_27b_a4w4_blockscale_untuned_gemm.csv)
- [aiter/configs/model_configs/qwen38_flash_next_a8w8_bpreshuffle_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_a8w8_bpreshuffle_tuned_gemm.csv)
- [aiter/configs/model_configs/qwen38_flash_next_a8w8_bpreshuffle_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_a8w8_bpreshuffle_untuned_gemm.csv)
- [aiter/configs/mxfp8fp4_asm_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/mxfp8fp4_asm_tuned_gemm.csv)
- [aiter/configs/mxfp8fp4_asm_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/mxfp8fp4_asm_untuned_gemm.csv)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py)
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
- [op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py)
- [op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py)
- [op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py)

## Performance and validation

Compare identical quantized inputs and scale tensors. Count quantization, shuffle and reduction kernels when they execute on the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
