---
title: "Grouped GEMM and MoE expert computation: AITER aiter"
kind: implementation_knowledge
operator: grouped_gemm_moe
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Grouped GEMM and MoE expert computation — AITER aiter

MoE execution couples token/expert sorting with gate/up projection, activation and down projection. The row distribution per expert, top-k, padded block count and placement of routing weights affect both work and correctness.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/fhmoe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/fhmoe.py) · **Recent**

Source documentation: Fused heterogeneous MoE (FHMoE) MXFP4/FP8 and MXFP8/FP8 dispatch. The notation names routed/shared storage respectively. Shared FP8 means E4M3 payloads with per-32 E8M0 block scales, stored separately from routed weights. FHMoE requires gfx950, SiLU, per-1x32 quantization, contiguous preshuffled weights/scales, all four shared tensors, and a dummy final routed row identified by shared_expert_id == E - 1. Expert bias and expert-mask dispatch are not supported.

Symbols: `_validate_fhmoe_contract`, `_flydsl_fhmoe_stage1_wrapper`, `_flydsl_fhmoe_stage2_wrapper`, `_use_fhmoe_wrappers`, `_validate_fhmoe_metadata_consistency`, `_dtype_name`, `_is_fhmoe_metadata_compatible`, `_is_dsv4_i384_metadata`, `_valid_fhmoe_kernel_geometry`, `_fhmoe_config_file`, `_config_has_mxfp8_fhmoe_geometry`, `_config_has_mxfp8_fhmoe_contract`, `_dsv4_i384_fhmoe_config_file`, `_default_fhmoe_bucket`, `_supports_fhmoe_config`, `_supports_dsv4_i384_fhmoe_config`, `supports_dsv4_i384_fhmoe`, `_supports_mxfp8_fhmoe_config`, `supports_mxfp8_fhmoe_config`, `_resolve_mxfp8_fhmoe_config_file`, `_is_dsv4_i384_fhmoe_contract`, `_uses_dsv4_fhmoe_config`, `fhmoe_fake`, `fhmoe_`, `_fhmoe`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/fhmoe_contract.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/fhmoe_contract.py) · **Recent**

Source documentation: Lightweight FHMoE safety helpers shared by runtime, AOT, and tuners. This module must remain free of imports from the top-level aiter namespace so it is safe when AITER_AOT_IMPORT=1 exposes only the lightweight JIT core.

Symbols: `fhmoe_reduce_output_nbytes`, `max_fhmoe_reduce_tokens`.

### `aiter/fused_moe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/fused_moe.py) · **Recent**

Source documentation: Build valid_mask [token_num, topk] for EP mode. valid_mask[t, k] = 1 if topk_ids[t, k] points to a local (non-fake) expert, else 0. When expert_mask is None (non-EP), returns all-ones.

Symbols: `_get_flydsl_moe_kernels`, `_aux_uses_opus`, `_is_fused_decode_sort_quant_enabled`, `_fused_decode_sort_quant`, `_moe_prepare_unsorted_input`, `_adaptive_moe_sort`, `_validate_output_buffer_metadata`, `_validate_output_buffer_no_overlap`, `_return_output`, `_padded_scale_cols`, `_is_inline_sort_kname`, `_is_inline_sort_cfg`, `_is_mxfp4_inline_sort`, `_mxfp4_aux_instance_supported`, `_mxfp4_inline_sort_unsupported`, `_moe_sorting_impl`, `_flydsl_moe_sorting`, `moe_sorting`, `get_topk_valid_mask`, `_is_flydsl_stage2`, `_stage2_honors_runtime_reduce_override`, `stage2_uses_route_reduce`, `_stage2_runtime_uses_route_reduce`, `_stage2_uses_fp8_route_output`, `_flydsl_route_output_nbytes`, `_validate_flydsl_route_output_nbytes`, `get_inter_dim`, `_bound_split`, `resolve_activation_dtype`, `fused_moe`, `fused_moe_fake`, `fused_moe_`, `_fused_moe_impl`, `fused_moe_1stage`, `get_block_size_M`, `use_nt`, `get_ksplit`, `_mxfp4_per_1x32_weight_expert_to_f32`, `_int4_per_1x32_weight_expert_to_f32`, `nextPow2`, `get_padded_M`, `MOEMetadata`, `_needs_swiglu_bias_support`, `_normalize_mxfp4_activation_params`, `_normalize_bias_for_kernel`, `_get_padding_for_flydsl`, `_flydsl_stage1_wrapper`, `_flydsl_stage2_wrapper`, `_empty_bf16`, `_empty_u8`, `_mxfp4_a4w4_stage1`, `_mxfp4_a4w4_stage2`, `_mxfp4_a4w4_stage1_fw`, `_mxfp4_a4w4_stage2_fw`, `_mxfp4_scale_u8`, `_flydsl_stage2_fp8_enabled`, `_mxfp4_intermediate_enabled`, `_opus_stage2_fp8_enabled`, `_flydsl_v2_stage2_wrapper`, `_make_mxfp4_metadata`, `_flydsl_mxfp4_layout_is_compatible`, `_is_cktile_mxfp4_stage2_name`, `_can_reroute_mxfp4_to_flydsl`, `get_2stage_cfgs`, `fused_moe_2stages`, `torch_moe_act`, `asm_stage1`, `torch_moe`, `swiglu`, `situv2`, `torch_moe_stage1`, `torch_moe_stage2`, `ck_moe_stage1`, `cktile_moe_stage1`, `cktile_moe_stage2`, `fused_topk`.

Parameters referenced in this file: `BLOCK_SIZE_M`, `split_k`, `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/fused_moe_bf16_asm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/fused_moe_bf16_asm.py) · **Recent**

Symbols: `moe_sorting_ck`, `asm_moe`, `asm_moe_tkw1`, `get_block_size`, `ck_moe_2stages`, `torch_moe`, `torch_moe_tkw1`, `fused_topk`.

Parameters referenced in this file: `BLOCK_SIZE_M`.

### `aiter/fused_moe_dp_shared_expert.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/fused_moe_dp_shared_expert.py) · **Recent**

Source documentation: user API

Symbols: `get_dp_shared_expert_token_range`, `get_dp_shared_expert_stage1_moe_sorting_result`, `get_dp_shared_expert_stage2_moe_sorting_result`, `get_inter_dim`, `nextPow2`, `get_padded_M`, `fused_moe_dp_share_expert`, `fused_moe_1stage`, `get_block_size_M`, `MOEMetadata`, `get_2stage_cfgs`, `fused_moe_2stages`, `torch_moe_act`, `asm_stage1`, `torch_moe`.

Parameters referenced in this file: `BLOCK_SIZE_M`.

### `aiter/fused_moe_registry.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/fused_moe_registry.py) · **Recent**

Symbols: `FusedMoeRequest`, `FusedMoeImplResolutionError`, `register_fused_moe_impl`, `make_fused_moe_impl_kernel_name`, `resolve_fused_moe_impl`.

### `aiter/ops/moe_mxfp4_aux.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/moe_mxfp4_aux.py) · **Recent**

Source documentation: Return whether the generated MXFP4 auxiliary kernels cover this shape.

Symbols: `is_mxfp4_moe_shape_supported`, `_mxfp4_moe_sort_internal_is_supported`, `mxfp4_moe_sort_quant`, `mxfp4_moe_sort`, `mxfp4_moe_quant`, `mxfp4_moe_sort_scales`, `mxfp4_moe_scatter_reduce`, `mxfp4_moe_scatter_reduce_q`.

### `aiter/ops/moe_op.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/moe_op.py) · **Recent**

Symbols: `_topk_softmax`, `topk_softmax`, `_topk_softmax_fused_shared_gate`, `topk_softmax_fused_shared_gate`, `topk_softmax_asm`, `topk_sigmoid`, `_moe_sum`, `moe_sum`, `moe_align_block_size`, `fmoe`, `fmoe_int8_g1u0`, `fmoe_g1u1`, `fmoe_g1u1_tkw1`, `fmoe_int8_g1u0_a16`, `fmoe_g1u1_a16`, `fmoe_fp8_blockscale_g1u1`, `moe_stage1_g1u1`, `cmdGenFunc_ck_moe_stage`, `cmdGenFunc_ck_moe_stage2`, `ck_moe_stage1`, `ck_moe_stage2`, `moe_cktile2stages_gemm1_ck`, `moe_cktile2stages_gemm1`, `moe_cktile2stages_gemm2_ck`, `moe_cktile2stages_gemm2`, `get_moe_stage_module`, `ck_moe_stage1_fwd`, `ck_moe_stage2_fwd`.

Parameters referenced in this file: `split_k`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_ds_v3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_ds_v3.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_1.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_1.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_gfx950.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_gfx950.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_shared_gfx942.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_shared_gfx942.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_shared_gfx950.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_shared_gfx950.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_minimax-m2_5.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_minimax-m2_5.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_qwen3_235b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_qwen3_235b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_qwen3_5_397b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_qwen3_5_397b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_ds_v3.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_ds_v3.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5_3_flash_gfx950.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5_3_flash_gfx950.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5_3_flash_shared_gfx942.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5_3_flash_shared_gfx942.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5_3_flash_shared_gfx950.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5_3_flash_shared_gfx950.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_minimax-m2_5.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_minimax-m2_5.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_qwen3_235b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_qwen3_235b.csv)
- [aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_qwen3_5_397b.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_qwen3_5_397b.csv)
- [aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/dsv3_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv3_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/dsv41_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/dsv41_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/dsv41_fp8_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_fp8_tuned_fmoe.csv)
- [aiter/configs/model_configs/dsv41_fp8_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_fp8_untuned_fmoe.csv)
- [aiter/configs/model_configs/dsv41_fp8fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_fp8fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/dsv4_fp8fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_fp8fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/glm47_fp8_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm47_fp8_tuned_fmoe.csv)
- [aiter/configs/model_configs/glm47_fp8_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm47_fp8_untuned_fmoe.csv)
- [aiter/configs/model_configs/glm53_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm53_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/glm53_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm53_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/glm5_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/glm5_mxfp8_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_mxfp8_tuned_fmoe.csv)
- [aiter/configs/model_configs/glm5_mxfp8_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_mxfp8_untuned_fmoe.csv)
- [aiter/configs/model_configs/glm5_ptpc_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_ptpc_tuned_fmoe.csv)
- [aiter/configs/model_configs/glm5_ptpc_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_ptpc_untuned_fmoe.csv)
- [aiter/configs/model_configs/gptoss_a16w4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_a16w4_tuned_fmoe.csv)
- [aiter/configs/model_configs/gptoss_a16w4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_a16w4_untuned_fmoe.csv)
- [aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/gptoss_fp8fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_fp8fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/hunyuan3_fp8_per_tensor_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/hunyuan3_fp8_per_tensor_tuned_fmoe.csv)
- [aiter/configs/model_configs/hunyuan3_fp8_per_tensor_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/hunyuan3_fp8_per_tensor_untuned_fmoe.csv)
- [aiter/configs/model_configs/k2_horizon_375b_bf16_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/k2_horizon_375b_bf16_tuned_fmoe.csv)
- [aiter/configs/model_configs/k2_horizon_375b_bf16_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/k2_horizon_375b_bf16_untuned_fmoe.csv)
- [aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/kimik2_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/kimik2_fp8fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_fp8fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/kimik2_fp8fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_fp8fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/kimik2_i4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_i4_tuned_fmoe.csv)
- [aiter/configs/model_configs/kimik2_i4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_i4_untuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_a16w4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a16w4_tuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_a16w4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a16w4_untuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_a4w4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a4w4_untuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_a8w4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_a8w4_untuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_i4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_i4_tuned_fmoe.csv)
- [aiter/configs/model_configs/kimik3_i4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_i4_untuned_fmoe.csv)
- [aiter/configs/model_configs/mimo_v2_5_pro_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/mimo_v2_5_pro_tuned_fmoe.csv)
- [aiter/configs/model_configs/mimo_v2_5_pro_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/mimo_v2_5_pro_untuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m25_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m25_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m25_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m25_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_a16w4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_a16w4_tuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_fp8_ptpc_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_fp8_ptpc_tuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_fp8_ptpc_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_fp8_ptpc_untuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_mxfp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_mxfp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_mxfp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_mxfp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv)
- [aiter/configs/model_configs/minimax_m3_mxfp8_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_mxfp8_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen38_flash_next_bf16_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_bf16_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen38_flash_next_bf16_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_bf16_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen38_flash_next_fp8_ptpc_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_fp8_ptpc_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_235b_bf16_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_235b_bf16_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_235b_bf16_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_235b_bf16_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_5_397b_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_397b_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_5_397b_fp8_ptpc_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_397b_fp8_ptpc_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_5_397b_fp8_ptpc_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_397b_fp8_ptpc_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_5_397b_tuned_grouped_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_397b_tuned_grouped_fmoe.csv)
- [aiter/configs/model_configs/qwen3_6_35b_a3b_bf16_tuned_fmoe_gfx1201.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_6_35b_a3b_bf16_tuned_fmoe_gfx1201.csv)
- [aiter/configs/model_configs/qwen3_6_35b_a3b_bf16_untuned_fmoe_gfx1201.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_6_35b_a3b_bf16_untuned_fmoe_gfx1201.csv)
- [aiter/configs/model_configs/qwen3_6_35b_a3b_fp8_tuned_fmoe_gfx1201.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_6_35b_a3b_fp8_tuned_fmoe_gfx1201.csv)
- [aiter/configs/model_configs/qwen3_6_35b_a3b_fp8_untuned_fmoe_gfx1201.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_6_35b_a3b_fp8_untuned_fmoe_gfx1201.csv)
- [aiter/configs/model_configs/qwen3_8_2400b_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_8_2400b_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_8_2400b_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_8_2400b_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_vl_fp4_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_vl_fp4_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3_vl_fp4_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_vl_fp4_untuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3next_80b_fp8_tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3next_80b_fp8_tuned_fmoe.csv)
- [aiter/configs/model_configs/qwen3next_80b_fp8_untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3next_80b_fp8_untuned_fmoe.csv)
- [aiter/configs/tuned_fhmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/tuned_fhmoe.csv)
- [aiter/configs/tuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/tuned_fmoe.csv)
- [aiter/configs/tuned_grouped_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/tuned_grouped_fmoe.csv)
- [aiter/configs/untuned_fhmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/untuned_fhmoe.csv)
- [aiter/configs/untuned_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/untuned_fmoe.csv)
- [aiter/configs/untuned_grouped_fmoe.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/untuned_grouped_fmoe.csv)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/bench_mega_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/bench_mega_moe.py)
- [op_tests/multigpu_tests/bench_mega_moe_v2.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/bench_mega_moe_v2.py)
- [op_tests/multigpu_tests/test_mega_moe_v2.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_mega_moe_v2.py)
- [op_tests/multigpu_tests/test_wide_ep_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_wide_ep_moe.py)
- [op_tests/test_fhmoe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fhmoe.py)
- [op_tests/test_flydsl_moe_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_moe_a16wfp4.py)
- [op_tests/test_flydsl_moe_aux.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_moe_aux.py)
- [op_tests/test_flydsl_moe_gfx942.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_moe_gfx942.py)
- [op_tests/test_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe.py)
- [op_tests/test_moeTopkSoftmax.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moeTopkSoftmax.py)
- [op_tests/test_moe_2stage.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_2stage.py)
- [op_tests/test_moe_a16w16_default_dispatch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_a16w16_default_dispatch.py)
- [op_tests/test_moe_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_blockscale.py)
- [op_tests/test_moe_dp_share_expert.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_dp_share_expert.py)
- [op_tests/test_moe_ep.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_ep.py)
- [op_tests/test_moe_glm5_fp4_ep_dispatch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_glm5_fp4_ep_dispatch.py)
- [op_tests/test_moe_local_expert_ids.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_local_expert_ids.py)
- [op_tests/test_moe_mxfp4_inter_dim_dispatch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_mxfp4_inter_dim_dispatch.py)
- [op_tests/test_moe_mxfp8_passthrough.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_mxfp8_passthrough.py)
- [op_tests/test_moe_tkw1.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_tkw1.py)
- [op_tests/test_mxfp4_moe_aux_codegen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp4_moe_aux_codegen.py)
- [op_tests/tuners/tune_fhmoe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuners/tune_fhmoe.py)
- [op_tests/tuning_tests/test_fhmoe_tuner.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_fhmoe_tuner.py)
- [op_tests/flydsl_tests/test_flydsl_grouped_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py)
- [op_tests/flydsl_tests/test_flydsl_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_moe.py)
- [op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_a16w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_a16w4.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_cudagraph.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_cudagraph.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_proton.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_proton.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_proton.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_proton.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_moe_gemm_int8_smoothquant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_gemm_int8_smoothquant.py)
- [op_tests/op_benchmarks/triton/bench_moe_situ_epilogue.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_moe_situ_epilogue.py)
- [op_tests/op_benchmarks/triton/bench_sonicmoe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_sonicmoe.py)
- [op_tests/triton_tests/moe/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/__init__.py)
- [op_tests/triton_tests/moe/moe_test_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_test_utils.py)
- [op_tests/triton_tests/moe/test_moe_gemm_a16w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_moe_gemm_a16w4.py)
- [op_tests/triton_tests/moe/test_moe_gemm_a4w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_moe_gemm_a4w4.py)
- [op_tests/triton_tests/moe/test_moe_gemm_a8w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_moe_gemm_a8w4.py)
- [op_tests/triton_tests/moe/test_moe_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_moe_gemm_a8w8.py)
- [op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py)
- [op_tests/triton_tests/moe/test_moe_gemm_int8_smoothquant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_moe_gemm_int8_smoothquant.py)
- [op_tests/triton_tests/moe/test_moe_situ_epilogue.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_moe_situ_epilogue.py)
- [op_tests/triton_tests/moe/test_quant_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_quant_moe.py)
- [op_tests/triton_tests/moe/test_sonicmoe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/test_sonicmoe.py)

## Performance and validation

Benchmark representative expert-load distributions, not only a uniform dense matrix. Include sorting, zeroing, scatter and reduction when evaluating the complete operator.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
