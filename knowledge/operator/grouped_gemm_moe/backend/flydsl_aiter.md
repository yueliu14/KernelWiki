---
title: "Grouped GEMM and MoE expert computation: AITER flydsl"
kind: implementation_knowledge
operator: grouped_gemm_moe
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Grouped GEMM and MoE expert computation — AITER flydsl

MoE execution couples token/expert sorting with gate/up projection, activation and down projection. The row distribution per expert, top-k, padded block count and placement of routing weights affect both work and correctness.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/fhmoe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fhmoe.py) · **Recent**

Source documentation: FlyDSL runtime support for fused heterogeneous MoE (FHMoE).

Symbols: `compile_flydsl_fhmoe_stage1`, `compile_flydsl_fhmoe_stage2`, `_s1_args_fhmoe`, `_s2_args_fhmoe`, `flydsl_fhmoe_stage1`, `flydsl_fhmoe_stage2`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/fused_moe_gfx942.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fused_moe_gfx942.py) · **Recent**

Symbols: `Config`, `_Problem`, `get_tune_space`, `_get_compiled_kernel`, `_launch`, `_quant_per_tensor`, `_empty_scale`, `_gateup_output`, `_run_prefill`, `_run_batch1`, `_run_decode`, `run_flydsl_moe_gfx942`, `run_flydsl_moe_gfx942_impl`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_TILE_SIZE_K`, `BLOCK_TILE_SIZE_M`, `BLOCK_TILE_SIZE_N`, `tile_k`.

### `aiter/ops/flydsl/grouped_gemm_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/grouped_gemm_mxfp4.py) · **Recent**

Source documentation: Grouped MXFP4 GEMM launchers.

Symbols: `_select_bool_env`, `_select_tdm_b_th`, `_select_cluster_n`, `_select_cluster_m`, `_select_num_waves_per_tensor_tdm`, `flydsl_grouped_gemm_a8w4_masked`.

Parameters referenced in this file: `num_buffers`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/grouped_moe_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/grouped_moe_gfx1250.py) · **Recent**

Source documentation: gfx1250 grouped MoE GEMM (a8w4 / a4w4) via FlyDSL. Supports optional DeepGEMM-style contiguous-M scheduler (AITER_GROUPED_DEEPGEMM_CONTIGUOUS=1 or CSV grouped_contiguous_m=1).

Symbols: `set_flydsl_dispatch_context`, `_flydsl_dispatch_context`, `_grouped_weight_uint8`, `_as_bool`, `_as_int`, `_dtype_name`, `_enum_name`, `_load_grouped_config_rows`, `_next_pow2`, `_get_padded_m`, `_find_grouped_config`, `_use_grouped_gemm_enabled`, `_align_up`, `_make_contiguous_psum_layout`, `_grouped_a8w4_preshuffle_e8m0_scale`, `_grouped_a8w4_prepare_scale_batch`, `_compact_scale_min_tokens`, `_use_fused_quant_preshuffle`, `_get_compiled_g2l_lut`, `route_counter_buffer`, `route_fused_workspace`, `_build_g2l_lut`, `_tdm_align_up`, `get_wmma_m_rep`, `_grouped_a8w4_tdm_moe`, `grouped_gemm_gfx1250_a8w4`, `_get_compiled_gather_reduce`, `_choose_gather_reduce_vec`, `_get_compiled_route_maps`, `_get_compiled_contiguous_psum`, `_get_compiled_contiguous_psum_remap`, `_get_compiled_route_psum_fused`, `fused_route_psum_remap`, `contiguous_psum`, `_get_compiled_contiguous_psum_remap_ep`, `contiguous_psum_remap`, `build_route_maps`, `flydsl_moe_gather_reduce`, `_get_compiled_scatter_copy`, `flydsl_moe_scatter_copy_token`, `_get_compiled_scatter_preshuffle_scale`, `flydsl_moe_scatter_preshuffle_scale`, `flydsl_moe_preshuffle_scale`.

Parameters referenced in this file: `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/moe_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/moe_common.py) · **Recent**

Source documentation: Common types shared across MoE FlyDSL kernel modules.

Symbols: `get_flydsl_activation_name`, `GateMode`, `apply_gate_up`.

### `aiter/ops/flydsl/moe_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/moe_kernels.py) · **Recent**

Source documentation: FlyDSL MOE kernel management: naming, compilation, and high-level API.

Symbols: `_warn_tile_override`, `flydsl_kernel_name`, `pick_flydsl_stage2_tile_k`, `pick_flydsl_stage1_tile_n`, `resolve_flydsl_grid_y_persist_m`, `requires_flydsl_stage2_reduce`, `resolve_flydsl_stage2_mode`, `requires_flydsl_stage2_global_a`, `resolve_flydsl_stage2_tile_k`, `resolve_flydsl_stage1_tile_n`, `get_flydsl_kernel_params`, `get_flydsl_stage1_kernels`, `get_flydsl_stage2_kernels`, `build_flydslv2_gemm2_name`, `get_flydsl_stage2_v2_kernels`, `_register_production_variants_stage2`, `_gemm1_lds_bytes`, `get_flydsl_stage1_kernels_int4_bf16`, `get_flydsl_stage2_kernels_int4_bf16`, `_register_all_configs`, `compile_flydsl_moe_stage1`, `compile_flydsl_moe_stage2`, `_view_safe`, `runtime_swiglu_limit`, `_s1_args_fp4`, `_s1_args_std`, `_s2_args_fp4`, `_s2_args_std`, `_run_compiled`, `_run_moe_reduction`, `_mxscale_pad_cache_key`, `_mxscale_pad_cache_get`, `_mxscale_pad_cache_put`, `_mxscale_align_up`, `_mxscale_pick_tile_n`, `_mxscale_zero_pad_last`, `_mxscale_pad_weight_k`, `_get_compiled_silu_fused`, `_get_compiled_swiglu`, `flydsl_swiglu_and_mul_interleaved`, `flydsl_silu_and_mul_interleaved`, `_flydsl_moe_stage1_impl`, `flydsl_moe_stage1`, `_flydsl_moe_stage2_impl`, `flydsl_moe_stage2`, `_get_compiled_fused_route_quant_scatter`, `_get_compiled_fused_route_quant_scatter_st_ksplit`, `_get_compiled_topids_to_rows`, `_get_compiled_topids_to_rows_g2l`, `_get_compiled_route_g2l_fused`, `_get_compiled_route_g2l_lds`, `flydsl_moe_topids_to_rows`, `flydsl_moe_fused_route_quant_scatter`, `_get_compiled_fused_route_psum_quant_scatter`, `flydsl_moe_fused_route_psum_quant_scatter`, `flydsl_moe_fused_ep_route_quant_compact`, `_get_compiled_fused_quant_preshuffle`, `token_multidest_eligible`, `_get_compiled_token_multidest_compact_quant`, `_get_compiled_token_multidest_quant`, `_get_compiled_fused_quant_preshuffle_route_ksplit`, `_routeks_ep_psum_args`, `flydsl_moe_fused_quant_preshuffle`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fhmoe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fhmoe.py) · **Recent**

Source documentation: Heterogeneous MoE facades for MXFP4/MXFP8 routed and FP8 shared weights.

Symbols: `compile_mixed_fhmoe_gemm1`, `compile_mixed_fhmoe_gemm2`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py) · **Recent**

Source documentation: Ordinary MoE facades for the shared MXFP4/MXFP8 kernel builders.

Symbols: `compile_mixed_moe_gemm1`, `compile_mixed_moe_gemm2`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py) · **Recent**

Source documentation: Shared MXFP4/MXFP8 MoE and heterogeneous MoE kernel builders.

Symbols: `validate_moe_dtypes`, `_barrier`, `_idx_to_fly_ptr`, `compile_mixed_moe_gemm1_common`, `compile_mixed_moe_gemm2_common`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm1.py) · **Recent**

Source documentation: Flattened grid for a16w4 gemm1: (m-blocks) x (inter/tile_n) n-blocks.

Symbols: `_gemm1_body_a16w4`, `gemm1_a16w4_grid`, `compile_gemm1_a16w4_port`.

Parameters referenced in this file: `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_2stage_a16wmix/gemm2.py) · **Recent**

Source documentation: Flattened launch grid for a16w4 gemm2. Non-persistent (default): one CTA per (m-block x n-block) tile over padded max_m_blocks. Persistent: cap to min(total_work, NUM_CU) CTAs (only when padded work > NUM_CU*4); each CTA loops over its real work-tiles.

Symbols: `_atomic_bf16_epilog`, `_reduce_bf16_epilog`, `_gemm2_body_a16w4`, `gemm2_a16w4_grid`, `compile_gemm2_a16w4_port`.

Parameters referenced in this file: `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/moe_2stage_a16wmix/utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_2stage_a16wmix/utils.py) · **Recent**

Source documentation: Shared operand paths and low-level helpers for the a16w4/a16wi4/a16w16 MoE kernels. Used by both stage1 (:mod:gemm1) and stage2 (:mod:gemm2): * :func:make_a_loader the A (activation) path -- global->LDS staging plus the v8bf16 MMA-fragment reads that must match its LDS layout. * :func:make_b_loader the B (weight) path -- N addressing, raw W + scale gathers and the upconvert, all specialized for w_dtype so the stages carry no weight-dtype branches.

Symbols: `_udiv`, `_umod`, `_global_i32_at`, `_buffer_i32_scalar_read`, `_int_to_llvm_ptr`, `_lds_ptr3`, `_global_base_ptr1`, `_gep`, `_cvt_pk_bf16_f32_se`, `_int4_nibble_to_bf16x8`, `_e8m0_byte_to_f32`, `_a16w4_swizzle_xor16`, `_bf16_frag8`, `_bf16_frag4`, `_mma_bf16`, `_a_col_bytes_for_ku`, `_ALoader`, `make_a_loader`, `_BCol`, `_BLoader`, `_fp4x4_e2m1_to_f32x4`, `_fp4_nibble_to_bf16x8`, `make_b_loader`.

### `aiter/ops/flydsl/kernels/moe_contiguous_psum.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_contiguous_psum.py) · **Recent**

Source documentation: DeepGEMM-contiguous M-tile prefix sum (FlyDSL). Computes tile-aligned exclusive prefix sum of per-expert counts for the contiguous grouped-GEMM scheduler. The parallel scan replaces torch.cumsum (avoids rocprim trampoline overhead for small E). The block is MAX_EXPERTS_PER_BLOCK threads wide but E is not bounded by it: the scan sweeps the experts in block-sized chunks and carries the running offset between chunks in LDS. Kimi-K3 (E=896) is the first model to exceed one chunk.

Symbols: `_PsumStorage`, `_RoutePsumStorage`, `_PsumRemapEpStorage`, `_PsumRemapStorage`, `build_moe_contiguous_psum_module`, `build_moe_contiguous_psum_remap_module`, `build_moe_contiguous_psum_remap_ep_module`, `build_moe_route_psum_fused_module`.

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py) · **Recent**

Source documentation: Fused MoE route-map + MX quant + scatter-copy + scale-preshuffle (FlyDSL). The grouped a8w4/fp4 MoE stage1 input prep is normally four kernels (see grouped_moe_gfx1250.py): 1. build_route_maps route i -> grouped row (atomic argsort) 2. per_1x32 MX quant hidden(T, model_dim) -> payload + e8m0 scale 3. scatter_copy_token payload[token] -> grouped_payload[row] 4.

Symbols: `_i32_ptr`, `_EpPsumScanStorage`, `_arch_has_pk8`, `_cvt_scalef32_pk8_fp4_bf16`, `_cvt_scalef32_pk8_fp8_bf16`, `_emit_pk8_lane_amax`, `_arch_has_native_scaled_cvt`, `_quant_layout`, `_emit_quant_block_loop`, `_emit_quant_result_stores`, `_pack_block_group_dword`, `_emit_interleaved_scale_dwords`, `_emit_compact_scale_dwords`, `_emit_payload_stores`, `_emit_quant_one_k_group`, `_emit_prequant_copy_preshuffle`, `build_moe_fused_route_quant_scatter_module`, `build_moe_fused_route_quant_scatter_st_ksplit_module`, `build_moe_fused_quant_preshuffle_module`, `build_moe_fused_quant_preshuffle_route_ksplit_module`, `_token_multidest_scale_base`, `token_multidest_ksplit`, `token_multidest_tdm_chunks`, `build_moe_token_multidest_quant_module`, `fused_quant_preshuffle_supported`, `_fused_preshuffle_k_chunks`, `build_moe_token_multidest_compact_quant_module`, `build_moe_fused_route_psum_quant_scatter_module`.

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`, `tile_m`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/common.py) · **Recent**

Source documentation: Shared numerics, layout helpers, and host specialization support.

Symbols: `_f32_to_bf16_rta`, `_f32_to_bf16_rte`, `_f32_to_bf16`, `_ptr`, `get_down_device_config`, `BufferTensor`, `LdsTensor`, `validate_gemm_options`, `div_up`, `div_e`, `split_works`, `load_fragment`, `all_elements`, `get_d1_shape`, `inner_most_stride`, `all_copy_atoms`, `_as_ptr`, `atomic_add_bf16`, `make_1d_coord_tensor`, `sub_tensor`, `atom_tensor`, `torch_layout`, `view_as_torch_tensor`, `FlyObjCache`, `asm_mark`, `dump_ir`, `_select`, `_cvt_fp8_bf16`, `make_gemm_helpers`.

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_TILE_SIZE_M`, `tile_k`, `tile_m`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm1.py) · **Recent**

Source documentation: Baseline Gate/Up kernels for split-K, batch1, and prefill.

Symbols: `compile_moe_gemm1`.

Parameters referenced in this file: `BLOCK_TILE_SIZE_M`, `BLOCK_TILE_SIZE_N`, `tile_k`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2.py) · **Recent**

Source documentation: Baseline Down kernels for split-K, batch1, and prefill.

Symbols: `compile_moe_gemm2`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_TILE_SIZE_M`, `BLOCK_TILE_SIZE_N`, `tile_k`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2_1x4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2_1x4.py) · **Recent**

Source documentation: MoE stage2 1x4 down-projection kernel for N256 tiles.

Symbols: `_flatten_rows`, `_gather_sorted_rows`, `_act_scale_tensor`, `_apply_token_scales`, `_set_priority`, `_pack_scaled_bf16_pairs`, `_store_scaled_bf16`, `_write_cshuffle_vector`, `_store_cshuffle_row_pair`, `_build_moe_gemm2_1x4_n256`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_TILE_SIZE_M`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2_8x1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2_8x1.py) · **Recent**

Source documentation: 8x1 Down kernels: shared primitives, task setup and N schedule, the entry point, then K128n and K192.

Symbols: `stage_end`, `priority`, `cshuffle_plane_offset`, `read_a_g2r`, `read_8x1_scale_g2r`, `_fma_f32`, `pack_8x1_record`, `store_8x1_c_r2g`, `store_8x1_c_tile_r2g`, `clear_8x1_record`, `mfma_8x1_record`, `_pin_vgpr`, `_input_tensor`, `_sorted_ids_tensor`, `_routing_scale`, `_expert_weights`, `_load_row_scale`, `_output_tensor`, `_weight_scale_buffer`, `_make_a_fragments`, `_b_coords`, `packing_events`, `vmem_wait_schedule`, `_b_parts`, `save_8x1_state`, `restore_8x1_state`, `_prepare_8x1_loop_state`, `run_8x1_tile`, `_run_8x1_pipeline`, `_build_moe_gemm2_8x1`, `_emit_k128n_body`, `k192_wait_schedule`, `_emit_k192_body`, `run_k192_tile`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2_8x1_compact.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/gemm2_8x1_compact.py) · **Recent**

Source documentation: Split M64 sorted metadata into M256 full and M64 tail tasks [row_begin, expert_id]. Rows keep their original positions; each expert must be one contiguous run of blocks.

Symbols: `task_capacities`, `allocate_task_buffers`, `build_task_table`, `_build_moe_gemm2_8x1_compact`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/quant.py) · **Recent**

Source documentation: Baseline per-tensor FP8 quantization with cached launchers.

Symbols: `flydsl_absmax`, `flydsl_quant_per_tensor`.

### `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxmoe_dispatcher.py) · **Recent**

Source documentation: Compile + launch dispatch for the layout-API MXFP4 MoE gemm (BM32, opus-sort); a4w4/a8w4 entry point.

Symbols: `_norm_sbm`, `_active_m_blocks_upper_bound`, `_validate_v2_gemm2_dtypes`, `_spart_output_tile_index`, `_flat_persistent_tile`, `_pick_epi_lanes`, `compile_gemm2_a4w4_port`, `get_g2`, `mxfp4_moe_gemm2`.

### `aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py) · **Recent**

Source documentation: Layout-API MXFP4 MoE GEMM device body (BM32): gemm2 down.

Symbols: `bq_view`, `bq_view_fp8`, `scale_view`, `scale_mma_atoms`, `mma_one_j`, `issue_a_load_lds_dt`, `gemm2_body_v2`, `atomic_bf16_epilog`, `nonatomic_bf16_epilog`, `nonatomic_mxfp4_epilog`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/test_flydsl_grouped_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py)
- [op_tests/flydsl_tests/test_flydsl_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_moe.py)
- [op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py)
- [op_tests/multigpu_tests/bench_mega_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/bench_mega_moe.py)
- [op_tests/multigpu_tests/bench_mega_moe_v2.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/bench_mega_moe_v2.py)
- [op_tests/multigpu_tests/test_mega_moe_v2.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_mega_moe_v2.py)
- [op_tests/multigpu_tests/test_wide_ep_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_wide_ep_moe.py)
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
- [op_tests/tuners/tune_fhmoe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuners/tune_fhmoe.py)
- [op_tests/tuning_tests/test_fhmoe_tuner.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_fhmoe_tuner.py)

## Performance and validation

Benchmark representative expert-load distributions, not only a uniform dense matrix. Include sorting, zeroing, scatter and reduction when evaluating the complete operator.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
