---
title: "Fused MoE compute and communication: AITER flydsl"
kind: implementation_knowledge
operator: fused_moe_grouped_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused MoE compute and communication — AITER flydsl

These paths fuse several stages of MoE, sometimes including inter-GPU communication. Their synchronization and buffer ownership are part of correctness, alongside local GEMM arithmetic.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/comm_fused_moe_host.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/comm_fused_moe_host.py) · **Recent**

Source documentation: Production host runtime for communication-fused FlyDSL MoE. Set AITER_COMM_FUSED_WINDOW_MXFP8_FALLBACK=1 before runner creation to temporarily replay Window AR with dynamic MXFP8 partial and reduced payloads.

Symbols: `ShapeKey`, `_gpu_model_from_pci`, `get_gpu_model`, `_window_mxfp8_fallback_enabled`, `_mori_communicator`, `is_flydsl_comm_fused_moe_available`, `_int_value`, `_mega_defaults`, `config_name`, `config_record`, `_config`, `_check_block_m`, `_optional_float`, `_select_candidate`, `_comm_configs`, `_winner_table`, `winners_for`, `_align_bytes`, `_symmetric_allocation_bytes`, `_symmetric`, `_packed_offsets`, `_packed_symmetric`, `_window_workspace_sizes`, `_runner_workspace_bytes`, `_WindowWorkspaceLayout`, `_optional_ptr`, `_WindowPhaseView`, `_create_communicator`, `_register`, `_barrier`, `_barrier_pair`, `_bind_inter_layout`, `_stage2_args`, `_clear_missing_routes`, `_RuntimeLayoutRunner`, `_MegakernelRunner`, `_WindowRunner`, `_DirectRunner`, `create_runner`, `_LazyRunners`, `create_flydsl_comm_fused_runners`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/mega_moe_tp.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mega_moe_tp.py) · **Recent**

Source documentation: Tensor-parallel MoE layer as one kernel (MegaMoE TP), gfx950. Experts are replicated over the TP group and inter_dim is sharded; one launch runs the collectives around GEMM1 + activation + GEMM2 of this rank's inter slice. comm_mode: "ag_rs" (sequence-parallel in/out) or "ar" (replicated in, all-reduced out). Weights are MXFP4 (shuffle_weight(16, 16) + e8m0_shuffle).

Symbols: `mega_moe_tp_supported`, `MegaMoeTPConfig`, `MegaMoeTP`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/collectives.py) · **Recent**

Source documentation: Communication and synchronization primitives for comm-fused MoE.

Symbols: `peer_base`, `spin_until_epoch_i32_system`, `spin_until_epoch_i32_agent`, `buffer_tensor_from_addr`, `load_buffer`, `store_buffer`, `load_bf16`, `store_bf16`, `compile_epoch_barrier`, `compile_epoch_barrier_pair`, `e8m0_scale`, `pack_fp8_words`, `mxfp4_e8m0_scale`, `pack_mxfp4_8`, `_pack_mxfp4_32`, `_unpack_mxfp4_32`, `quantize_group32`, `decode_scaled_fp8_f32`, `decode_fp8_words`, `load_fp8_words`, `load_e8m0_scale`, `store_fp8_words`, `_reduce_tp_partial_group`, `emit_tp_reduce_scatter`, `emit_tp_all_gather`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py) · **Recent**

Source documentation: Configuration and workspace layouts for gfx950 A8W4 comm-fused MoE.

Symbols: `Shape`, `_align_up`, `_mxfp8_scale_bytes`, `MegakernelConfig`, `WindowConfig`, `DirectConfig`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/direct.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/direct.py) · **Recent**

Source documentation: Direct BF16 reduce-scatter and shared-expert addition.

Symbols: `_reduce`, `compile_reduce_scatter_add`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py) · **Recent**

Source documentation: Single-launch Stage2 producer/consumer kernel. Producer workgroups run the natural-grid route GEMM. Completion is tracked per N tile; the final service_groups producers become communication consumers and execute the selected direct, reduce-broadcast, or reduce-scatter/all-gather path. This keeps the native GEMM tiling and avoids an artificial window.

Symbols: `_decode_scaled_fp8_bf16`, `_mxfp8_scale`, `_store_mxfp8_scale`, `_load_mxfp8_scale`, `emit_service_tile`, `compile_megakernel`.

Parameters referenced in this file: `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py) · **Recent**

Source documentation: Prepare route storage and adapt its ABI to the MXMoE GEMM2 producer.

Symbols: `compile_mark_written_routes`, `compile_clear_missing_routes`, `resolve_route_input_row`, `_input_row_resolver_for`, `compile_megakernel_producer`, `compile_window_producer`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/window.py) · **Recent**

Source documentation: Windowed GEMM2 and TP communication family.

Symbols: `_producer_spatial_tag`, `_payload_tags`, `_producer_tile_index`, `_compose_compute`, `_compile_compute`, `compile_compute`, `_emit_local`, `_compose_cycle`, `compile_cycle`, `compile_drain`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/dispatch.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/dispatch.py) · **Recent**

Source documentation: Compact dispatch path for MegaMoE v2 stage1.

Symbols: `DispatchSlot`, `_load_fanout_pair`, `_wave_inclusive_scan_i32`, `_wave_reduce_max_i32`, `_increment_i32`, `_classify_fanout_wave_route`, `_configure_payload_geometry`, `_store_expert_metadata`, `_initialize_section_ready`, `_copy_token_row`, `_publish_tile_range`, `emit_direct_fixed_slot_payload`, `emit_direct_fixed_slot_finalize`, `_derive_allgather_offsets`, `_derive_next_fanout_pairs`, `emit_dispatch_plan`, `emit_dispatch_group`, `emit_dispatch_payload`.

Parameters referenced in this file: `BLOCK_HIST`.

### `aiter/ops/flydsl/kernels/mega_moe/envs.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/envs.py) · **Recent**

Source documentation: Every environment variable MegaMoEV2 reads, in one place. Read through this module (envs.AITER_MEGA_...) rather than os.environ. A value is read when it is accessed, so the caller decides when it takes effect: at import, when a kernel is compiled, or when an instance is built.

Symbols: `_flag`, `__getattr__`, `__dir__`.

### `aiter/ops/flydsl/kernels/mega_moe/gemm1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm1.py) · **Recent**

Source documentation: GEMM1 compute shared by fused MegaMoE v2 stage1 and its standalone interface.

Symbols: `_LdsF32View`, `do_tile`, `build_fused_gemm1`, `compile_gemm1`, `gemm1_kernel`.

Parameters referenced in this file: `tile_k`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mega_moe/gemm2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm2.py) · **Recent**

Source documentation: GEMM2 compute for fused MegaMoE v2 stage2.

Symbols: `scale_view`, `scale_mma_atoms`, `mma_one_j`, `issue_a_load_lds_dt`, `gemm2_compute_v2`, `_spart_output_tile_index`, `_resolve_g2_knobs`.

### `aiter/ops/flydsl/kernels/mega_moe/gemm_util.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/gemm_util.py) · **Recent**

Source documentation: A8W4/A4W4 GEMM utilities for MegaMoE v2 Stage1.

Symbols: `_load_row_map_subgroup`, `wait_lds_barrier`, `TileScheduler`, `ATileLoader`, `AS2RLoader`, `BWeightLoader`, `BScaleLoader`, `AScaleLoader`, `MfmaScaleGU`, `_group_absmax`, `_e8m0_scale`, `SiluQuantEpilogue`.

Parameters referenced in this file: `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py) · **Recent**

Source documentation: Static MegaMoEV2 configuration rules for MI355X.

Symbols: `fixed_stage1_epoch_slot`, `fixed_stage1_epoch_slot_count`, `Stage1Config`, `stage1_bundle_identity`, `Stage2Config`, `MegaMoEConfig`, `Stage2BundleKey`, `MegaMoEBundleEntry`, `MegaMoEBundlePlan`, `nearest_token_bucket`, `mtpr_config_class`, `_fixed_dispatch_cu`, `_select_fixed_stage1`, `_select_bounded_stage1`, `_select_large_stage1`, `_select_bounded_stage2`, `_select_large_stage2`, `_select_bucket_config`, `_replace_config`, `_TuningContext`, `_fp4_common_prelude`, `_tune_r1`, `_tune_v4_pro`, `_tune_kimi_k3`, `NetworkProfile`, `_match_network`, `select_mega_moe_config`, `build_mega_moe_bundle_plan`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py) · **Recent**

Source documentation: Low-resource compact-dispatch prepare kernel for MegaMoE Stage1.

Symbols: `compile_mega_moe_prepare`, `run_mega_moe_prepare`, `preload_mega_moe_prepare`.

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py) · **Recent**

Source documentation: Fused Stage1 dispatch with FP8/FP4 activation and FP4-weight GEMM1.

Symbols: `_Stage1KernelSpec`, `_validate_fixed_slot_geometry`, `_validate_dispatch_capacity`, `compile_mega_moe_stage1`, `compile_mega_moe_stage1_bundle`, `run_mega_moe_stage1`, `run_mega_moe_stage1_bundle`, `preload_mega_moe_stage1_bundle`.

Parameters referenced in this file: `tile_k`, `tile_m`, `tile_n`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py) · **Recent**

Source documentation: Fused GEMM2 and weighted cross-rank P2P scatter.

Symbols: `_fp8_scale_for_leader`, `p2p_scatter_epilog`, `_stage2_lds_bytes`, `compile_mega_moe_stage2`, `_get_g2_launch`, `run_mega_moe_stage2`, `preload_mega_moe_stage2`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2_aligned_pair.py) · **Recent**

Source documentation: Aligned common-row fusion for MegaMoE Stage2. The Stage1 fanout planner lays the selected experts' common routes out in the same order. This kernel consumes those two contiguous sections, evaluates the two experts sequentially with one accumulator set, and scatters their weighted sum once. The ordinary Stage2 kernel concurrently handles all residual rows.

Symbols: `_store_accumulator_tile`, `_pair_routewise_half_scatter`, `compile_mega_moe_stage2_aligned_pair`, `run_mega_moe_stage2_aligned_pair`, `preload_mega_moe_stage2_aligned_pair`.

### `aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py) · **Recent**

Source documentation: MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation.

Symbols: `MegaMoEV2`.

Parameters referenced in this file: `BLOCK_HIST`, `tile_k`, `tile_n`.

### `aiter/ops/flydsl/kernels/mega_moe/quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe/quant.py) · **Recent**

Source documentation: FlyDSL 1x32 MXFP4/MXFP8 quantization with E8M0 scales.

Symbols: `_load_i32x4`, `_store_i32x4`, `emit_per_1x32_mx_fp8_group`, `emit_per_1x32_mx_fp4_group`, `build_per_1x32_mx_quant_module`, `_get_launcher`, `per_1x32_mx_quant`, `build_mxfp4_moe_scale_sort_module`, `mxfp4_moe_scale_sort`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/combine.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/combine.py) · **Recent**

Source documentation: Barrier-wait and top-k reduce kernel for gfx1250 Stage2-fused MegaMoE.

Symbols: `_make_combine_fused_sync`, `combine_reduce_lds_bytes`, `_make_combine_fused_reduce`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/compact_plan.py) · **Recent**

Source documentation: Send-side compact route plan for gfx1250 TDM dispatch. A low-LDS persistent-enough grid counts every local route into (dest_rank, local_expert) buckets, allgathers the histogram, then writes the destination compact row into tok_map. Dispatch copies payload straight onto that row, so the receiver never runs moe_route_g2l_lds.

Symbols: `compact_plan_waves`, `_align32`, `compact_hist_layout`, `compact_hist_stride`, `compact_done_nbytes`, `_wave32_inclusive_scan_i32`, `compact_row_capacity`, `compile_tdm_compact_plan`.

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/config.py) · **Recent**

Source documentation: gfx1250-only launch geometry for Stage2-fused MegaMoE.

Symbols: `_select_dispatch_config`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py) · **Recent**

Source documentation: FlyDSL intranode EP dispatch driven by the gfx1250 Tensor Data Mover. A TDM moves a whole payload row on one descriptor. Routes are one lane each (WAVE / topk tokens per wave); a wave dedups a token's same-peer routes with a ballot + mbcnt_lo match-any. Recv slots are reserved one remote atomic per (block, peer) off an LDS histogram.

Symbols: `_align`, `tdm_tokens_per_wave`, `tdm_stage_capacity`, `tdm_lds_bytes`, `_tile_bytes`, `tdm_max_warps`, `_make_dispatch_tdm`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py) · **Recent**

Source documentation: gfx1250 Stage2-fused MegaMoE host pipeline.

Symbols: `_combine_tile`, `read_dispatch_wire_env`, `_DispatchWire`, `_align_up`, `_compact_gemm_align_m`, `_mori_dispatch_schedule`, `SymmetricArena`, `MegaMoEConfig`, `Routing`, `MegaMoEGfx1250`.

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_gather_shim.py) · **Recent**

Source documentation: Vendored TDM gather-mode API for gfx1250 (self-contained, no FlyDSL patch). Lets aiter kernels issue indexed TDM stores against a *stock* FlyDSL wheel -- i.e. without the FlyDSL-side change that taught make_tensor_gather_descriptor to accept a fly lds_view / pointer-based LDS index. Adapted from FlyDSL's expr/rocdl/tdm_ops.py (MI400 ISA S4.10.3.2 gather mode); it bottoms out on the low-level rocdl.tensor_store_from_lds 5-group intrinsic, which the wheel exposes.

Symbols: `compute_padding_encoding`, `TDMGatherDescriptor`, `_byte_offset_to_i64`, `_zero_dgroup_v8i32`, `make_tensor_gather_descriptor`, `make_tensor_gather_dgroup0`, `tensor_store_gather`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/tdm_prims.py) · **Recent**

Source documentation: gfx1250 TDM (Tensor Data Mover) descriptors over RAW 64-bit addresses. tdm_ops.make_tensor_descriptor_2d derives the global base from a fly memref/tensor, which the EP dispatch cannot supply: a payload's destination is a *peer's* copy of a symmetric region, i.e. the integer lsa_ptr(pe, off), a runtime value rather than a kernel tensor argument. So GROUP0 is packed by hand from an i64 address plus an LDS byte address, and GROUP1 from the compile-time tile geometry.

Symbols: `_i32`, `tdm_group1`, `tdm_group1_rows_4b`, `tdm_plan_xfer_4b`, `tdm_group0`, `tdm_load`, `tdm_store`, `tdm_wait`.

### `aiter/ops/flydsl/kernels/mega_moe_gfx1250/types.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_gfx1250/types.py) · **Recent**

Source documentation: Private host-side types for the gfx1250 MegaMoE pipeline.

Symbols: `GpuPointerView`, `_from_gpu_ptr`, `Stage2ScatterContext`.

Parameters referenced in this file: `tile_m`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/bigm_compute.py) · **Recent**

Source documentation: Host side of the big-M path of MegaMoeTP.forward (the kernel: compile_bigm_kernel in mega_moe_tp_kernel.py): MegaMoeTPBigM (the SDMA queues and hipIpc arena, buffers, per-batch plans, launches), the disk cache of the compiled kernels, the bf16 row gather of the fused tail.

Symbols: `_kernel_sources`, `_bigm_digest`, `_bigm_cache_dir`, `_warn_once`, `_private`, `_bigm_cached`, `_warn_spills`, `_hip`, `_hip_check`, `_kcopy`, `_CudaBuf`, `MegaMoeTPBigM`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/common.py) · **Recent**

Source documentation: Device helpers shared by the MegaMoE TP and SpRsNorm kernels.

Symbols: `_attr`, `_u`, `i32`, `_LazyTy`, `_ty`, `lds_ptr`, `lds_ld`, `lds_ld_i32`, `lds_st`, `lds_p`, `gptr`, `lds_atomic_add`, `lds_atomic_or`, `lds_ld_acq`, `lds_st_rel`, `lds_cas`, `_ctpop`, `g_ld_rel`, `g_ld_sys`, `g_st_sys`, `g_add_agent`, `g_or_agent`, `g_ld_i32`, `uni`, `uni64`, `rsrc`, `bld`, `bst`, `_dma`, `dma16`, `dma4`, `wait_vm`, `wait_lgkm0`, `amax`, `fp4_pack`, `fp8x4_pack`, `fp8x4_unpack`, `e8_scale`, `fp8x8_decode`, `mxfp8x8`, `scales4`, `bf16_bits`, `pack_bf16x2`, `bf16x8_to_f32`, `pack_bf16x8`, `sum_live`, `wave_red`, `wave_rank`, `_swap`, `swap16`, `swap32`, `widen`, `mfma`, `cat8`, `now`, `before`, `alive`, `spin_lds_ge`, `spin0`, `poll_sys_ge`, `i8x4_pack`, `i8x4_unpack`, `f32_bits_add`, `f32_bits_max`, `i4x8_pack`, `i4x8_unpack`, `_read`, `tuned_rows`, `nearest`, `device_key`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/communication.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/communication.py) · **Recent**

Source documentation: Per-instance device helpers (ctrl / peer addressing, LL packets, barriers, error reports, route gather) and the communication: the input all-gather, and the ReduceScatter of the GEMM2 partials (ar: plus the all-gather of the sums) with its final sums, LL packets, comm wave roles and the per-launch init / finish.

Symbols: `build_communication`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/gemm.py) · **Recent**

Source documentation: GEMM1 (gate / up + activation + quant of the intermediate into LDS) with its A loader wave, and GEMM2 (down projection) of a unit into the route regions.

Symbols: `build_gemm`, `bigm_lds`, `build_bigm_unit`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py) · **Recent**

Source documentation: Host side of the single-kernel TP MegaMoE (launch configs, buffers, launches).

Symbols: `LaunchCfg`, `_allocation_base`, `_ipc_open`, `_ipc_close`, `_no_expandable_segments`, `_Slice`, `SymmetricArena`, `MegaMoeTPLDSError`, `_mt_table`, `MegaMoeTPEngine`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py) · **Recent**

Source documentation: Compile-time layout (flag / ctrl / LDS slots), shape limits and the per-instance KernelCtx shared by the parts of the fused TP MegaMoE kernel.

Symbols: `mega_moe_tp_shape_supported`, `_nsk2_for`, `gemm2_group_step`, `gemm2_chunk_groups`, `mega_moe_tp_consts`, `KernelCtx`, `kernel_config`, `i8_row_bytes`, `unit_cap`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py) · **Recent**

Source documentation: Fused TP MegaMoE kernel (a4w4, gfx950): the fused tail norm, the wave roles, the kernel entry and its launcher. The device code is built per instance by the build_* parts (communication, gemm, schedule) over one shared KernelCtx.

Symbols: `compile_mega_moe_tp`, `compile_bigm_kernel`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/schedule.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/schedule.py) · **Recent**

Source documentation: Unit scheduling: dynamic (active-expert row-chunk plan) and LB (two-phase GEMM1 / GEMM2 units claimed by whichever CTA is free).

Symbols: `build_schedule`.

### `aiter/ops/flydsl/kernels/mega_moe_tp/sp_rs_norm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mega_moe_tp/sp_rs_norm.py) · **Recent**

Source documentation: Sequence-parallel int8 reduce-scatter + residual add + RMSNorm (+ MoE router), one kernel.

Symbols: `compile_sp_rs_norm`, `SpRsNorm`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_comm_fused_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_comm_fused_moe.py)
- [op_tests/multigpu_tests/test_mega_moe_tp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_mega_moe_tp.py)
- [op_tests/tuners/tune_comm_fused_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuners/tune_comm_fused_moe.py)

## Performance and validation

Record TP/EP topology, expert placement, token counts and collective completion. A local GEMM speedup alone does not establish a fused-MoE gain.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
