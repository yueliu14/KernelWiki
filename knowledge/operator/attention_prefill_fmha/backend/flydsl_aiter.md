---
title: "Prefill and full attention: AITER flydsl"
kind: implementation_knowledge
operator: attention_prefill_fmha
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Prefill and full attention — AITER flydsl

Tiled attention interleaves QK products, online softmax and probability-value products. Inference paths require the correct causal or variable-length mask, head layout and KV-cache contract. Use forward entry points with dropout disabled.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/fmha_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fmha_kernels.py) · **Recent**

Source documentation: High-level FlyDSL Flash Attention APIs. flydsl_flash_attn_batch_func / flydsl_flash_attn_varlen_func dispatch by arch and dtype: gfx950 fp8 to kernels/fmha_gfx950, gfx1250 bf16/f16 to the m32x8 prefill kernel, anything else None so the caller falls through to CK/Triton. flydsl_flash_attn_func (gfx1201 / RDNA4) wraps the flash_attn_func_gfx1201 kernel with: - Build cache keyed by (num_heads, head_dim, causal, dtype, waves_per_eu, daz).

Symbols: `_torch_dtype_to_str`, `_get_kernel`, `flydsl_flash_attn_func`, `_check_out`, `_fp8_gfx950_supported`, `flydsl_flash_attn_varlen_func`, `flydsl_flash_attn_batch_func`.

Parameters referenced in this file: `BLOCK_M`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_fp8_gfx950.py) · **Recent**

Source documentation: FlyDSL Flash Attention fp8 (e4m3fn) forward for gfx950. Q/K/V are pre-quantized e4m3fn with per-tensor fp32 shape-[1] descales; the output is bf16. Dense, packed-varlen and split-K.

Symbols: `_is_valid_softmax_scale`, `_fp8_rescale_threshold`, `_gpu_arch_cached`, `_num_cu_cached`, `_gpu_arch`, `_num_cu`, `_fp8_config_reason`, `flydsl_flash_attn_fp8_supported`, `_fp8_auto_block_m`, `_fp8_batch_interleave_group`, `_fp8_auto_kv_splits`, `_build_fp8`, `flydsl_flash_attn_fp8_func`.

Parameters referenced in this file: `BLOCK_M`.

### `aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py) · **Recent**

Source documentation: Flash Attention kernel for gfx1201 (RDNA4). Uses 16x16x16 wave32 WMMA, online softmax, pipelined V loads, and flattened BSHD inputs. Requires head_dim >= 64 and head_dim % 32 == 0.

Symbols: `build_flash_attn_func_module_primary`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_N_OUT`, `BLOCK_SIZE`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_b16_buffer_managers.py) · **Recent**

Source documentation: 16-bit (bf16/fp16) Q/K/V staging managers for the gfx1250 MHA kernels. Each manager owns the global -> LDS (async) -> VGPR (WMMA fragment) path for one 16-bit-element operand (Q, K or V): the LDS layout, the global->LDS copy schedule and the fragment read. They are self-contained — a caller passes only the configuration it already maintains (hdim, gqa_ratio, kv block width, wave count) through the constructor plus the runtime warp_idx/lane_idx.

Symbols: `_assert_multiple`, `_as_bases`, `_lds_ptr_b128`, `_async_load_to_lds`, `ProducerCtx`, `BufferOpDescriptor`, `AsyncCopyDescriptor`, `TdmCopyDescriptor`, `_warp_band`, `_async_band_descriptor`, `_pow2_segments`, `_tdm_load_views`, `_tdm_band_descriptor`, `_AsyncTransport`, `_TdmTransport`, `QManager16bV1`, `KManager16bV1`, `VManager16bV1`, `QManager16bV2`, `KManager16bV2`, `VManager16bV2`, `OManager16bV1`, `OManager16bV2`, `OManager16bV3`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py) · **Recent**

Source documentation: MHA Forward Prefill kernel — m32x8 design, gfx1250 (MI400 / mi450). A clean FlyDSL kernel written in the high-level layout-algebra style (tiled copy / tiled MMA + SharedAllocator). m32x8 names the threadgroup shape: **8 waves per threadgroup**, each wave owning a **32-row** Q span (2 adjacent 16-row WMMA tiles). gfx1250 runs wave32, so a threadgroup is 8 * 32 = 256 threads and BLOCK_M = 32 * 8 = 256 Q rows.

Symbols: `supported_num_q_tiles`, `supports_gqa_ratio`, `_block_m`, `_m_tile_idx`, `WarpType`, `_warp_id`, `_lane_id`, `_copy_wait`, `_kv_drain_depths`, `_bare_barrier`, `_copy_fence`, `_load_seqlen_pair`, `_load_sink_logit`, `_packed_tile_indices`, `_wmma`, `_convert_vectors`, `_keepalive`, `_ring_num_prefetch`, `_ring_drive`, `_tree_reduce_multi`, `_softmax`, `_pv_qk_gemm`, `_wg_kv_span`, `_alloc_lds`, `_prologue_attention`, `schedule_bank`, `_o_pack`, `_o_unpack`, `_core_attention_multi_kv_tiles`, `_core_attention_one_kv_tile`, `_o_manager`, `_store_o_lse`, `_epi_state`, `_epi_state_zeros`, `_epilogue_attention`, `_zero_fill_attention`, `kv_split_bytes`, `pick_n_block`, `build_fmha_fwd_prefill_a16w16_m32x8`, `_pick_one_kv_tile`, `_pick_num_q_tiles`, `_ensure_thd_kernel`, `_ensure_bshd_kernel`, `flash_attn_varlen_m32x8`, `flash_attn_batch_m32x8`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_SIZE`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/flash_attn_fp8_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/flash_attn_fp8_gfx950.py) · **Recent**

Source documentation: gfx950 DUALWAVE_SWP FP8 flash attention.

Symbols: `build_flash_attn_dualwave_swp_fp8_module`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_combine.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_combine.py) · **Recent**

Source documentation: Split-K combine pass and workspace sizing.

Symbols: `DualwaveSplitKCombineContext`, `DualwaveSplitKCombineHelper`, `dualwave_splitk_workspace_elems`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_epilog.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_epilog.py) · **Recent**

Source documentation: Epilogue: output packing and the dense / split-K partial stores.

Symbols: `DualwaveFp8StoreHelper`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_gemm.py) · **Recent**

Source documentation: The two GEMMs: QK^T and PV.

Symbols: `DualwaveFp8GemmHelper`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_lds.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_lds.py) · **Recent**

Source documentation: LDS data movement: Q load, K/V global->LDS, LDS->VGPR.

Symbols: `DualwaveFp8QLoader`, `DualwaveFp8KvGmemToLdsLoader`, `DualwaveFp8KvLdsToVgprLoader`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/op_softmax.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/op_softmax.py) · **Recent**

Source documentation: Online softmax, with causal / seq-pad masking.

Symbols: `DualwaveFp8SoftmaxHelper`.

Parameters referenced in this file: `BLOCK_N`.

### `aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py) · **Recent**

Source documentation: Per-block forward pass: shared primitives, traits, and the kernel context.

Symbols: `_waitcnt_vm_n`, `_s_setprio`, `_read_exec_i64`, `_ds_read_tr8_b64_imm`, `_bitcast_i32`, `_bitcast_f32`, `_attn_mask_vec2_imm`, `_reduction_pair`, `_anchor_scalar_f32`, `_anchor_v_o`, `_score_pair_to_lists`, `_score_lists_to_vecs`, `_reduce_score_pair`, `_lane_pair_reduce`, `_score_pair_max`, `_score_pair_sum`, `_scale_sub_score_pair`, `_exp2_score_slice`, `_scale_o_accs`, `_causal_pair_thresholds`, `_apply_dualwave_causal_mask_pair`, `_cu_load`, `_make_ws_rsrc`, `_p_headroom_log2`, `_store_lse`, `_buffer_load_128`, `_buffer_load_lds_128`, `_buffer_store_128`, `_stagger_extra_barrier_if_one`, `DualwaveSwpFp8Traits`, `_make_dualwave_swp_fp8_traits`, `dualwave_fp8_dma_per_iter`, `_init_dualwave_thread_mapping`, `_init_dualwave_q_row`, `DualwaveFp8KernelContext`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/flydsl/bench_mha_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_fwd.py)
- [op_tests/cpp/mha/compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/mha/compile.py)
- [op_tests/op_benchmarks/triton/bench_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_attn_res.py)
- [op_tests/op_benchmarks/triton/bench_deepgemm_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_deepgemm_attention.py)
- [op_tests/op_benchmarks/triton/bench_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_extend_attention.py)
- [op_tests/op_benchmarks/triton/bench_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mha.py)
- [op_tests/op_benchmarks/triton/bench_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_prefill.py)
- [op_tests/op_benchmarks/triton/compare_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/compare_mha.py)
- [op_tests/test_flydsl_fmha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fmha.py)
- [op_tests/test_fmha_fwd_mxfp8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_mxfp8_asm.py)
- [op_tests/test_fmha_fwd_with_sink_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_asm.py)
- [op_tests/test_fmha_fwd_with_sink_varlen_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_varlen_asm.py)
- [op_tests/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha.py)
- [op_tests/test_mha_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_flydsl.py)
- [op_tests/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_fp8.py)
- [op_tests/test_mha_native_splitkv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_native_splitkv.py)
- [op_tests/test_mha_optional_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_optional_flydsl.py)
- [op_tests/test_mha_v4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4.py)
- [op_tests/test_mha_v4_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4_sparse.py)
- [op_tests/test_mha_varlen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen.py)
- [op_tests/test_mha_varlen_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_fp8.py)
- [op_tests/test_mha_varlen_large_kv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_large_kv.py)
- [op_tests/triton_tests/attention/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/__init__.py)
- [op_tests/triton_tests/attention/mha_test_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/mha_test_utils.py)
- [op_tests/triton_tests/attention/test_chunked_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_chunked_pa_prefill.py)
- [op_tests/triton_tests/attention/test_dsv4_indexer.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_dsv4_indexer.py)
- [op_tests/triton_tests/attention/test_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_extend_attention.py)
- [op_tests/triton_tests/attention/test_fav3_sage.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage.py)
- [op_tests/triton_tests/attention/test_fav3_sage_compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage_compile.py)
- [op_tests/triton_tests/attention/test_la_paged.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_la_paged.py)
- [op_tests/triton_tests/attention/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha.py)
- [op_tests/triton_tests/attention/test_mha_dao_ai.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_dao_ai.py)
- [op_tests/triton_tests/attention/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_fp8.py)
- [op_tests/triton_tests/attention/test_mha_spatial_swizzle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_spatial_swizzle.py)
- [op_tests/triton_tests/attention/test_mha_v3.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_v3.py)
- [op_tests/triton_tests/attention/test_mha_with_pe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_pe.py)
- [op_tests/triton_tests/attention/test_mha_with_sink.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_sink.py)
- [op_tests/triton_tests/attention/test_na3d_flash.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_na3d_flash.py)
- [op_tests/triton_tests/attention/test_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill.py)
- [op_tests/triton_tests/attention/test_prefill_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_prefill_attention.py)
- [op_tests/triton_tests/attention/test_qsa.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_qsa.py)
- [op_tests/triton_tests/fusions/test_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_attn_res.py)

## Performance and validation

Record sequence lengths, masks, heads and dtypes. Use the matching forward reference with dropout disabled; do not compare kernel latency with an end-to-end serving metric.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
