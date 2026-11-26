---
title: "Multi-head latent attention: AITER opus"
kind: implementation_knowledge
operator: mla_attention
backend: opus
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Multi-head latent attention — AITER opus

MLA combines latent KV storage with non-positional and positional query/key components. Cache representation and projection placement can move substantial work between GEMM and attention.

OPUS implementations combine architecture-aware policies and native kernel instances. Follow policy and dispatch code to the launched entry and verify that the intended instance engages.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/kernels/mla/opus/mla_decode_a16w16_16mx4_64nx1.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a16w16_16mx4_64nx1.cu) · **Recent**

Symbols: `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a16w16_16mx4_64nx1.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a16w16_16mx4_64nx1.hpp) · **Recent**

Symbols: `pin_vgpr`, `pin_agpr`, `make_layout_gq`, `make_layout_rq`, `make_layout_kv_indices`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `make_layout_rv`, `make_layout_o`, `make_layout_o_packed`, `kv_lds_sub_tile_base`, `kv_lds_bases`, `kv_gmem_bases`, `async_load_kv_inst`, `async_load_kv_tile`, `tr_load_v`, `q_lds_warp_off`, `async_load_q_inst`, `async_load_q_tile`, `load_q_lds`, `store_o_packed`, `load_k_step`, `load_v_step`, `compute_qk`, `compute_pv`, `attn_mask_vec2_imm`, `attn_mask_oob_kv_tile`, `attn_row_max`, `attn_scale_sub`, `attn_exp2`, `attn_row_sum`, `attn_cast_p`, `scale_o_tile`, `softmax_tile`, `attention_tiles`, `decode_one_work`, `__attribute__`.

### `csrc/kernels/mla/opus/mla_decode_a16w16_32mx1_16nx4.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a16w16_32mx1_16nx4.cu) · **Recent**

Symbols: `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a16w16_32mx1_16nx4.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a16w16_32mx1_16nx4.hpp) · **Recent**

Symbols: `pin_vgpr`, `pin_agpr`, `make_layout_gq`, `make_layout_rq`, `make_layout_kv_indices`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `make_layout_rv`, `make_layout_o`, `make_layout_o_packed`, `kv_lds_sub_tile_base`, `kv_lds_bases`, `kv_gmem_bases`, `async_load_kv_inst`, `async_load_kv_tile`, `tr_load_v`, `q_lds_warp_off`, `async_load_q_inst`, `async_load_q_tile`, `load_q_lds`, `store_o_packed`, `load_k_step`, `load_v_step`, `compute_qk`, `compute_pv`, `attn_mask_vec2_imm`, `attn_mask_oob_kv_tile`, `attn_tile_max`, `attn_tile_sum`, `reduce_ml_across_waves`, `attn_scale_sub`, `attn_exp2`, `scale_o_tiles`, `softmax_tile`, `attention_tiles`, `decode_one_work`, `__attribute__`.

### `csrc/kernels/mla/opus/mla_decode_a16w16_32mxt_32nx1.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a16w16_32mxt_32nx1.cu) · **Recent**

Symbols: `__launch_bounds__`, `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a16w16_32mxt_32nx1.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a16w16_32mxt_32nx1.hpp) · **Recent**

Symbols: `pin_vgpr`, `pin_agpr`, `update_o_agpr`, `read_o_agpr`, `read_o_agpr_all`, `clear_o_agpr`, `make_layout_gq`, `make_layout_rq`, `make_layout_kv_indices`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `make_layout_rv`, `make_layout_o`, `make_layout_o_packed`, `kv_lds_sub_tile_base`, `kv_lds_bases`, `q_lds_bases`, `kv_gmem_bases`, `async_load_kv_inst`, `async_load_kv_tile`, `tr_load_v`, `q_lds_warp_off`, `async_load_q_inst`, `async_load_q_tile`, `load_q_lds`, `store_o_packed_half`, `store_o_packed`, `store_o_accum_half`, `store_o_accum`, `load_k_step`, `load_v_step`, `compute_qk`, `compute_pv`, `attn_mask_vec2_imm`, `attn_mask_oob_kv_tile`, `attn_row_max`, `attn_row_sum`, `attn_scale_sub`, `attn_exp2`, `attn_cast_p`, `rescale_o_tiles`, `softmax_tile`, `attention_tiles`, `decode_one_work`, `__attribute__`.

### `csrc/kernels/mla/opus/mla_decode_a8w8_16mx1_16nx8.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a8w8_16mx1_16nx8.hpp) · **Recent**

Symbols: `opus_mla_decode_a8w8_16mx1_16nx8_kernel`, `readfirstlane_f32`, `sched_pv`, `make_layout_gq_nope`, `make_layout_sq_nope`, `make_layout_rq_nope`, `make_layout_gq_rope`, `make_layout_kv_indices`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `p_row_perm`, `p_half_swz`, `make_layout_sp`, `make_layout_rp`, `make_layout_rv`, `make_layout_o`, `attn_row_max`, `attn_scale_sub_row`, `attn_exp2_slice`, `attn_row_sum_local`, `merge_row_sum`, `scale_output_tile`, `pin_output_tile`, `attn_mask_vec2_imm`, `causal_kv_bound`, `attn_mask_kv_tile`, `__attribute__`, `__attribute__`, `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a8w8_16mx1_32nx4.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a8w8_16mx1_32nx4.hpp) · **Recent**

Symbols: `opus_mla_decode_a8w8_16mx1_32nx4_kernel`, `readfirstlane_f32`, `make_layout_gq_nope`, `make_layout_sq_nope`, `make_layout_rq_nope`, `make_layout_gq_rope`, `make_layout_kv_indices_nope`, `make_layout_kv_indices_rope`, `make_layout_gkv_nope`, `make_layout_skv_nope`, `make_layout_gkv_rope`, `make_layout_skv_rope`, `make_layout_rk_nope`, `make_layout_rk_rope`, `make_layout_rv`, `p_half_swz`, `p_row_offset`, `make_layout_sp`, `make_layout_rp`, `make_layout_o`, `attn_row_max`, `attn_scale_sub_row`, `attn_exp2_slice`, `attn_row_sum_local`, `merge_row_sum`, `scale_output_tile`, `pin_output_tile`, `attn_mask_vec2_imm`, `attn_mask_kv_tile`, `__attribute__`, `__attribute__`, `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a8w8_16mx8_32nx1.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a8w8_16mx8_32nx1.hpp) · **Recent**

Symbols: `opus_mla_decode_a8w8_16mx8_32nx1_kernel`, `readfirstlane_f32`, `sched_compute_qk`, `make_layout_q_nope`, `make_layout_q_rope`, `make_layout_kv_indices`, `make_layout_kv_indices_rope`, `make_layout_gk_nope`, `make_layout_sk_nope`, `make_layout_rk_nope`, `make_layout_gk_rope`, `make_layout_sk_rope`, `make_layout_rk_rope`, `make_layout_rv`, `make_layout_o`, `attn_row_max`, `attn_scale_sub_row`, `attn_exp2_slice`, `attn_row_sum`, `scale_output_tile`, `pin_output_tile`, `attn_mask_vec2_imm`, `causal_kv_bound`, `attn_mask_kv_tile`, `__attribute__`, `__attribute__`, `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a8w8_32mx2_32nx2.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a8w8_32mx2_32nx2.cu) · **Recent**

Symbols: `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a8w8_32mx2_32nx2.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a8w8_32mx2_32nx2.hpp) · **Recent**

Symbols: `opus_mla_decode_a8w8_32mx2_32nx2_kernel`, `pin_vgpr`, `pin_agpr`, `readfirstlane_f32`, `buffer_load_lds_asm`, `buffer_load_lds`, `lds_load`, `tr_load_b8`, `make_layout_kv_indices`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `make_layout_rv`, `make_layout_sp`, `make_layout_rp`, `make_layout_srow`, `make_layout_o`, `build_p_half`, `row_combine_max`, `row_combine_sum`, `attn_exp2_slice`, `attn_mask_vec2_imm`, `attn_mask_kv_tile`, `lds_addr`, `smem_at`, `__attribute__`, `__attribute__`, `__attribute__`, `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a8w8_32mx4_64nx1.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a8w8_32mx4_64nx1.cu) · **Recent**

Symbols: `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_a8w8_32mx4_64nx1.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_a8w8_32mx4_64nx1.hpp) · **Recent**

Symbols: `opus_mla_decode_a8w8_32mx4_64nx1_kernel`, `pin_vgpr`, `pin_agpr`, `readfirstlane_f32`, `buffer_load_lds_asm`, `buffer_load_lds`, `tr_load_b8`, `make_layout_q`, `make_layout_kv_indices`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `make_layout_rv`, `make_layout_o`, `build_p_tile`, `row_combine_max`, `row_combine_sum`, `attn_row_max`, `attn_exp2_slice`, `scale_tile_agpr`, `attn_mask_vec2_imm`, `attn_mask_kv_tile`, `lds_addr`, `smem_at`, `__attribute__`, `__attribute__`, `__attribute__`, `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_kargs.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_kargs.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/opus/mla_decode_mxfp8_16mx8_32nx1.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_mxfp8_16mx8_32nx1.hpp) · **Recent**

Symbols: `opus_mla_decode_mxfp8_16mx8_32nx1_kernel`, `sched_compute_qk`, `make_layout_q_nope`, `make_layout_q_rope`, `make_layout_q_mxscl`, `make_layout_kv_indices`, `make_layout_gk_nope`, `make_layout_sk_nope`, `make_layout_rk_nope`, `make_layout_gk_rope`, `make_layout_sk_rope`, `make_layout_rk_rope`, `make_layout_gk_mxscl`, `make_layout_sk_mxscl`, `make_layout_rk_mxscl`, `make_layout_rv_dequant`, `make_layout_sv_dequant`, `make_layout_rv`, `make_layout_o`, `attn_row_max`, `attn_sub_row`, `attn_exp2_slice`, `attn_row_sum`, `scale_output_tile`, `pin_output_tile`, `attn_mask_vec2_imm`, `attn_mask_causal_kv_tile`, `mla_decode_le2_tiles`, `mla_decode_pipelined`, `mla_decode_one_work`, `__launch_bounds__`.

### `csrc/kernels/mla/opus/mla_decode_traits.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_decode_traits.h) · **Recent**

Symbols: `ceil_div`, `cp_local_bound`.

### `csrc/kernels/mla/opus/mla_global_load.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus/mla_global_load.hpp) · **Recent**

Symbols: `global_load_base`, `global_load`, `llvm_amdgcn_global_load_lds`, `global_load_lds`.

### `csrc/kernels/mla/opus_mla_decode_fwd.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/opus_mla_decode_fwd.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode_rope.py)
- [op_tests/test_concat_cache_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_concat_cache_mla.py)
- [op_tests/test_concat_cache_mla_seg.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_concat_cache_mla_seg.py)
- [op_tests/test_flydsl_gather_kv_b_proj.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_gather_kv_b_proj.py)
- [op_tests/test_flydsl_mla_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_mla_reduce.py)
- [op_tests/test_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla.py)
- [op_tests/test_mla_decode_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_decode_gate.py)
- [op_tests/test_mla_decode_pagesize64.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_decode_pagesize64.py)
- [op_tests/test_mla_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_flydsl.py)
- [op_tests/test_mla_ltx.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_ltx.py)
- [op_tests/test_mla_persistent.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_persistent.py)
- [op_tests/test_mla_persistent_round_robin.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_persistent_round_robin.py)
- [op_tests/test_mla_prefill_ps.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_prefill_ps.py)
- [op_tests/test_mla_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_reduce.py)
- [op_tests/test_mla_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_sparse.py)
- [op_tests/test_mla_stage2_merge.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_stage2_merge.py)
- [op_tests/test_mla_v40_persistent.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v40_persistent.py)
- [op_tests/test_mla_v4_kargpreld.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v4_kargpreld.py)
- [op_tests/test_mla_v4_nm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v4_nm.py)
- [op_tests/triton_tests/attention/test_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla.py)
- [op_tests/triton_tests/attention/test_mla_decode_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla_decode_rope.py)
- [op_tests/triton_tests/attention/test_mla_v4_triton.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla_v4_triton.py)
- [op_tests/triton_tests/fusions/test_fused_mla_prefill_qkv_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_mla_prefill_qkv_fp8.py)
- [op_tests/triton_tests/test_gather_kv_b_proj.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_gather_kv_b_proj.py)
- [op_tests/triton_tests/utils/mla_decode_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mla_decode_ref.py)
- [op_tests/triton_tests/utils/mla_extend_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mla_extend_ref.py)

## Performance and validation

Record cache layout, head counts, latent/rope dimensions and query length. Compare the complete wrapper and its reduction/projection helpers.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
