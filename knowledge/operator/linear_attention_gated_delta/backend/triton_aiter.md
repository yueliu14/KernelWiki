---
title: "Linear attention and gated delta rules: AITER triton"
kind: implementation_knowledge
operator: linear_attention_gated_delta
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Linear attention and gated delta rules — AITER triton

Prefill commonly uses chunked state updates while decode applies a recurrent update. State layout, gate transforms, convolution state and in-place update order must match the wrapper.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/chunk_kda.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunk_kda.py) · **Recent**

Source documentation: Chunked Kimi Delta Attention prefill, Gluon on gfx1250 and gfx950: prepare, then the walk. Runs on gluon for 950/1250, takes the triton path for 942

Symbols: `prepare_chunk_kda_metadata`, `_empty`, `_check_tokens`, `_check_index`, `chunk_kda_prepare`, `chunk_kda_walk`, `_walk_gfx950`, `chunk_kda`.

Parameters referenced in this file: `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/kda.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/kda.py) · **Recent**

Source documentation: Fused recurrent Kimi Delta Attention (KDA), gfx950/gfx1250 Gluon decode path. Per token: S = exp(g_t) * S, then S += beta_t * k_t (x) (v_t - S^T k_t), then o_t = S^T q_t. Requires triton >= 3.6.0 on gfx1250. Args: q: [B, T, H, K] queries k: [B, T, H, K] keys v: [B, T, HV, V] values g: [B, T, HV, K] log-space decay gates beta: [B, T, HV] scalar or [B, T, HV, V] headwise write strengths. A_log: [HV] gate parameter dt_bias: [HV, K] optional bias added to g inside the gate chain.

Symbols: `fused_recurrent_kda`, `fused_recurrent_kda_packed_decode`.

Parameters referenced in this file: `num_buffers`, `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_decode_qkvz.py) · **Recent**

Source documentation: Fused Qwen3-Next Gated DeltaNet decode, with an optional FP8 quant epilogue. One Gluon kernel launch replaces the four-kernel decode chain fused_qkvzba_split_reshape_cat -> causal_conv1d_update -> fused_recurrent_gated_delta_rule_packed_decode -> layer_norm_gated and can additionally emit the per-head group-128 FP8 activations that a block-FP8 out_proj consumes directly, removing a separate quantization launch.

Symbols: `_arch_supported`, `_gluon_supported`, `fused_gdn_decode_qkvz_supported`, `fused_gdn_decode_qkvz`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py) · **Recent**

Source documentation: Fused Qwen3-Next Gated DeltaNet *prefill*, with an FP8 group-quant epilogue.

Symbols: `_arch_supported`, `_gluon_supported`, `_select_tile_key`, `FP8PrecisionConfig`, `_run_chunked_delta`, `gdn_prefill_group_fp8_quant_m1024_3071`, `_Schedule`, `_schedule`, `gdn_prefill_group_fp8_quant_m12289_16384_b1_5`, `gdn_prefill_group_fp8_quant_m12289_16384_b6_15`, `gdn_prefill_group_fp8_quant_m3072_16384`, `_load_tile`, `fused_gdn_prefill_qkvz_supported`, `fused_gdn_prefill_qkvz`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_kda_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_kda_decode.py) · **Recent**

Source documentation: Fused KDA decode: conv1d + delta-rule recurrence + gated RMSNorm. Single Triton kernel launch. Fuses three separate operations into one, eliminating q/k/v intermediate HBM traffic and kernel launch overhead.

Symbols: `_launch_config`, `fused_kda_decode`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_kda_decode_unified.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_kda_decode_unified.py) · **Recent**

Source documentation: Unified fused KDA decode: conv1d + recurrence + gated RMSNorm. Python wrapper for the unified kernel. Mode is auto-selected from args: - state_indices only → normal decode - state_indices + num_accepted → DSpark spec decode - slot_idx + buf_k/u/g → ReplaySSM - slot_idx + buf_k/u/g + conv_state_indices → DSpark + ReplaySSM

Symbols: `fused_kda_decode_unified`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py) · **Recent**

Source documentation: Fused sigmoid-gated delta rule over packed QKV. draft_window is host metadata required for FlyDSL dispatch. FlyDSL falls back to Triton during CUDA Graph capture.

Symbols: `_flydsl_gdr_enabled`, `_try_flydsl_mtp`, `fused_rearrange_sigmoid_gated_delta_rule`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/gated_delta_net/gated_delta_rule.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/gated_delta_rule.py) · **Recent**

Source documentation: Gated Delta Rule Operations (Forward Only). This module provides high-level interfaces for gated delta rule computations, including both fused recurrent and chunk-based implementations. Important Note: Only forward pass is implemented in aiter. These functions do NOT support gradient computation or backward pass. For training with autograd, please use the flash-linear-attention library instead. These implementations are optimized for inference and forward-only operations.

Symbols: `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `chunk_gated_delta_rule_opt`, `chunk_gated_delta_rule_opt_vk`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py) · **Recent**

Source documentation: Kimi Delta Attention (KDA) chunked forward pass (Forward Only). This is the public entry point for the chunk_delta_attn Triton kernels. The signature mirrors fla.ops.kda.chunk_kda so serving stacks can swap the two backends without reshaping tensors or re-deriving the gate. Important Note: Only the forward pass is implemented. These functions do NOT support gradient computation. For training, use the flash-linear-attention library.

Symbols: `chunk_kimi_delta_attn`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py) · **Recent**

Source documentation: Fused KDA decode kernel: conv1d + delta-rule recurrence + gated RMSNorm. Single Triton kernel. Grid: (batch, heads).

Symbols: `_valid_spec_prefix`, `fused_conv_recurrent_norm_kernel`, `fused_kda_spec_parallel_v_kernel`, `fused_kda_spec_finalize_kernel`.

Parameters referenced in this file: `num_stages`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm_unified.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm_unified.py) · **Recent**

Source documentation: Unified fused KDA decode kernel: conv1d + recurrence + gated RMSNorm.

Symbols: `_fused_kda_decode_unified_kernel`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_rearrange_sigmoid_gdr.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_rearrange_sigmoid_gdr.py) · **Recent**

Symbols: `fused_rearrange_sigmoid_gated_delta_rule_update_kernel`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_recurrent.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_recurrent.py) · **Recent**

Source documentation: Fused recurrent gated delta rule forward kernel (Forward only). This module provides an optimized fused recurrent implementation of the gated delta rule. Note: Only forward pass is implemented. Backward pass is not supported in aiter.

Symbols: `_fused_recurrent_gated_delta_rule_fwd_kernel`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_sigmoid_gating_recurrent.py) · **Recent**

Source documentation: Fused kernel that combines sigmoid gating computation with recurrent delta rule update.

Symbols: `is_cuda`, `get_cuda_autotune_config`, `get_hip_autotune_config`, `get_autotune_config`, `fused_sigmoid_gating_delta_rule_update_kernel`, `fused_sigmoid_gating_delta_rule_update`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/fused_qkvzba_split.py) · **Recent**

Source documentation: Fused kernels for QKVZBA split, reshape and concatenation.

Symbols: `_fused_qkvzba_split_reshape_cat_decode_kernel`, `fused_qkvzba_split_reshape_cat_decode`, `_fused_qkvzba_prefill_kernel_nqk1`, `_fused_qkvzba_prefill_kernel_nqk2`, `_fused_qkvzba_prefill_kernel_nqk4`, `_fused_qkvzba_prefill_kernel_nqk8`, `_fused_qkvzba_split_reshape_cat_prefill_kernel`, `fused_qkvzba_split_reshape_cat_prefill`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/gated_delta_rule_utils.py) · **Recent**

Source documentation: Checks the current operating system, Triton version, and Python version, issuing warnings if they don't meet recommendations. This function's body only runs once due to lru_cache.

Symbols: `check_environments`, `get_abs_err`, `get_err_ratio`, `assert_close`, `tensor_cache`, `input_guard`, `require_version`, `Action`, `deprecate_kwarg`, `checkpoint`, `check_pytorch_version`, `_cpu_device_warning`, `get_multiprocessor_count`, `get_available_device`, `map_triton_backend_to_torch_device`, `get_all_max_shared_mem`, `Backend`, `check_shared_mem`, `_register_aliases`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk.py) · **Recent**

Source documentation: Chunk-based gated delta rule forward computation. This module implements the chunk-based parallel computation for the gated delta rule. Note: Only forward pass is implemented. Backward pass is not supported in aiter.

Symbols: `_is_unsupported_gfx12_runtime`, `_is_gfx12_runtime`, `chunk_gated_delta_rule_fwd`, `chunk_gated_delta_rule_fwd_opt`, `chunk_gated_delta_rule_fwd_opt_vk`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_delta_h.py) · **Recent**

Source documentation: Chunk-based hidden state computation for gated delta rule (Forward only). This module computes the per-chunk hidden states and the recomputed value tensor (v_new) consumed by the chunked gated delta rule, supporting both the [K, V] and the transposed [V, K] hidden-state layouts.

Symbols: `_gate_exp`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_gated_delta_rule_fwd_h`, `chunk_gated_delta_rule_fwd_kernel_h_opt`, `chunk_gated_delta_rule_fwd_h_opt`, `chunk_gated_delta_rule_fwd_kernel_h_opt_vk`, `chunk_gated_delta_rule_fwd_h_opt_vk`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/chunk_o.py) · **Recent**

Source documentation: Chunk-based output computation (Forward only). This module provides functions for computing the final output in chunk mode.

Symbols: `_bp_ld1d`, `_bp_st1d`, `_bp_ld2d`, `_bp_st2d`, `chunk_fwd_kernel_o`, `chunk_fwd_o`, `chunk_fwd_kernel_o_opt`, `chunk_fwd_o_opt`, `chunk_fwd_kernel_o_opt_vk`, `chunk_fwd_o_opt_vk`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_cumsum_kkt.py) · **Recent**

Source documentation: Fused cumsum + KKT. Args: g: [B, T, H] k: [B, T, Hg, K] beta: [B, T, H] Returns: g_cumsum: [B, H, T] A: [B, T, H, chunk_size], strictly lower triangular

Symbols: `safe_exp`, `_fused_cumsum_kkt_kernel`, `fused_cumsum_kkt`, `fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel`, `fused_chunk_local_cumsum_scaled_dot_kkt_fwd`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_gdn_gating_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_gdn_gating_prefill.py) · **Recent**

Source documentation: Fused sigmoid + gdn_gating kernel for prefill.

Symbols: `fused_gdn_gating_sigmoid_kernel`, `fused_gdn_gating_and_sigmoid`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gated_delta_net/prefill/fused_solve_tril_recompute.py) · **Recent**

Source documentation: Fused triangular solve + recompute w, u in a single kernel. Eliminates the intermediate Ai tensor (64x64 per chunk x head) global memory round-trip by keeping the inverse blocks in registers.

Symbols: `_should_use_split_path`, `_bp_ld1d`, `_bp_ld2d`, `_bp_st2d`, `fused_solve_tril_recompute_w_u_kernel`, `recompute_w_u_head_major_kernel`, `_run_split_path`, `fused_solve_tril_recompute_w_u`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_delta_attn_utils.py) · **Recent**

Source documentation: Shared utilities for chunk_delta_attn kernels.

Symbols: `chunk_delta_attn_tuned_config`, `chunk_delta_attn_tuned_config_shortlist`, `_same_arg`, `_same_call`, `tensor_cache`, `_get_available_device`, `check_shared_mem`, `softplus`, `input_guard`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_fwd.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_fwd.py) · **Recent**

Source documentation: Top-level forward function for chunk_delta_attn. Pipeline: 1. Gate cumsum: apply fused A_log / dt_bias / softplus (or sigmoid) gating and chunk-local prefix sum to produce gk (in log2 space). 2. Intra-chunk: compute Aqk, Akk inverse, and auxiliary W/U/KG tensors. 3. Inter-chunk: update recurrent hidden state H via chunk_gated_delta_rule_fwd_h. 4. Output: compute final output via chunk_gla_fwd_o (gated q * h + A * v).

Symbols: `chunk_delta_attn_fwd`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/fast_launch.py) · **Recent**

Source documentation: Launch a Triton kernel without redoing the part that only depends on shapes. JITFunction.run binds the arguments, specializes each one, builds a cache key and looks the kernel up, and it does all of that on every call.

Symbols: `bypassed`, `recording`, `_tensor_key`, `_FastLaunch`, `fast_launch`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py) · **Recent**

Source documentation: Two-kernel fused KDA forward, following the FlashKDA kernel split. The default chunk_delta_attn_fwd pipeline runs five kernels (l2norm, beta sigmoid, gate cumsum, intra, delta_h, gla_o) and materializes g_cumsum, Aqk, Akk, w, u and h in HBM between them.

Symbols: `_flash_kda_prepare_kernel`, `_seg_occupancy_class`, `_flash_kda_segment_kernel`, `_flash_kda_seg_scan_kernel`, `flash_kda_supported`, `_gluon_k1_usable`, `_gluon_k2_usable`, `_log_route`, `_k2_gluon_schedule`, `_num_cus`, `_scan_bv`, `_choose_chunks_per_seg`, `_seq_bounds`, `_build_segments`, `_check_paged_state_cache`, `flash_kda_fwd`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gate.py) · **Recent**

Source documentation: Gate activation kernels for chunk_delta_attn. Provides: - beta_sigmoid_fwd: elementwise sigmoid for the beta gate (forward only). - chunk_delta_attn_gate_fwd: per-token fused gate (-exp(A)*softplus or lower_bound*sigmoid(exp(A)*g)) without cumsum (forward only).

Symbols: `beta_sigmoid_fwd_kernel`, `beta_sigmoid_fwd`, `chunk_delta_attn_gate_fwd_kernel`, `chunk_delta_attn_gate_fwd`.

Parameters referenced in this file: `BLOCK_SIZE`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/gla_output.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/gla_output.py) · **Recent**

Source documentation: GLA output computation for chunk_delta_attn forward pass. chunk_gla_fwd_o computes the final attention output: o = (q * exp2(gk)) @ h * scale + A @ v_new where h is the recurrent hidden state and A is the local intra-chunk attention matrix (Aqk).

Symbols: `chunk_gla_fwd_kernel_o`, `chunk_gla_fwd_o`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/intra_attn.py) · **Recent**

Source documentation: Intra-chunk attention kernels for chunk_delta_attn forward pass. Three-step pipeline: 1. Token-parallel kernel: computes diagonal Akk blocks + full Aqk row-by-row. 2. Sub-chunk intra kernel (safe-gate path): alternative diagonal-block kernel. 3. Fused inter + solve_tril kernel: computes off-diagonal Akk, then solves the full chunk triangular system in one pass → writes Akk_inv.

Symbols: `chunk_delta_attn_fwd_kernel_intra_token_parallel`, `_chunk_delta_attn_fwd_intra_token_parallel`, `chunk_delta_attn_fwd_kernel_intra_sub_chunk`, `chunk_delta_attn_fwd_kernel_inter_solve_fused`, `chunk_delta_attn_fwd_intra`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/kimi_delta_attn/wy_fast.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/kimi_delta_attn/wy_fast.py) · **Recent**

Source documentation: Recompute W and U tensors for chunk_delta_attn forward pass. recompute_w_u_fwd takes the chunk Akk inverse matrix (A), key (k), value (v), gate-cumsum (gk) and beta scalar to produce: - w = A @ (k * beta * exp2(gk)) [chunk-gated key correction] - u = A @ (v * beta) [chunk-gated value target] - qg = q * exp2(gk) [gated query, optional] - kg = k * exp2(gk_last - gk) [boundary-normalised key]

Symbols: `recompute_w_u_fwd_kernel`, `recompute_w_u_fwd`.

Parameters referenced in this file: `num_stages`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx950/triton/attention/fused_kda_decode/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/fused_kda_decode/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_chunk_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_chunk_kda.py)
- [op_tests/op_benchmarks/triton/bench_flash_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_flash_kda.py)
- [op_tests/op_benchmarks/triton/bench_fused_gdn_decode_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gdn_decode_qkvz.py)
- [op_tests/op_benchmarks/triton/bench_fused_gdn_prefill_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gdn_prefill_qkvz.py)
- [op_tests/op_benchmarks/triton/bench_fused_kda_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_kda_decode.py)
- [op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py)
- [op_tests/op_benchmarks/triton/bench_gated_delta_rule_snapshot_dtype.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gated_delta_rule_snapshot_dtype.py)
- [op_tests/op_benchmarks/triton/bench_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_kda.py)
- [op_tests/triton_tests/attention/test_chunk_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_chunk_kda.py)
- [op_tests/triton_tests/attention/test_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_kda.py)
- [op_tests/triton_tests/gated_delta_net/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/__init__.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_gdn_decode_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_gdn_decode_qkvz.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_gdn_prefill_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_gdn_prefill_qkvz.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_kda_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_kda_decode.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_kda_decode_unified.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_kda_decode_unified.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_rearrange_sigmoid_gdr.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_rearrange_sigmoid_gdr.py)
- [op_tests/triton_tests/gated_delta_net/test_gated_delta_rule_strided_qk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_gated_delta_rule_strided_qk.py)
- [op_tests/triton_tests/kimi_delta_attn/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/__init__.py)
- [op_tests/triton_tests/kimi_delta_attn/test_chunk_delta_attn_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_chunk_delta_attn_fwd.py)
- [op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py)
- [op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py)
- [op_tests/triton_tests/utils/gdr_mtp_refs.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/gdr_mtp_refs.py)
- [op_tests/triton_tests/utils/kda_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/kda_ref.py)
- [op_tests/test_flydsl_gdr_mtp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_gdr_mtp.py)
- [op_tests/test_flydsl_linear_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention.py)
- [op_tests/test_flydsl_linear_attention_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention_prefill.py)
- [op_tests/test_gated_delta_rule.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gated_delta_rule.py)
- [op_tests/test_gdn_prepare.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdn_prepare.py)
- [op_tests/test_gdr_decode_packed_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdr_decode_packed_bf16.py)
- [op_tests/test_split_gdr_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_split_gdr_update.py)

## Performance and validation

Validate a sequence of updates against the reference, not just one isolated call. Include preparation, triangular solves and normalization for the chosen path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
