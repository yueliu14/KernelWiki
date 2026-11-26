---
title: "Prefill and full attention: AITER triton"
kind: implementation_knowledge
operator: attention_prefill_fmha
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Prefill and full attention — AITER triton

Tiled attention interleaves QK products, online softmax and probability-value products. Inference paths require the correct causal or variable-length mask, head layout and KV-cache contract. Use forward entry points with dropout disabled.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/chunked_pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/chunked_pa_prefill.py) · **Recent**

Source documentation: Unified attention for mixed prefill (multi-token) and decode (single-token) sequences with paged KV cache. Args: query (torch.Tensor): Query tensor with shape (total_tokens, num_q_heads, head_dim). key (torch.Tensor): Key tensor for prefill portion with shape (total_tokens, num_kv_heads, head_dim). value (torch.Tensor): Value tensor for prefill portion with shape (total_tokens, num_kv_heads, head_dim).

Symbols: `chunked_prefill_paged_decode`.

Parameters referenced in this file: `BLOCK_SIZE`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/dsv4_indexer.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/dsv4_indexer.py) · **Recent**

Source documentation: Compute indexer scores. Args: q: [S, H, Hd] bf16 k: [P, Hd] bf16 w: [S, H] fp32 compress_ratio: int Returns: scores: [S, P] fp32

Symbols: `_causal_mask`, `indexer_fwd`, `dsv4_indexer`.

### `aiter/ops/triton/attention/extend_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/extend_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supports page size = 1 and prefill with KV cache (i.e. extend).

Symbols: `extend_attention_fwd`, `redundant_attention`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DPE`, `BLOCK_DV`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/attention/fav3_sage.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage.py) · **Recent**

Source documentation: Return Triton launch config for fav3_sage sage_fwd. Default behaviour is unchanged from upstream aiter (gfx942: num_warps=8, etc.). Set a3_tuned=True to opt into MI308/gfx942 perf tuning for 128/64 tiles (num_warps=4, waves_per_eu=2). Numerics match the default launch params; use this when long sequences (e.g. Qwen edit ~8400 tokens) are register-bound.

Symbols: `_load_sage_fwd_tables`, `get_sage_fwd_configs`, `_FAv3SageWrapperFunc`, `fav3_sage_wrapper_func`, `fav3_sage_func`.

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_N`, `BLOCK_R`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py) · **Recent**

Source documentation: Returns tuned config for MXFP4 on supported architectures.

Symbols: `get_sage_fwd_configs_mxfp4`, `_FAv3SageMXFP4WrapperFunc`, `fav3_sage_mxfp4_wrapper`, `fav3_sage_mxfp4_func`.

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_R`.

### `aiter/ops/triton/attention/mha.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha.py) · **Recent**

Source documentation: True when the Gluon MHA forward kernel can actually run on this device.

Symbols: `is_gluon_available`, `_resolve_backend`, `mha_set_impl`, `mha_set_use_int64_strides`, `_fwd_offsets_fit_int32`, `mha_set_swizzle`, `_get_sliding_window_size`, `gluon_forward_unsupported_reason`, `check_gluon_forward_support`, `_get_softmax_scale`, `_pack_attn_returns`, `_gluon_flash_attn_forward`, `_flash_attn_forward`, `_FlashAttnFunc`, `flash_attn_func`, `_FlashAttnVarlenFunc`, `flash_attn_varlen_func`, `flash_attn_with_kvcache`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_OUT`, `BLOCK_DMODEL_PE`, `BLOCK_DMODEL_PE_POW2`, `BLOCK_DMODEL_POW2`, `BLOCK_M`, `BLOCK_N`, `num_stages`.

### `aiter/ops/triton/attention/mha_v3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/mha_v3.py) · **Recent**

Source documentation: FlashAttention v3 entry point.

Symbols: `_FlashAttnV3Func`, `flash_attn_func`, `_FlashAttnVarlenV3Func`, `flash_attn_varlen_func`, `flash_attn_with_kvcache_fake_tensor`, `_flash_attn_with_kvcache`, `flash_attn_with_kvcache`, `_quantize_bshd`, `_quantize_thd`, `_FlashAttnFP8Wrapper`, `flash_attn_fp8_func`, `_FlashAttnVarlenFP8Wrapper`, `flash_attn_varlen_fp8_func`.

### `aiter/ops/triton/attention/na3d_flash.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/na3d_flash.py) · **Recent**

Source documentation: 3D neighborhood flash attention launcher. Each query token at grid position (t, h, w) attends to the KT x KH x KW neighborhood centered on it, with inward border-shift so every window has exactly KT x KH x KW keys. Usage:: from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn out = na3d_flash_attn(q, k, v, kernel_size=(11, 11, 11)) # q, k, v: (B, T, H, W, NH, HD) bfloat16, Q already scaled by head_dim**-0.5 # out : (B, T, H, W, NH, HD) bfloat16

Symbols: `na3d_flash_attn`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_Q`, `num_stages`, `num_warps`.

### `aiter/ops/triton/attention/pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_prefill.py) · **Recent**

Source documentation: Paged attention prefill for multi-token context processing with paged KV cache. Supports variable-length sequences, GQA, FP8 quantization, ALiBi, and sliding window. Args: q (torch.Tensor): Query tensor with shape (total_tokens, num_q_heads, head_dim). k (torch.Tensor): Key tensor for prefill tokens with shape (total_tokens, num_kv_heads, head_dim). v (torch.Tensor): Value tensor for prefill tokens with shape (total_tokens, num_kv_heads, head_dim).

Symbols: `context_attention_fwd`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_PADDED`, `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/pod_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pod_attention.py) · **Recent**

Source documentation: POD (Prefill-On-Decode) fused attention for simultaneous prefill and decode execution. Launches persistent kernels that execute both operations concurrently on different CUs for improved hardware utilization. Args: cu_ctr (torch.Tensor): CU (Compute Unit) counter for workload distribution. q (torch.Tensor): Decode query with shape (batch_size * 1, num_heads, head_dim). k (torch.Tensor): Decode key with shape (total_tokens, num_heads, head_dim).

Symbols: `pod_attention`, `get_num_splits_and_buffer_sizes`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/prefill_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/prefill_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supporst page size = 1.

Symbols: `context_attention_fwd`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`.

### `aiter/ops/triton/attention/utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/utils.py) · **Recent**

Source documentation: Convert a dense block attention mask to a ragged look-up table of KV block indices per (batch, head, q_block). Used for block-sparse attention with no per-iteration branching in the kernel. block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks) for per-head masks. True = may attend, False = must not attend. num_heads: Required when block_attn_mask is 3D (number of Q heads).

Symbols: `block_attn_mask_to_ragged_lut`.

Parameters referenced in this file: `BLOCK_KB`.

### `aiter/ops/triton/fusions/attn_res.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/attn_res.py) · **Recent**

Source documentation: Attention-residual (residual candidate gating) forward.

Symbols: `_tensor_spec`, `_fresh_tensor_spec`, `_int_spec`, `_next_pow2`, `_dtype_max`, `_pick_attn_res_separate_bl`, `_get_attn_res_config`, `_pick_attn_res_separate_num_warps`, `_pick_attn_res_seq_config`, `_fast_reshape2d`, `_fast_reshape3d`, `_fast_flatten1d`, `_pick_attn_res_packed_config`, `_launch_tune_kwargs`, `_assert_launch_cache_key`, `_launch_hooks_idle`, `_launch_attn_res`, `_build_ptr_table`, `attn_res_fwd`, `_run_sequence`, `_run_packed`, `attn_res_gate`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/quant/mxfp6_fmha_pack.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/mxfp6_fmha_pack.py) · **Recent**

Source documentation: MXFP6-E2M3 packing utilities for the gfx950 MHA v4 kernels.

Symbols: `fp6_k_raw_buffer_sizes`, `_k_lds_order_gather_index`, `_v_field_perm`, `_v_kvtab_dev`, `quantize_fp6_v_clean_triton`, `quantize_fp6_v_data_scale_triton`, `_v_noswap_kvtab`, `_v_direct_kvtab`, `_qk_field_perm`, `_qk_field_perm_dev`, `quantize_fp6_lastdim_triton`, `_k_lds_gather_index`, `_k_lds_src_within`, `reorder_fp6_k_lds_order_triton`, `quantize_fp6_k_lds_order_triton`, `fp6_k_lds_order_views_from_raw`, `_qk_field_perm_pt`, `_e2m3_encode_torch`, `quantize_fp6_lastdim_torch`, `_k_scale_tail_index`, `quantize_fp6_k_lds_order_torch`, `pack_fp6_v_data_scale_views`.

Parameters referenced in this file: `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/quant/sage_attention_quant_wrappers.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/quant/sage_attention_quant_wrappers.py) · **Recent**

Source documentation: Round a V sequence length up to the 128-token FP4 packing tile.

Symbols: `_bshd_order`, `sage_quant_mxfp4`, `_apply_int8_q_smoothing`, `_f4f4_v_kperm`, `fp4_v_padded_sequence`, `fp4_v_raw_buffer_size`, `sage_quant_v_f4f4`, `pack_v_mxfp4_colmajor_raw`, `_pack_v_mxfp4_colmajor_raw_fake`, `sage_quant_v_mxfp4`, `sage_quant_mxfp6`, `sage_quant`, `rotation_smooth_qk`, `smooth_rotate_downcast_qk`, `create_hadamard_matrix`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_M`, `BLOCK_N`, `BLOCK_R`, `BLOCK_SIZE_M`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/chunked_pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/chunked_pa_prefill.py) · **Recent**

Symbols: `cdiv_fn`, `_kernel_paged_attention_2d`.

Parameters referenced in this file: `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/attention/extend_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/extend_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supports page size = 1 and prefill with KV cache (i.e. extend).

Symbols: `_fwd_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DPE`, `BLOCK_DV`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py) · **Recent**

Source documentation: Calculate the window boundaries for a query block.

Symbols: `map_dims`, `_sage_fwd_no_mask`, `_sage_fwd_blocksparse_nomask`, `_sage_fwd_blocksparse_mask`, `_sage_fwd_mask`, `compute_window_bounds`, `classify_window_blocks`, `handle_padded_last_block`, `compute_padding_info`, `compute_block_masking`, `sage_fwd`.

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py) · **Recent**

Source documentation: Calculate padding information for the last K block.

Symbols: `compute_padding_info`, `compute_block_masking`, `_sage_fwd_no_mask_mxfp4`, `_sage_fwd_mask_mxfp4`, `_sage_fwd_blocksparse_nomask_mxfp4`, `_sage_fwd_blocksparse_mask_mxfp4`, `sage_fwd_mxfp4`.

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/mha.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/mha.py) · **Recent**

Symbols: `_cdiv_fn`, `_load_fn`, `_compute_alibi_block`, `_attn_fwd_inner`, `_attn_fwd`, `_get_config`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_PE`, `BLOCK_DMODEL_POW2`, `BLOCK_M`, `BLOCK_N`, `num_stages`.

### `aiter/ops/triton/_triton_kernels/attention/na3d_flash.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/na3d_flash.py) · **Recent**

Source documentation: Triton flash-style 3D neighborhood attention. Each query token attends to the (KT x KH x KW) neighborhood centered on its (t, h, w) grid position, with inward border-shift so every window has the full KT x KH x KW keys. Algorithm: - Grid dim[0] = T × H × ⌈W / BLOCK_Q⌉; each program covers one (t, h) row. - Q is loaded once into registers and held for all KT x KH inner iterations. - Each iteration loads one (t_kv, h_kv) row of K and V at the shared W window.

Symbols: `_get_config`, `_na3d_flash_fwd`.

Parameters referenced in this file: `BLOCK_KV`, `BLOCK_Q`, `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/attention/pa_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_prefill.py) · **Recent**

Symbols: `_fwd_kernel`, `_fwd_kernel_alibi`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_PADDED`, `BLOCK_M`, `BLOCK_MODEL`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/pod_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pod_attention.py) · **Recent**

Symbols: `read_realtime`, `get_cu_id`, `pod_persistent`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/prefill_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/prefill_attention.py) · **Recent**

Source documentation: Memory-efficient attention for prefill. It supporst page size = 1.

Symbols: `_fwd_kernel`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/common/mha_kernel_utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/mha_kernel_utils.py) · **Recent**

Symbols: `_compute_fp8_scaling_factors`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py) · **Recent**

Source documentation: Triton kernel helper functions shared across flash attention modules. This module contains Triton JIT-compiled helper functions that are used within the main attention kernels (fwd_prefill, fwd_decode, bwd). These are kept separate from utils.py to allow stricter type checking on pure Python utilities.

Symbols: `compute_alibi_block`, `compute_fp8_scaling_factors`, `_cast_varlen_to_fp8_kernel_2d`, `_rotary_kernel`, `cast_to_fp8`, `_apply_rotary_kernel`, `_ApplyRotary`, `apply_rotary_emb`, `apply_rotary`.

Parameters referenced in this file: `BLOCK_H`, `BLOCK_K`, `BLOCK_M`, `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py) · **Recent**

Source documentation: Allowed query/key pairs; negative window sizes mean unbounded.

Symbols: `get_fwd_prefill_configs`, `attention_mask`, `_attn_fwd_inner`, `compute_window_bounds`, `classify_window_blocks`, `handle_padded_last_block`, `compute_padding_info`, `compute_block_masking`, `attn_fwd`, `attention_forward_prefill_triton_impl`.

Parameters referenced in this file: `BLOCK_DMODEL_QK`, `BLOCK_DMODEL_V`, `BLOCK_M`, `BLOCK_M_OPTIONS`, `BLOCK_N`, `BLOCK_N_OPTIONS`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py) · **Recent**

Symbols: `fwd`, `varlen_fwd`, `fwd_kvcache`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py) · **Recent**

Source documentation: Flash Attention v3 forward pass compatible interface for AMD Triton implementation. This function maps v3 parameters to the existing AMD Triton implementation.

Symbols: `fwd`, `fwd_combine`, `get_scheduler_metadata`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py) · **Recent**

Source documentation: Utilities for Flash Attention Triton AMD backend. This module contains essential runtime utilities: - GPU architecture detection - Global configuration flags - Tensor shape/stride helpers - FP8 type detection

Symbols: `GpuArch`, `is_fp8`, `get_shape_from_layout`, `get_stride_from_layout`, `get_padded_headsize`, `round_multiple`, `is_hip`, `get_arch`, `remap_xcd`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/fusions/attn_res.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/attn_res.py) · **Recent**

Source documentation: AttnRes forward, ported from fla 0.5.2 attnres_fwd_kernel. Per token: RMS-normalize each of the L residual candidates, score each with a dot product against q * w, softmax over the candidate axis, and return the weighted sum (optionally output-RMSNorm'd). One program per row; the whole D stays in registers (BD = next_pow2(D)) and the candidate axis is streamed in BL tiles with an online softmax, so each candidate tile is read once.

Symbols: `attnres_fwd_kernel`.

Parameters referenced in this file: `num_stages`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py) · **Recent**

Source documentation: Pack per-channel fp4 (E2M1) V into the f4f4 kernel's col-major LDS operand layout: per 128-kv tile, 8 blocks of 1024 B (u = 2*n + k; n = head-dim 32-block 0..3, k = kv 64-half 0..1); each block is 64 kv-cols x 16 nibble-bytes (even chan -> low nibble, odd chan -> high nibble). One program packs one 1024 B block. Cosine-equivalent to the numpy packer; only exact E2M1 tie-midpoints may differ by one code (arbitrary, cosine-neutral).

Symbols: `mha_v4_per_tensor_amax_kernel`, `mha_v4_per_tensor_scale_kernel`, `mha_v4_per_tensor_quant_kernel`, `_compute_mx_quant_and_scale_rne`, `sage_quant_v_kernel`, `sage_quant_v_amax_partial_kernel`, `sage_quant_v_amax_finalize_kernel`, `_e2m1_code`, `_mxfp4_scale_from_amax`, `_e2m1_pack_native`, `sage_quant_v_fp4_colmajor_kernel`, `sage_quant_v_mxfp4_colmajor_kernel`, `_rotate_quantize_q_kernel`, `_rotate_quantize_k_kernel`, `_rotate_quantize_qk_kernel`, `_rot_q_kernel`, `_rot_k_only_kernel`, `_compute_delta_s_kernel`, `_q_smooth_int8_kernel`, `sage_quant_kernel`, `_general_quant_kernel`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_R`, `BLOCK_SIZE`, `BLOCK_SIZE_OUT_DIM`, `BLOCK_SIZE_QUANT_DIM`, `BLOCK_SIZE_QUANT_MX_SCALE`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1101/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1101/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/attention/chunk_delta_attn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/attention/chunk_delta_attn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/attention/extend_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/attention/extend_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/chunk_delta_attn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/chunk_delta_attn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/extend_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/extend_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/fav3_sage/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/fav3_sage/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/na3d_flash/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/na3d_flash/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/chunk_delta_attn/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/chunk_delta_attn/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/extend_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/extend_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage_mxfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/fav3_sage_mxfp4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/mha/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/mha/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/na3d_flash/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/na3d_flash/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate_close_block/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_gate_separate_close_block/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_sequence/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/attn_res_sequence/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_attn_res.py)
- [op_tests/op_benchmarks/triton/bench_deepgemm_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_deepgemm_attention.py)
- [op_tests/op_benchmarks/triton/bench_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_extend_attention.py)
- [op_tests/op_benchmarks/triton/bench_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mha.py)
- [op_tests/op_benchmarks/triton/bench_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_prefill.py)
- [op_tests/op_benchmarks/triton/compare_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/compare_mha.py)
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
- [op_tests/cpp/mha/compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/mha/compile.py)
- [op_tests/op_benchmarks/flydsl/bench_mha_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_fwd.py)
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

## Performance and validation

Record sequence lengths, masks, heads and dtypes. Use the matching forward reference with dropout disabled; do not compare kernel latency with an end-to-end serving metric.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
