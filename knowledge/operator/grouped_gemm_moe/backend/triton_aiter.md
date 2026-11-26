---
title: "Grouped GEMM and MoE expert computation: AITER triton"
kind: implementation_knowledge
operator: grouped_gemm_moe
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Grouped GEMM and MoE expert computation — AITER triton

MoE execution couples token/expert sorting with gate/up projection, activation and down projection. The row distribution per expert, top-k, padded block count and placement of routing weights affect both work and correctness.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/gemm/grouped/gmm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/grouped/gmm.py) · **Recent**

Source documentation: Perform Group Matrix Multiplication (GMM): out = lhs @ rhs + bias lhs rows are divided into G groups. Each group of lhs rows is matrix multiplied with a plane of rhs 3D tensor and then stored in a slice of out. In PyTorch parlance, it can be implemented as follows for a given group g: out[group_start:group_end, :] = lhs[group_start:group_end, :] @ rhs[g] + bias[g] The size of each group, and their respective start and end positions are specified by group_sizes tensor.

Symbols: `_get_gmm_tile_counter`, `_gmm_grid`, `gmm`, `_ptgmm_grid`, `ptgmm`, `_nptgmm_grid`, `nptgmm`.

Parameters referenced in this file: `BLOCK_SIZE_`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/moe/moe_op_gemm_a16w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_op_gemm_a16w4.py) · **Recent**

Source documentation: Computes MoE GEMM with 16-bit activations and MxFP4 weights Y[:, :] = 0. for e in num_experts: Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :]) Args: x (torch.Tensor): Activations with shape (num_tokens, K/hidden_dim). Must be 16-bit w (torch.Tensor): Weight tensor with shape(E/num_experts_total, N/int_dim, K/hidden_dim/emb_dim).

Symbols: `_warn_gluon_fallback_once`, `_is_gluon_available`, `can_overflow_int32`, `should_upcast_indices`, `allocate_output`, `get_kernel_config_triton`, `get_kernel_config_gluon_gfx950_pipelined`, `get_kernel_config_gluon_gfx950`, `get_kernel_config_gluon_gfx1250`, `moe_gemm_a16w4`, `swiglu_torch`, `moe_gemm_torch`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `split_k`, `waves_per_eu`.

### `aiter/ops/triton/moe/moe_op_gemm_a4w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_op_gemm_a4w4.py) · **Recent**

Source documentation: Returns: - A packed MXFP4 tensor x_fp4 of shape [M, N // 2] (stored as uint8), where each byte stores two 4-bit values. - A block-scale tensor x_scale of shape [M, N / 32], where each entry corresponds to one MXFP4 quantization block of 32 elements along the N dimension.

Symbols: `_prefill_lds_bytes`, `_workgroups_per_wgp`, `can_overflow_int32`, `should_upcast_indices`, `allocate_output`, `get_kernel_config_triton`, `m2bucket`, `get_kernel_config_gluon`, `mxfp4_quant`, `_gather_idx_bitwidth`, `moe_gemm_a4w4`, `swiglu_torch`, `moe_gemm_torch`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `matrix_instr_nonkdim`, `num_buffers`, `num_stages`, `num_warps`, `split_k`, `waves_per_eu`.

### `aiter/ops/triton/moe/moe_op_gemm_a8w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_op_gemm_a8w4.py) · **Recent**

Source documentation: Y[:, :] = 0. for e in num_experts: Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])

Symbols: `can_overflow_int32`, `should_upcast_indices`, `allocate_output`, `get_kernel_config_triton`, `m2bucket`, `get_gluon_a8w4_ctas_per_cga`, `get_kernel_config_gluon`, `moe_gemm_a8w4`, `swiglu_torch`, `moe_gemm_torch`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `matrix_instr_nonkdim`, `num_buffers`, `num_stages`, `num_warps`, `persistent_iters`, `split_k`, `waves_per_eu`.

### `aiter/ops/triton/moe/moe_op_gemm_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_op_gemm_a8w8.py) · **Recent**

Source documentation: Y[:, :] = 0. for e in num_experts: Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :]) x_token_scale: optional fp32 per-token activation scale, one entry per row of x; applied to the accumulator before bias / activation. w_expt_scale: optional fp32 per-expert weight scale [n_expts].

Symbols: `can_overflow_int32`, `should_upcast_indices`, `allocate_output`, `get_kernel_config`, `moe_gemm_a8w8`, `swiglu_torch`, `moe_gemm_torch`.

Parameters referenced in this file: `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `split_k`, `waves_per_eu`.

### `aiter/ops/triton/moe/moe_op_gemm_a8w8_blockscale.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_op_gemm_a8w8_blockscale.py) · **Recent**

Source documentation: Y[:, :] = 0. for e in num_experts: Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])

Symbols: `can_overflow_int32`, `should_upcast_indices`, `allocate_output`, `get_kernel_config`, `moe_gemm_a8w8_blockscale`, `swiglu_torch`, `moe_gemm_torch`.

Parameters referenced in this file: `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `split_k`, `waves_per_eu`.

### `aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py) · **Recent**

Source documentation: Preshuffle int8 weight from (E, K, N) to the MFMA-friendly tile layout (E, K*16, N//16). This is the same transpose-first per-expert (16, 16) tiling that aiter.ops.triton.utils.shuffle.shuffle_weight produces on its gfx1250 path, so the host-side shuffle stays single-sourced in aiter.ops.triton.utils.shuffle. The matching in-kernel inverse is unshuffle_weights in the int8 smoothquant kernel. Args: w: int8 weight tensor of shape (E, K, N) where K % 32 == 0 and N % 16 == 0.

Symbols: `can_overflow_int32`, `should_upcast_indices`, `preshuffle_weights`, `allocate_output`, `get_kernel_config`, `moe_gemm_int8_smoothquant`, `swiglu_torch`, `moe_gemm_smoothquant_torch`, `fused_moe_int8_smoothquant`.

Parameters referenced in this file: `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `split_k`, `waves_per_eu`.

### `aiter/ops/triton/moe/moe_situ_epilogue.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_situ_epilogue.py) · **Recent**

Source documentation: Public wrapper for a merged MoE SiTU epilogue.

Symbols: `moe_situ_epilogue`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/moe/quant_moe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/quant_moe.py) · **Recent**

Source documentation: Convert the src weights to mx format. The src weight is quantized along the axis dimension. If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte. Note that this means the k_dim of the tensor will be half of the logical k_dim. If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored in their respective formats.

Symbols: `downcast_to_static_fp8_3d`, `downcast_to_static_fp8`, `DequantScaleRoundingMode`, `downcast_to_mxfp`, `dequant_x_blockscale`, `dequant_w_blockscale`, `smoothquant_quantize`, `quantize_weights_int8`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_OUT_DIM`, `BLOCK_QUANT_DIM`, `num_warps`.

### `aiter/ops/triton/moe/reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/reduce.py) · **Recent**

Source documentation: Where a grouped GEMM's un-reduced rows go, in place of a local reduce. out is the destination the rows are placed in -- for expert-parallel combine, a strided view over the symmetric combine-staging window covering every peer's slot region, so one row index selects both the peer and the slot within it. dst_row[m] is that row index for sorted row m, negative where the row must not be delivered at all.

Symbols: `EpCombineScatter`, `scatter_grouped`, `validate_reduce_out`, `reduce_grouped`.

Parameters referenced in this file: `BLOCK_N`, `num_warps`, `split_k`.

### `aiter/ops/triton/moe/sonicmoe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/sonicmoe.py) · **Recent**

Source documentation: Run grouped GEMM, optionally with 1x128 activation and 128x128 weight scales.

Symbols: `_local_tensor`, `_use_qwen3_tuned_configs`, `grouped_gemm`, `_grouped_gemm_triton`, `token_gather_and_sum_varlen_K_triton`, `_launch_config`, `activation_fwd`, `TC_topk_router_metadata_triton`, `general_routing_router_metadata_triton`, `_router_forward`, `_topk_softmax_fwd`, `ActivationType`, `is_glu`, `TC_Softmax_Topk_Router_Function`, `_UpProjection`, `_DownProjection`, `moe_TC_softmax_topk_layer`, `moe_general_routing_inputs`, `moe_pre_routed_inputs`.

Parameters referenced in this file: `BLOCK_H`, `BLOCK_H_MAX`, `BLOCK_I`, `BLOCK_I_MAX`, `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/common/moe_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/common/moe_common.py) · **Recent**

Symbols: `_write_zeros_to_output`.

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/gemm/grouped/gmm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/grouped/gmm.py) · **Recent**

Symbols: `get_config`, `_remap_xcd_tile_grid`, `_total_gmm_tiles`, `_resolve_gmm_tile`, `_process_gmm_tile`, `_gmm`, `_work_stealing_gmm`, `gmm_kernel`, `tgmm_persistent_kernel`, `tgmm_non_persistent_kernel`.

Parameters referenced in this file: `BLOCK_SIZE_G`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/triton/_triton_kernels/moe/activations.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/activations.py) · **Recent**

Symbols: `clip`, `_swiglu`, `silu`, `silu_grad`, `gelu_tanh`, `gelu_tanh_grad`, `relu`, `relu_grad`, `relu_sq`, `relu_sq_grad`, `_glu_fwd_kernel`, `_pointwise_act_fwd_kernel`.

Parameters referenced in this file: `BLOCK_I`, `BLOCK_M`.

### `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py) · **Recent**

Source documentation: Swizzle the program id based on integer XCD_SWIZZLE. This is useful for reording how blocks are ordered. A scheduler may, for example, assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2. This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to the same hardware unit.

Symbols: `matmul_launch_metadata`, `xcd_swizzle`, `unswizzle_mx_scale_cdna4`, `_moe_gemm_a16w4`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py) · **Recent**

Source documentation: Swizzle the program id based on integer XCD_SWIZZLE. This is useful for reording how blocks are ordered. A scheduler may, for example, assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2. This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to the same hardware unit.

Symbols: `matmul_launch_metadata`, `xcd_swizzle`, `unswizzle_mx_scale_cdna4`, `unswizzle_mx_scale_gfx1250`, `_moe_gemm_a4w4`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py) · **Recent**

Source documentation: Swizzle the program id based on integer XCD_SWIZZLE. This is useful for reording how blocks are ordered. A scheduler may, for example, assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2. This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to the same hardware unit.

Symbols: `matmul_launch_metadata`, `xcd_swizzle`, `unswizzle_mx_scale_cdna4`, `unswizzle_mx_scale_gfx1250`, `_moe_gemm_a8w4`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py) · **Recent**

Source documentation: Swizzle the program id based on integer XCD_SWIZZLE. This is useful for reording how blocks are ordered. A scheduler may, for example, assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2. This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to the same hardware unit.

Symbols: `matmul_launch_metadata`, `xcd_swizzle`, `unswizzle_mx_scale_cdna4`, `_moe_gemm_a8w8`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py) · **Recent**

Source documentation: Swizzle the program id based on integer XCD_SWIZZLE. This is useful for reording how blocks are ordered. A scheduler may, for example, assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2. This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to the same hardware unit.

Symbols: `matmul_launch_metadata`, `xcd_swizzle`, `_moe_gemm_a8w8_blockscale`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py) · **Recent**

Symbols: `matmul_launch_metadata`, `unshuffle_weights`, `_moe_gemm_int8_smoothquant`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_situ_epilogue.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_situ_epilogue.py) · **Recent**

Source documentation: Triton kernel for a merged MoE SiTU epilogue.

Symbols: `_moe_situ_epilogue_kernel`.

### `aiter/ops/triton/_triton_kernels/moe/quant_moe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/quant_moe.py) · **Recent**

Symbols: `_compute_static_fp8_quant`, `_get_max_quant_val`, `_compute_mx_quant_and_scale`, `_downcast_to_mxfp`, `_smoothquant_fuse_quant_kernel`, `_smoothquant_fuse_quant_kernel_single_pass`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_SIZE_OUT_DIM`, `BLOCK_SIZE_QUANT_DIM`, `BLOCK_SIZE_QUANT_MX_SCALE`, `BLOCK_SIZE_QUANT_MX_TENSOR`.

### `aiter/ops/triton/_triton_kernels/moe/reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/reduce.py) · **Recent**

Symbols: `_scatter_grouped`, `_reduce_grouped`.

Parameters referenced in this file: `BLOCK_N`, `BLOCK_N_OUT`, `split_k`.

### `aiter/ops/triton/_triton_kernels/moe/sonicmoe/grouped_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/sonicmoe/grouped_gemm.py) · **Recent**

Symbols: `_grouped_gemm_kernel`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_T`.

### `aiter/ops/triton/_triton_kernels/moe/sonicmoe/token_gather.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/sonicmoe/token_gather.py) · **Recent**

Symbols: `token_gather_sum_kernel`.

Parameters referenced in this file: `BLOCK_H`, `BLOCK_K`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/triton/moe/a8w4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/moe/a8w4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/moe/a16w4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/moe/a16w4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/moe/mxfp8_fnuz/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/moe/mxfp8_fnuz/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/moe/sonicmoe_bf16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/moe/sonicmoe_bf16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/moe/a8w4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/moe/a8w4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/moe/mxfp8_fnuz/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/moe/mxfp8_fnuz/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

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
- [op_tests/flydsl_tests/test_flydsl_grouped_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py)
- [op_tests/flydsl_tests/test_flydsl_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_moe.py)
- [op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py)
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

## Performance and validation

Benchmark representative expert-load distributions, not only a uniform dense matrix. Include sorting, zeroing, scatter and reduction when evaluating the complete operator.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
