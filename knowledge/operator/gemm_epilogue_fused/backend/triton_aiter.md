---
title: "Gemm epilogue fused: AITER triton"
kind: implementation_knowledge
operator: gemm_epilogue_fused
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Gemm epilogue fused — AITER triton

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/gemm/feed_forward/ff_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/feed_forward/ff_a16w16.py) · **Recent**

Source documentation: Full feed-forward block with gating (e.g swiglu). x: torch.Tensor (M, K) w_up: torch.Tensor (N, K) w_down: torch.Tensor (N, K) intermediate: torch.Tensor (M, N) y: torch.Tensor (M, K) activation: One of ("silu", "relu", "gelu", "gelu_tanh", "silu_exp2", None)

Symbols: `ff_a16w16_nogate`, `ff_a16w16_gated`.

### `aiter/ops/triton/gemm/feed_forward/ff_a16w16_fused_gated.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/feed_forward/ff_a16w16_fused_gated.py) · **Recent**

Source documentation: Computes a full feed-forward operation with a gated activation (e.g FF with SwiGLU) Uses the first half of the output (along the N dim) as a gate for the second half. Key parameters: - X: Matrix X with shape (M, K). - w_up: Up-projection W with shape (N, K). - w_down: Down-projection W with shape (N//2, K). - dtype: Optional parameter to specify bf16 or fp16 datatype. Default is bf16 - Y: Output Matrix Y with shape (M, K).

Symbols: `ff_a16w16_fused_gated`.

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/feed_forward/ff_a16w16_fused_ungated.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/feed_forward/ff_a16w16_fused_ungated.py) · **Recent**

Source documentation: Computes a full feed-forward operation with activation (e.g FF with Relu) Key parameters: - X: Matrix X with shape (M, K). - w_up: Up-projection W with shape (N, K). - w_down: Down-projection W with shape (N, K). - dtype: Optional parameter to specify bf16 or fp16 datatype. Default is bf16 - Y: Output Matrix Y with shape (M, K). If this is none, then it's created by this API and returned as output. - activation: Optional activation function to apply.

Symbols: `ff_a16w16_fused_ungated`.

Parameters referenced in this file: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/fused/fused_gemm_a16w16_quant_x.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/fused/fused_gemm_a16w16_quant_x.py) · **Recent**

Source documentation: Computes 16-bit matmul Y = X @ W^T and also emits an MXFP8-quantized X. This fuses the GEMM with the activation per-1x32 MXFP8 quantization that immediately follows the router-gate GEMM in MoE flows (e.g. DSv4), avoiding a separate kernel/DRAM pass to cast X from BF16 to FP8 + e8m0 scales.

Symbols: `fused_gemm_a16w16_quant_x`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py) · **Recent**

Source documentation: Computes the 8 bit matmul Y = X x WT + B using the block-scale quantization approach for x_fp8 and w_fp8. Computes the 16 bit matmul Y = X x WT + B for x_bf16 and w_bf16 This fusion is primarily aiming for fusing the gate up-projections and MOE gating: gate up-projections: (M, K) x (2N, K) = (M, 2N) MOE gating: (M, K) x (N, K) + (N, ) = (M, N) Key parameters: - x_fp8: Matrix X with shape (M, K). - w_fp8: Matrix W with shape (N_fp8, K).

Symbols: `fused_gemm_a8w8_blockscale_a16w16`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py) · **Recent**

Source documentation: Computes matrix multiplication Y = X @ W^T with FP8 activations and FP8 weights. if fuse_type == 0: the final output = a * Y + b elif fuse_type == 1 the final output = a * b + Y Args: x (torch.Tensor): FP8 input matrix with shape (M, K). w (torch.Tensor): FP8 weight matrix with shape (N, K), internally transposed. x_scale (torch.Tensor): Block-wise scale for x with shape (M, scale_k). scale_k = ceil(K / scale_block_size_k).

Symbols: `set_use_gemm_splitk_bf16`, `get_splitk`, `fused_gemm_a8w8_blockscale_mul_add`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_K_STEP`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `split_k`.

### `aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py) · **Recent**

Source documentation: Computes the 8 bit matmul C = X @ W^T using the block-scale quantization approach. Then split the product C into C1 and C2 with sizes S1 and S2 at the last dimension respectively. Finally concatenate Y to C1 at the last dimension. Equivalent to the following sequence: c = (x @ w).view(-1, y.shape(1), S1 + S2) c1, c2 = c.split([S1, S2], dim=-1) c1 = c1.cat(y, dim=-1) return c1, c2 Key parameters: - x: Matrix X with shape (M, K). - w: Matrix W with shape (N, K).

Symbols: `fused_gemm_a8w8_blockscale_split_cat`, `fused_gemm_a8w8_blockscale_preshuffle_split_cat`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_S3`, `NUM_KSPLIT`, `split_k`.

### `aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_a16w16.py) · **Recent**

Source documentation: Computes the 8 bit matmul Y = X x WT + B using the block-scale quantization approach for x_fp4 and w_fp4. Computes the 16 bit matmul Y = X x WT + B for x_bf16 and w_bf16 This fusion is primarily aiming for fusing the gate up-projections and MOE gating: gate up-projections: (M, K) x (2N, K) = (M, 2N) MOE gating: (M, K) x (N, K) + (N, ) = (M, N) Key parameters: - x_fp4: Matrix X with shape (M, K). - w_fp4: Matrix W with shape (N_fp4, K).

Symbols: `fused_gemm_afp4wfp4_a16w16`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_mul_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_mul_add.py) · **Recent**

Source documentation: Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights. if fuse_type == 0: the final output = a * Y + b elif fuse_type == 1 the final output = a * b + Y Args: x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K). w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K), internally transposed. x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M, K//32). One scale per 32 elements in K dimension.

Symbols: `set_use_gemm_splitk_bf16`, `get_splitk`, `fused_gemm_afp4wfp4_mul_add`, `fused_gemm_afp4wfp4_preshuffle_add_mul`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_K_STEP`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py) · **Recent**

Source documentation: Computes the 4 bit matmul C = X @ W^T using the block-scale quantization approach. Then split the product C into C1 and C2 with sizes S1 and S2 at the last dimension respectively. Finally concatenate Y to C1 at the last dimension. Equivalent to the following sequence: c = (x @ w).view(-1, y.shape(1), S1 + S2) c1, c2 = c.split([S1, S2], dim=-1) c1 = c1.cat(y, dim=-1) return c1, c2 Key parameters: - x: Matrix X with shape (M, K // 2). - w: Matrix W with shape (N, K // 2).

Symbols: `fused_gemm_afp4wfp4_split_cat`, `fused_gemm_afp4wfp4_preshuffle_split_cat`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_S3`, `NUM_KSPLIT`, `split_k`.

### `aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_gated.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_gated.py) · **Recent**

Symbols: `_ff_a16w16_fused_gated`, `_get_config`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_ungated.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_ungated.py) · **Recent**

Symbols: `_ff_a16w16_fused_ungated`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a16w16_quant_x.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a16w16_quant_x.py) · **Recent**

Symbols: `_fused_gemm_a16w16_quant_x_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py) · **Recent**

Symbols: `_fused_gemm_a8w8_blockscale_a16w16_kernel`, `_fused_gemm_a8w8_blockscale_a16w16_reduce_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py) · **Recent**

Symbols: `_fused_gemm_a8w8_blockscale_mul_add_kernel`, `_fused_gemm_a8w8_blockscale_mul_add_reduce_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py) · **Recent**

Symbols: `_fused_gemm_a8w8_blockscale_split_cat`, `_fused_gemm_a8w8_blockscale_preshuffle_split_cat`, `_fused_gemm_a8w8_blockscale_split_cat_reduce`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_S3`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_a16w16.py) · **Recent**

Symbols: `_fused_gemm_afp4wfp4_a16w16_kernel`, `_fused_gemm_afp4wfp4_preshuffle_a16w16_kernel`, `_fused_gemm_afp4wfp4_a16w16_reduce_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_mul_add.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_mul_add.py) · **Recent**

Symbols: `_fused_gemm_afp4wfp4_mul_add_kernel`, `_fused_gemm_afp4wfp4_preshuffle_mul_add_kernel`, `_fused_gemm_afp4wfp4_mul_add_reduce_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_split_cat.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_split_cat.py) · **Recent**

Symbols: `_fused_gemm_afp4wfp4_split_cat`, `_fused_gemm_afp4wfp4_preshuffle_split_cat`, `_fused_gemm_afp4wfp4_split_cat_reduce`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_S3`, `NUM_KSPLIT`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/triton/gemm/ff_a16w16_fused/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/ff_a16w16_fused/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/fused_gemm_afp4wfp4_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/fused_gemm_afp4wfp4_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/fused_gemm_afp4wfp4_preshuffled_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/fused_gemm_afp4wfp4_preshuffled_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/ff_a16w16_fused/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/ff_a16w16_fused/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=11264-K=2048.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=11264-K=2048.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=16384-K=1280.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=16384-K=1280.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=16384-K=2048.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=16384-K=2048.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=16384-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=16384-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=22016-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=22016-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=27648-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=27648-K=5120.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=28672-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/ff_a16w16_fused/FF-A16W16-fused-N=28672-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_a16w16/FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_mul_add/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_mul_add/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_mul_add/FUSED-GEMM-A8W8_BLOCKSCALE-MUL_ADD-N=7168-K=256.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_a8w8_blockscale_mul_add/FUSED-GEMM-A8W8_BLOCKSCALE-MUL_ADD-N=7168-K=256.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_a16w16/FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_a16w16/FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_mul_add/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_mul_add/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_mul_add/FUSED-GEMM-AFP4WFP4-MUL_ADD-N=7168-K=256.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_mul_add/FUSED-GEMM-AFP4WFP4-MUL_ADD-N=7168-K=256.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_preshuffled_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/fused_gemm_afp4wfp4_preshuffled_a16w16/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_ff_a16w16_fused.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_ff_a16w16_fused.py)
- [op_tests/op_benchmarks/triton/bench_fused_gemm_a8w8_blockscale_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gemm_a8w8_blockscale_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_fused_gemm_a8w8_blockscale_mul_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gemm_a8w8_blockscale_mul_add.py)
- [op_tests/op_benchmarks/triton/bench_fused_gemm_afp4wfp4_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gemm_afp4wfp4_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_fused_gemm_afp4wfp4_mul_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gemm_afp4wfp4_mul_add.py)
- [op_tests/op_benchmarks/triton/bench_fused_gemm_commons.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gemm_commons.py)
- [op_tests/triton_tests/gemm/feed_forward/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/feed_forward/__init__.py)
- [op_tests/triton_tests/gemm/feed_forward/ff_test_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/feed_forward/ff_test_utils.py)
- [op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py)
- [op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py)
- [op_tests/triton_tests/gemm/fused/test_fused_gemm_a16w16_quant_x.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/test_fused_gemm_a16w16_quant_x.py)
- [op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py)
- [op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py)
- [op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_split_cat.py)
- [op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py)
- [op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py)
- [op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_split_cat.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_split_cat.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
