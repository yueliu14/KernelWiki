---
title: "Batched matrix multiplication: AITER triton"
kind: implementation_knowledge
operator: batched_gemm
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Batched matrix multiplication — AITER triton

Batch strides and batch size affect dispatch as well as M/N/K. A batch axis can expose parallelism that a single small GEMM lacks. Quantized variants add per-batch, per-token or block scale layouts.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py) · **Recent**

Source documentation: Fused FP4 BMM + RoPE + concat + KV-cache write for MLA. This function fuses two INDEPENDENT operations to reduce kernel launch overhead: 1. Batched FP4 GEMM: ql_nope = q_nope @ w_k^T (with FP4 quantization) - Writes to q_out[:, :, :kv_lora_rank] 2.

Symbols: `set_use_gemm_splitk_bf16`, `fused_fp4_bmm_rope_cat_and_cache_mla`, `fused_fp8_bmm_rope_cat_and_cache_mla`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/gemm/batched/batched_gemm_a16w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/batched/batched_gemm_a16w8.py) · **Recent**

Source documentation: Whether batched_gemm_a16w8 can take (B, M, K) X and (B, N, K) WQ as given.

Symbols: `is_batched_gemm_a16w8_supported`, `batched_gemm_a16w8`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py) · **Recent**

Source documentation: Computes batched matrix multiplication Y[i] = X[i] @ W[i]^T with BF16 activations and FP4 weights. Args: x (torch.Tensor): BF16/FP16 input matrix with shape (B, M, K). Quantized to MXFP4 on-the-fly during GEMM. w (torch.Tensor): FP4 E2M1 weight batch with shape (B, N, K//2), internally transposed. w_scales (torch.Tensor): E8M0 per-group scale for w with shape (B, N, K//32). One scale per 32 elements in K dimension.

Symbols: `set_use_gemm_splitk_bf16`, `batched_gemm_a16wfp4_fake_tensor`, `batched_gemm_a16wfp4_`, `batched_gemm_a16wfp4`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/batched/batched_gemm_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/batched/batched_gemm_a8w8.py) · **Recent**

Source documentation: Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with per-batch scaling. Each batch element is independently scaled back to higher precision. Args: XQ (torch.Tensor): INT8 input batch with shape (B, M, K). WQ (torch.Tensor): INT8 weight batch with shape (B, N, K), internally transposed. x_scale (torch.Tensor): Scale for XQ with shape (B, M, 1). w_scale (torch.Tensor): Scale for WQ with shape (B, 1, N).

Symbols: `batched_gemm_a8w8`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `splitK`.

### `aiter/ops/triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py) · **Recent**

Source documentation: Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization. X is quantized to INT8 during computation using per-token grouped quantization. W is pre-quantized INT8 with per-batch-element scaling. Args: X (torch.Tensor): Higher precision input batch with shape (B, M, K) or (M, B, K) if transpose_bm_in=True. Quantized to INT8 on-the-fly during GEMM.

Symbols: `_is_gluon_available`, `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `splitK`.

### `aiter/ops/triton/gemm/batched/batched_gemm_afp4wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/batched/batched_gemm_afp4wfp4.py) · **Recent**

Source documentation: Computes batched FP4 matrix multiplication Y[i] = X[i] @ W[i]^T with FP4 activations and weights. Args: x (torch.Tensor): FP4 E2M1 input batch with shape (B, M, K). w (torch.Tensor): FP4 E2M1 weight batch with shape (B, N, K), internally transposed. x_scales (torch.Tensor): E8M0 per-group scale for x with shape (B, M, K//32). One scale per 32 elements in K dimension. w_scales (torch.Tensor): E8M0 per-group scale for w with shape (B, N, K//32).

Symbols: `set_use_gemm_splitk_bf16`, `batched_gemm_afp4wfp4`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/batched/batched_gemm_afp4wfp4_pre_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/batched/batched_gemm_afp4wfp4_pre_quant.py) · **Recent**

Symbols: `set_use_gemm_splitk_bf16`, `batched_gemm_afp4wfp4_pre_quant`.

### `aiter/ops/triton/gemm/batched/batched_gemm_bf16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/batched/batched_gemm_bf16.py) · **Recent**

Source documentation: Computes batched 16 bit matrix multiplication Y[i] = X[i] @ W[i]^T with optional bias. Args: XQ (torch.Tensor): Input batch with shape (B, M, K) (BF16 or FP16). WQ (torch.Tensor): Weight batch with shape (B, N, K), internally transposed. bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N). dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16). splitK (Optional[int]): Not supported. Must be None.

Symbols: `_is_gluon_available`, `batched_gemm_bf16`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_warps`, `splitK`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py) · **Recent**

Symbols: `_unit_rope`, `_fused_fp4_bmm_rope_cat_and_cache_mla_kernel`, `_fused_fp4_bmm_reduce_kernel`, `_fused_fp8_bmm_rope_cat_and_cache_mla_kernel`, `_get_fp4_config`, `_get_fp8_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a16wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a16wfp4.py) · **Recent**

Symbols: `_batched_gemm_a16wfp4_kernel`, `_batched_gemm_a16wfp4_reduce_kernel`, `get_splitk`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_K_STEP`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a8w8.py) · **Recent**

Symbols: `_batched_gemm_a8w8_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py) · **Recent**

Symbols: `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_afp4wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_afp4wfp4.py) · **Recent**

Symbols: `_batched_gemm_afp4_wfp4_kernel`, `_batched_gemm_afp4_wfp4_reduce_kernel`, `get_splitk`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_K_STEP`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_bf16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_bf16.py) · **Recent**

Symbols: `_batched_gemm_bf16_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1201/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx1201/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx1201/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=8-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=8-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16wfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a16wfp4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_afp4wfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/batched_gemm_afp4wfp4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-N=1024-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16wfp4/BATCHED_GEMM-A16WFP4-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16wfp4/BATCHED_GEMM-A16WFP4-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16wfp4/BATCHED_GEMM-A16WFP4-N=8192-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16wfp4/BATCHED_GEMM-A16WFP4-N=8192-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16wfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a16wfp4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=256-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=256-K=512.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=8192-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=8192-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_afp4wfp4/BATCHED_GEMM-AFP4WFP4-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_afp4wfp4/BATCHED_GEMM-AFP4WFP4-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_afp4wfp4/BATCHED_GEMM-AFP4WFP4-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_afp4wfp4/BATCHED_GEMM-AFP4WFP4-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_afp4wfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/batched_gemm_afp4wfp4/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py)
- [op_tests/op_benchmarks/triton/bench_fused_bmm_rope_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_bmm_rope_kv_cache.py)
- [op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a16w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a16w8.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py)
- [op_tests/test_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8.py)
- [op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py)
- [op_tests/test_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_bf16.py)
- [op_tests/test_flydsl_batched_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_batched_gemm.py)

## Performance and validation

Use the wrapper and matching reference for that variant. Report batch size, all matrix strides and any broadcasted operands.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
