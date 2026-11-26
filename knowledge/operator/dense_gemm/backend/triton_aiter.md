---
title: "Dense matrix multiplication: AITER triton"
kind: implementation_knowledge
operator: dense_gemm
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Dense matrix multiplication — AITER triton

Separate the arithmetic kernel from weight preparation and dispatch. Compare tiles at the actual M/N/K, strides, output type and weight layout. Small M often needs a different grid or split-K policy from prefill.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/gemm/basic/gemm_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/basic/gemm_a16w16.py) · **Recent**

Source documentation: Computes 16 bit matrix multiplication Y = X @ W^T Uses the gluon backend automatically on supported architectures (gfx1250) and the triton backend everywhere else. Pass backend to force a choice. Args: x (torch.Tensor): Input matrix with shape (M, K). w (torch.Tensor): Weight matrix with shape (N, K), internally transposed. bias (Optional[torch.Tensor]): Bias vector with shape (N,). dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).

Symbols: `_is_gluon_available`, `gemm_a16w16_fake_tensor`, `gemm_a16w16_`, `gemm_a16w16`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/gemm/basic/gemm_a16w16_atomic.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/basic/gemm_a16w16_atomic.py) · **Recent**

Source documentation: Computes 16 bit matrix multiplication Y = X @ W^T using atomic operations for split-K reduction. Args: x (torch.Tensor): BF16/FP16 input matrix matrix with shape (M, K). w (torch.Tensor): BF16/FP16 weight matrix with shape (N, K), internally transposed. dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16). Note: BF16 atomic aggregation may have slight precision loss. y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).

Symbols: `gemm_a16w16_atomic_fake_tensor`, `gemm_a16w16_atomic_`, `gemm_a16w16_atomic`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/gemm/basic/gemm_a16w16_gated.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gemm/basic/gemm_a16w16_gated.py) · **Recent**

Source documentation: Computes 16 bit gated matrix multiplication Y = X @ W^T with gating mechanism (e.g., SwiGLU). Uses first half of W output as gate for second half, producing (M, N//2) output. Args: x (torch.Tensor): Input matrix with shape (M, K). w (torch.Tensor): Weight matrix with shape (N, K), internally transposed. N must be even. dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16). y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N//2).

Symbols: `gemm_a16w16_gated`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`.

### `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py) · **Recent**

Symbols: `_gemm_a16_w16_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py) · **Recent**

Symbols: `_gemm_a16_w16_atomic_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_gated.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_gated.py) · **Recent**

Symbols: `_gemm_a16_w16_gated_kernel`, `_get_config`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_stages`.

### `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_persistent.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_persistent.py) · **Recent**

Source documentation: Kernel for computing the matmul C = A x B. A has shape (M, K), B has shape (K, N) and C has shape (M, N)

Symbols: `gemm_a16w16_persistent_kernel_`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=34816-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=34816-K=5120.json)
- [aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=17408.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=17408.json)
- [aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=5120.json)
- [aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=5120.json)
- [aiter/ops/triton/configs/gfx1150/triton/gemm/gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/gemm/gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=10248-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=10248-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=3072.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=3072.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=32320-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=32320-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=36864-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=36864-K=6144.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=384-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=384-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=3072.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=3072.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=55296-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=55296-K=6144.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=18432.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=18432.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=24576.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=24576.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=6144.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=16384.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=16384.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_atomic/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_atomic/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_gated/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_gated/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=1024-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=1024-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=5120-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=5120-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/triton/gemm/grouped/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/gemm/grouped/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=10240-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=10240-K=8192.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=11008-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=11008-K=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=12288-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=12288-K=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=13824-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=13824-K=5120.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=15360-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=15360-K=5120.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=22016-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=22016-K=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=24576-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=24576-K=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=27648-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=27648-K=5120.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=28672-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=28672-K=8192.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=11008.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=11008.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=12288.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=12288.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=22016.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=22016.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=24576.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=24576.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=6144.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=13824.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=13824.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=15360.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=15360.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=27648.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=27648.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=5120.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=57344-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=57344-K=8192.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=4096.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=10240.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=10240.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=28672.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=28672.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=57344.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=57344.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=8192.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16_atomic/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16_atomic/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16_gated/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16_gated/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16_persistent/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16_persistent/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/gemm/grouped/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/gemm/grouped/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=10240-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=10240-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=2880.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=5120.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1280-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1280-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=14336-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=14336-K=5120.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1536-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1536-K=128.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=16384-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=16384-K=5120.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=2048.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=2048.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2816-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2816-K=6144.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=288-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=288-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=512.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=3328-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=3328-K=6144.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=34816-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=34816-K=5120.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=384-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=384-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=6656.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4096-K=6656.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4608-K=6656.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=4608-K=6656.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=50016-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=50016-K=6144.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=17408.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=17408.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=2880.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=6144.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=57344-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=57344-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=640-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=640-K=2880.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6656-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=6656-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=28672.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=28672.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=8192-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=96-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/GEMM-A16W16-N=96-K=5120.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/GEMM-A16W16-ATOMIC-N=256-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/GEMM-A16W16-ATOMIC-N=256-K=6144.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/GEMM-A16W16-ATOMIC-N=256-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/GEMM-A16W16-ATOMIC-N=256-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/GEMM-A16W16-ATOMIC-N=896-K=3584.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_atomic/GEMM-A16W16-ATOMIC-N=896-K=3584.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_gated/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_gated/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=10240-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=10240-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=2880.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=5120.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=1280-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=1280-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2048-K=2048.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2048-K=2048.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=256-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=256-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=512.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=384-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=384-K=7168.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=4096-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=4096-K=4096.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=5120-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=5120-K=2880.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=57344-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=57344-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=640-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=640-K=2880.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=8192-K=28672.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=8192-K=28672.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=8192-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=8192-K=8192.json)
- [aiter/ops/triton/configs/gfx950/triton/gemm/grouped/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/gemm/grouped/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py)
- [op_tests/triton_tests/gemm/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/__init__.py)
- [op_tests/triton_tests/gemm/basic/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/__init__.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py)
- [op_tests/triton_tests/gemm/batched/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/__init__.py)
- [op_tests/triton_tests/gemm/fused/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/__init__.py)
- [op_tests/triton_tests/gemm/grouped/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/__init__.py)
- [op_tests/triton_tests/gemm/grouped/test_gmm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/test_gmm.py)
- [op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py)
- [op_tests/flydsl_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_gemm_a16w16.py)
- [op_tests/test_deepgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_deepgemm.py)
- [op_tests/test_f4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_f4gemm.py)
- [op_tests/test_flydsl_hgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_hgemm.py)
- [op_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w16.py)
- [op_tests/test_gemm_codegen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_codegen.py)
- [op_tests/test_gemm_group32_interface.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_group32_interface.py)
- [op_tests/test_opus_a16w16_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_opus_a16w16_gemm.py)

## Performance and validation

Measure packing and split-K reduction separately, then include both in the deployed operator cost. Keep accumulation and epilogue semantics unchanged.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
