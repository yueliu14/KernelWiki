---
title: "Dense matrix multiplication: AITER gluon"
kind: implementation_knowledge
operator: dense_gemm
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Dense matrix multiplication — AITER gluon

Separate the arithmetic kernel from weight preparation and dispatch. Compare tiles at the actual M/N/K, strides, output type and weight layout. Small M often needs a different grid or split-K policy from prefill.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_a16w16.py) · **Recent**

Source documentation: elem_bits sizes A's pad interval; elem_bits_b sizes B's (defaults to elem_bits).

Symbols: `_pad_interval`, `create_shared_layouts`, `create_wmma_layouts`, `_gemm_a16w16_bandwidth_bound_kernel`, `_gemm_a16w16_compute_bound_kernel`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_a16w16_persistent.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_a16w16_persistent.py) · **Recent**

Source documentation: Gluon persistent Triton GEMM.

Symbols: `gemm_a16w16_persistent_kernel_`, `gemm_a16w16_persistent_compute_bound_kernel_`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=1024-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=10248-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=10248-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=12448-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=12448-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=3072.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=128-K=3072.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=13824-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=13824-K=5120.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=16-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=16-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=1792-K=3584.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=1792-K=3584.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2048-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2112-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2112-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=24576-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=24576-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=256-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2560-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2560-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2880-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=3072-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=3072-K=128.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=3072-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=3072-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=32320-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=32320-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=3584-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=3584-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=36864-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=36864-K=6144.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=384-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=384-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=4608-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=4608-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=512-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=13824.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=13824.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=3072.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=3072.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5120-K=5120.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5184-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=5184-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=55296-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=55296-K=6144.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=18432.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=18432.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=24576.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=24576.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=512.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=6144.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=6144-K=6144.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=5120.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=64-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=16384.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=16384.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=3072.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7168-K=3072.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7552-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=7552-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=896-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=896-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=1024-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=1024-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=2880.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=5120.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=128-K=5120.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=256-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=256-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=2880-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=5120-K=2880.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16_persistent/GEMM-A16W16-PERSISTENT-N=5120-K=2880.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_gemm_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gated.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py)
- [op_tests/test_deepgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_deepgemm.py)
- [op_tests/test_f4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_f4gemm.py)
- [op_tests/test_flydsl_hgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_hgemm.py)
- [op_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w16.py)
- [op_tests/test_gemm_codegen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_codegen.py)
- [op_tests/test_gemm_group32_interface.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_group32_interface.py)
- [op_tests/test_opus_a16w16_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_opus_a16w16_gemm.py)
- [op_tests/triton_tests/gemm/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/__init__.py)
- [op_tests/triton_tests/gemm/basic/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/__init__.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py)
- [op_tests/triton_tests/gemm/batched/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/__init__.py)
- [op_tests/triton_tests/gemm/fused/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/fused/__init__.py)
- [op_tests/triton_tests/gemm/grouped/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/__init__.py)
- [op_tests/triton_tests/gemm/grouped/test_gmm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/grouped/test_gmm.py)
- [op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py)

## Performance and validation

Measure packing and split-K reduction separately, then include both in the deployed operator cost. Keep accumulation and epilogue semantics unchanged.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](gluon.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
