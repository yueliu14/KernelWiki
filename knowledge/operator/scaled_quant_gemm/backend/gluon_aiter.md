---
title: "Scaled and mixed-precision GEMM: AITER gluon"
kind: implementation_knowledge
operator: scaled_quant_gemm
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Scaled and mixed-precision GEMM — AITER gluon

The quantization format, scale granularity and scale layout are part of the operator contract. Follow the wrapper from logical K to packed storage K and from source weights to the kernel-specific preshuffle.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_a8w8_blockscale.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_a8w8_blockscale.py) · **Recent**

Source documentation: Unshuffle preshuffled weight tile in shared memory. Host shuffle: (N//16, 16, K//32, 2, 16) -> permute(0, 2, 3, 1, 4) -> stored as (N//16, K*16) Inverse: (N//16, K//32, 2, 16, 16) -> permute(0, 3, 1, 2, 4) -> (N, K) then transpose to (K, N)

Symbols: `_gemm_a8w8_blockscale_bandwidth_bound_kernel`, `_gemm_a8w8_blockscale_compute_bound_kernel`, `depreshuffle_b`, `_gemm_a8w8_blockscale_preshuffle_bandwidth_bound_kernel`, `_gemm_a8w8_blockscale_preshuffle_compute_bound_kernel`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_mxfp4.py) · **Recent**

Symbols: `cluster_shape`, `get_gemm_afp4wfp4_preshuffle_layouts`, `_build_gemm_afp4wfp4_preshuffle_layouts`, `depreshuffle_scales`, `depreshuffle_b_raw_to_kn`, `_issue_tdm_stage`, `gemm_mxfp4_preshuffle_gfx1250`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_K_BYTES`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `tile_m`, `tile_n`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_mxfp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_mxfp8.py) · **Recent**

Source documentation: Pure-python make_cga_layout([m, n], [m, n], [0, 1]). Deliberately not a call into libtriton's make_cga_layout: gluon's dependency walker inspects the globals of every function a kernel body references and rejects nanobind builtins outright ("Unsupported function referenced"), so even a constexpr_function wrapper around it cannot be used from inside a kernel. Equivalence is asserted in the op test.

Symbols: `cga_bases`, `cga_drop_k`, `cga_swap`, `cga_bcast`, `cga_flat`, `make_wmma_layout`, `depreshuffle_b`, `_load_scale_tile`, `_gather_scale_tile`, `_gather_scale_vec`, `_read_tile`, `_gemm_mxfp8_preshuffle_bandwidth_bound_kernel`, `_gemm_mxfp8_preshuffle_compute_bound_kernel`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a8w8.py) · **Recent**

Source documentation: Gluon INT8/FP8 GEMM kernels for gfx950.

Symbols: `_gemm_a8w8_kernel`, `_gemm_a8w8_preshuffled_kernel`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_stages`.

### `aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a8w8_blockscale.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a8w8_blockscale.py) · **Recent**

Source documentation: Gluon FP8 block-scale GEMM kernels for gfx950.

Symbols: `_is_supported_tile`, `_linear_a_reg`, `_linear_a_lane`, `_linear_a_warp`, `_linear_b_reg`, `_linear_b_lane`, `_linear_b_warp`, `_shared_a_bases`, `_shared_b_bases`, `_prefetch_tensors`, `_load_shared`, `_prefetch_scales`, `_compute_MN_tile`, `_gemm_a8w8_blockscale_kernel`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`.

### `aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_afp4wfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_afp4wfp4.py) · **Recent**

Source documentation: Gluon MXFP4 GEMM kernel for gfx950.

Symbols: `_gemm_afp4wfp4_kernel`.

Parameters referenced in this file: `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=1536-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=1536-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=1536-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=1536-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=16384-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=16384-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2048-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2048-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=32768-K=1024.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=32768-K=1024.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=2048.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=2048.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=8192.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=8192.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=6144-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=6144-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=65536-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=65536-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=16384.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=16384.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=3072.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=3072.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=768.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=768.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=8192-K=1024.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=8192-K=1024.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=8192-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale_preshuffled/GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=8192-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/GEMM-AFP4WFP4_PRESHUFFLED-N=24576-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/GEMM-AFP4WFP4_PRESHUFFLED-N=24576-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/GEMM-AFP4WFP4_PRESHUFFLED-N=7168-K=16384.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp4wfp4_preshuffled/GEMM-AFP4WFP4_PRESHUFFLED-N=7168-K=16384.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=1536-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=1536-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=16384-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=16384-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=2048-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=2048-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=6144-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=6144-K=7168.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=65536-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=65536-K=1536.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=16384.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=16384.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=3072.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=3072.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=768.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=7168-K=768.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=8192-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_afp8wfp8_preshuffled/GEMM-AFP8WFP8_PRESHUFFLED-N=8192-K=1536.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=2112-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=2112-K=7168.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=3072-K=1536.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=3072-K=1536.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=4608-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=4608-K=7168.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=512-K=7168.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=512-K=7168.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=7168-K=2048.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=7168-K=2048.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=7168-K=256.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_a8w8_blockscale/GEMM-A8W8_BLOCKSCALE-N=7168-K=256.json)
- [aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_afp4wfp4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/gemm/gemm_afp4wfp4/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/bench_flydsl_gemm_a16wfp4.py)
- [op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/hip/bench_gemm_mixed_mxfp.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a16w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py)
- [op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py)
- [op_tests/test_gemm_a16w8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w8_asm.py)
- [op_tests/test_gemm_a4w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w4.py)
- [op_tests/test_gemm_a4w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a4w6.py)
- [op_tests/test_gemm_a6w4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w4.py)
- [op_tests/test_gemm_a6w6.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a6w6.py)
- [op_tests/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8.py)
- [op_tests/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale.py)
- [op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_cktile_aq_rowmajor.py)
- [op_tests/test_gemm_a8w8_blockscale_fallback.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_blockscale_fallback.py)
- [op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py)
- [op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a8w8_mxfp8_bpreshuffle.py)
- [op_tests/test_mxfp8fp4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mxfp8fp4gemm.py)
- [op_tests/test_quant_mxfp4_gemm_fused.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp4_gemm_fused.py)
- [op_tests/test_quant_mxfp6_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_quant_mxfp6_gemm.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/basic/test_gemm_afp8wfp8.py)
- [op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_gemm_a8w8_blockscale_cktile_instances.py)

## Performance and validation

Compare identical quantized inputs and scale tensors. Count quantization, shuffle and reduction kernels when they execute on the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](gluon.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
