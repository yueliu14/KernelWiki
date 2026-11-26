---
title: "Batched matrix multiplication: AITER flydsl"
kind: implementation_knowledge
operator: batched_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Batched matrix multiplication — AITER flydsl

Batch strides and batch size affect dispatch as well as M/N/K. A batch axis can expose parallelism that a single small GEMM lacks. Quantized variants add per-batch, per-token or block scale layouts.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/batched_gemm_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/batched_gemm_a8w8.py) · **Recent**

Source documentation: FlyDSL batched mxfp8 GEMM on a (16, 16)-preshuffled weight, on whichever arch has a kernel for the operands' w_scale block. XQ [M, B, K] fp8, WQ [B, N, K] fp8 preshuffled, x_scale and w_scale e8m0 blocks, Out [M, B, N] bf16. x_scale is row-major, or with x_scale_transposed (B = 1, 1x128 blocks) column-major [K / 128, M] under the same shape, the blockscale convention.

Symbols: `_runner`, `bmm_a8w8_mxfp8_supported`, `run_bmm_a8w8_mxfp8`.

### `aiter/ops/flydsl/batched_gemm_a8w8_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/batched_gemm_a8w8_gfx1250.py) · **Recent**

Source documentation: gfx1250 strided-batched mxfp8_128 GEMM.

Symbols: `_lazy_import`, `parse_bmm_kernel_name`, `bmm_kernel_name`, `_align`, `preload_lds_bytes`, `bmm_candidates`, `pick_bmm_kernel_name`, `run_bmm_a8w8_mxfp8_128_gfx1250`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_N`, `num_buffers`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/batched_gemm_a8w8_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/batched_gemm_a8w8_gfx950.py) · **Recent**

Source documentation: gfx950 strided-batched mxfp8 GEMM with 32- or 128-wide e8m0 blocks.

Symbols: `_launch_config`, `parse_bmm_kernel_name`, `bmm_kernel_name`, `_runs`, `_variants`, `_max_split`, `pick_bmm_kernel_name`, `_x_scale_column_major`, `_split_counters`, `_scale_block`, `_require_config`, `_constexprs`, `_launch`, `compile_bmm_a8w8_mxfp8_gfx950`, `run_bmm_a8w8_mxfp8_gfx950`.

Parameters referenced in this file: `num_buffers`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/batched_gemm_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/batched_gemm_mxfp4.py) · **Recent**

Source documentation: Thin strided-batched MXFP4/MXFP6/MXFP8 preshuffle GEMM launcher: out[b] = dequant(x[b]) @ dequant(w[b]).T, per-1x32 e8m0 scales folded into a scaled 16x16x128 matrix op. gfx950 uses the wave64 MFMA path (a4w4/a8w4, fp4/fp6/fp8 A); gfx1250 uses the wave32 WMMA path (a8w4 only: MXFP8 E4M3 A x MXFP4 B). Operands are preshuffled + laid out by the caller (once, off the launch path) -- see the arch-specific preshuffle in the tests.

Symbols: `flydsl_batched_gemm_mxfp4`, `_run_gfx1250`, `preshuffle_a8w4_weight_mbn`, `quant_act_mxfp8_mbn`, `flydsl_batched_gemm_a8w4_v2`.

Parameters referenced in this file: `num_buffers`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/bmm_a8w8_mxscale_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/bmm_a8w8_mxscale_gfx950.py) · **Recent**

Source documentation: MXFP8 strided-batched GEMM on gfx950 with E8M0 block scales. The structure follows the gfx1250 batched kernel (kernels/gemm_a8w8_gfx1250.py): an m_warp x n_warp wave grid, a num_buffers-deep LDS ring filled by async copies, and an LDS-staged epilogue. gfx950 has no TDM, so every stage is filled by buffer_load...lds, and the MFMA fragments and scale words follow the gfx950 MXFP preshuffle kernel (scaled 16x16x128 MFMA, fp8 halves 64 K apart).

Symbols: `_LdsLayout`, `_lds_layout`, `_padded_k`, `check_bmm_config`, `_byte_slots`, `_scale_panels`, `launch_bmm_a8w8_mxscale`.

Parameters referenced in this file: `num_buffers`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mxfp4_preshuffle_batched_gemm_gfx1250_tdm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mxfp4_preshuffle_batched_gemm_gfx1250_tdm.py) · **Recent**

Source documentation: Strided-batched A8W4 (MXFP8 E4M3 A x MXFP4 B) preshuffle GEMM for gfx1250

Symbols: `launch_gemm_a8w4_tdm`.

Parameters referenced in this file: `num_buffers`, `num_warps`, `tile_k`, `tile_m`, `tile_n`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py)
- [op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py)
- [op_tests/op_benchmarks/triton/bench_fused_bmm_rope_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_bmm_rope_kv_cache.py)
- [op_tests/test_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8.py)
- [op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py)
- [op_tests/test_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_bf16.py)
- [op_tests/test_flydsl_batched_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_batched_gemm.py)
- [op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a16w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a16w8.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a16wfp4.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_afp4wfp4.py)
- [op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py)

## Performance and validation

Use the wrapper and matching reference for that variant. Report batch size, all matrix strides and any broadcasted operands.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
