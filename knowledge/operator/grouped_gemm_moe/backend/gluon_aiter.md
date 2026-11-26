---
title: "Grouped GEMM and MoE expert computation: AITER gluon"
kind: implementation_knowledge
operator: grouped_gemm_moe
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Grouped GEMM and MoE expert computation — AITER gluon

MoE execution couples token/expert sorting with gate/up projection, activation and down projection. The row distribution per expert, top-k, padded block count and placement of routing weights affect both work and correctness.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a16w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a16w4.py) · **Recent**

Source documentation: Swizzle the program id based on integer XCD_SWIZZLE.

Symbols: `matmul_launch_metadata`, `xcd_swizzle`, `unswizzle_mx_scale_gfx1250`, `_moe_gemm_a16w4`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a4w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a4w4.py) · **Recent**

Symbols: `matmul_launch_metadata`, `get_moe_a4w4_layouts_prefill`, `get_moe_a4w4_layouts_decode`, `unswizzle_scales_gfx1250`, `unshuffle_weights_gfx1250`, `_moe_gemm_a4w4_prefill`, `_moe_gemm_a4w4_decode`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py) · **Recent**

Symbols: `matmul_launch_metadata`, `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_gather_idx_dtype`, `get_moe_a8w4_layouts`, `_build_moe_a8w4_layouts`, `_moe_gemm_a8w4_prefill`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `BLOCK_N_PERSISTENT`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/moe/reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/moe/reduce.py) · **Recent**

Source documentation: Gluon grouped row-reduce for the gfx1250 MoE scatter-combine (replaces the Triton _reduce_grouped). One workgroup per group sums the K*B rows indx[g, :] (TDM-loaded, summed in-register, no cross-wave communication) into out[g, :N], with optional external residual fold-in.

Symbols: `reduce_grouped_gluon`, `reduce_grouped_gluon_num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx942/moe/moe_op_gemm_int8_smoothquant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx942/moe/moe_op_gemm_int8_smoothquant.py) · **Recent**

Symbols: `_gluon_moe_gemm_int8_smoothquant`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/moe/moe_op_gemm_a16w4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/moe/moe_op_gemm_a16w4.py) · **Recent**

Symbols: `matmul_launch_metadata`, `unswizzle_mx_scale_cdna4`, `_moe_gemm_a16w4_swizzle`, `_moe_gemm_a16w4_swizzle_pipelined`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `matrix_instr_nonkdim`, `num_warps`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/moe/a4w4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/moe/a4w4/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/moe/a8w4/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/moe/a8w4/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/flydsl_tests/test_flydsl_grouped_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py)
- [op_tests/flydsl_tests/test_flydsl_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_flydsl_moe.py)
- [op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py)
- [op_tests/multigpu_tests/bench_mega_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/bench_mega_moe.py)
- [op_tests/multigpu_tests/bench_mega_moe_v2.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/bench_mega_moe_v2.py)
- [op_tests/multigpu_tests/test_mega_moe_v2.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_mega_moe_v2.py)
- [op_tests/multigpu_tests/test_wide_ep_moe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_wide_ep_moe.py)
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
- [op_tests/tuners/tune_fhmoe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuners/tune_fhmoe.py)
- [op_tests/tuning_tests/test_fhmoe_tuner.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_fhmoe_tuner.py)

## Performance and validation

Benchmark representative expert-load distributions, not only a uniform dense matrix. Include sorting, zeroing, scatter and reduction when evaluating the complete operator.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
