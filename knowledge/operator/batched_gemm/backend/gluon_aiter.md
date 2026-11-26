---
title: "Batched matrix multiplication: AITER gluon"
kind: implementation_knowledge
operator: batched_gemm
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Batched matrix multiplication — AITER gluon

Batch strides and batch size affect dispatch as well as M/N/K. A batch axis can expose parallelism that a single small GEMM lacks. Quantized variants add per-batch, per-token or block scale layouts.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/batched/batched_gemm_a16w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/batched/batched_gemm_a16w8.py) · **Recent**

Symbols: `_tile_coords`, `_issue_load`, `_batched_gemm_a16w8_gfx1250_persistent_kernel`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/batched/batched_gemm_bf16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/gemm/batched/batched_gemm_bf16.py) · **Recent**

Symbols: `_batched_gemm_bf16_bandwidth_bound_kernel`, `_batched_gemm_bf16_compute_bound_kernel`.

Parameters referenced in this file: `BLOCK_K`, `BLOCK_M`, `BLOCK_N`, `NUM_KSPLIT`, `num_warps`, `waves_per_eu`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=128-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=128-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=128-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=128-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=16-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=16-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=8-N=1024-K=4096.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=8-N=1024-K=4096.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w16/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-B=128-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-B=128-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-B=128-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-B=128-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-N=128-K=512.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-N=128-K=512.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-N=512-K=128.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-N=512-K=128.json)
- [aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/DEFAULT.json)

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
