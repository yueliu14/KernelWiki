---
title: "Batched matrix multiplication: AITER aiter"
kind: implementation_knowledge
operator: batched_gemm
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Batched matrix multiplication — AITER aiter

Batch strides and batch size affect dispatch as well as M/N/K. A batch axis can expose parallelism that a single small GEMM lacks. Quantized variants add per-batch, per-token or block scale layouts.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/batched_gemm_op_a8w8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/batched_gemm_op_a8w8.py) · **Recent**

Source documentation: Exact tuned row for this shape and weight-scale block, else one at a padded M; None when no level hits. The row is shared: treat it as read-only.

Symbols: `gen_batched_gemm_a8w8_fake_tensors`, `batched_gemm_a8w8`, `compute_batched_gemm_SplitK`, `get_CKBatchedGEMM_config`, `batched_gemm_a8w8_CK`, `_mxscale_bmm_tuned_path`, `_get_mxscale_bmm_launchers`, `_load_mxscale_bmm_tuned`, `lookup_mxscale_bmm_config`, `_w_scale_block_of`, `_get_mxscale_bmm_launch_plan`, `_batched_gemm_a8w8_mxscale_impl`, `_batched_gemm_a8w8_mxscale_fake`, `batched_gemm_a8w8_mxscale`, `_batched_gemm_a8w8_mxscale_bpreshuffle_impl`, `batched_gemm_a8w8_mxscale_bpreshuffle`, `gen_batched_gemm_a8w8_tune_fake_tensors`, `batched_gemm_a8w8_tune`.

Parameters referenced in this file: `splitK`, `split_k`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/batched_gemm_op_bf16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/batched_gemm_op_bf16.py) · **Recent**

Symbols: `gen_batched_gemm_bf16_tune_fake_tensor`, `batched_gemm_bf16`, `compute_batched_gemm_SplitK`, `get_CKBatchedGEMM_config`, `batched_gemm_bf16_CK`, `batched_gemm_bf16_tune`.

Parameters referenced in this file: `splitK`, `tile_k`, `tile_m`, `tile_n`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/configs/a8w8_tuned_batched_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_tuned_batched_gemm.csv)
- [aiter/configs/a8w8_untuned_batched_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/a8w8_untuned_batched_gemm.csv)
- [aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv)
- [aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv)
- [aiter/configs/bf16_tuned_batched_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/bf16_tuned_batched_gemm.csv)
- [aiter/configs/bf16_untuned_batched_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/bf16_untuned_batched_gemm.csv)
- [aiter/configs/model_configs/dsr1_a8w8_tuned_batched_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsr1_a8w8_tuned_batched_gemm.csv)
- [aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv)
- [aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_untuned.csv)
- [aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_batched_gemm_a8w8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8.py)
- [op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py)
- [op_tests/test_batched_gemm_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_batched_gemm_bf16.py)
- [op_tests/test_flydsl_batched_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_batched_gemm.py)
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

## Performance and validation

Use the wrapper and matching reference for that variant. Report batch size, all matrix strides and any broadcasted operands.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
