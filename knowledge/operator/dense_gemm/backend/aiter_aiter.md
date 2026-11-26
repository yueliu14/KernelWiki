---
title: "Dense matrix multiplication: AITER aiter"
kind: implementation_knowledge
operator: dense_gemm
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Dense matrix multiplication — AITER aiter

Separate the arithmetic kernel from weight preparation and dispatch. Compare tiles at the actual M/N/K, strides, output type and weight layout. Small M often needs a different grid or split-K policy from prefill.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/deepgemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/deepgemm.py) · **Recent**

Source documentation: DeepGEMM front-end (CK backend). Hosts the CK-backed deepgemm_ck binding plus a thin deepgemm() wrapper. Opus entries have been extracted under aiter.ops.opus.*; see aiter.ops.opus.gemm_a16w16_opus for BF16 matmul and aiter.ops.opus.opus_gemm_a16w16_tune for id-based kernel selection. opus_gemm_a16w16_tune is kept here as a deprecation shim for one release to ease migration from the old aggregate entry.

Symbols: `deepgemm_ck`, `deepgemm`, `opus_gemm_a16w16_tune`.

Parameters referenced in this file: `splitK`.

### `aiter/ops/gemm_op_a16w16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_a16w16.py) · **Recent**

Source documentation: Return a per-(device, stream) zero-initialized semaphore workspace. SplitK a16w16 ASM kernels use an atomic-counter protocol where the last workgroup performs the reduction phase. Concurrent launches on different streams must not share the same atomic counter, or the counts get mixed and the reduction phase never fires (deadlock).

Symbols: `_gemm_a16w16_asm`, `_get_semaphore_workspace_keyed`, `get_semaphore_workspace`, `gemm_a16w16_asm`.

Parameters referenced in this file: `splitK`.

### `aiter/ops/gemm_op_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/gemm_op_common.py) · **Recent**

Source documentation: (row, M it was tuned at): the row for M itself, else for M padded at the two getPaddedM granularities, the walk the CK / asm lookups make. key maps an M to the table key; row is None when no level hits.

Symbols: `get_padded_m`, `find_padded_m_row`, `mxscale_w_scale_block`, `with_mxscale_w_scale_block`.

### `aiter/tuned_gemm.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/tuned_gemm.py) · **Recent**

Source documentation: * Copyright (C) Advanced Micro Devices, Inc. All rights reserved. * Copyright (C) 2024-2026, The vLLM team. * * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use this file except in compliance with the License.

Symbols: `_get_flydsl_gemm_kernels`, `get_GEMM_A16W16_config_`, `is_skinny_default_shape`, `get_GEMM_A16W16_tuned_config`, `get_GEMM_A16W16_config`, `save_shapes`, `gen_gemm_a16w16_fake_tensor`, `gemm_a16w16`, `skinny_gemm`, `hipb_gemm`, `torch_gemm`, `asm_gemm`, `flydsl_gemm`, `opus_gemm`, `triton_gemm`, `TunedGemm`.

Parameters referenced in this file: `splitK`, `split_k`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/configs/bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/bf16_tuned_gemm.csv)
- [aiter/configs/bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/dsr1_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsr1_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv3_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv3_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/dsv41_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv41_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/dsv4_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/dsv4_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/flux2_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/flux2_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/glm47_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm47_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/glm47_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm47_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/glm53_flash_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm53_flash_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/glm53_flash_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm53_flash_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/glm5_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/glm5_untuned_gemm_bf16.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/glm5_untuned_gemm_bf16.csv)
- [aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/inkling_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/inkling_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/kimi_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimi_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/kimi_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimi_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/kimik2_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik2_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/kimik3_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/kimik3_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/llama405B_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/llama405B_untuned_gemm.csv)
- [aiter/configs/model_configs/llama405B_untuned_gemm_bf16.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/llama405B_untuned_gemm_bf16.csv)
- [aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/llama70B_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/llama70B_untuned_gemm.csv)
- [aiter/configs/model_configs/llama70B_untuned_gemm_bf16.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/llama70B_untuned_gemm_bf16.csv)
- [aiter/configs/model_configs/minimax_m3_eagle_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_eagle_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/minimax_m3_eagle_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/minimax_m3_eagle_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/muse_glimmer_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/muse_glimmer_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/q3vl_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/q3vl_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/q3vl_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/q3vl_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/qwen32B_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen32B_untuned_gemm.csv)
- [aiter/configs/model_configs/qwen32B_untuned_gemm_bf16.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen32B_untuned_gemm_bf16.csv)
- [aiter/configs/model_configs/qwen38_flash_next_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/qwen38_flash_next_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen38_flash_next_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/qwen3_5_397b_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_397b_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/qwen3_5_397b_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_5_397b_untuned_gemm.csv)
- [aiter/configs/model_configs/qwen3_8_27b_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_8_27b_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/qwen3_8_2p4t_a95b_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_8_2p4t_a95b_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/qwen3_8_2p4t_a95b_bf16_untuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_8_2p4t_a95b_bf16_untuned_gemm.csv)
- [aiter/configs/model_configs/qwen3_8b_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/qwen3_8b_bf16_tuned_gemm.csv)
- [aiter/configs/model_configs/wan22_bf16_tuned_gemm.csv](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/configs/model_configs/wan22_bf16_tuned_gemm.csv)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_deepgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_deepgemm.py)
- [op_tests/test_f4gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_f4gemm.py)
- [op_tests/test_flydsl_hgemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_hgemm.py)
- [op_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_a16w16.py)
- [op_tests/test_gemm_codegen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_codegen.py)
- [op_tests/test_gemm_group32_interface.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gemm_group32_interface.py)
- [op_tests/test_opus_a16w16_gemm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_opus_a16w16_gemm.py)
- [op_tests/flydsl_tests/test_gemm_a16w16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/flydsl_tests/test_gemm_a16w16.py)
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

## Performance and validation

Measure packing and split-K reduction separately, then include both in the deployed operator cost. Keep accumulation and epilogue semantics unchanged.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
