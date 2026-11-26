---
title: "Splitk streamk gemm: AITER flydsl"
kind: implementation_knowledge
operator: splitk_streamk_gemm
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Splitk streamk gemm — AITER flydsl

Entry points in this family have distinct input, layout and numerical contracts. Read the wrapper and its reference before choosing a replacement.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/kernels/gemm_a8w8_splitk_reduce_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gemm_a8w8_splitk_reduce_gfx1250.py) · **Recent**

Source documentation: Split-K partial reduction epilogue for the gfx1250 a8w8 preshuffle GEMM.

Symbols: `compile_gemm_a8w8_splitk_reduce`.

Parameters referenced in this file: `split_k`.

### `aiter/ops/flydsl/kernels/splitk_epilogue.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/splitk_epilogue.py) · **Recent**

Source documentation: One-stage split-K epilogue: last arriving split reduces the fp32 partials.

Symbols: `pairwise_sum`, `reduce_thread_split`, `splitk_reduce_epilogue`, `_semaphore_addr`.

Parameters referenced in this file: `split_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/splitk_fused_epilogue_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/splitk_fused_epilogue_gfx1250.py) · **Recent**

Source documentation: Fused split-K reduction shared by the K-split workgroups in one cluster.

Symbols: `emit_fused_splitk_epilogue`.

Parameters referenced in this file: `num_warps`, `split_k`, `tile_m`, `tile_n`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/tuning_tests/test_asm_splitk_guard.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/tuning_tests/test_asm_splitk_guard.py)

## Performance and validation

Compare identical inputs, layouts and outputs on the target architecture. Include preparation and postprocessing that execute in the live path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
