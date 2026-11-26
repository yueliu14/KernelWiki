---
title: "MoE routing, sorting and token assignment: AITER flydsl"
kind: implementation_knowledge
operator: moe_routing_topk
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# MoE routing, sorting and token assignment — AITER flydsl

Routing selects experts and constructs the token order consumed by expert GEMMs. Softmax/sigmoid, grouped selection, padding and expert-id conventions vary across entry points.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/kernels/moe_route_maps.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_route_maps.py) · **Recent**

Source documentation: One-pass MoE route -> grouped-row map kernel (FlyDSL), atomic-scatter argsort. Computes topids_to_rows (route -> grouped row) and rows_to_tokens (inverse) via per-expert atomicAdd. One thread per route, no host-side argsort.

Symbols: `_RouteCntStorage`, `_RouteG2LStorage`, `_i32_ptr`, `build_moe_route_maps_module`, `build_moe_topids_to_rows_module`, `build_moe_topids_to_rows_g2l_module`, `build_moe_route_g2l_lds_module`, `build_moe_route_g2l_fused_module`.

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/moe_sorting_kernel.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_sorting_kernel.py) · **Recent**

Source documentation: MoE token sorting kernel (FlyDSL). Implements the MoE sorting operation used in DeepSeek R1 and similar MoE models. Given router top-k selections (topk_ids, topk_weights), reorganizes tokens by expert for efficient batched expert GEMM execution. Algorithm: counting sort in LDS (histogram → prefix-sum → scatter).

Symbols: `_unwrap_val`, `_dpp_intra_wave_prefix_sum`, `_allwave_inclusive_prefix_sum`, `_zero_moe_buf_grid_stride`, `_extend_prefix_sum_serial`, `_write_expert_id_blocks`, `_fill_sentinel_slots`, `_lds_load_raw`, `_lds_store_raw`, `_buf_iter`, `_i8_global_ptr`, `_gld`, `_gst`, `_compile_moe_sorting_oneshot`, `_p23_block_size`, `_compile_moe_sorting_multiphase`, `_compute_sub_tokens`, `moe_sorting_get_workspace_size`, `compile_moe_sorting`, `moe_sorting_flydsl`.

Parameters referenced in this file: `BLOCK_SIZE`.

### `aiter/ops/flydsl/moe_sorting.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/moe_sorting.py)

Source documentation: FlyDSL MoE sorting kernel — drop-in replacement for CK/Opus moe_sorting_fwd. Provides flydsl_moe_sorting_fwd() with the same signature as aiter.moe_sorting_fwd() so it can be used as a direct dispatch target in _moe_sorting_impl(). Workspace is pre-allocated here (not inside the kernel) so that CUDA graph capture sees deterministic allocations.

Symbols: `flydsl_moe_sorting_fwd`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/hip/bench_mxfp4_moe_sort_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/hip/bench_mxfp4_moe_sort_quant.py)
- [op_tests/test_moe_sorting.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_sorting.py)
- [op_tests/test_moe_sorting_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_sorting_mxfp4.py)
- [op_tests/test_moe_topk_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_topk_gating.py)
- [op_tests/test_topk_softmax.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_softmax.py)
- [op_tests/test_topk_softmax_shared_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_softmax_shared_gate.py)
- [op_tests/triton_tests/moe/moe_routing/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/__init__.py)
- [op_tests/triton_tests/moe/moe_routing/test_moe_routing.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/test_moe_routing.py)
- [op_tests/triton_tests/moe/moe_routing/test_moe_routing_herd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/test_moe_routing_herd.py)
- [op_tests/triton_tests/moe/moe_routing/test_topk_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/test_topk_gating.py)
- [op_tests/triton_tests/torch_compile/test_compile_moe_routing.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_moe_routing.py)

## Performance and validation

Check ties, repeated experts, masked tokens, output weights and padding. For small batches measure host launch cost as well as device time.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
