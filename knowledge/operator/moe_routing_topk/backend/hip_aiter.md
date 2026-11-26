---
title: "MoE routing, sorting and token assignment: AITER hip"
kind: implementation_knowledge
operator: moe_routing_topk
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# MoE routing, sorting and token assignment — AITER hip

Routing selects experts and constructs the token order consumed by expert GEMMs. Softmax/sigmoid, grouped selection, padding and expert-id conventions vary across entry points.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/moe_sorting_opus.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/moe_sorting_opus.h) · **Recent**

Symbols: `__launch_bounds__`, `wave_cumsum`, `wave_reduce`.

### `csrc/kernels/moe_align_block_size_kernels.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/moe_align_block_size_kernels.cu) · **Recent**

Symbols: `index`, `moe_align_block_size_kernel`.

### `csrc/kernels/moe_fused_gate.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/moe_fused_gate.cu) · **Recent**

Symbols: `cmp_gt`, `cmp_eq`, `moe_fused_gate_impl`, `moe_fused_gate_kernel`, `moe_fused_gate_kernel_dynamic`.

### `csrc/kernels/topk_gating/topk_sigmoid.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/topk_gating/topk_sigmoid.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/topk_gating/topk_softmax.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/topk_gating/topk_softmax.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/topk_gating/topk_sqrtsoftplus.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/topk_gating/topk_sqrtsoftplus.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/topk_gating_kernels.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/topk_gating_kernels.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/topk_softmax_kernels.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/topk_softmax_kernels.cu) · **Recent**

Symbols: `moeSoftmax`, `moeTopK`, `topkGatingSoftmax`, `applySharedExpertSigmoid`, `moe_sum_kernel`, `topkGatingSoftmaxFusedGate`.

### `csrc/kernels/topk_softmax_kernels_group.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/topk_softmax_kernels_group.cu) · **Recent**

Symbols: `wave_reduce`, `cumsum_topk_with_pivot`, `wave_reduce_max2`, `wave_reduce_argmax2`, `warpReduceMax`, `legacy_topk_lane_rank`, `legacy_topk_tie_rank`, `topk_candidate_better`, `emit_legacy_ordered_topk`, `blockReduceMax`, `grouped_topk_kernel`, `topk_reg_kernel`, `grouped_topk_opt_sort_kernel`.

### `csrc/include/moe_sorting.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/moe_sorting.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

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

## Historical knowledge

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
