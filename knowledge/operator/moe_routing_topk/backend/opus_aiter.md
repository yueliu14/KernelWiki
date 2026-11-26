---
title: "MoE routing, sorting and token assignment: AITER opus"
kind: implementation_knowledge
operator: moe_routing_topk
backend: opus
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# MoE routing, sorting and token assignment — AITER opus

Routing selects experts and constructs the token order consumed by expert GEMMs. Softmax/sigmoid, grouped selection, padding and expert-id conventions vary across entry points.

OPUS implementations combine architecture-aware policies and native kernel instances. Follow policy and dispatch code to the launched entry and verify that the intended instance engages.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/moe_sorting_opus.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/moe_sorting_opus.py) · **Recent**

Source documentation: Fused route sort + compact MXFP4 activation quantization. gfx950 only, model_dim=4096, block_m=32, M in [1, 128]. (num_experts, topk) must be one of (512, 10) or (513, 11) -- the expert histogram is a kernel template parameter.

Symbols: `moe_sorting_opus_get_workspace_size`, `moe_sorting_opus_fwd`, `mxfp4_moe_sort_quant_fwd`.

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
