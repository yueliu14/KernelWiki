---
title: "MoE routing, sorting and token assignment: AITER triton"
kind: implementation_knowledge
operator: moe_routing_topk
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# MoE routing, sorting and token assignment — AITER triton

Routing selects experts and constructs the token order consumed by expert GEMMs. Softmax/sigmoid, grouped selection, padding and expert-id conventions vary across entry points.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/fused_routing_from_topk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_routing_from_topk.py) · **Recent**

Source documentation: Sort (token, slot) pairs by their expert id via a single Triton kernel. Replaces the multi-kernel torch chain (per-row sort, gather, two stable argsorts, advanced indexing, fp32 histc) with a single counting-sort kernel launch. Intended for use in the triton_kernels.matmul_ogs MoE pipeline, where the launch overhead of the original chain is a measurable fraction of decode TPOT. Args: topk_weights: [n_tokens, n_expts_act] per-token routing weights. Must be contiguous.

Symbols: `fused_routing_from_topk`.

Parameters referenced in this file: `BLOCK_E`, `BLOCK_NK`, `num_warps`.

### `aiter/ops/triton/moe/moe_routing/bitmatrix.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_routing/bitmatrix.py) · **Recent**

Source documentation: Represents a boolean matrix in a packed format where each element occupies a single bit of memory. _scratchpad is either None or an all-zero array of size >= shape[-1]; we pass it along with the actual bitmatrix to avoid having to launch a separate memset kernel when we call Bitmatrix::sum().

Symbols: `Bitmatrix`, `clear_sums`, `sum_bitmatrix_rows`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_MM`, `num_warps`.

### `aiter/ops/triton/moe/moe_routing/minunique.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_routing/minunique.py) · **Recent**

Source documentation: Fusion-2 driver: (k+1) candidates + popularity -> kept-k (Vout/Iout) + post-drop histogram + cross-block prefix offsets. hist/part are PRE-ZEROED.

Symbols: `keepk_sort0`, `_minunique_common`, `routing_minunique`, `routing_minunique_fused`.

Parameters referenced in this file: `BLOCK_A`, `num_warps`.

### `aiter/ops/triton/moe/moe_routing/routing.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_routing/routing.py) · **Recent**

Source documentation: Routing entry point. score_mode selects the path: * score_mode is None (default) -> the plain flat top-k routing process: flat top-k with softmax. sm_first controls whether softmax is applied to the logits before the top-k (True) or inside the top-k (False). The fused-V4-only arguments (bias, use_grouped_topk ...) are ignored on this path. * score_mode is not None -> the fused V4 (DeepSeek) routing process: fused score transform + (optionally grouped) top-k.

Symbols: `ExptData`, `RoutingData`, `sort_tokens`, `sort_tokens_fused`, `log2_power_of_two`, `_compute_expt_data_internal`, `routing`, `routing_from_hash`, `EpScatterGeometry`, `_ep_sort_scratch`, `ep_sort_routing`, `compute_expt_data_torch`, `routing_torch`.

Parameters referenced in this file: `BLOCK_A`, `num_warps`.

### `aiter/ops/triton/moe/moe_routing/topk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_routing/topk.py) · **Recent**

Source documentation: Triton grouped top-k expert selection. See module docstring. Returns (y_vals, y_indx, bitmatrix) matching the contract of aiter.ops.triton.moe.moe_routing.topk.topk: - y_vals: (n_rows, k) in x.dtype. - y_indx: (n_rows, k) int16. - bitmatrix: real :class:Bitmatrix; same uint32 (n_cols_words, n_rows_pad32).T storage / scratchpad layout the _topk kernel emits, so sort_tokens and sort_tokens_fused consume it unchanged.

Symbols: `grouped_topk`, `topk`, `hash_routing`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_MM`, `BLOCK_N`, `BLOCK_S`, `BLOCK_SP`, `num_warps`.

### `aiter/ops/triton/moe/moe_routing/topk_gating.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/moe/moe_routing/topk_gating.py) · **Recent**

Source documentation: Caller-owned MoE gating launchers without bitmatrix routing metadata.

Symbols: `exact_sigmoid_biased_topk`.

Parameters referenced in this file: `BLOCK_N`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_routing_from_topk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_routing_from_topk.py) · **Recent**

Symbols: `_fused_routing_from_topk_hist_kernel`, `_fused_routing_from_topk_offset_kernel`, `_fused_routing_from_topk_place_kernel`.

Parameters referenced in this file: `BLOCK_E`, `BLOCK_NK`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix.py) · **Recent**

Source documentation: Vertical popcount Input x : uint32[..., N] Output y : uint32[..., 32] semantics : y[..., i] = sum_j((x[..., j] >> i) & 1) credits: @apgoucher

Symbols: `vpopc`, `_sum_bitmatrix_memset`, `_sum_bitmatrix_rows`, `_sum_bitmatrix_rows_fused`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_MM`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix_sonicmoe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix_sonicmoe.py) · **Recent**

Symbols: `_sonicmoe_bitmatrix_metadata_compute_stage1`, `_sonicmoe_bitmatrix_metadata_compute_stage2`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/expt_data.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/expt_data.py) · **Recent**

Symbols: `_cdiv_pow2`, `_expt_data_compute_stage1`, `_expt_data_compute_stage2`, `_expt_data_compute_stage2_fused`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/minunique.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/minunique.py) · **Recent**

Symbols: `_keepk_sort0`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py) · **Recent**

Symbols: `_routing_compute_indx`, `_routing_compute_indx_fused`, `_combined_routing`, `_combined_routing_fused`, `_ep_gate_prep_scan_kernel`, `_ep_scatter_atomic_expt_data_kernel`.

Parameters referenced in this file: `BLOCK_A`, `BLOCK_M`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/routing_sonicmoe.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/routing_sonicmoe.py) · **Recent**

Symbols: `_sonicmoe_compute_col_partial_sum_kernel`, `_sonicmoe_general_compute_col_partial_sum_kernel`, `_sonicmoe_general_metadata_compute_stage2`, `_sonicmoe_token_offset_searchsorted_kernel`.

Parameters referenced in this file: `BLOCK_SIZE`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py) · **Recent**

Symbols: `get_topmask_and_fullmask`, `fpval_to_key`, `key_to_fpval`, `_apply_score_mode`, `streaming_topk`, `_topk`, `_hash_routing`, `_grouped_topk`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `BLOCK_S`, `BLOCK_SP`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/topk_gating.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/topk_gating.py) · **Recent**

Source documentation: Caller-owned MoE gating kernels without bitmatrix routing metadata.

Symbols: `_exact_sigmoid_biased_topk`.

Parameters referenced in this file: `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/moe/moe_routing/utils.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/moe/moe_routing/utils.py) · **Recent**

Source documentation: Add packed [16-bit key \| 16-bit count] values by equal-key runs.

Symbols: `keyed_add`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/triton_tests/moe/moe_routing/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/__init__.py)
- [op_tests/triton_tests/moe/moe_routing/test_moe_routing.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/test_moe_routing.py)
- [op_tests/triton_tests/moe/moe_routing/test_moe_routing_herd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/test_moe_routing_herd.py)
- [op_tests/triton_tests/moe/moe_routing/test_topk_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/moe/moe_routing/test_topk_gating.py)
- [op_tests/triton_tests/torch_compile/test_compile_moe_routing.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/torch_compile/test_compile_moe_routing.py)
- [op_tests/op_benchmarks/hip/bench_mxfp4_moe_sort_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/hip/bench_mxfp4_moe_sort_quant.py)
- [op_tests/test_moe_sorting.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_sorting.py)
- [op_tests/test_moe_sorting_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_sorting_mxfp4.py)
- [op_tests/test_moe_topk_gating.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_moe_topk_gating.py)
- [op_tests/test_topk_softmax.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_softmax.py)
- [op_tests/test_topk_softmax_shared_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_topk_softmax_shared_gate.py)

## Performance and validation

Check ties, repeated experts, masked tokens, output weights and padding. For small batches measure host launch cost as well as device time.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
