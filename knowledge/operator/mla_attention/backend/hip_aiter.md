---
title: "Multi-head latent attention: AITER hip"
kind: implementation_knowledge
operator: mla_attention
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Multi-head latent attention — AITER hip

MLA combines latent KV storage with non-positional and positional query/key components. Cache representation and projection placement can move substantial work between GEMM and attention.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/mla_metadata.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_metadata.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla_v4_prefill_opus.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_v4_prefill_opus.h) · **Recent**

Symbols: `ceil_div`, `opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernel`, `opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel`, `opus_mla_v4_prefill_a8w8_16mx8_32nx1_kernel`, `opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel`, `opus_mla_v4_prefill_a16w16_16mx8_32nx1_kernel`, `opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel`, `opus_mla_v4_prefill_a8w8_16mx8_32nx1_kernel`, `opus_mla_v4_prefill_a8w8_16mx1_16nx4_kernel`, `global_load`, `global_load`, `global_load`, `sched_compute_qk`, `make_layout_q`, `make_layout_o`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `make_layout_rv`, `make_layout_kv_indices`, `attn_row_max`, `attn_row_scale_sub`, `attn_exp2_slice`, `attn_row_sum`, `scale_output_tile`, `attn_mask_vec2_imm`, `attn_mask_oob_score`, `attn_mask_oob_value`, `mla_v4_prefill_accum_le2_tiles`, `mla_v4_prefill_accum_pipelined`, `__launch_bounds__`, `make_layout_q`, `make_layout_o`, `make_layout_gkv`, `make_layout_skv`, `make_layout_rk`, `make_layout_rv`, `make_layout_kv_indices`, `attn_row_max`, `attn_row_scale_sub`, `attn_exp2_slice`, `attn_row_sum`, `scale_output_tile`, `attn_mask_oob_score`, `attn_mask_oob_value`, `mla_v4_prefill_accum_pipelined`, `__launch_bounds__`, `sched_compute_qk`, `make_layout_q_nope`, `make_layout_q_rope`, `make_layout_q_mxscl`, `make_layout_kv_indices`, `make_layout_gk_nope`, `make_layout_sk_nope`, `make_layout_rk_nope`, `make_layout_rk_mxscl`, `make_layout_gk_rope`, `make_layout_sk_rope`, `make_layout_cp_rope`, `make_layout_rk_rope`, `make_layout_rv_dequant`, `make_layout_sv_dequant`, `make_layout_rv`, `make_layout_o`, `scale_output_tile`, `attn_row_max`, `attn_row_scale_sub`, `attn_exp2_slice`, `attn_row_sum`, `attn_mask_vec2_imm`, `attn_mask_oob_score`, `attn_mask_oob_value`, `mla_v4_prefill_accum_le2_tiles`, `mla_v4_prefill_accum_pipelined`, `__launch_bounds__`, `make_layout_q_nope`, `make_layout_q_rope`, `make_layout_q_mxscl`, `make_layout_rk_nope`, `make_layout_rk_rope`, `make_layout_sk_nope`, `make_layout_sk_rope`, `make_layout_rv`, `make_layout_kv_indices`, `make_layout_o`, `attn_row_max`, `attn_row_scale_sub`, `attn_exp2_slice`, `attn_row_sum`, `scale_output_tile`, `attn_mask_oob_score`, `attn_mask_oob_value`, `reorder_mxscl_for_opsel`, `mla_v4_prefill_accum_pipelined`, `__launch_bounds__`.

### `csrc/kernels/mla/metadata.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/metadata.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh10_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh10_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh112_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh112_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh128_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh128_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh128_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh128_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh12_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh12_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh16_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh16_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh16_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh16_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh1_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh1_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh24_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh24_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh2_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh2_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh32_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh32_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh32_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh32_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh40_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh40_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh48_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh48_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh48_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh48_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh4_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh4_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh64_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh64_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh64_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh64_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh64_hd64.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh64_hd64.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh80_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh80_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh8_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh8_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh8_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh8_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh96_hd128.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh96_hd128.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/reduce_instances/nh96_hd512.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/reduce_instances/nh96_hd512.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla_hk.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_hk.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/mla_reduce.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mla_reduce.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/hk_v32_decode_fwd.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/hk_v32_decode_fwd.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mla/hk_v40_decode_fwd.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mla/hk_v40_decode_fwd.cu)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode_rope.py)
- [op_tests/test_concat_cache_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_concat_cache_mla.py)
- [op_tests/test_concat_cache_mla_seg.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_concat_cache_mla_seg.py)
- [op_tests/test_flydsl_gather_kv_b_proj.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_gather_kv_b_proj.py)
- [op_tests/test_flydsl_mla_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_mla_reduce.py)
- [op_tests/test_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla.py)
- [op_tests/test_mla_decode_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_decode_gate.py)
- [op_tests/test_mla_decode_pagesize64.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_decode_pagesize64.py)
- [op_tests/test_mla_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_flydsl.py)
- [op_tests/test_mla_ltx.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_ltx.py)
- [op_tests/test_mla_persistent.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_persistent.py)
- [op_tests/test_mla_persistent_round_robin.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_persistent_round_robin.py)
- [op_tests/test_mla_prefill_ps.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_prefill_ps.py)
- [op_tests/test_mla_reduce.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_reduce.py)
- [op_tests/test_mla_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_sparse.py)
- [op_tests/test_mla_stage2_merge.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_stage2_merge.py)
- [op_tests/test_mla_v40_persistent.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v40_persistent.py)
- [op_tests/test_mla_v4_kargpreld.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v4_kargpreld.py)
- [op_tests/test_mla_v4_nm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mla_v4_nm.py)
- [op_tests/triton_tests/attention/test_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla.py)
- [op_tests/triton_tests/attention/test_mla_decode_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla_decode_rope.py)
- [op_tests/triton_tests/attention/test_mla_v4_triton.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mla_v4_triton.py)
- [op_tests/triton_tests/fusions/test_fused_mla_prefill_qkv_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_mla_prefill_qkv_fp8.py)
- [op_tests/triton_tests/test_gather_kv_b_proj.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_gather_kv_b_proj.py)
- [op_tests/triton_tests/utils/mla_decode_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mla_decode_ref.py)
- [op_tests/triton_tests/utils/mla_extend_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/mla_extend_ref.py)

## Performance and validation

Record cache layout, head counts, latent/rope dimensions and query length. Compare the complete wrapper and its reduction/projection helpers.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](hip.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
