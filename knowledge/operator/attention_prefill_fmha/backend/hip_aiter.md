---
title: "Prefill and full attention: AITER hip"
kind: implementation_knowledge
operator: attention_prefill_fmha
backend: hip
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Prefill and full attention — AITER hip

Tiled attention interleaves QK products, online softmax and probability-value products. Inference paths require the correct causal or variable-length mask, head layout and KV-cache contract. Use forward entry points with dropout disabled.

HIP implementations are reached through JIT bindings or native wrappers. Track compile-time templates and dispatch conditions as well as launch dimensions; changing a source file requires confirming that the rebuilt module is loaded.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `csrc/include/fmha_fwd_bf16_opus_launch.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fmha_fwd_bf16_opus_launch.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/fmha_fwd_hd128_bf16_opus.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fmha_fwd_hd128_bf16_opus.h) · **Recent**

Symbols: `gqa_d128_kernel`, `gqa_d128_kernel`.

### `csrc/include/fmha_fwd_hd128_bf16_opus_defs.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fmha_fwd_hd128_bf16_opus_defs.h) · **Recent**

Symbols: `ceil_div`.

### `csrc/include/fmha_fwd_hd128_bf16_opus_kernel.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fmha_fwd_hd128_bf16_opus_kernel.hpp) · **Recent**

Symbols: `sched_barrier_pairs`, `sched_barrier_exp_pairs`, `sched_barrier_exp_valu_pairs`, `sched_barrier_valu_exp_pairs`, `sched_mfma_tail`, `make_layout_q`, `make_layout_o`, `make_layout_o_x4`, `make_layout_gk_gv`, `make_layout_sk_sv`, `make_layout_rk`, `make_layout_rv`, `make_layout_gq`, `make_layout_sq`, `make_layout_rq`, `attn_row_max`, `attn_sub_row`, `attn_scale_sub_row`, `attn_exp2_slice`, `load_range`, `attn_sum`, `scale_output_tile`, `pin_output_tile`, `attn_mask_vec2_imm`, `attn_mask_causal_tile`, `attn_mask_border_tile`, `__attribute__`, `kv_frag_t`, `s_frag_t`, `__launch_bounds__`.

### `csrc/include/fmha_fwd_hd192_v128_bf16_opus_defs.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fmha_fwd_hd192_v128_bf16_opus_defs.h) · **Recent**

Symbols: `ceil_div`.

### `csrc/include/fmha_fwd_hd192_v128_bf16_opus_kernel.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fmha_fwd_hd192_v128_bf16_opus_kernel.hpp) · **Recent**

Symbols: `sched_mfma_valu`, `sched_mfma_exp`, `sched_mfma_exp_valu`, `sched_mfma_tail`, `attn_row_max`, `attn_scale_sub_row`, `attn_scale_sub_row_slice`, `attn_exp2_slice`, `attn_sum`, `scale_output_tile`, `pin_output_tile`, `async_load_range`, `make_layout_o_x4`, `make_layout_g`, `make_layout_sq`, `make_layout_s`, `make_layout_rq`, `make_layout_rk_su`, `make_layout_rv_su`, `attn_mask_vec2_imm`, `attn_mask_causal_tile`, `attn_mask_border_tile`, `__attribute__`, `s_frag_t`, `__launch_bounds__`.

### `csrc/include/mha_v4_quant.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mha_v4_quant.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/torch/fmha_fwd_bf16_opus.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/torch/fmha_fwd_bf16_opus.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/torch/mha_v4_fwd.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/torch/mha_v4_fwd.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mha_native/mha_fwd_native_splitkv.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/mha_fwd_native_splitkv.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mha_native/mha_native.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/mha_native.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mha_v4_quant.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_v4_quant.cu) · **Recent**

Symbols: `swap_thread_data`, `subtract_k_mean`, `hadamard_rotate_activation_hd128_kernel`, `hadamard_rotate_activation_mxfp8_quant_kernel`, `hadamard_rotate_activation_mxfp6_quant_kernel`, `hadamard_rotate_activation_mxfp4_quant_kernel`, `fp6_p_v_token`, `mxfp4_v_token`, `__launch_bounds__`, `__launch_bounds__`.

### `csrc/include/attention_ck.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/attention_ck.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/attention_dtypes.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/attention_dtypes.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/fmha_fwd_hd192_v128_bf16_opus.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/fmha_fwd_hd192_v128_bf16_opus.h)

Symbols: `gqa_d192_v128_kernel`, `gqa_d192_v128_kernel`.

### `csrc/include/mha_common.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mha_common.h)

Symbols: `ParsePhiloxCudaState`.

### `csrc/include/mha_fwd.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/mha_fwd.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/torch/mha_batch_prefill.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/torch/mha_batch_prefill.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/torch/mha_fwd.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/torch/mha_fwd.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/torch/mha_v3_fwd.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/torch/mha_v3_fwd.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/torch/mha_v3_varlen_fwd.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/torch/mha_v3_varlen_fwd.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/include/torch/mha_varlen_fwd.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/include/torch/mha_varlen_fwd.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mha_common.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_common.cu)

Symbols: `ParsePhiloxCudaState`.

### `csrc/kernels/mha_native/fmha_fwd_d64_bf16_combine.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fmha_fwd_d64_bf16_combine.cu)

Symbols: `__launch_bounds__`.

### `csrc/kernels/mha_native/fmha_fwd_d64_bf16_msk0_split.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fmha_fwd_d64_bf16_msk0_split.cu)

Symbols: `__launch_bounds__`.

### `csrc/kernels/mha_native/fmha_fwd_d64_bf16_msk1_split.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fmha_fwd_d64_bf16_msk1_split.cu)

Symbols: `__launch_bounds__`.

### `csrc/kernels/mha_native/fused/op_combine.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fused/op_combine.hpp)

Symbols: `combine_split`.

### `csrc/kernels/mha_native/fused/op_epilog.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fused/op_epilog.hpp)

Symbols: `epilog_store`, `epilog_store_split`.

### `csrc/kernels/mha_native/fused/op_gemm.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fused/op_gemm.hpp)

Symbols: `pack_short4`, `swz`, `lds_elem_offset`, `v_lds_elem_offset`, `f32_to_bf16_trunc`, `bf16_to_f32`, `clear_acc`, `slice_q`, `gemm0_subtile`, `pack_p_subtile`, `slice_p`, `gemm1_subtile`, `gemm0`, `gemm1_bpermute`.

### `csrc/kernels/mha_native/fused/op_lds.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fused/op_lds.hpp)

Symbols: `s_waitcnt_vmcnt`, `s_waitcnt_vmcnt_0`, `s_waitcnt_lgkmcnt_0`, `s_waitcnt_lgkmcnt_1`, `async_load_fence`, `s_barrier`, `block_sync_lds`, `bpermute_f32`, `buf_base_elems`, `buf_base_bytes`, `async_copy_k_subtile`, `load_v_from_dram`, `store_v_to_lds`, `k_lds_offset`, `copy_k_to_lds_2x_guarded`.

### `csrc/kernels/mha_native/fused/op_softmax.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fused/op_softmax.hpp)

Symbols: `softmax_mask`, `softmax_row_max`, `softmax_exp2`, `softmax_row_sum`, `rescale_o_acc`, `rescale_o_acc`, `softmax_row_max`, `softmax_exp`, `softmax_row_sum`, `rescale_o_acc`.

### `csrc/kernels/mha_native/fused/pipeline.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/fused/pipeline.hpp)

Symbols: `make_buffer_resource`, `clamp_num_records`, `fmha_fwd_d64_device`.

### `csrc/kernels/mha_native/mha_native_launch.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/mha_native_launch.h)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/mha_native/runner/params.hpp`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/mha_native/runner/params.hpp)

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/cpp/mha/compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/mha/compile.py)
- [op_tests/op_benchmarks/flydsl/bench_mha_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_fwd.py)
- [op_tests/op_benchmarks/triton/bench_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_attn_res.py)
- [op_tests/op_benchmarks/triton/bench_deepgemm_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_deepgemm_attention.py)
- [op_tests/op_benchmarks/triton/bench_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_extend_attention.py)
- [op_tests/op_benchmarks/triton/bench_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mha.py)
- [op_tests/op_benchmarks/triton/bench_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_prefill.py)
- [op_tests/op_benchmarks/triton/compare_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/compare_mha.py)
- [op_tests/test_flydsl_fmha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fmha.py)
- [op_tests/test_fmha_fwd_mxfp8_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_mxfp8_asm.py)
- [op_tests/test_fmha_fwd_with_sink_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_asm.py)
- [op_tests/test_fmha_fwd_with_sink_varlen_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fmha_fwd_with_sink_varlen_asm.py)
- [op_tests/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha.py)
- [op_tests/test_mha_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_flydsl.py)
- [op_tests/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_fp8.py)
- [op_tests/test_mha_native_splitkv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_native_splitkv.py)
- [op_tests/test_mha_optional_flydsl.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_optional_flydsl.py)
- [op_tests/test_mha_v4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4.py)
- [op_tests/test_mha_v4_sparse.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_v4_sparse.py)
- [op_tests/test_mha_varlen.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen.py)
- [op_tests/test_mha_varlen_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_fp8.py)
- [op_tests/test_mha_varlen_large_kv.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_mha_varlen_large_kv.py)
- [op_tests/triton_tests/attention/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/__init__.py)
- [op_tests/triton_tests/attention/mha_test_utils.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/mha_test_utils.py)
- [op_tests/triton_tests/attention/test_chunked_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_chunked_pa_prefill.py)
- [op_tests/triton_tests/attention/test_dsv4_indexer.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_dsv4_indexer.py)
- [op_tests/triton_tests/attention/test_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_extend_attention.py)
- [op_tests/triton_tests/attention/test_fav3_sage.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage.py)
- [op_tests/triton_tests/attention/test_fav3_sage_compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fav3_sage_compile.py)
- [op_tests/triton_tests/attention/test_la_paged.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_la_paged.py)
- [op_tests/triton_tests/attention/test_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha.py)
- [op_tests/triton_tests/attention/test_mha_dao_ai.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_dao_ai.py)
- [op_tests/triton_tests/attention/test_mha_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_fp8.py)
- [op_tests/triton_tests/attention/test_mha_spatial_swizzle.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_spatial_swizzle.py)
- [op_tests/triton_tests/attention/test_mha_v3.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_v3.py)
- [op_tests/triton_tests/attention/test_mha_with_pe.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_pe.py)
- [op_tests/triton_tests/attention/test_mha_with_sink.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_mha_with_sink.py)
- [op_tests/triton_tests/attention/test_na3d_flash.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_na3d_flash.py)
- [op_tests/triton_tests/attention/test_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_prefill.py)
- [op_tests/triton_tests/attention/test_prefill_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_prefill_attention.py)
- [op_tests/triton_tests/attention/test_qsa.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_qsa.py)
- [op_tests/triton_tests/fusions/test_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_attn_res.py)

## Performance and validation

Record sequence lengths, masks, heads and dtypes. Use the matching forward reference with dropout disabled; do not compare kernel latency with an end-to-end serving metric.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
