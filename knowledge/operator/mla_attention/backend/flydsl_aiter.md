---
title: "Multi-head latent attention: AITER flydsl"
kind: implementation_knowledge
operator: mla_attention
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Multi-head latent attention — AITER flydsl

MLA combines latent KV storage with non-positional and positional query/key components. Cache representation and projection placement can move substantial work between GEMM and attention.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/gather_kv_b_proj.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj.py) · **Recent**

Source documentation: FlyDSL backend for the MLA fused gather + kv_b_proj expansion. Supported: fp8 KV cache (OCP e4m3), fp8 weight in either row-major or shuffle_weight((16,16)) layout, per-output-row *or* 128x128 block weight scale, per-tensor activation scale, page_size 1, bf16 or scaled fp8 outputs on gfx950. The gfx1250 128+128 ptpc specialization is dispatched separately.

Symbols: `lds_bytes`, `_lds_capacity`, `block_n_of`, `_default_head_tiles`, `_cu_count`, `_default_block_m`, `_config_reason`, `_raise`, `_validate`, `_is_per_row_scale`, `_unsupported_reason`, `gather_kv_b_proj_flydsl_supported`, `_output_scale_reason`, `gather_kv_b_proj_flydsl_fp8_supported`, `compile_gather_kv_b_proj`, `_arch_of`, `_as_i8`, `_default_xcd`, `gather_kv_b_proj_flydsl`.

Parameters referenced in this file: `BLOCK_M`, `waves_per_eu`.

### `aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/gather_kv_b_proj_gfx1250.py) · **Recent**

Source documentation: gfx1250 MLA gather + FlyDSL ptpc projection.

Symbols: `_projection_runtime`, `_cluster_n`, `_run_projection`, `unsupported_reason`, `gather_kv_b_proj_flydsl_gfx1250`.

### `aiter/ops/flydsl/mla_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_kernels.py) · **Recent**

Source documentation: BF16 Q/KV form of :func:flydsl_mla_pagesize1_fp8_fp8; no scales.

Symbols: `_require`, `_require_runtime`, `_require_layout`, `_validate_pagesize1_inputs`, `_run_pagesize1`, `flydsl_mla_pagesize1_fp8_fp8`, `flydsl_mla_pagesize1_bf16_bf16`, `_validate_pagesize64_inputs`, `flydsl_mla_pagesize64_fp8_fp8`, `flydsl_mla_decode_reduce`.

### `aiter/ops/flydsl/mla_reduce_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/mla_reduce_kernels.py) · **Recent**

Source documentation: High-level FlyDSL MLA decode reduce API. Drop-in alternative for the HIP aiter.mla_reduce_v1 (csrc/kernels/mla/reduce.cu): same signature and in/out contract. Opt-in via AITER_MLA_REDUCE_FLYDSL=1; production keeps the HIP kernel by default. Kernels are compiled per (H, Dv, out_dtype, tier, output_lse) and lru_cache-backed; Tier.ALL (one kernel, device-side per-tile tier selection) mirrors HIP.

Symbols: `_out_dtype_str`, `_pointer_arg`, `_require_tensor`, `_validate_actual_max_splits`, `_validate_mla_reduce_inputs`, `_resolve_actual_max_splits`, `flydsl_mla_reduce_v1`.

Parameters referenced in this file: `waves_per_eu`.

### `aiter/ops/flydsl/kernels/gather_kv_b_proj_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gather_kv_b_proj_gfx1250.py) · **Recent**

Source documentation: MLA gather projection WMMA kernel for gfx1250. This is deliberately not a generic GEMM entry point. It gathers page-size-1 cache rows directly into the projection pipeline's padded LDS layout, multiplies the 512-wide FP8 latent by a ptpc-preshuffled FP8 weight, and writes BF16 128-wide K-nope, 64-wide RoPE, and 128-wide V head slices.

Symbols: `launch_gather_kv_b_proj_8wave_gfx1250`.

Parameters referenced in this file: `num_buffers`, `num_warps`, `tile_k`, `tile_m`, `tile_n`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_common.py) · **Recent**

Source documentation: Helpers shared by the gfx1250 MLA kernels.

Symbols: `_instruction_prefetch`, `_xor16_f32`, `_concat_wmma_operand`, `_concat_wmma_operand_k64`, `_concat_ds_tr8_b64`, `_rmem_i32`, `make_fp8_wmma_atom`, `_pack_fp8x4`, `_dwordx4_iter`, `make_global_load_b128`, `wave_layout`, `_unhoisted`, `_softmax_all`, `_round_robin`, `_drain`, `_min_u32`, `_wait_scalars`.

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_bf16_bf16.py) · **Recent**

Source documentation: gfx1250 page-size-1 BF16 persistent MLA decode stage-1 kernel, four waves.

Symbols: `keep_alive`, `kv_row_offset`, `_softmax_group`, `launch_mla_pagesize1_bf16_bf16`.

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py) · **Recent**

Source documentation: gfx1250 page-size-1 FP8 persistent MLA decode stage-1 kernel, four waves.

Symbols: `kv_row_offset`, `_softmax_group`, `launch_mla_pagesize1_fp8_fp8`.

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize64_fp8_fp8.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize64_fp8_fp8.py) · **Recent**

Source documentation: gfx1250 MLA page-size-64 FP8 stage-1 kernel.

Symbols: `_dword_iter`, `_rope_row_to_lds`, `launch_mla_pagesize64_fp8_fp8`.

Parameters referenced in this file: `BLOCK_THREADS`, `num_warps`.

### `aiter/ops/flydsl/kernels/mla_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mla_reduce.py) · **Recent**

Source documentation: FlyDSL MLA decode reduce/combine epilogue. Stage-2 of split-KV MLA decode: merges per-split partial outputs O_i (fp32) weighted by exp(LSE_i - LSE_max) (online softmax) into the final output (bf16/fp16), and optionally the merged LSE.

Symbols: `_out_numeric_t`, `_vector_elements`, `_make_vector`, `_pointer_view`, `_pointer_buffer_tensor`, `_f32_copy_atom`, `_chunk_tile_idx`, `_load_partial_out`, `_out_copy_atom`, `_store_final_out`, `_store_lse`, `_exp`, `_is_zero_or_nan`, `Tier`, `select_tier`, `should_use_persistent_launch`, `compile_mla_reduce`, `compile_mla_decode_reduce`, `plan_splitk`, `derive_actual_max_splits`, `plan_splitk_capture_safe`, `_get_splitk_scratch`, `compile_mla_reduce_splitk`.

Parameters referenced in this file: `waves_per_eu`.

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

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
