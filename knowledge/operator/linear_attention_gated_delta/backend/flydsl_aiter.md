---
title: "Linear attention and gated delta rules: AITER flydsl"
kind: implementation_knowledge
operator: linear_attention_gated_delta
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Linear attention and gated delta rules — AITER flydsl

Prefill commonly uses chunked state updates while decode applies a recurrent update. State layout, gate transforms, convolution state and in-place update order must match the wrapper.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/linear_attention_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_kernels.py) · **Recent**

Source documentation: High-level FlyDSL Linear Attention APIs.

Symbols: `_mtp_variant`, `_tile_warps`, `_decode_warp_shape`, `_decode_tiling`, `get_default_kwargs`, `_mtp_rung`, `_mtp_shape`, `_mtp_tiling`, `get_mtp_default_kwargs`, `_mtp_kwargs`, `flydsl_gdr_decode`, `_is_supported_arch_index`, `_is_supported_arch`, `_unit_strided`, `_on_device`, `_mtp_shapes_supported`, `_flydsl_gdr_mtp_supported`, `_flydsl_gdr_mtp_sglang_supported`, `_snapshot_store_bytes`, `_require_index`, `_mtp_common_checks`, `_mtp_stream`, `_mtp_launch`, `_launch_flydsl_gdr_mtp`, `flydsl_gdr_mtp`, `flydsl_gdr_mtp_sglang`.

Parameters referenced in this file: `num_warps`, `waves_per_eu`.

### `aiter/ops/flydsl/linear_attention_prefill_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/linear_attention_prefill_kernels.py) · **Recent**

Source documentation: FlyDSL host wrappers for GDN prefill. The module exposes hidden-state and prepare kernels. The prepare wrapper matches the Triton prepare layout and exponent-domain contract, so both paths can be selected independently through chunk_gated_delta_rule_opt_vk. Call gdn_prepare_flydsl_supported before selecting the fused prepare path.

Symbols: `_hipeq_device_idx`, `_hipeq_shared_memory_per_cu`, `_hipeq_compute_bv`, `_hipeq_select_bv`, `_hipeq_varlen_host_metadata`, `_opt_check`, `_placeholder_pair`, `_tuned_bv_runtime`, `_require_contiguous`, `_as_contiguous`, `_load_tuned_bv_table`, `reload_tuned_bv_table`, `_tuned_bv`, `_as_int32`, `_as_state_indices`, `_resolve_prologue`, `_resolve_state_dtype`, `_resolve_snapshot_dtype`, `_get_or_compile_opt`, `chunk_gated_delta_rule_fwd_h_flydsl_opt`, `_device_index`, `_pad_grid_x_odd`, `_is_cdna_mfma_arch`, `gdn_prepare_flydsl_supported`, `gdn_prepare_fwd_flydsl`.

### `aiter/ops/flydsl/kernels/gdr_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_common.py) · **Recent**

Source documentation: Vector-copy helpers shared by GDR decode and prefill kernels.

Symbols: `_gview`, `_load_vec`, `_store_vec`.

### `aiter/ops/flydsl/kernels/gdr_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_decode.py) · **Recent**

Source documentation: Gated delta rule over a speculative draft window. Undoing a rejected token needs a rollback point and a per-token record; mode picks whose contract to follow. MTP_MODE_CHAIN is vLLM's: rollback at state_indices[n, num_accepted - 1], and every token checkpoints into state_indices[n, t], the last of which is the final store. MTP_MODE_SNAPSHOT is SGLang's: rollback at state_indices[n], record in intermediate_states_buffer.

Symbols: `_gview64`, `_fast_exp`, `create_vk_gdr_decode_kernel`, `create_vk_gdr_mtp_kernel`.

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/chunk_gated_delta_h.py) · **Recent**

Source documentation: Gated Delta Net K5 hidden-state recurrence kernel (@flyc.kernel API). HIP-aligned fork: same mfma_f32_16x16x16bf16_1k instruction and warp partition (BT split-M, K split across waves, V not split) as the hand-tuned HIP/C++ K5 kernel; writes the public [..., V, K] layout through a [V][K] transpose buffer + b128 store (fp32 snapshots use two half-BV LDS rounds).

Symbols: `_make_fast_exp`, `_make_bf16_converter`, `compile_chunk_gated_delta_h`.

Parameters referenced in this file: `BLOCK_THREADS`.

### `aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/gdr_prefill/gdn_prepare.py) · **Recent**

Source documentation: FlyDSL GDN prefill chunk-prepare kernel for the BT=64 path. For each chunk, form the strictly lower-triangular gated KKT matrix A and C = (I + A)^-1, then produce: g_cumsum : [B, H, T] in-chunk prefix sum of g w_bar : [B, H, T, K] C @ (k * beta * exp(g_cumsum)) u_bar : [B, H, T, V] C @ (v * beta) Inputs are token-major and outputs are head-major. g_scale controls whether g_cumsum is published in natural-log or log2 space.

Symbols: `_exp2_f32`, `_mfma16`, `_mfma16_f32`, `_acc16_n4`, `_gemm16_n4`, `_accum_to_bf16x4`, `_swizzled_vt_view`, `_load_mfma_tile_vt_tiled`, `_tile16_view`, `_load_fp32_tile`, `_load_fp32_tile_raw`, `_store_fp32_tile`, `_identity_frag`, `_wy_prefetch`, `_wy_epilogue_to_lds`, `_wy_scatter`, `_wave_inclusive_scan`, `compile_gdn_prepare`.

Parameters referenced in this file: `BLOCK_THREADS`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_chunk_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_chunk_kda.py)
- [op_tests/op_benchmarks/triton/bench_flash_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_flash_kda.py)
- [op_tests/op_benchmarks/triton/bench_fused_gdn_decode_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gdn_decode_qkvz.py)
- [op_tests/op_benchmarks/triton/bench_fused_gdn_prefill_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_gdn_prefill_qkvz.py)
- [op_tests/op_benchmarks/triton/bench_fused_kda_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_kda_decode.py)
- [op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gated_delta_rule_block.py)
- [op_tests/op_benchmarks/triton/bench_gated_delta_rule_snapshot_dtype.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_gated_delta_rule_snapshot_dtype.py)
- [op_tests/op_benchmarks/triton/bench_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_kda.py)
- [op_tests/test_flydsl_gdr_mtp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_gdr_mtp.py)
- [op_tests/test_flydsl_linear_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention.py)
- [op_tests/test_flydsl_linear_attention_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_linear_attention_prefill.py)
- [op_tests/test_gated_delta_rule.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gated_delta_rule.py)
- [op_tests/test_gdn_prepare.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdn_prepare.py)
- [op_tests/test_gdr_decode_packed_bf16.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_gdr_decode_packed_bf16.py)
- [op_tests/test_split_gdr_update.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_split_gdr_update.py)
- [op_tests/triton_tests/attention/test_chunk_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_chunk_kda.py)
- [op_tests/triton_tests/attention/test_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_kda.py)
- [op_tests/triton_tests/gated_delta_net/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/__init__.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_gdn_decode_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_gdn_decode_qkvz.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_gdn_prefill_qkvz.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_gdn_prefill_qkvz.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_kda_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_kda_decode.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_kda_decode_unified.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_kda_decode_unified.py)
- [op_tests/triton_tests/gated_delta_net/test_fused_rearrange_sigmoid_gdr.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_fused_rearrange_sigmoid_gdr.py)
- [op_tests/triton_tests/gated_delta_net/test_gated_delta_rule_strided_qk.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/gated_delta_net/test_gated_delta_rule_strided_qk.py)
- [op_tests/triton_tests/kimi_delta_attn/__init__.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/__init__.py)
- [op_tests/triton_tests/kimi_delta_attn/test_chunk_delta_attn_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_chunk_delta_attn_fwd.py)
- [op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py)
- [op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py)
- [op_tests/triton_tests/utils/gdr_mtp_refs.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/gdr_mtp_refs.py)
- [op_tests/triton_tests/utils/kda_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/kda_ref.py)

## Performance and validation

Validate a sequence of updates against the reference, not just one isolated call. Include preparation, triangular solves and normalization for the chosen path.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](flydsl.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
