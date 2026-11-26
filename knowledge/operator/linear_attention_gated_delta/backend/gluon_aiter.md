---
title: "Linear attention and gated delta rules: AITER gluon"
kind: implementation_knowledge
operator: linear_attention_gated_delta
backend: gluon
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Linear attention and gated delta rules — AITER gluon

Prefill commonly uses chunked state updates while decode applies a recurrent update. State layout, gate transforms, convolution state and in-place update order must match the wrapper.

Gluon code exposes layouts and asynchronous data movement explicitly. Keep architecture-specific layouts, buffer counts and synchronization tied to the selected implementation; a similarly named Triton wrapper may select Gluon only for particular conditions.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/chunk_kda/prepare.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/chunk_kda/prepare.py) · **Recent**

Source documentation: One (chunk, head) per program: l2norm, gate cumsum, the intra-chunk products and the (I + L)^-1 solve, written as the walk's workspace (qg, w, u, aqk, kg_t, decay).

Symbols: `_get_config`, `chunk_kda_prepare_kernel`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/chunk_kda/walk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/chunk_kda/walk.py) · **Recent**

Source documentation: One (sequence, head, BV rows of V) per program: the [BV, K] state stays in WMMA accumulators across the chunks, fed by a NUM_STAGES-deep TDM ring.

Symbols: `_get_config`, `_load_chunk`, `_chunk`, `_step`, `_store_o`, `chunk_kda_walk_kernel`.

### `aiter/ops/triton/_gluon_kernels/gfx1250/attention/kda_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx1250/attention/kda_decode.py) · **Recent**

Symbols: `_state_layout`, `_smem_layout`, `_k_row_layout`, `_v_row_layout`, `_state_row`, `_fetch_token`, `_stage_token`, `_staged_token`, `_conv_qkv_load`, `_conv_qkv_roll`, `_conv_qkv_finish`, `_process_token`, `fused_recurrent_kda_packed_decode_kernel`, `get_kda_config`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/chunk_kda/prepare.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/chunk_kda/prepare.py) · **Recent**

Source documentation: One (chunk, head) per program: l2norm, gate cumsum, the intra-chunk products and the (I + L)^-1 solve, written as the walk's workspace (qg, w, u, aqk, kg_t, decay).

Symbols: `_get_config`, `chunk_kda_prepare_kernel`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/chunk_kda/walk.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/chunk_kda/walk.py) · **Recent**

Source documentation: One (sequence, head, V slice[, group]) per program, the state in MFMA accumulators. PASS 0 walks the whole sequence; PASS 1 and PASS 2 walk the chunk groups around the scan.

Symbols: `_get_config`, `_fetch`, `_stash`, `_chunk`, `_split_a`, `_split_b`, `_store_o`, `_ring`, `_step`, `_tile_off`, `_frag_off`, `_load_state`, `chunk_kda_walk_kernel`, `_scan_load`, `_scan_step`, `chunk_kda_scan_kernel`.

### `aiter/ops/triton/_gluon_kernels/gfx950/attention/kda_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/attention/kda_decode.py) · **Recent**

Symbols: `_conv_load`, `_token`, `_finish`, `fused_recurrent_kda_packed_decode_kernel`, `get_kda_config`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_decode_qkvz.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_decode_qkvz.py) · **Recent**

Source documentation: Gluon kernels for the fused Qwen3-Next GDN decode step (gfx950/CDNA4). One eight-wave workgroup owns a token/key-head group's convolution history and both FP32 value-head states, fusing: the packed qkvz/ba split, a depthwise causal conv1d with bias and SiLU, the delta-rule gating, the FP32 recurrent state update, a gated RMSNorm with a SiLU gate, and an optional per-head group-128 FP8 quantization epilogue. Partial-column prefetch overlaps state reads with QKV preparation.

Symbols: `_exp`, `_sigmoid`, `_quantization_scale`, `_convolve`, `_head_dynamics`, `_load_epilogue_inputs`, `_wave_maximum_pair`, `_wave_sum_pair`, `_finish_pair`, `_load_state`, `_store_state`, `_normalize_qk`, `_state_dot`, `_quadrants`, `_fused_decode`, `_recurrent_tile`, `_fused_decode_tiled`, `_conv_channels`, `_decode_group`.

### `aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_prefill_qkvz.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/gated_delta_net/fused_gdn_prefill_qkvz.py) · **Recent**

Source documentation: Gluon kernels for the fused Qwen3-Next GDN *prefill* step (gfx950/CDNA4). A fused Gluon op replaces the prefill chain causal_conv1d_split_qkv -> fused_gdn_gating -> chunk_gated_delta_rule -> gated_rmsnorm_fp8_group_quant with six tight launches that fuse the work intra-kernel and drop the intermediate HBM round-trips; the epilogue launch emits the per-head group-128 FP8 activations a block-FP8 out_proj consumes directly (no separate quant kernel).

Symbols: `_add`, `_packed_channel`, `_prepare_inputs_tiled`, `_update_conv_state`, `_matrix_product`, `_store_packed_operand`, `_load_packed_operand`, `_inverse16`, `_assemble_inverse`, `_inverse32`, `_four_diagonal_inverses`, `_block_inverse`, `_chunk_offsets`, `_find_sequence`, `_chunk_location`, `_prepare_chunk_factors`, `_propagate_chunks`, `_output_norm_quant`, `_prepare_segment_maps`, `_propagate_segments`, `_render_segment_chunks`, `_gate_values`, `_prepare_qkv_window`, `_prepare_gates`, `_multiply`, `_dot_f32`, `_dot_bf16`, `_chunk_bounds`, `_invert_diagonal_sixteen`, `_solve_two_halves`, `_chunk_transform`, `_load_recurrent_factors`, `_chunk_state_rows`, `_load_key_decay`, `_load_tail_key`, `_advance_chunk`, `_recurrence_ids`, `_build_segments`, `_build_segments_reverse`, `_build_segments_stacked`, `_prefix_segments`, `_gated_rms`, `_quantize_group`, `_chunk_output_quant`, `_state_and_core`, `_normalize_quantize`, `_projection_channel`, `_load_history`, `_prepare_tokens`, `_update_conv_state_b6_15`, `_chunk_dot`, `_multiply_b6_15`, `_chunk_coefficients`, `_prepare_chunks_full`, `_prepare_chunks_compact`, `_recurrence_full`, `_recurrence_compact`, `_group_bounds`, `_build_group_maps`, `_propagate_group_maps`, `_evaluate_groups`, `_gated_rms_quant`, `_convolve_sliding`, `_update_conv_pool`, `_add_prefix`, `_multiply_prefix`, `_write_offsets`, `_float_matrix_product`, `_store_chunk_rhs`, `_chunk_rhs_offset`, `_accumulator_offset`, `_store_chunk_accumulator`, `_load_chunk_u`, `_load_matrix_rhs`, `_load_scaled_qk`, `_solve_diagonal_blocks`, `_prepare_chunk_matrices`, `_store_core_output`, `_chunk_delta_recurrence`, `_scan_chunk_states`, `_emit_chunk_outputs`, `_store_packed_linear`, `_load_packed_linear`, `_block_transforms`, `_propagate_block_states`, `_finish_blocks`, `_norm_and_quantize`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/kimi_delta_attn/flash_kda_k1.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/kimi_delta_attn/flash_kda_k1.py) · **Recent**

Symbols: `_add`, `_exp`, `_exp2`, `_sigmoid`, `_l2norm`, `_via_lds`, `_dot_f32`, `k1_prepare_gluon`, `gluon_k1_prepare`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/_gluon_kernels/gfx950/kimi_delta_attn/flash_kda_k2.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_gluon_kernels/gfx950/kimi_delta_attn/flash_kda_k2.py) · **Recent**

Source documentation: The layout set, derived from one decision: the state stays in registers. instr_shape[0:2] = [16, 16] with transposed=False is what makes an MFMA accumulator a legal B operand, so the state can be the accumulator of dot(kr^T, U) and the B operand of dot(kd, h) without a round trip. kw names the dots that contract over C and kw_big the one that contracts over K; the two accumulator distributions are the same, since an accumulator's layout is set by M and N and not by K.

Symbols: `build_layouts`, `_sigmoid`, `_recur`, `_kr_operand`, `k2_ab_fused_gluon`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1250/gluon/attention/chunk_kda/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/attention/chunk_kda/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/gluon/attention/kda_decode/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/gluon/attention/kda_decode/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/attention/chunk_kda/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/attention/chunk_kda/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/gluon/attention/kda_decode/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/gluon/attention/kda_decode/DEFAULT.json)

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

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
