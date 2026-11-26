---
title: "Prefill and full attention: AITER aiter"
kind: implementation_knowledge
operator: attention_prefill_fmha
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Prefill and full attention — AITER aiter

Tiled attention interleaves QK products, online softmax and probability-value products. Inference paths require the correct causal or variable-length mask, head layout and KV-cache contract. Use forward entry points with dropout disabled.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

This page selects LLM inference paths. Shared upstream files and test suites can also contain training code; use their forward cases with gradients and dropout disabled. Explicit gradient-only implementations, tests and configurations are omitted.

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/attention.py) · **Recent**

Source documentation: Paged attention forward pass with automatic kernel selection. ASM is favored for int8 kv caches, for short ctx_len, or when the workload exceeds the heuristic thresholds for larger ctx_len values. PA is normally using per tensor quant and this is what has been tested, however, per head quant can be supported as well in principle, but not tested.

Symbols: `gen_pa_fwd_native_fake`, `gen_pa_fwd_asm`, `pa_fwd_naive`, `_pa_fwd_asm`, `pa_fwd_asm`, `_should_use_asm_kernel`, `paged_attention_common`, `gen_pa_ps_fwd_asm`, `_pa_ps_fwd_asm`, `pa_ps_fwd_asm`, `_pa_decode_bf16_asm`, `pa_decode_bf16_asm`, `pa_reduce_v1`, `_attention_ps_reduce_asm`, `pa_persistent_fwd`, `paged_attention_rocm`, `paged_attention_v1`, `paged_attention_ragged`, `mla_ps64_qh128_fp8_asm_fwd`, `mla_decode_stage1_asm_fwd`, `mla_ps1_fp8_asm_fwd`, `mla_prefill_asm_fwd`, `get_pa_metadata_info_v1`, `get_pa_metadata_v1`, `get_ps_metadata_info_v1`, `get_ps_metadata_v1`, `mla_prefill_ps_asm_fwd`, `get_mla_decode_fwd_occupancy`, `get_mla_decode_fwd_max_splits`, `get_mla_metadata_info_v1`, `_get_mla_metadata_v1_impl`, `get_mla_metadata_v1`, `mla_reduce_v1`, `decode_update_mla_metadata_v1_kernel`, `decode_update_mla_metadata_v1`, `hk_mla_v32_decode_fwd_mi3xx`, `hk_mla_v32_decode_fwd`, `hk_mla_v40_decode_fwd_mi3xx`, `hk_mla_v40_decode_fwd`, `opus_mla_decode_mxfp8_fwd`, `opus_mla_decode_fwd`.

### `aiter/ops/mha.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mha.py) · **Recent**

Source documentation: Public wrapper for the OPUS gfx950 bf16 dense (batch) forward (D=128 and D_QK=192/D_V=128). q/k/v are dense bshd [B, S, H, D]; allocates out ([B, S, H_q, D_v]) if needed and forwards. The kernel applies softmax_scale to Q·K^T internally and handles GQA fan-out. lse is an output buffer for the log-sum-exp of the scaled scores ([B, H_q, S] float32, natural log; rows that see no keys get -inf), filled when supplied and allocated here when return_lse is set.

Symbols: `_fmha_kv_byte_extent_ge_u32`, `cmdGenFunc_mha_fwd`, `common_mha_fwd_fake_tensors`, `gen_mha_fwd_fake_tensors`, `mha_fwd`, `gen_mha_fwd_native_splitkv_fake_tensors`, `_mha_fwd_native_splitkv`, `mha_fwd_native_splitkv`, `gen_fmha_v3_fwd_fake_tensors`, `fmha_v3_fwd`, `gen_fmha_fwd_bf16_opus_fwd_fake`, `_fmha_fwd_bf16_opus_fwd`, `fmha_fwd_bf16_opus_fwd`, `fmha_fwd_bf16_opus_varlen_fwd`, `_fmha_fwd_with_sink_asm`, `fmha_fwd_with_sink_asm`, `_fmha_fwd_with_sink_varlen_asm`, `fmha_fwd_with_sink_varlen_asm`, `_fmha_fwd_mxfp8_asm`, `fmha_fwd_mxfp8_asm`, `cmdGenFunc_mha_varlen_fwd`, `gen_mha_varlen_fwd_fake_tensor`, `mha_varlen_fwd`, `gen_fmha_v3_varlen_fwd_fake_tensor`, `fmha_v3_varlen_fwd`, `cmdGenFunc_mha_batch_prefill`, `maybe_contiguous`, `_native_splitkv_heuristic`, `_flash_attn_forward`, `FlashAttnFunc`, `flash_attn_func`, `_flash_attn_varlen_forward`, `FlashAttnVarlenFunc`, `flash_attn_varlen_func`, `mha_batch_prefill_fake_tensors`, `mha_batch_prefill`, `_mha_batch_prefill`, `mha_batch_prefill_func`, `flash_attn_fp8_pertensor_func`, `flash_attn_varlen_fp8_pertensor_func`.

### `aiter/ops/mha_v4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mha_v4.py) · **Recent**

Source documentation: MHA v4 recipe selection, validation, and launch APIs. Raw BF16 BSHD operands are delegated to producers in :mod:mha_v4_quant. Format and scale-mode IDs are part of the launcher ABI. Optional block-sparse execution uses a boolean tile mask on the raw API and a ragged LUT triple on the packed API; the work table is built inside the sparse custom op.

Symbols: `_mha_v4_sparse_work_table_fake`, `mha_v4_sparse_work_table`, `AttentionFormat`, `AttentionPack`, `AttentionScaleMode`, `_RawRecipeKind`, `_RawRecipePlan`, `native_fp8_format`, `mha_v4_kv_tile`, `_mha_v4_kv_tile_from_manifest`, `_is_fp8_format`, `_validate_format_contract`, `_validate_pack_contract`, `scale_modes_for_formats`, `_validate_scale_recipe`, `_raw_scale_recipe`, `_resolve_raw_recipe`, `_packed_lut_triple`, `_block_mask_to_lut`, `_fmha_v4_fwd_fake`, `_fmha_v4_fwd`, `_mha_v4_fwd_launch`, `_mha_v4_fwd_launch_fake`, `_fmha_v4_fwd_sparse_fake`, `_fmha_v4_fwd_sparse`, `_mha_v4_fwd_sparse_launch`, `_mha_v4_fwd_sparse_launch_fake`, `_check_lse_capable`, `_check_varlen_capable`, `_empty_lse`, `_validate_gqa_heads`, `mha_v4_packed`, `_launch_mxfp4_coalesced`, `_launch_mxfp4_coalesced_fake`, `_launch_mxfp6`, `_launch_mxfp6_fake`, `_k_mean`, `_restore_k_mean_in_lse`, `_validate_mha_v4_raw_inputs`, `mha_v4`.

### `aiter/ops/mha_v4_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/mha_v4_quant.py) · **Recent**

Source documentation: Quantization and packed-layout producers for MHA v4.

Symbols: `mha_v4_q_multiplier`, `_rotate_activation_hd128`, `_or_empty`, `rotate_activation_hd128`, `rotate_activation_mxfp8_quant`, `rotate_activation_mxfp6_quant`, `rotate_activation_mxfp6_quant_k`, `_quantize_v_mxfp6_fp6_p_hip`, `rotate_activation_mxfp4_quant`, `rotate_activation_mxfp4_quant_k`, `_quantize_v_mxfp4_hip`, `_quantize_v_mxfp4_fp6_p_hip`, `_validate_bshd_hd128`, `_quantize_per_tensor`, `quantize_int8`, `_quantize_int8_fake`, `quantize_fp8`, `_quantize_fp8_fake`, `quantize_fp8_rotated`, `block_scale_storage`, `query_block_scale`, `_plain_block_scale`, `quantize_mxfp8_q`, `_quantize_mxfp8_q_fake`, `quantize_mxfp8_k`, `_quantize_mxfp8_k_fake`, `quantize_mxfp4_q`, `_quantize_mxfp4_q_fake`, `mxfp4_k_raw_buffer_size`, `mxfp4_v_tiles`, `mxfp4_v_padded_sequence`, `mxfp4_v_raw_buffer_size`, `mxfp6_v_tiles`, `mxfp6_v_padded_sequence`, `mxfp6_v_raw_buffer_size`, `quantize_mxfp4_k`, `_quantize_mxfp4_k_fake`, `mxfp4_k_view`, `mxfp6_k_view`, `quantize_mxfp6_q`, `_quantize_mxfp6_q_fake`, `quantize_mxfp6_k`, `_quantize_mxfp6_k_raw_fake`, `quantize_v_fp8`, `_quantize_v_fp8_fake`, `_mxfp4_v_buffers`, `quantize_v_mxfp4_fp6_p`, `_quantize_v_mxfp4_fp6_p_raw_fake`, `quantize_v_mxfp6`, `_quantize_v_mxfp6_fake`, `quantize_v_mxfp6_fp6_p`, `_quantize_v_mxfp6_fp6_p_fake`, `mxfp4_v_view`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_K`, `BLOCK_N`, `BLOCK_SIZE`, `num_stages`, `num_warps`.

### `aiter/test_mha_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/test_mha_common.py) · **Recent**

Source documentation: FlashAttention stores the S matrix in a different way. Arguments: S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded) query_padding_mask: (batch_size, seqlen_q_rounded) key_padding_mask: (batch_size, seqlen_k_rounded)

Symbols: `convert_flash_attn_S_to_softmax`, `attn_bias_from_alibi_slopes`, `generate_random_padding_mask`, `generate_qkv`, `quantize_fp8_per_bh`, `construct_local_mask`, `block_attn_mask_to_token_mask`, `attention_ref_block_sparse`, `attention_ref`, `attention_ref_with_tol`, `opus_ref_lse`, `opus_check_lse`, `sparse_mla_dsv4_ref`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/cpp/mha/compile.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/mha/compile.py)
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
- [op_tests/op_benchmarks/flydsl/bench_mha_fwd.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/flydsl/bench_mha_fwd.py)
- [op_tests/op_benchmarks/triton/bench_attn_res.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_attn_res.py)
- [op_tests/op_benchmarks/triton/bench_deepgemm_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_deepgemm_attention.py)
- [op_tests/op_benchmarks/triton/bench_extend_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_extend_attention.py)
- [op_tests/op_benchmarks/triton/bench_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mha.py)
- [op_tests/op_benchmarks/triton/bench_pa_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_prefill.py)
- [op_tests/op_benchmarks/triton/compare_mha.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/compare_mha.py)
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

## Historical knowledge

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
