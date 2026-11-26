---
title: "Paged and unified attention decode: AITER triton"
kind: implementation_knowledge
operator: attention_decode_paged
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Paged and unified attention decode — AITER triton

Paged attention must preserve the block table, sequence lengths, causal mask, query-head to KV-head mapping and cache dtype. Split-KV exposes parallelism at the cost of partial buffers and a reduction pass.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/attention/lean_atten_paged.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/lean_atten_paged.py) · **Recent**

Source documentation: Lean Attention + Paged Attention ================================ This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480, enhanced with Paged Attention from https://arxiv.org/abs/2309.06180, for the decode phase. Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.

Symbols: `persistent_lean_attention_paged`, `get_num_splits_and_buffer_sizes`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/pa_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/pa_decode.py) · **Recent**

Source documentation: Paged attention decode with automatic V1/V2 dispatch and quantization support. If the arch has a PA-DECODE config, bf16/fp16 caches with scalar scales use V1 only up to v1_max_partitions partitions. Other cases use V1 up to 8192 tokens when there is one partition or num_seqs * num_q_heads > 512, and V2 otherwise. Args: output (torch.Tensor): Pre-allocated output with shape (num_seqs, num_q_heads, head_dim).

Symbols: `paged_attention_decode`, `paged_attn_decode_v1`, `paged_attn_decode_v2`, `paged_attn_decode_v1_per_token_quant`, `paged_attn_decode_v2_per_token_quant`.

Parameters referenced in this file: `num_stages`, `waves_per_eu`.

### `aiter/ops/triton/attention/paged_attention_output_gate.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/paged_attention_output_gate.py) · **Recent**

Source documentation: Paged decode attention with a sigmoid output gate and an optional group-128 FP8 epilogue, for Qwen3-Next-style full-attention layers. Replaces three launches -- paged decode attention, then out * sigmoid(gate), then the per-group activation quant the next GEMM needs -- with two, by folding the gate and the quant into the split-K merge the attention already performs. Why this is a new op rather than a flag on an existing one: no aiter paged-decode op has this KV layout.

Symbols: `_gluon_supported`, `paged_attention_output_gate_supported`, `_short_body_selected`, `paged_attention_output_gate_group_fp8_quant`.

Parameters referenced in this file: `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/attention/unified_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/attention/unified_attention.py) · **Recent**

Symbols: `_is_gluon_available`, `_UAParams`, `_gfx950_gluon_supported`, `_gfx950_split_params`, `_gfx950_split_grid`, `use_2d_kernel`, `unified_attention`, `is_2d_gluon_available`, `is_3d_gluon_available`, `is_reduce_gluon_available`, `_unified_attention_2d_triton`, `_unified_attention_3d_triton`, `_reduce_segments_triton`, `_unified_attention_2d_gfx1250`, `_unified_attention_3d_gfx1250`, `_reduce_segments_gfx1250`, `_unified_attention_gfx950`, `_unified_attention_2d_gfx950`, `_unified_attention_3d_gfx950`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_Q`, `BLOCK_SCALES_SIZE`, `BLOCK_SIZE`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/attention/lean_atten_paged.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/lean_atten_paged.py) · **Recent**

Source documentation: Lean Attention + Paged Attention ================================ This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480, enhanced with Paged Attention from https://arxiv.org/abs/2309.06180, for the decode phase. Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.

Symbols: `find_group`, `la_persistent_paged`, `_attn_lean_tile`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_N`.

### `aiter/ops/triton/_triton_kernels/attention/pa_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/pa_decode.py) · **Recent**

Symbols: `_get_dispatch_config`, `_paged_attn_decode_v1_wo_dot_kernel`, `_paged_attn_decode_v1_w_dot_kernel`, `_paged_attn_decode_v2_wo_dot_kernel`, `_paged_attn_decode_v2_wo_dot_reduce_kernel`, `_paged_attn_decode_v2_w_dot_kernel`, `_paged_attn_decode_v2_w_dot_reduce_kernel`, `_paged_attn_decode_v1_wo_dot_kernel_per_token_quant`, `_paged_attn_decode_v1_w_dot_kernel_per_token_quant`, `_paged_attn_decode_v2_wo_dot_kernel_per_token_quant`, `_paged_attn_decode_v2_wo_dot_reduce_kernel_per_token_quant`, `_paged_attn_decode_v2_w_dot_kernel_per_token_quant`, `_paged_attn_decode_v2_w_dot_reduce_kernel_per_token_quant`.

### `aiter/ops/triton/_triton_kernels/attention/unified_attention.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/attention/unified_attention.py) · **Recent**

Symbols: `fast_exp`, `cdiv_fn`, `apply_softcap`, `find_seq_idx`, `kernel_unified_attention_2d`, `kernel_unified_attention_3d`, `reduce_segments`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_Q`, `BLOCK_SIZE`, `num_stages`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py) · **Recent**

Source documentation: Returns configs for both the splitK kernel and reduce kernel. Returns: (splitk_configs, reduce_config): Tuple of config lists for each kernel

Symbols: `get_fwd_decode_configs`, `_attn_fwd_inner`, `_fwd_kernel_splitK`, `_splitK_reduce`, `cast_uint32_to_half2`, `dequantize`, `quantize_kv_int4`, `dequantize_kv_fp16`, `get_split_k`, `attention_forward_decode_triton_impl`.

Parameters referenced in this file: `BLOCK_DMODEL`, `BLOCK_DMODEL_PACKED`, `BLOCK_M`, `BLOCK_M_OPTIONS`, `BLOCK_N`, `BLOCK_N_OPTIONS`, `BLOCK_N_PER_SPLIT`, `BLOCK_SIZE_K`, `num_stages`, `num_warps`, `splitK`, `split_k`, `waves_per_eu`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx1100/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1100/triton/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1150/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1150/triton/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1151/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1151/triton/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1200/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1200/triton/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1201/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1201/triton/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx1250/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx1250/triton/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx942/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx942/triton/attention/unified_attention/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/pa_decode/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/pa_decode/DEFAULT.json)
- [aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/attention/unified_attention/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_decode.py)
- [op_tests/op_benchmarks/triton/bench_paged_attention_output_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_paged_attention_output_gate.py)
- [op_tests/op_benchmarks/triton/bench_unified_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_unified_attention.py)
- [op_tests/triton_tests/attention/test_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_decode.py)
- [op_tests/triton_tests/attention/test_paged_attention_output_gate.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_paged_attention_output_gate.py)
- [op_tests/triton_tests/attention/test_unified_attention.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_unified_attention.py)
- [op_tests/triton_tests/test_pa_decode_gluon.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/test_pa_decode_gluon.py)
- [op_tests/cpp/test_pa_decode_gluon_aot.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/cpp/test_pa_decode_gluon_aot.py)
- [op_tests/test_flydsl_pa_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_decode.py)
- [op_tests/test_pa_decode_bf16_asm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_decode_bf16_asm.py)

## Performance and validation

Record page size, KV length distribution, head dimensions, number of query tokens and split count. Time the full split/reduce path; verify multi-token decode causality.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Historical knowledge

[GEAK implementation notes](triton.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
