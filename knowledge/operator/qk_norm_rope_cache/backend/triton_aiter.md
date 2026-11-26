---
title: "Fused Q/K normalization, RoPE and cache writes: AITER triton"
kind: implementation_knowledge
operator: qk_norm_rope_cache
backend: triton
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Fused Q/K normalization, RoPE and cache writes — AITER triton

This fusion couples normalization with rotary position encoding and sometimes quantization, splitting and paged cache writes. Position ids, Q/K head counts, rotary convention and cache slot mappings are part of the contract.

Follow the public Python wrapper into the launchable Triton function. Tune only parameters that reach that launch; architecture-specific JSON and wrapper-derived split counts can override apparent defaults.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/triton/fusions/fused_qk_norm_rope_group_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_qk_norm_rope_group_quant.py) · **Recent**

Source documentation: Q/K RMSNorm, RoPE, fp8 group quant and SWA cache write. Each row of 512 values gets an RMSNorm. K also uses k_weight. Values [0, 448) go to fp8 e4m3 with one e8m0 scale for each 64 values. Values [448, 512) get RoPE on adjacent value pairs with cos/sin[positions]. The K row of token t also goes to row swa_dest_rows[t] of the SWA cache. Args: q: [T, H, 512] contiguous bf16. H must be a multiple of HEADS_PER_PROGRAM in the config. kv: [T, 512] bf16.

Symbols: `_get_config`, `fused_qk_norm_rope_group_quant`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/fusions/fused_reduce_qk_norm_rope_swa_write.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/fused_reduce_qk_norm_rope_swa_write.py) · **Recent**

Source documentation: Fused split-K reduce + per-head weighted RMSNorm + RoPE tail on Q, weighted RMSNorm + RoPE tail on KV (+ optional SWA). Replaces the (split-K reduce → q-norm → q-rope → kv-norm → kv-rope → swa-write) post-GEMM chain. Shapes: q: [M, N] or [num_splitk, M, N] where N = num_local_heads * head_dim. The 3D form lets the caller pass an unreduced split-K GEMM output; the kernel sums across num_splitk while loading.

Symbols: `_pick_block_size_m`, `fused_reduce_qk_norm_rope_swa_write`.

Parameters referenced in this file: `BLOCK_SIZE_M`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py) · **Recent**

Source documentation: RMSNorm -> GPT-J RoPE -> MXFP4 -> paged cache store of one key per token. MXFP4 is e2m1 values, two per byte (low nibble first), with one e8m0 scale per 32 values chosen as 2^ceil(log2(amax / 6)), so no value saturates. Args: k: [T, head_size] bf16, the key before its norm. Rows past slot_mapping.numel() are not read. positions: [T] token positions.

Symbols: `k_norm_rope_mxfp4_cache`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/triton/rope/fused_qk_norm_rope_cached.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/fused_qk_norm_rope_cached.py) · **Recent**

Source documentation: Per-head RMSNorm then partial NeoX RoPE on q and k, in place. q[t, h] = rope(rmsnorm(q[t, h], q_weight), cos_sin_cache[t]) k[t, h] = rope(rmsnorm(k[t, h], k_weight), cos_sin_cache[t]) Written for diffusion transformers, where the rotated subspace is a fraction of the head that is neither the whole head nor half of it -- 96 of 128 for MiniMax-H3 -- which the existing rope ops do not cover, and where there is no KV cache to write through.

Symbols: `fused_qk_norm_rope_cached`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`, `num_warps`.

### `aiter/ops/triton/rope/fused_qk_norm_rope_gate_fp8_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/fused_qk_norm_rope_gate_fp8_quant.py) · **Recent**

Source documentation: Fused Q/K RMSNorm, RoPE, gate extraction, and grouped FP8 Q/K/V quantization. This operator is intentionally cache-independent. Sequence metadata and the decode-prefix boundary are explicit inputs so framework integrations retain ownership of scheduling, cache updates, and execution ordering.

Symbols: `FusedQKNormRopeGateFp8QuantOutput`, `_validate_gfx950_fp8`, `_require_inner_contiguous`, `fused_qk_norm_rope_gate_fp8_quant`.

Parameters referenced in this file: `BLOCK_B`, `BLOCK_D`, `BLOCK_T`, `num_stages`, `num_warps`.

### `aiter/ops/triton/rope/fused_qkv_split_qk_norm_rope_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/fused_qkv_split_qk_norm_rope_cache.py) · **Recent**

Source documentation: Pick Triton token tile BLOCK_T for :func:fused_qkv_split_qk_norm_rope_cache. Same heuristic as vLLM input_quant_fp8.calc_rows_per_block: scale tile size with sequence length and SM/CU count so the launch grid stays in a sensible range — use ceil(T / (2 * sm_count)) rounded up to the next power of two, then cap (the vLLM quant path caps at 4 rows; this kernel keeps a larger upper bound due to per-token work and tl.arange(0, BLOCK_T) register pressure).

Symbols: `infer_rope_cache_triton_block_t`, `fused_qkv_split_qk_norm_rope_cache`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_D_HALF`, `BLOCK_SIZE`, `BLOCK_T`, `num_warps`.

### `aiter/ops/triton/rope/fused_qkv_split_qk_rope.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/fused_qkv_split_qk_rope.py) · **Recent**

Symbols: `fused_qkv_split_qk_rope`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_D_HALF`, `BLOCK_T`, `num_warps`, `waves_per_eu`.

### `aiter/ops/triton/rope/q_rope_mxfp4_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/rope/q_rope_mxfp4_quant.py) · **Recent**

Source documentation: GPT-J RoPE on the last rope_dim dims -> MXFP4 query, plus scaled weights. MXFP4 is e2m1 values, two per byte (low nibble first), with one e8m0 scale per 32 values chosen as 2^ceil(log2(amax / 6)), so no value saturates. Args: q: [T, H, head_size] bf16/fp16 (strided is fine, the last dim must be unit-stride). positions: [T] token positions. cos_sin_cache: [max_pos, rope_dim], the cos half then the sin half. weights: optional [T, H] per-head weights.

Symbols: `_launch_config`, `q_rope_mxfp4_quant`.

Parameters referenced in this file: `BLOCK_H`, `num_warps`.

### `aiter/ops/triton/_triton_kernels/fusions/fused_reduce_qk_norm_rope_swa_write.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/fused_reduce_qk_norm_rope_swa_write.py) · **Recent**

Source documentation: Fused split-K reduce + per-head weighted RMSNorm + RoPE (tail) on Q, per-row weighted RMSNorm + RoPE (tail) on KV (+ optional SWA KV write). Grid: (cdiv(M, BLOCK_SIZE_M), num_local_heads + 1). Each program tile handles BLOCK_SIZE_M tokens.

Symbols: `_get_config`, `_batched_rmsnorm_op`, `_batched_unit_rope`, `_fused_reduce_qk_norm_rope_swa_write_kernel`.

Parameters referenced in this file: `BLOCK_M`, `BLOCK_SIZE_M`.

### `aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py) · **Recent**

Symbols: `_k_norm_rope_mxfp4_cache_kernel`.

### `aiter/ops/triton/_triton_kernels/rope/fused_qk_norm_rope_cached.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/fused_qk_norm_rope_cached.py) · **Recent**

Symbols: `_fused_qk_norm_rope_cached_kernel`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_H`.

### `aiter/ops/triton/_triton_kernels/rope/fused_qk_norm_rope_gate_fp8_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/fused_qk_norm_rope_gate_fp8_quant.py) · **Recent**

Source documentation: Fused Q/K RMSNorm, RoPE, gate extraction, and FP8 quantization kernels.

Symbols: `_find_sequence_for_token`, `persistent_qk_norm_rope_gate_token_amax_kernel`, `v_token_amax_kernel`, `segmented_qkv_partial_amax_kernel`, `segmented_qkv_scale_kernel`, `quantize_qk_grouped_kernel`, `quantize_qk_grouped_offset_kernel`, `quantize_v_grouped_offset_kernel`.

Parameters referenced in this file: `BLOCK_B`, `BLOCK_D`, `BLOCK_T`.

### `aiter/ops/triton/_triton_kernels/rope/fused_qkv_split_qk_norm_rope_cache.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/fused_qkv_split_qk_norm_rope_cache.py) · **Recent**

Symbols: `_rms_norm`, `_rms_norm_returning_inv`, `_partial_neox_rotated`, `_partial_gptj_rotated`, `_fused_qkv_split_qk_norm_rope_cache_kernel`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_D_HALF`, `BLOCK_M`, `BLOCK_SIZE`, `BLOCK_T`.

### `aiter/ops/triton/_triton_kernels/rope/fused_qkv_split_qk_rope.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/fused_qkv_split_qk_rope.py) · **Recent**

Symbols: `_fused_qkv_split_qk_rope_kernel`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_D_HALF`, `BLOCK_T`.

### `aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py) · **Recent**

Symbols: `_q_rope_mxfp4_quant_kernel`.

Parameters referenced in this file: `BLOCK_H`.

## Configuration references

Use the matching wrapper/config loader. These files are tuning choices, not standalone proof of a performance gain.

- [aiter/ops/triton/configs/gfx950/triton/fusions/fused_reduce_qk_norm_rope_swa_write/DEFAULT.json](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/configs/gfx950/triton/fusions/fused_reduce_qk_norm_rope_swa_write/DEFAULT.json)

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_gate_fp8_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_gate_fp8_quant.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_norm_rope_group_quant.py)
- [op_tests/op_benchmarks/triton/bench_fused_reduce_qk_norm_rope_swa_write.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_reduce_qk_norm_rope_swa_write.py)
- [op_tests/triton_tests/fusions/test_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_qk_norm_rope_group_quant.py)
- [op_tests/triton_tests/fusions/test_fused_reduce_qk_norm_rope_swa_write.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/fusions/test_fused_reduce_qk_norm_rope_swa_write.py)
- [op_tests/triton_tests/rope/test_fused_qk_norm_rope_cached.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_fused_qk_norm_rope_cached.py)
- [op_tests/triton_tests/rope/test_fused_qk_norm_rope_gate_fp8_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_fused_qk_norm_rope_gate_fp8_quant.py)
- [op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/rope/test_fused_qkv_split_qk_rope.py)
- [op_tests/multigpu_tests/test_fused_qknorm_ar.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_qknorm_ar.py)
- [op_tests/test_flydsl_qk_norm_rope_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_qk_norm_rope_quant.py)
- [op_tests/test_fused_qk_norm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm.py)
- [op_tests/test_fused_qk_norm_mrope_cache_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_mrope_cache_quant.py)
- [op_tests/test_fused_qk_norm_rope_1way_perhead.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_1way_perhead.py)
- [op_tests/test_fused_qk_norm_rope_2way_perhead.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_2way_perhead.py)
- [op_tests/test_fused_qk_norm_rope_cache_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_cache_quant.py)
- [op_tests/test_fused_qk_norm_rope_group_quant.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qk_norm_rope_group_quant.py)
- [op_tests/test_fused_qknorm_idxrqknorm.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_fused_qknorm_idxrqknorm.py)

## Performance and validation

Validate Q and K outputs plus modified cache locations. Include position tails, partial rotary dimensions and the quantization scale layout.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
