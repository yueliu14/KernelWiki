---
title: "Compressed attention: AITER flydsl"
kind: implementation_knowledge
operator: compressed_attention
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Compressed attention — AITER flydsl

Compression changes the representation consumed by attention; fused paths combine preparation and attention steps. Compression ratio, positional features and cache layout must match the source wrapper.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/kernels/fused_compress_attn.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fused_compress_attn.py) · **Recent**

Source documentation: Fused Compressor boundary kernel for V4 attention (FlyDSL). Drop-in replacement for the Triton _fused_compress_attn_kernel in atom/model_ops/v4_kernels/fused_compress.py. Per-boundary online-softmax pool over K=2*RATIO source positions, RMSNorm (fp32), GPT-J interleaved RoPE, optional FP8 (ue8m0 scale + MFMA 16x16 preshuffle) cache scatter. Two kernel families share this file: - Legacy single-wave (_build_kernel): 1 wave64 per boundary, K iters serialized.

Symbols: `_load_f32_vec`, `_load_bf16_vec_then_f32`, `_build_kernel`, `_build_kernel_ksplit`, `hca_per_n_config`, `compile_flydsl_fused_compress_attn`, `csa_ksplit_num_waves`, `compile_flydsl_fused_compress_attn_ksplit`, `flydsl_fused_compress_attn`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_TH`, `BLOCK_THREADS`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fused_compress_attn_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fused_compress_attn_common.py) · **Recent**

Source documentation: Shared flydsl emitters for the V4 compress-attn kernels. Single source of truth for the FP8 group_fp8 (V4 nm-asm) scatter tail used by both the CSA single-kernel (fused_compress_attn) and the HCA 2-kernel (fused_compress_attn_hca) paths, on wave64 (VEC=8) and wave32 (VEC=16). Keeping it here avoids drift between the two kernels' fp8 entry layouts (they MUST stay byte-identical so the V4 nm-asm sparse-attn reader sees one layout).

Symbols: `_ptr_at_byte_off`, `_fp8_const`, `_fexp_f32`, `_wave_reduce_add`, `_wave_reduce_max`, `_split_state`, `group_fp8_mx_dtype`, `state_slot_byte_offset`, `block_base_bytes_i64`, `_global_ptr`, `emit_group_fp8_nm_asm_scatter`.

### `aiter/ops/flydsl/kernels/fused_compress_attn_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fused_compress_attn_gfx1250.py) · **Recent**

Source documentation: Fused Compressor boundary kernel for V4 attention -- **gfx1250 (RDNA4, wave32)**. Port of fused_compress_attn.py (wave64) to gfx1250 wave32.

Symbols: `_load_f32_vec`, `_load_bf16_vec_then_f32`, `_wave_reduce_add`, `_wave_reduce_max`, `_build_kernel`, `_build_kernel_ksplit`, `hca_per_n_config_gfx1250`, `compile_flydsl_fused_compress_attn_gfx1250`, `csa_ksplit_num_waves_gfx1250`, `compile_flydsl_fused_compress_attn_ksplit_gfx1250`, `flydsl_fused_compress_attn_gfx1250`.

Parameters referenced in this file: `BLOCK_D`, `BLOCK_TH`, `BLOCK_THREADS`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fused_compress_attn_hca.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fused_compress_attn_hca.py) · **Recent**

Source documentation: HCA-path FlyDSL compress + norm+rope+scatter kernels (2-kernel split). Targeted optimization for V4-Pro HCA Main: D=512, RD=64, ratio=128, overlap=False.

Symbols: `_build_compress_forward_kernel`, `_build_norm_rope_scatter_kernel`, `compile_hca_compress_forward`, `compile_hca_norm_rope_scatter`, `flydsl_hca_compress_attn`, `flydsl_hca_compress_forward`.

Parameters referenced in this file: `BLOCK_TH`, `BLOCK_THREADS`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/fused_compress_attn_hca_gfx1250.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/fused_compress_attn_hca_gfx1250.py) · **Recent**

Source documentation: HCA-path compress + norm+rope+scatter kernels -- **gfx1250 (RDNA4, wave32)**. Port of fused_compress_attn_hca.py (wave64) to gfx1250 wave32. Key differences: - BLOCK_THREADS = 32 (wave32) - SLICE = 32 (head_dim elements per block) - VEC = SLICE_SZ / 32 (vs /64) - Kernel B: D=512 -> VEC=16, requires split load/store paths - Kernel names suffixed with "w32" See fused_compress_attn_hca.py for the original wave64 documentation.

Symbols: `_build_compress_forward_kernel`, `_NormRopeScatterCfg`, `_norm_rope_scatter_cfg`, `_emit_norm_rope_scatter`, `_build_norm_rope_scatter_kernel`, `_sync_counter`, `compile_hca_compress_forward_gfx1250`, `compile_hca_norm_rope_scatter_gfx1250`, `flydsl_hca_compress_attn_gfx1250`.

Parameters referenced in this file: `BLOCK_TH`, `BLOCK_THREADS`, `waves_per_eu`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_flydsl_compress_attn.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_compress_attn.py)

## Performance and validation

Validate the compression result and final attention output. Include preparation cost when comparing against a decomposed reference.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
