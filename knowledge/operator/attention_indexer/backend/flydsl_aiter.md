---
title: "Attention index scoring and selection: AITER flydsl"
kind: implementation_knowledge
operator: attention_indexer
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Attention index scoring and selection — AITER flydsl

These kernels compute or select scores used to choose sparse-attention tokens/pages. Paged and contiguous caches, FP8/MXFP4 storage and prefill/decode schedules are separate contracts.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/dcp_topk_merge.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/dcp_topk_merge.py) · **Recent**

Source documentation: FlyDSL DCP decode TopK merge interface.

Symbols: `_get_launcher`, `_validate`, `flydsl_dcp_topk_merge`.

### `aiter/ops/flydsl/fp8_mqa_logits_kernels.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/fp8_mqa_logits_kernels.py) · **Recent**

Source documentation: High-level FlyDSL FP8 MQA logits (DeepSeek lightning indexer) API. Wraps the gfx942/gfx950 kernel builders in aiter.ops.flydsl.kernels.mqa_logits.fp8_mqa_logits with: - The kernel-variant registry and shape-adaptive auto-selection (_auto_variant / KERNEL_VARIANTS / DEFAULT_VARIANT). - A build cache keyed by shape/variant/dtype-conversion flags (compile_fp8_mqa_logits).

Symbols: `_SplitPolicy`, `_split_policy`, `_device_cu_count`, `_auto_num_splits`, `_mk_builder`, `_register_variants`, `_parse_variant`, `_auto_variant`, `_resolve_variant`, `compile_fp8_mqa_logits`, `flydsl_fp8_mqa_logits`.

Parameters referenced in this file: `num_buffers`, `waves_per_eu`.

### `aiter/ops/flydsl/kernels/dcp_topk_merge.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/dcp_topk_merge.py) · **Recent**

Source documentation: FlyDSL DCP decode TopK merge: global threshold select + owned-slot emit.

Symbols: `_f32_to_ord`, `_make_storage`, `build_dcp_topk_merge_module`.

### `aiter/ops/flydsl/kernels/minimax_m3_index_score.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/minimax_m3_index_score.py) · **Recent**

Source documentation: MiniMax-M3 lightning-indexer decode block-score kernel (FlyDSL, gfx950). For each request b, each 128-token page p and each (query row, index head), score[head, row, p] = max over t in [0,128) of dot(K[page, t, :], Q[row, head, :]) * scale with a per-query causal cutoff. K is the fp8 index cache fused_qknorm_idxrqknorm writes; Q is bf16.

Symbols: `fp8_dwords_to_bf16`, `fragment_helpers`, `bf16_dwords_to_fp8`, `_decode_depth_table`, `_cp_blocks`, `feature_tiles`, `reduce_lds_bytes`, `work_chunk`, `_grid_chunks`, `_spread_fits`, `_cu_count`, `geometry`, `kernel_name`, `build_index_score`, `_get`, `_plan_spread_map`, `_run_spread_map`, `_plan_chunk_map`, `_zero_rows`, `_run_chunk_map`, `work_map_size`, `work_map_capacity`, `build_work_map`, `_make_map`, `_span`, `_validate_tensor`, `_validate_arch`, `_validate_rows`, `_validate_map`, `_validate_bounds`, `_validate_metadata`, `index_score_supported`, `alloc_score`, `score_flydsl`.

Parameters referenced in this file: `BLOCK_B`, `BLOCK_K`, `BLOCK_N`.

### `aiter/ops/flydsl/kernels/minimax_m3_index_score_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/minimax_m3_index_score_prefill.py) · **Recent**

Source documentation: MiniMax-M3 lightning-indexer PREFILL block-score kernel (FlyDSL). Same operator as the decode scorer next door, different machine. For request b, 128-token page p, query row r and index head h: score[h, r, p] = max over t in [0,128) of dot(K[page, t, :], Q[r, h, :]) * scale masked to t's global position <= r's. Replaces ATOM's Triton _index_block_score_kernel.

Symbols: `lds_bytes`, `feature_tiles`, `_auto_tile_q`, `_auto_ppw`, `_auto_waves`, `geometry`, `chunk_blocks`, `kernel_name`, `build_prefill_score`, `_get`, `alloc_score`, `score_prefill_flydsl`.

### `aiter/ops/flydsl/kernels/mqa_logits/fp8_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mqa_logits/fp8_mqa_logits.py) · **Recent**

Source documentation: FP8 MQA logits (DeepSeek lightning indexer) -- FlyDSL gfx942/gfx950 kernel builders. Compute for each query row m and KV position n inside that row's window [cu_starts[m], cu_ends[m]):: logits[m, n] = sum_h ReLU(<Q[m, h, :], K[n, :]> * kv_scale[n]) * weights[m, h] Low-level @flyc.kernel/@flyc.jit builders only. The public host-facing API (flydsl_fp8_mqa_logits, variant selection/registry) lives in aiter.ops.flydsl.fp8_mqa_logits_kernels.

Symbols: `_make_out_row_t`, `_load_pack_i32x8`, `_make_weight_copy`, `_load_row_weights`, `_emit_col_sum`, `_emit_row_neg_inf_fill`, `_no_gemm_kwargs`, `_identity_scale_kwargs`, `_frag_i64`, `_frag_i32x8`, `MfmaAtom`, `_build_kernel_mfma_r_w`, `_build_kernel_mfma_lds_pipe`.

Parameters referenced in this file: `num_buffers`.

### `aiter/ops/flydsl/kernels/mqa_logits/fp8_paged_mqa_logits.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mqa_logits/fp8_paged_mqa_logits.py) · **Recent**

Source documentation: Paged FP8 MQA logits (decode) -- gfx950 ragged LDS mapping. The implementation lives in fp8_paged_mqa_logits_gfx950. This module re-exports the public entry so existing import paths keep working.

### `aiter/ops/flydsl/kernels/mqa_logits/fp8_paged_mqa_logits_gfx950.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mqa_logits/fp8_paged_mqa_logits_gfx950.py) · **Recent**

Source documentation: gfx950 H32/D128/KVB64 ragged paged FP8 MQA-logits kernel. One 8-wave workgroup owns one (sequence, SplitKV slice). Waves 0..3 compute the four N=16 tiles in one page and replay every live query row. Waves 4..7 stream the next page directly into the idle LDS bank.

Symbols: `udiv`, `umod`, `uceildiv`, `_wait_all`, `_guarded`, `_lds_f32x4`, `_lds_f32`, `_concat_i32x4`, `_lds_i32x4`, `_load_q_pack_lds`, `_load_k_pack_lds`, `_load_weight_frag_lds`, `_mfma_score`, `_reduce_scores`, `_make_out_row_view`, `device_cu_count`, `_build_kernel`, `_compile`, `flydsl_fp8_paged_mqa_logits`.

Parameters referenced in this file: `waves_per_eu`.

### `aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4.py) · **Recent**

Source documentation: Decode/varctx FP4 paged MQA logits (gfx950). parallel_unit_num is the persistent-grid CTA count; when None it is auto-derived (cudagraph-safe, no device→host sync) as batch * next_n * ceil(max_seq_len / block_k), which is a multiple of next_n and >= batch*next_n by construction. Pass a smaller explicit value to trade parallelism for fewer no-op CTAs.

Symbols: `build_pa_mqa_logits_fp4_module`, `compile_pa_mqa_logits_fp4`, `flydsl_pa_mqa_logits_fp4`.

Parameters referenced in this file: `num_warps`.

### `aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_common.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_common.py) · **Recent**

Symbols: `_i32_buffer`, `_load_vec4_i32`, `_varctx_cta_info_kernel`, `compute_varctx_schedule`.

Parameters referenced in this file: `BLOCK_B`, `BLOCK_S`.

### `aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py) · **Recent**

Source documentation: Compute the persistent-grid schedule for ragged-prefill MQA logits. Pass cta_info_out (a fixed [parallel_unit_num, CTA_INFO_WIDTH] int32 buffer) to write the schedule into a stable address (CUDAGraph decode: the captured kernel replays from this pointer while build() refreshes its contents). Returns (safe, cta_info, parallel_unit_num), safe being the [1] int32 split factor the schedule was built with.

Symbols: `compute_prefill_schedule`, `_RowPlan`, `_row_plan`, `_row_plan_torch`, `_prefill_row_plan_kernel`, `_prefill_cta_info_kernel`, `build_pa_mqa_logits_fp4_prefill_module`, `compile_pa_mqa_logits_fp4_prefill`, `flydsl_pa_mqa_logits_fp4_prefill`, `_varqlen_windows_kernel`, `compute_varqlen_windows`, `flydsl_pa_mqa_logits_fp4_varqlen`.

Parameters referenced in this file: `BLOCK_P`, `BLOCK_T`, `num_warps`.

### `aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_rowgroup.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_rowgroup.py) · **Recent**

Source documentation: FP4 paged MQA logits (gfx950), a group of a sequence's query rows a wave. pa_mqa_logits_fp4 scores one query row a workgroup, so a speculative step's rows each read the whole context.

Symbols: `page_bytes`, `pack_kv_cache`, `pack_q_scales`, `_permlane_swap`, `_vop`, `_global_i32`, `_addr_i32_buffer`, `_read_all`, `_lane_xor`, `_wave_sum`, `_wave_max`, `_wave_inclusive_scan`, `_ceil_div`, `_readlane`, `_relu2`, `_fma`, `_lds_barrier`, `_add`, `build_pa_mqa_logits_fp4_rowgroup_module`, `compile_pa_mqa_logits_fp4_rowgroup`, `default_rows_per_wave`, `_light_waves`, `_single_wave_workgroups`, `default_kv_nt`, `default_workgroups`, `Fp4MqaPlan`, `make_fp4_mqa_plan`, `flydsl_pa_mqa_logits_fp4_rowgroup`.

Parameters referenced in this file: `num_warps`, `tile_m`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_msa_block_select_cp.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_msa_block_select_cp.py)
- [op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fp8_mqa_logits.py)
- [op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_pa_mqa_logits_mxfp4.py)
- [op_tests/test_flydsl_dcp_topk_merge.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_dcp_topk_merge.py)
- [op_tests/test_flydsl_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_mqa_logits.py)
- [op_tests/test_flydsl_fp8_paged_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_fp8_paged_mqa_logits.py)
- [op_tests/test_flydsl_minimax_m3_index_score.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_minimax_m3_index_score.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4.py)
- [op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py)
- [op_tests/test_msa_block_select.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_msa_block_select.py)
- [op_tests/test_pa_mqa_logits_mxfp4_opus.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_mxfp4_opus.py)
- [op_tests/test_pa_mqa_logits_offset.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_offset.py)
- [op_tests/triton_tests/attention/test_fp8_mqa_logits.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_fp8_mqa_logits.py)
- [op_tests/triton_tests/attention/test_pa_mqa_logits_2d_context_lens.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_mqa_logits_2d_context_lens.py)
- [op_tests/triton_tests/attention/test_pa_mqa_logits_mxfp4.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/attention/test_pa_mqa_logits_mxfp4.py)
- [op_tests/triton_tests/utils/pa_mqa_logits_mxfp4_ref.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/utils/pa_mqa_logits_mxfp4_ref.py)

## Performance and validation

Check selected indices as well as logits, including ties and masked pages. Keep scoring and top-k costs visible when evaluating sparse attention.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
