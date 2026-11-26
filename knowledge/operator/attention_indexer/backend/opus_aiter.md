---
title: "Attention index scoring and selection: AITER opus"
kind: implementation_knowledge
operator: attention_indexer
backend: opus
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Attention index scoring and selection — AITER opus

These kernels compute or select scores used to choose sparse-attention tokens/pages. Paged and contiguous caches, FP8/MXFP4 storage and prefill/decode schedules are separate contracts.

OPUS implementations combine architecture-aware policies and native kernel instances. Follow policy and dispatch code to the launched entry and verify that the intended instance engages.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/opus/pa_mqa_logits_mxfp4.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/opus/pa_mqa_logits_mxfp4.py) · **Recent**

Source documentation: MXFP4 paged MQA logits for DeepSeek-style sparse attention (OPUS kernels), gfx950 and gfx1250. Per query row r over a window [s, e): out[r, s:e] = sum_H( relu(Q[r] . K^T) * weight[r] ) * weight_scale Prefill and decode share one launch over a per-tile schedule built on device.

Symbols: `MqaLogitsVariant`, `MqaLogitsBuffers`, `_check_variant_tables`, `_fwd_sched_raw`, `_build_tiles_raw`, `_build_sched_raw`, `_variants_for`, `_default_variant`, `_as_i32`, `_max_tiles_for`, `_sched_slots`, `_cta_info`, `_cu_tiles`, `_compute_tiles`, `_compute_schedule`, `_launch`, `MqaLogitsPlan`, `pa_mqa_logits_mxfp4_variants`, `_as_variant`, `pa_mqa_logits_mxfp4_block_table_width`, `_arch_of`, `_check_layout`, `pa_mqa_logits_mxfp4_plan_buffers`, `pa_mqa_logits_mxfp4_plan`, `pa_mqa_logits_mxfp4`.

### `csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_kernels.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_kernels.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_opus.h`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_opus.h) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

### `csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_pybind.cu`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/opus_mqa_logits/pa_mqa_logits_mxfp4/pa_mqa_logits_mxfp4_pybind.cu) · **Recent**

Native implementation or support code; inspect its callers and bindings to identify the selected specialization.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/test_pa_mqa_logits_mxfp4_opus.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_pa_mqa_logits_mxfp4_opus.py)
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
