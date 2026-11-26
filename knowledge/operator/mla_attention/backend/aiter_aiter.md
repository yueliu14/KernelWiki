---
title: "Multi-head latent attention: AITER aiter"
kind: implementation_knowledge
operator: mla_attention
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Multi-head latent attention — AITER aiter

MLA combines latent KV storage with non-positional and positional query/key components. Cache representation and projection placement can move substantial work between GEMM and attention.

This page describes AITER public wrappers and dispatch. AITER is the implementation provider; inspect the called Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS path before attributing performance to a backend.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/mla.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/mla.py) · **Recent**

Source documentation: num_kv_splits for mla_decode_fwd_v4_nm (host only, no device work). num_seqs / num_heads are the decode call's qo_indptr.shape[0] - 1 and q.size(1); kv_len is the per-seq KV length to plan for (the longest one the call can see). Minimizes the latency model above where the arch has coefficients, and is the wrapper's own occupancy-only pick elsewhere.

Symbols: `_flydsl_mla_reduce_supported`, `_flydsl_mla_reduce_enabled`, `_mla_decode_reduce_v1_dispatch`, `_fwd_kernel_stage2_asm`, `get_meta_param`, `_v4_nm_pick_num_kv_splits`, `get_mla_v4_nm_num_kv_splits`, `MlaV4NmPsWorkspace`, `get_mla_v4_nm_ps_workspace`, `_persistent_mla_decode_max_batch`, `_fold_seqlen_indptr`, `_use_persistent_mla_decode`, `mla_decode_fwd_ds32`, `_mla_cpu_scale_on_device`, `_mla_scale_on_device`, `_check_mla_qh128_inputs`, `mla_decode_fwd_qh128_asm`, `_mla_decode_fwd_kv_shuffled`, `_check_mla_qh64_shuffled_inputs`, `_mla_page_csr_indptr`, `_mla_page_csr_indices`, `mla_decode_prepare_page_csr`, `mla_decode_fwd`, `mla_v40_decode_fwd`, `mla_prefill_fwd`, `mla_prefill_ps_fwd`, `_mla_prefill_reduce_kernel`, `mla_prefill_reduce_triton`, `mla_prefill_reduce`, `mla_decode_fwd_v4_nm`, `mla_decode_fwd_v4_nm_ps`.

Parameters referenced in this file: `BLOCK_DIM`, `BLOCK_DV`, `num_stages`, `num_warps`, `waves_per_eu`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

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
- [op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_mla_prefill_qkv_fp8.py)
- [op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_fused_qk_cat_and_cache_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode.py)
- [op_tests/op_benchmarks/triton/bench_mla_decode_rope.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/op_benchmarks/triton/bench_mla_decode_rope.py)
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

[GEAK implementation notes](aiter.md) retain older implementation explanations and dated performance evidence. Check their source revision before transferring a claim to this snapshot.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
