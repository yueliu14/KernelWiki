# TP4/EP1 conc1: V09 → V11

[Recovered-image comparisons](../recovered_image_history.md) · [Full matrix](../configuration_history.md)

**vLLM commits:** [e68efc685ab1](https://github.com/vllm-project/vllm/commit/e68efc685ab1e6f1eda004ced7157e3ca411da2f) → [4fab5a074758](https://github.com/vllm-project/vllm/commit/4fab5a07475822c6eaa971dceb2e2220d0446808). [Kernel implementation browser](../kernel_implementations.md). Container file identity is retained separately from the reported framework commit.

[Original pair and exact launch changes](../pairs/tp4_ep1_conc1_off_2026-10-20_to_2026-11-09.md).

| Endpoint | Docker image | Point / exact CI | (p90 interactivity, output/chip) | Server artifact |
| --- | --- | --- | ---: | --- |
| Before: 2026-10-20 | `vllm/vllm-openai-rocm:nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f` | [445393](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39295252770/job/118206654303) | (588.235294, 45.71031) | `12228583807` |
| After: 2026-11-09 | `vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808` | [447804](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42335637576/job/128292105363) | (675.675676, 47.42262) | `13561965484` |

Interactivity **+14.86%**; throughput/chip **+3.75%**. E2E measurements do not isolate kernel causality.

## Exact log evidence and measurement windows

Both artifacts are joined through their upload IDs in their own endpoint job logs. Server messages are aligned against that job's AIPerf/CI phase boundaries; timestamps are treated as UTC, and events within one second of a boundary are marked uncertain. AIPerf `profiling` means E2E measurement.

| Endpoint | E2E measurement start | E2E measurement end |
| --- | --- | --- |
| Before | 2026-10-19T22:24:20.162000 | 2026-10-19T23:24:20.163000 |
| After | 2026-11-07T04:04:29.114000 | 2026-11-07T05:04:29.115000 |

| Diagnostic | Before observed phases | After observed phases | Interpretation |
| --- | --- | --- | --- |
| mono_dispatch | server_setup_or_graph_warmup: 2 diagnostic messages | server_setup_or_graph_warmup: 2 diagnostic messages | Reached guarded mono path; no sustained coverage fraction |
| markov_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 1 diagnostic messages | Reached guarded sampling path; first message only |
| quant_gemm_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 46 diagnostic messages | Explicit HIT includes row/weight/tile shape; logs once per shape |
| jit_compile | benchmark_warmup: 7 diagnostic messages | benchmark_warmup: 12 diagnostic messages; e2e_measurement: 1 diagnostic messages | Named compilation events; not launches or total GPU time |

Selected exact log lines (all retained events and source mappings are in [runtime facts](../evidence/recovered_runtime_facts.json)):

- **Before, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g09_agg_w0.out:528`: `(Worker_TP0 pid=4047031) INFO 10-19 22:06:45 [mono_decode.py:435] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:472`: `(Worker_TP0 pid=1566566) INFO 11-07 03:57:16 [mono_decode.py:693] DSv4.1 mono decode draft layer active (decode steps of <= 48 rows)`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:408`: `(Worker_TP0 pid=1566566) INFO 11-07 03:56:46 [fused_markov.py:274] DSpark fused Markov sampling: 5 steps, vocabulary 129280, rank 256, 505 blocks`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:389`: `(Worker_TP0 pid=1566566) WARNING 11-07 03:56:43 [rocm_block32_gemm.py:989] tiny-M fused MXFP8 quant+GEMM HIT (M=5, N=1792, K=5120, packed tile 8x16x1024 k_pack=2)`
- **After, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:940`: `(Worker_TP0 pid=1566566) WARNING 11-07 04:16:56 [jit_monitor.py:143] FlyDSL JIT compilation during inference: launch_radix_topk_one_block (kernels=radix_topk_one_block_decode_short_k512_b256_w64_v0_s0). This causes a latency spike; consider extending warmup to cover this shape/config.`

## Applicable source changes

| Shared kernel/body/fusion record | Applicability / log limit |
| --- | --- |
| [Tiny-M activation quantization fused into block32 GEMM](../recovered_changes/v09_to_v11.md#tiny_quant) | Raw contiguous BF16/FP16, packed config, 0<M≤8. Client concurrency does not equal M. Most recorded HITs occur before measurement; every pair reports its own phase and shapes. |
| [DSpark Markov sampling on gfx950](../recovered_changes/v09_to_v11.md#markov) | gfx950 --monokernel, at most five request rows, supported sampler/dtypes. All observed first Markov messages precede E2E measurement; they do not prove sustained coverage at higher concurrency. |
| [SWA metadata helper admitted on gfx950](../recovered_changes/v09_to_v11.md#swa_meta) | --monokernel, pure decode, ≤32 decode token rows, supported causal or DSpark non-causal metadata. No specific HIT message was found; source eligibility only, not measured launch elimination. |
| [Ordinary delayed mHC device bodies unchanged](../recovered_changes/v09_to_v11.md#mhc_same) | Ordinary/fallback mHC path only; mono bypasses it. The added gfx942-only ASM seam in aiter/ops/mhc.py is not attributed to MI355X. |
| [mHC overlap and Engram gate sharding](../recovered_changes/v09_to_v11.md#overlap_engram) | gfx950 ordinary/fallback path; mHC ≤128 tokens, hc_mult4, no ubatching and FULL capture (not eager/piecewise). No direct overlap/Engram HIT log was found. |
| [MXFP4 Q/K producer and score bodies retained](../recovered_changes/v09_to_v11.md#indexer_same) | Index-source work; still outside persistent mono launches. JIT warnings are observations of compilation, not every invocation. |
| [Ordinary sparse MLA body/dispatch changes](../recovered_changes/v09_to_v11.md#sparse_body) | Ordinary/fallback sparse attention. Exact _sparse_mla/_sparse_mla_reduce JIT messages identify compiled candidates, not the full serving trace. |

Excluded from this configuration:

- TP2 ratio-2 internal mono tuning: excluded by TP guard.

Pinned source evidence: [inventory](../evidence/recovered_source_inventory.json), [reviewed rules](../evidence/recovered_kernel_reviews.json), [V09→V11 source diff](../evidence/recovered_source_diffs/v09_to_v11.diff). Installation paths are preserved in the inventory; canonical comparison paths above normalize the earlier deepseek_v4_1 directory name.

## Named JIT kernels: observed sets and body comparisons

This table lists every symbol mentioned by the selected endpoints' JIT diagnostics. Missing messages do not imply missing execution: the monitor activates after server warmup and uses warning_once. A changed file can contain an unchanged symbol; AST comparisons below separate those cases. A name match is a source candidate, not proof of a unique binary implementation.

| Symbol | Before logged phases | After logged phases | Candidate source / symbol-body relation |
| --- | --- | --- | --- |
| `_block32_splitk_reduce_kernel` | not logged | benchmark_warmup | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: identical_ast [before source](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L276) [after source](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L400) |
| `_block32_tiled_kernel` | benchmark_warmup | benchmark_warmup | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: identical_ast [before source](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L101) [after source](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L107) |
| `_mask_candidates_kernel` | benchmark_warmup | not logged | `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`: identical_ast [before source](../kernel_sources/blobs/604f588fcf96d546/candidate_blocks.py#L125) [after source](../kernel_sources/blobs/0990a1da8a18d9e5/candidate_blocks.py#L125) |
| `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast [before source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53) [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53) |
| `_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast [before source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230) [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230) |
| `_mxfp8_quant_triton_kernel` | not logged | benchmark_warmup | `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`: identical_ast [before source](../kernel_sources/blobs/03a16184c905b3a5/mxfp8_utils.py#L88) [after source](../kernel_sources/blobs/327be3cf903c648b/mxfp8_utils.py#L116) |
| `_mxfp8_wo_a_bmm_kernel` | benchmark_warmup | benchmark_warmup | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`: identical_ast [before source](../kernel_sources/blobs/fa6a9c47537abdc1/rocm_aiter_mla_sparse.py#L1743) [after source](../kernel_sources/blobs/8b5e4ad1de1d9eaa/rocm_aiter_mla_sparse.py#L1965) |
| `_pa_mqa_logits_mxfp4_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`: identical_ast [before source](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392) [after source](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392) |
| `_prepare_dflash_inputs_kernel` | benchmark_warmup | not logged | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`: changed_ast [before source](../kernel_sources/blobs/28fb34eedddf62ac/speculator.py#L624) [after source](../kernel_sources/blobs/61be7bf7dd274a71/speculator.py#L706) |
| `_q_rope_mxfp4_quant_kernel` | not logged | benchmark_warmup | `aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py`: identical_ast [before source](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py#L20) [after source](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py#L20) |
| `_sparse_mla` | not logged | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`: changed_ast [before source](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L2160) [after source](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2163) |
| `_sparse_mla_reduce` | not logged | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`: changed_ast [before source](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L2794) [after source](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2798) |
| `launch_radix_topk_one_block` | not logged | benchmark_warmup, e2e_measurement | `aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py`: identical_ast [before source](../kernel_sources/blobs/b58fab682f7e4a85/radix_topk_one_block.py#L1400) [after source](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py#L1469) |

The [structured pair record](../data/recovered_pair_records.json) retains complete generated kernel names (including FlyDSL variant parameters), candidate source paths, file hashes, symbol code/AST hashes and line ranges. Triton JIT log labeling can include Gluon implementations; source decorators distinguish them.

No GPU trace, kernel time ranking, per-step launch count, or causal kernel speedup is claimed. Admission, graph capture, batching, quick-reduce quantization and synthetic DSpark acceptance are retained in the original CI evidence and can affect E2E deltas.
