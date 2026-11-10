# TP8/EP1 conc1: V10 → V11

[Recovered-image comparisons](../recovered_image_history.md) · [Full matrix](../configuration_history.md)

**vLLM commits:** [335dd77b0916](https://github.com/vllm-project/vllm/commit/335dd77b09161c5d63139b4e7ae51bb9928305b1) → [4fab5a074758](https://github.com/vllm-project/vllm/commit/4fab5a07475822c6eaa971dceb2e2220d0446808). [Kernel implementation browser](../kernel_implementations.md). Container file identity is retained separately from the reported framework commit.

[Original pair and exact launch changes](../pairs/tp8_ep1_conc1_off_2026-11-03_to_2026-11-09.md).

| Endpoint | Docker image | Point / exact CI | (p90 interactivity, output/chip) | Server artifact |
| --- | --- | --- | ---: | --- |
| Before: 2026-11-03 | `vllm/vllm-openai-rocm:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1` | [447088](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/41394320841/job/125113160330) | (684.931507, 23.65412) | `13122361038` |
| After: 2026-11-09 | `vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808` | [447806](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42335637576/job/128309287110) | (704.225352, 24.00536) | `13566143184` |

Interactivity **+2.82%**; throughput/chip **+1.48%**. E2E measurements do not isolate kernel causality.

## Exact log evidence and measurement windows

Both artifacts are joined through their upload IDs in their own endpoint job logs. Server messages are aligned against that job's AIPerf/CI phase boundaries; timestamps are treated as UTC, and events within one second of a boundary are marked uncertain. AIPerf `profiling` means E2E measurement.

| Endpoint | E2E measurement start | E2E measurement end |
| --- | --- | --- |
| Before | 2026-11-02T05:48:33.193000 | 2026-11-02T06:48:33.193000 |
| After | 2026-11-07T06:47:38.996000 | 2026-11-07T07:47:38.997000 |

| Diagnostic | Before observed phases | After observed phases | Interpretation |
| --- | --- | --- | --- |
| mono_dispatch | server_setup_or_graph_warmup: 2 diagnostic messages | server_setup_or_graph_warmup: 2 diagnostic messages | Reached guarded mono path; no sustained coverage fraction |
| markov_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 1 diagnostic messages | Reached guarded sampling path; first message only |
| quant_gemm_dispatch | server_setup_or_graph_warmup: 41 diagnostic messages | server_setup_or_graph_warmup: 41 diagnostic messages | Explicit HIT includes row/weight/tile shape; logs once per shape |
| jit_compile | benchmark_warmup: 9 diagnostic messages; e2e_measurement: 2 diagnostic messages | benchmark_warmup: 8 diagnostic messages; e2e_measurement: 1 diagnostic messages; server_setup_or_graph_warmup: 1 diagnostic messages | Named compilation events; not launches or total GPU time |

Selected exact log lines (all retained events and source mappings are in [runtime facts](../evidence/recovered_runtime_facts.json)):

- **Before, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g11_agg_w0.out:784`: `(Worker_TP0 pid=3846495) INFO 11-02 05:41:41 [mono_decode.py:693] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`
- **Before, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g11_agg_w0.out:619`: `(Worker_TP0 pid=3846495) WARNING 11-02 05:41:17 [rocm_block32_gemm.py:989] tiny-M fused MXFP8 quant+GEMM HIT (M=5, N=1792, K=5120, packed tile 8x16x1024 k_pack=2)`
- **Before, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g11_agg_w0.out:1636`: `(Worker_TP0 pid=3846495) WARNING 11-02 05:50:45 [jit_monitor.py:143] Triton kernel JIT compilation during inference: _block32_packed_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **Before, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g11_agg_w0.out:1953`: `(Worker_TP0 pid=3846495) WARNING 11-02 05:57:10 [jit_monitor.py:143] FlyDSL JIT compilation during inference: gemm_a16w16_gfx950 (kernels=hgemm_bf16_t32x32x64x8_kst4_w2x1x1_gm0_bias0_lnt_pft_0). This causes a latency spike; consider extending warmup to cover this shape/config.`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:821`: `(Worker_TP0 pid=1691192) INFO 11-07 06:40:16 [mono_decode.py:693] DSv4.1 mono decode draft layer active (decode steps of <= 48 rows)`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:681`: `(Worker_TP0 pid=1691192) INFO 11-07 06:39:43 [fused_markov.py:274] DSpark fused Markov sampling: 5 steps, vocabulary 129280, rank 256, 505 blocks`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:622`: `(Worker_TP0 pid=1691192) WARNING 11-07 06:39:38 [rocm_block32_gemm.py:989] tiny-M fused MXFP8 quant+GEMM HIT (M=5, N=1792, K=5120, packed tile 8x16x1024 k_pack=2)`
- **After, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g16_agg_w0.out:1912`: `(Worker_TP0 pid=1691192) WARNING 11-07 06:59:58 [jit_monitor.py:143] FlyDSL JIT compilation during inference: launch_radix_topk_one_block (kernels=radix_topk_one_block_decode_short_k512_b256_w64_v0_s0). This causes a latency spike; consider extending warmup to cover this shape/config.`

## Applicable source changes

| Shared kernel/body/fusion record | Applicability / log limit |
| --- | --- |
| [Tiny-M activation quantization fused into block32 GEMM](../recovered_changes/v10_to_v11.md#tiny_quant) | Same tiny-M/packed/contiguous activation guards; no source-level change in this file. Logging once per shape prevents using the HIT count as call frequency. |
| [DSpark Markov sampling on gfx950](../recovered_changes/v10_to_v11.md#markov) | gfx950 --monokernel, at most five request rows, supported sampler/dtypes. All observed first Markov messages precede E2E measurement; they do not prove sustained coverage at higher concurrency. |
| [SWA metadata helper admitted on gfx950](../recovered_changes/v10_to_v11.md#swa_meta) | --monokernel, pure decode, ≤32 decode token rows, supported causal or DSpark non-causal metadata. No specific HIT message was found; source eligibility only, not measured launch elimination. |
| [Ordinary delayed mHC device bodies unchanged](../recovered_changes/v10_to_v11.md#mhc_same) | Ordinary/fallback mHC path only; mono bypasses it. The added gfx942-only ASM seam in aiter/ops/mhc.py is not attributed to MI355X. |
| [mHC overlap and Engram gate sharding](../recovered_changes/v10_to_v11.md#overlap_engram) | gfx950 ordinary/fallback path; mHC ≤128 tokens, hc_mult4, no ubatching and FULL capture (not eager/piecewise). No direct overlap/Engram HIT log was found. |
| [MXFP4 Q/K producer and score bodies retained](../recovered_changes/v10_to_v11.md#indexer_same) | Index-source work; still outside persistent mono launches. JIT warnings are observations of compilation, not every invocation. |
| [Ordinary sparse MLA body/dispatch changes](../recovered_changes/v10_to_v11.md#sparse_body) | Ordinary/fallback sparse attention. Exact _sparse_mla/_sparse_mla_reduce JIT messages identify compiled candidates, not the full serving trace. |
| [FlyDSL radix top-k body and observed symbol](../recovered_changes/v10_to_v11.md#topk) | TP8 c1 measured-phase JIT at 06:59:58 in the V11 endpoint; consult its exact phase window. Distinguish wrapper symbol and generated device symbol. |

Pinned source evidence: [inventory](../evidence/recovered_source_inventory.json), [reviewed rules](../evidence/recovered_kernel_reviews.json), [V10→V11 source diff](../evidence/recovered_source_diffs/v10_to_v11.diff). Installation paths are preserved in the inventory; canonical comparison paths above normalize the earlier deepseek_v4_1 directory name.

## Named JIT kernels: observed sets and body comparisons

This table lists every symbol mentioned by the selected endpoints' JIT diagnostics. Missing messages do not imply missing execution: the monitor activates after server warmup and uses warning_once. A changed file can contain an unchanged symbol; AST comparisons below separate those cases. A name match is a source candidate, not proof of a unique binary implementation.

| Symbol | Before logged phases | After logged phases | Candidate source / symbol-body relation |
| --- | --- | --- | --- |
| `_block32_packed_kernel` | e2e_measurement | not logged | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: identical_ast [before source](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L188) [after source](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L188) |
| `_block32_tiled_kernel` | benchmark_warmup | benchmark_warmup | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: identical_ast [before source](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L107) [after source](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L107) |
| `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast [before source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53) [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53) |
| `_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast [before source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230) [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230) |
| `_mxfp8_quant_triton_kernel` | benchmark_warmup | benchmark_warmup | `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`: identical_ast [before source](../kernel_sources/blobs/82a1a2f893b0d9d9/mxfp8_utils.py#L88) [after source](../kernel_sources/blobs/327be3cf903c648b/mxfp8_utils.py#L116) |
| `_mxfp8_wo_a_bmm_kernel` | benchmark_warmup | benchmark_warmup | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`: identical_ast [before source](../kernel_sources/blobs/f82fc79debbae469/rocm_aiter_mla_sparse.py#L1937) [after source](../kernel_sources/blobs/8b5e4ad1de1d9eaa/rocm_aiter_mla_sparse.py#L1965) |
| `_pa_mqa_logits_mxfp4_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`: identical_ast [before source](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392) [after source](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392) |
| `_q_rope_mxfp4_quant_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py`: identical_ast [before source](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py#L20) [after source](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py#L20) |
| `_sparse_mla` | benchmark_warmup | server_setup_or_graph_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`: changed_ast [before source](../kernel_sources/blobs/02c9ac2762fb8e61/sparse_mla.py#L2163) [after source](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2163) |
| `_sparse_mla_reduce` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`: changed_ast [before source](../kernel_sources/blobs/02c9ac2762fb8e61/sparse_mla.py#L2799) [after source](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2798) |
| `gemm_a16w16_gfx950` | e2e_measurement | not logged | `aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py`: identical_ast [before source](../kernel_sources/blobs/3357253816feb6d8/gemm_a16w16_gfx950.py#L1280) [after source](../kernel_sources/blobs/398666319e41f98b/gemm_a16w16_gfx950.py#L1291) |
| `launch_radix_topk_one_block` | not logged | e2e_measurement | `aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py`: identical_ast [before source](../kernel_sources/blobs/3e9e7f4386532334/radix_topk_one_block.py#L1463) [after source](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py#L1469) |

The [structured pair record](../data/recovered_pair_records.json) retains complete generated kernel names (including FlyDSL variant parameters), candidate source paths, file hashes, symbol code/AST hashes and line ranges. Triton JIT log labeling can include Gluon implementations; source decorators distinguish them.

No GPU trace, kernel time ranking, per-step launch count, or causal kernel speedup is claimed. Admission, graph capture, batching, quick-reduce quantization and synthetic DSpark acceptance are retained in the original CI evidence and can affect E2E deltas.
