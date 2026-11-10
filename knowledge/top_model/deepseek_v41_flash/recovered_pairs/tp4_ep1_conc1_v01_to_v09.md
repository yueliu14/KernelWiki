# TP4/EP1 conc1: V01 → V09

[Recovered-image comparisons](../recovered_image_history.md) · [Full matrix](../configuration_history.md)

**vLLM commits:** [79a7108d9aea](https://github.com/vllm-project/vllm/commit/79a7108d9aea27ddab99ce1779290d300b17fc23) → [e68efc685ab1](https://github.com/vllm-project/vllm/commit/e68efc685ab1e6f1eda004ced7157e3ca411da2f). [Kernel implementation browser](../kernel_implementations.md). Container file identity is retained separately from the reported framework commit.

Supplemental comparison across V02–V08; intermediate measured points remain in the original matrix. This is not an adjacent-image kernel attribution.

| Endpoint | Docker image | Point / exact CI | (p90 interactivity, output/chip) | Server artifact |
| --- | --- | --- | ---: | --- |
| Before: 2026-09-12 | `vllm/vllm-openai-rocm:deepseekv41-flash-0909` | [441676](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34657748450/job/103454471563) | (35.727045, 8.05237) | `10287797870` |
| After: 2026-10-20 | `vllm/vllm-openai-rocm:nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f` | [445393](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39295252770/job/118206654303) | (588.235294, 45.71031) | `12228583807` |

Interactivity **+1546.47%**; throughput/chip **+467.66%**. E2E measurements do not isolate kernel causality.

## Exact log evidence and measurement windows

Both artifacts are joined through their upload IDs in their own endpoint job logs. Server messages are aligned against that job's AIPerf/CI phase boundaries; timestamps are treated as UTC, and events within one second of a boundary are marked uncertain. AIPerf `profiling` means E2E measurement.

| Endpoint | E2E measurement start | E2E measurement end |
| --- | --- | --- |
| Before | 2026-09-11T23:44:43.164346 | 2026-09-12T00:45:23.169877 |
| After | 2026-10-19T22:24:20.162000 | 2026-10-19T23:24:20.163000 |

| Diagnostic | Before observed phases | After observed phases | Interpretation |
| --- | --- | --- | --- |
| mono_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 2 diagnostic messages | Reached guarded mono path; no sustained coverage fraction |
| markov_dispatch | No matching message (not proof of non-use) | No matching message (not proof of non-use) | Reached guarded sampling path; first message only |
| quant_gemm_dispatch | No matching message (not proof of non-use) | No matching message (not proof of non-use) | Explicit HIT includes row/weight/tile shape; logs once per shape |
| jit_compile | benchmark_warmup: 8 diagnostic messages; e2e_measurement: 1 diagnostic messages; phase_boundary_uncertain: 2 diagnostic messages; server_setup_or_graph_warmup: 2 diagnostic messages | benchmark_warmup: 7 diagnostic messages | Named compilation events; not launches or total GPU time |

Selected exact log lines (all retained events and source mappings are in [runtime facts](../evidence/recovered_runtime_facts.json)):

- **Before, e2e_measurement**, `results/server.log:26652`: `(Worker_TP0 pid=439645) WARNING 09-11 23:44:45 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _resample_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g09_agg_w0.out:528`: `(Worker_TP0 pid=4047031) INFO 10-19 22:06:45 [mono_decode.py:435] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`

## Applicable source changes

| Shared kernel/body/fusion record | Applicability / log limit |
| --- | --- |
| [Whole-layer mono decode](../recovered_changes/v01_to_v09.md#mono) | This supplemental V01→V09 comparison is TP4/EP1. Decode-only, ≤48 token rows, suitable cache/routing/graph/residency. Initialization/active logs do not establish measurement-phase coverage. |
| [Ordinary mHC seam: reference to fused Triton producer](../recovered_changes/v01_to_v09.md#mhc) | gfx950, hc_mult=4, supported contiguous BF16 residual/FP32 gates and norm weights; first entry broadcast and ordinary fallback exceptions retained. |
| [MXFP4 indexer producer fusion](../recovered_changes/v01_to_v09.md#indexer) | Applies to index-source layers using this cache/config. Such work also remains external between mono launches. |
| [Ordinary sparse attention output epilogue](../recovered_changes/v01_to_v09.md#attention) | Ordinary or fallback attention, gfx950 and supported fp8_ds_mla cache. Runtime JIT diagnostics and source dispatch are evidence; no per-kernel duration is available. |

Pinned source evidence: [inventory](../evidence/recovered_source_inventory.json), [reviewed rules](../evidence/recovered_kernel_reviews.json), [V01→V09 source diff](../evidence/recovered_source_diffs/v01_to_v09.diff). Installation paths are preserved in the inventory; canonical comparison paths above normalize the earlier deepseek_v4_1 directory name.

## Named JIT kernels: observed sets and body comparisons

This table lists every symbol mentioned by the selected endpoints' JIT diagnostics. Missing messages do not imply missing execution: the monitor activates after server warmup and uses warning_once. A changed file can contain an unchanged symbol; AST comparisons below separate those cases. A name match is a source candidate, not proof of a unique binary implementation.

| Symbol | Before logged phases | After logged phases | Candidate source / symbol-body relation |
| --- | --- | --- | --- |
| `_block32_tiled_kernel` | not logged | benchmark_warmup | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: added_or_removed_or_not_captured [after source](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L101) |
| `_combined_routing` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py`: changed_ast [before source](../kernel_sources/blobs/ddcf4a31fb2f3519/routing.py#L202) [after source](../kernel_sources/blobs/6b0fa3185ddee065/routing.py#L191) |
| `_compute_local_logits_stats_kernel` | phase_boundary_uncertain | not logged | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast [before source](../kernel_sources/blobs/20ca2e5ac34e9ef9/rejection_sampler_utils.py#L193) [after source](../kernel_sources/blobs/44de886702efc818/rejection_sampler_utils.py#L208) |
| `_dequantize_and_gather_k_kernel` | benchmark_warmup | not logged | `vllm/models/deepseek_v4/common/ops/cache_utils.py`: added_or_removed_or_not_captured [before source](../kernel_sources/blobs/ea905f610912d3f0/cache_utils.py#L229); `vllm/models/deepseek_v41/common/ops/cache_utils.py`: identical_ast [before source](../kernel_sources/blobs/bb6bd4fa9ddcf7fb/cache_utils.py#L233) [after source](../kernel_sources/blobs/d61dec31b6baaa8f/cache_utils.py#L326) |
| `_engram_lookup_kernel` | server_setup_or_graph_warmup | not logged | `vllm/models/deepseek_v41/common/engram.py`: changed_ast [before source](../kernel_sources/blobs/ac787367615f7e8f/engram.py#L565) [after source](../kernel_sources/blobs/ae2adbbe51f6b26d/engram.py#L639) |
| `_gluon_fp8_mqa_logits_kernel` | benchmark_warmup | not logged | `aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py`: changed_ast [before source](../kernel_sources/blobs/f15a92f78ea5de03/fp8_mqa_logits.py#L777) [after source](../kernel_sources/blobs/118caafa5b345448/fp8_mqa_logits.py#L895) |
| `_mask_candidates_kernel` | not logged | benchmark_warmup | `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`: identical_ast [before source](../kernel_sources/blobs/0d53e4e6a60a3799/candidate_blocks.py#L108) [after source](../kernel_sources/blobs/604f588fcf96d546/candidate_blocks.py#L125) |
| `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel` | not logged | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: added_or_removed_or_not_captured [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53) |
| `_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel` | not logged | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: added_or_removed_or_not_captured [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230) |
| `_moe_gemm_a16w4` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py`: changed_ast [before source](../kernel_sources/blobs/b4d2f4d871692339/moe_op_gemm_a16w4.py#L105) [after source](../kernel_sources/blobs/9f56a8a8dbd59d49/moe_op_gemm_a16w4.py#L105) |
| `_mxfp8_linear_kernel` | server_setup_or_graph_warmup | not logged | `vllm/model_executor/kernels/linear/mxfp8/rocm_native.py`: identical_ast [before source](../kernel_sources/blobs/06cb57d761f07344/rocm_native.py#L28) [after source](../kernel_sources/blobs/10fb66edb5c5d11e/rocm_native.py#L30) |
| `_mxfp8_wo_a_bmm_kernel` | not logged | benchmark_warmup | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`: added_or_removed_or_not_captured [after source](../kernel_sources/blobs/fa6a9c47537abdc1/rocm_aiter_mla_sparse.py#L1743) |
| `_pa_mqa_logits_mxfp4_kernel` | not logged | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`: added_or_removed_or_not_captured [after source](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392) |
| `_prepare_dflash_inputs_kernel` | benchmark_warmup | benchmark_warmup | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`: changed_ast [before source](../kernel_sources/blobs/c14815574ba63bac/speculator.py#L488) [after source](../kernel_sources/blobs/28fb34eedddf62ac/speculator.py#L624) |
| `_reduce_grouped` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/reduce.py`: changed_ast [before source](../kernel_sources/blobs/e5cd99082d78ed69/reduce.py#L23) [after source](../kernel_sources/blobs/d7e01e59585e719f/reduce.py#L86) |
| `_rejection_kernel` | phase_boundary_uncertain | not logged | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast [before source](../kernel_sources/blobs/20ca2e5ac34e9ef9/rejection_sampler_utils.py#L484) [after source](../kernel_sources/blobs/44de886702efc818/rejection_sampler_utils.py#L502) |
| `_resample_kernel` | e2e_measurement | not logged | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast [before source](../kernel_sources/blobs/20ca2e5ac34e9ef9/rejection_sampler_utils.py#L696) [after source](../kernel_sources/blobs/44de886702efc818/rejection_sampler_utils.py#L749) |
| `_sum_bitmatrix_rows` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix.py`: identical_ast [before source](../kernel_sources/blobs/3c30a0e4926b9430/bitmatrix.py#L73) [after source](../kernel_sources/blobs/3c30a0e4926b9430/bitmatrix.py#L73); `vllm/third_party/triton_kernels/tensor_details/bitmatrix_details/sum_bitmatrix_rows.py`: identical_ast [before source](../kernel_sources/blobs/55284ed2ee708a35/sum_bitmatrix_rows.py#L49) [after source](../kernel_sources/blobs/55284ed2ee708a35/sum_bitmatrix_rows.py#L49) |
| `_topk` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py`: identical_ast [before source](../kernel_sources/blobs/2ffa638e1454ab3e/topk.py#L186) [after source](../kernel_sources/blobs/2ffa638e1454ab3e/topk.py#L186); `vllm/model_executor/layers/logits_processor.py`: identical_ast [before source](../kernel_sources/blobs/6b0603d67b0c7562/logits_processor.py#L49) [after source](../kernel_sources/blobs/35cf8235cb529ba2/logits_processor.py#L49); `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`: identical_ast [before source](../kernel_sources/blobs/457015abe4165531/qsa_indexer.py#L450) [after source](../kernel_sources/blobs/8ee084541582f1ab/qsa_indexer.py#L477); `vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py`: added_or_removed_or_not_captured [after source](../kernel_sources/blobs/70bd6bfa497867c5/rocm_paged_mxfp4_indexer.py#L222) |

The [structured pair record](../data/recovered_pair_records.json) retains complete generated kernel names (including FlyDSL variant parameters), candidate source paths, file hashes, symbol code/AST hashes and line ranges. Triton JIT log labeling can include Gluon implementations; source decorators distinguish them.

No GPU trace, kernel time ranking, per-step launch count, or causal kernel speedup is claimed. Admission, graph capture, batching, quick-reduce quantization and synthetic DSpark acceptance are retained in the original CI evidence and can affect E2E deltas.
