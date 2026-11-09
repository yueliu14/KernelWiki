# TP4/EP1 conc16: V01 → V09

[Recovered-image comparisons](../recovered_image_history.md) · [Full matrix](../configuration_history.md)

Supplemental comparison across V02–V08; intermediate measured points remain in the original matrix. This is not an adjacent-image kernel attribution.

| Endpoint | Docker image | Point / exact CI | (p90 interactivity, output/chip) | Server artifact |
| --- | --- | --- | ---: | --- |
| Before: 2026-09-12 | `vllm/vllm-openai-rocm:deepseekv41-flash-0909` | [441675](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34657748450/job/103454471652) | (19.661817, 70.70759) | `10289485380` |
| After: 2026-10-20 | `vllm/vllm-openai-rocm:nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f` | [445392](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39295252770/job/118206654564) | (284.900285, 189.07649) | `12232727040` |

Interactivity **+1349.00%**; throughput/chip **+167.41%**. E2E measurements do not isolate kernel causality.

## Exact log evidence and measurement windows

Both artifacts are joined through their upload IDs in their own endpoint job logs. Server messages are aligned against that job's AIPerf/CI phase boundaries; timestamps are treated as UTC, and events within one second of a boundary are marked uncertain. AIPerf `profiling` means E2E measurement.

| Endpoint | E2E measurement start | E2E measurement end |
| --- | --- | --- |
| Before | 2026-09-12T00:36:47.190653 | 2026-09-12T01:37:27.192970 |
| After | 2026-10-19T23:53:31.256000 | 2026-10-20T00:53:31.257000 |

| Diagnostic | Before observed phases | After observed phases | Interpretation |
| --- | --- | --- | --- |
| mono_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 2 diagnostic messages | Reached guarded mono path; no sustained coverage fraction |
| markov_dispatch | No matching message (not proof of non-use) | No matching message (not proof of non-use) | Reached guarded sampling path; first message only |
| quant_gemm_dispatch | No matching message (not proof of non-use) | No matching message (not proof of non-use) | Explicit HIT includes row/weight/tile shape; logs once per shape |
| jit_compile | benchmark_warmup: 9 diagnostic messages; e2e_measurement: 4 diagnostic messages; server_setup_or_graph_warmup: 1 diagnostic messages | benchmark_warmup: 6 diagnostic messages; e2e_measurement: 1 diagnostic messages; server_setup_or_graph_warmup: 1 diagnostic messages | Named compilation events; not launches or total GPU time |

Selected exact log lines (all retained events and source mappings are in [runtime facts](../evidence/recovered_runtime_facts.json)):

- **Before, e2e_measurement**, `results/server.log:261132`: `(Worker_TP0 pid=456398) WARNING 09-12 00:36:49 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _compute_local_logits_stats_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **Before, e2e_measurement**, `results/server.log:261133`: `(Worker_TP0 pid=456398) WARNING 09-12 00:36:49 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _rejection_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **Before, e2e_measurement**, `results/server.log:261134`: `(Worker_TP0 pid=456398) WARNING 09-12 00:36:49 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _resample_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **Before, e2e_measurement**, `results/server.log:330089`: `(Worker_TP0 pid=456398) WARNING 09-12 00:45:22 [jit_monitor.py:141] Triton kernel JIT compilation during inference: _ring_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g09_agg_w0.out:532`: `(Worker_TP0 pid=4138890) INFO 10-19 23:34:44 [mono_decode.py:435] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`
- **After, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g09_agg_w0.out:1395`: `(Worker_TP0 pid=4138890) WARNING 10-19 23:58:33 [jit_monitor.py:140] Triton kernel JIT compilation during inference: _compute_local_logits_stats_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`

## Reviewed kernel bodies and fusion boundaries

These findings use recovered container file bytes. V01's tag was not historically digest-pinned; V09–V11 were. Source facts establish possible launch boundaries; only the log observations above establish which diagnostic path was reached.

### Whole-layer mono decode

**Before:** V01 model runs ordinary attention, mHC, experts and collectives; no mono module.

**After:** V09 has persistent FlyDSL K1/K2 for target and DSpark draft layers.

**Device language:** FlyDSL persistent kernels; Python dispatch

**Code/body:** New mono layer/runner/attention/MoE bodies use peer-memory synchronization and in-kernel TP sums. Exact target and draft active messages occur in each V09 endpoint before E2E measurement.

**Fusion boundaries:** K1 combines the attention seam (slice/gate/norm/MXFP8) and attention front. K2 combines attention back/wo_b peer push, TP sum + FFN seam, MoE and its reduction. Compressor/indexer run between K1 and K2 where required; Engram injection stays outside. This is two principal persistent launches, not one whole-model kernel.

**Applicability / log limit:** This supplemental V01→V09 comparison is TP4/EP1. Decode-only, ≤48 token rows, suitable cache/routing/graph/residency. Initialization/active logs do not establish measurement-phase coverage.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/model.py` | `b7f6050aa929a174a9dc850f345faf6be687f96169fd02861fea877c0beaf5e8` | `7cd417b4c36f9f5b32e8c82f68588c8df11d33e693d4e9f1d797642f62a546a3` | changed bytes |
| `vllm/models/deepseek_v41/amd/mono_decode.py` | `absent` | `19d9d75d959375dd1f487cb4670b26b61a59f6094fa312fe8bf4cca30166e14c` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/mono/layer.py` | `absent` | `c9e2791bce448a4041269053f97541b86aa085407e0d6e561eebb99acc20a5d6` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/mono/stages/seam.py` | `absent` | `0e196c1e62a4d9a41e99ec551eefb9680c800326a91bbe978d7ed3163dd15643` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/mono/stages/moe.py` | `absent` | `00476528913b1a046a65d1491d954020461c444dbb13a1c134ffe2a1624ee792` | added/absent in snapshot |

### Ordinary mHC seam: reference to fused Triton producer

**Before:** V01 AMD model directly calls mhc_pre_delayed_torch/mhc_post_torch and separate normalization.

**After:** V09 ordinary path calls the AITER delayed post/pre/RMSNorm implementation.

**Device language:** PyTorch/reference operations → two Triton device stages

**Code/body:** Recovered AITER bodies resolve the earlier API-only uncertainty: a split-K main kernel forms post-mix, gate-projection partials and carried-pre collapse; a reduce kernel forms gates/Sinkhorn and applies normalization.

**Fusion boundaries:** Two principal Triton launches (_main_kernel + _reduce_kernel), not one. Main/reduce share partials and layer-input buffers. Entry identity-pre preparation can add operations; mono-covered layers bypass this ordinary seam.

**Applicability / log limit:** gfx950, hc_mult=4, supported contiguous BF16 residual/FP32 gates and norm weights; first entry broadcast and ordinary fallback exceptions retained.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/model.py` | `b7f6050aa929a174a9dc850f345faf6be687f96169fd02861fea877c0beaf5e8` | `7cd417b4c36f9f5b32e8c82f68588c8df11d33e693d4e9f1d797642f62a546a3` | changed bytes |
| `vllm/model_executor/layers/mhc.py` | `47387df8343bcdaea65138d24079c8a10ea80dfba7822e243c2181354d8afdd8` | `433d750a4d669ad468679b1faf0a3f1e846410198fb4a1eeee2d14cfe55e6a55` | changed bytes |
| `vllm/_aiter_ops.py` | `4365cab5f1267018648c8c556707a74d042d571cab09fb0ea67c378afadf36d3` | `032f24a1dd76b84b03730804dfc9c45fe396249bf464aa0f01d7ac2a7109525d` | changed bytes |
| `aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py` | `absent` | `34003d031b0712e55b745c397e6c519b60ff32c2b6747024b42a7151592f949c` | added/absent in snapshot |
| `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py` | `absent` | `d60eafdcddcc2fbb9888f0fd4350f06cdac8122207ba8fc72fd1a326834813b2` | added/absent in snapshot |

### MXFP4 indexer producer fusion

**Before:** V01 uses its earlier indexer route; logs expose _gluon_fp8_mqa_logits_kernel and generic routing/reduction candidates.

**After:** V09 requests mxfp4 indexer cache and uses k_norm_rope_mxfp4_cache, q_rope_mxfp4_quant and MXFP4 score kernels.

**Device language:** Triton Q/K producers and Gluon sparse-score device kernel

**Code/body:** Recovered Triton producers combine K normalization/rotation/packing/cache storage and Q rotation/packing plus weights; scoring remains a separate Gluon kernel.

**Fusion boundaries:** Fused K and Q producer operations, followed by separate sparse scoring/top-k. Not a single indexer-plus-attention launch. Log name differences identify compilations, not a complete removed/added execution set.

**Applicability / log limit:** Applies to index-source layers using this cache/config. Such work also remains external between mono launches.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/rocm.py` | `b74d9d75b29b81530b0f458f7d87d80bc6907949f4b94d22b70c649f9b521bea` | `f121bd2ecf71dfc4074617b826415c8b81d18c93b07035254d083b2a23b63375` | changed bytes |
| `aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py` | `absent` | `0c824c1282c7348e43b954faf968eb78374e86684f03934541528a2fbbfe7a7c` | added/absent in snapshot |
| `aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py` | `absent` | `481e185d288607b1ab89eb6a7d3aa10d13c1ef0468bddd5d547ee9e2789ccf28` | added/absent in snapshot |
| `aiter/ops/triton/rope/q_rope_mxfp4_quant.py` | `absent` | `02a5ddb1835b90421e524b11ad7683da488cb18344c7306efdfe2dfb70ce7828` | added/absent in snapshot |
| `aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py` | `absent` | `5fd9e06fbb9b8ead7496a2305a221f81577c90b8b0351e27a219eb07424809bc` | added/absent in snapshot |

### Ordinary sparse attention output epilogue

**Before:** V01 performs sparse decode reduction, standalone inverse RoPE and BF16 wo_a contraction.

**After:** V09 ordinary gfx950 AITER attention supports inverse-RoPE/MXFP8 output epilogue; wo_a consumes the quantized output through a separate grouped BMM.

**Device language:** Triton baseline → AITER Gluon sparse main/reduce with a separate Triton wo_a BMM

**Code/body:** vLLM passes positions/cos-sin/out_mxfp8 through sparse_mla_fwd -> pa_decode_sparse -> gfx950 Gluon bodies for fp8_ds_mla packed caches. The device epilogue rotates and quantizes completed attention output.

**Fusion boundaries:** Reduction/inverse-RoPE/MXFP8 output share the selected output epilogue. Split attention still has main and reduce boundaries, and wo_a remains separate. No transfer of this ordinary-path fusion to already mono-covered work.

**Applicability / log limit:** Ordinary or fallback attention, gfx950 and supported fp8_ds_mla cache. Runtime JIT diagnostics and source dispatch are evidence; no per-kernel duration is available.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/rocm.py` | `b74d9d75b29b81530b0f458f7d87d80bc6907949f4b94d22b70c649f9b521bea` | `f121bd2ecf71dfc4074617b826415c8b81d18c93b07035254d083b2a23b63375` | changed bytes |
| `aiter/ops/triton/attention/sparse_mla.py` | `absent` | `91a6420d0c115eadb7a24d39d75412a7da7a3dba99e32163513ad3e899bdcf75` | added/absent in snapshot |
| `aiter/ops/triton/attention/pa_decode_sparse.py` | `ce3cb88029a184762c71e9d029650fb6d13b0e43357fa3973c773fb231a77f8c` | `2e107a1e78f3645a159831a191391c7dd602f7e64ba3a39e22a7fcec016c452c` | changed bytes |
| `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py` | `absent` | `c71422552e29403d5a3b4863d63b937d36e84109bc319e6531134eed1adaeeb7` | added/absent in snapshot |

Pinned source evidence: [inventory](../evidence/recovered_source_inventory.json), [reviewed rules](../evidence/recovered_kernel_reviews.json), [V01→V09 source diff](../evidence/recovered_source_diffs/v01_to_v09.diff). Installation paths are preserved in the inventory; canonical comparison paths above normalize the earlier deepseek_v4_1 directory name.

## Named JIT kernels: observed sets and body comparisons

This table lists every symbol mentioned by the selected endpoints' JIT diagnostics. Missing messages do not imply missing execution: the monitor activates after server warmup and uses warning_once. A changed file can contain an unchanged symbol; AST comparisons below separate those cases. A name match is a source candidate, not proof of a unique binary implementation.

| Symbol | Before logged phases | After logged phases | Candidate source / symbol-body relation |
| --- | --- | --- | --- |
| `_block32_tiled_kernel` | not logged | benchmark_warmup | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: added_or_removed_or_not_captured |
| `_combined_routing` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py`: changed_ast |
| `_compute_local_logits_stats_kernel` | e2e_measurement | e2e_measurement | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast |
| `_dequantize_and_gather_k_kernel` | benchmark_warmup | not logged | `vllm/models/deepseek_v4/common/ops/cache_utils.py`: added_or_removed_or_not_captured; `vllm/models/deepseek_v41/common/ops/cache_utils.py`: identical_ast |
| `_engram_lookup_kernel` | server_setup_or_graph_warmup | not logged | `vllm/models/deepseek_v41/common/engram.py`: changed_ast |
| `_gluon_fp8_mqa_logits_kernel` | benchmark_warmup | not logged | `aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py`: changed_ast |
| `_mask_candidates_kernel` | not logged | benchmark_warmup | `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`: identical_ast |
| `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel` | not logged | server_setup_or_graph_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: added_or_removed_or_not_captured |
| `_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel` | not logged | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: added_or_removed_or_not_captured |
| `_moe_gemm_a16w4` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py`: changed_ast |
| `_mxfp8_linear_kernel` | benchmark_warmup | not logged | `vllm/model_executor/kernels/linear/mxfp8/rocm_native.py`: identical_ast |
| `_mxfp8_wo_a_bmm_kernel` | not logged | benchmark_warmup | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`: added_or_removed_or_not_captured |
| `_pa_mqa_logits_mxfp4_kernel` | not logged | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`: added_or_removed_or_not_captured |
| `_prepare_dflash_inputs_kernel` | benchmark_warmup | benchmark_warmup | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`: changed_ast |
| `_reduce_grouped` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/reduce.py`: changed_ast |
| `_rejection_kernel` | e2e_measurement | not logged | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast |
| `_resample_kernel` | e2e_measurement | not logged | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast |
| `_ring_slot_mapping_kernel` | e2e_measurement | not logged | `vllm/models/deepseek_v41/compressor.py`: changed_ast |
| `_sum_bitmatrix_rows` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix.py`: identical_ast; `vllm/third_party/triton_kernels/tensor_details/bitmatrix_details/sum_bitmatrix_rows.py`: identical_ast |
| `_topk` | benchmark_warmup | not logged | `aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py`: identical_ast; `vllm/model_executor/layers/logits_processor.py`: identical_ast; `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`: identical_ast; `vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py`: added_or_removed_or_not_captured |

The [structured pair record](../data/recovered_pair_records.json) retains complete generated kernel names (including FlyDSL variant parameters), candidate source paths, file hashes, symbol code/AST hashes and line ranges. Triton JIT log labeling can include Gluon implementations; source decorators distinguish them.

No GPU trace, kernel time ranking, per-step launch count, or causal kernel speedup is claimed. Admission, graph capture, batching, quick-reduce quantization and synthetic DSpark acceptance are retained in the original CI evidence and can affect E2E deltas.
