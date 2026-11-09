# TP2/EP1 conc16: V09 → V11

[Recovered-image comparisons](../recovered_image_history.md) · [Full matrix](../configuration_history.md)

[Original pair and exact launch changes](../pairs/tp2_ep1_conc16_off_2026-10-20_to_2026-11-09.md).

| Endpoint | Docker image | Point / exact CI | (p90 interactivity, output/chip) | Server artifact |
| --- | --- | --- | ---: | --- |
| Before: 2026-10-20 | `vllm/vllm-openai-rocm:nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f` | [445385](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39295252770/job/118206654223) | (198.019802, 365.45014) | `12224605382` |
| After: 2026-11-09 | `vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808` | [447797](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42335637576/job/128309286758) | (302.114804, 370.94850) | `13564725051` |

Interactivity **+52.57%**; throughput/chip **+1.50%**. E2E measurements do not isolate kernel causality.

## Exact log evidence and measurement windows

Both artifacts are joined through their upload IDs in their own endpoint job logs. Server messages are aligned against that job's AIPerf/CI phase boundaries; timestamps are treated as UTC, and events within one second of a boundary are marked uncertain. AIPerf `profiling` means E2E measurement.

| Endpoint | E2E measurement start | E2E measurement end |
| --- | --- | --- |
| Before | 2026-10-19T20:43:15.170000 | 2026-10-19T21:43:21.892000 |
| After | 2026-11-07T05:32:12.051000 | 2026-11-07T06:32:18.106000 |

| Diagnostic | Before observed phases | After observed phases | Interpretation |
| --- | --- | --- | --- |
| mono_dispatch | server_setup_or_graph_warmup: 2 diagnostic messages | server_setup_or_graph_warmup: 2 diagnostic messages | Reached guarded mono path; no sustained coverage fraction |
| markov_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 1 diagnostic messages | Reached guarded sampling path; first message only |
| quant_gemm_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 42 diagnostic messages | Explicit HIT includes row/weight/tile shape; logs once per shape |
| jit_compile | benchmark_warmup: 6 diagnostic messages; e2e_measurement: 1 diagnostic messages; server_setup_or_graph_warmup: 1 diagnostic messages | benchmark_warmup: 6 diagnostic messages | Named compilation events; not launches or total GPU time |

Selected exact log lines (all retained events and source mappings are in [runtime facts](../evidence/recovered_runtime_facts.json)):

- **Before, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g37_agg_w0.out:419`: `(Worker_TP0 pid=3193506) INFO 10-19 20:24:04 [mono_decode.py:435] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`
- **Before, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g37_agg_w0.out:1234`: `(Worker_TP0 pid=3193506) WARNING 10-19 20:48:20 [jit_monitor.py:140] Triton kernel JIT compilation during inference: _compute_local_logits_stats_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g15_agg_w0.out:329`: `(Worker_TP0 pid=4156805) INFO 11-07 05:17:21 [mono_decode.py:693] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g15_agg_w0.out:297`: `(Worker_TP0 pid=4156805) INFO 11-07 05:16:51 [fused_markov.py:274] DSpark fused Markov sampling: 5 steps, vocabulary 129280, rank 256, 505 blocks`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g15_agg_w0.out:330`: `(Worker_TP0 pid=4156805) WARNING 11-07 05:17:27 [rocm_block32_gemm.py:989] tiny-M fused MXFP8 quant+GEMM HIT (M=2, N=12800, K=6144, packed tile 8x16x256 k_pack=2)`

## Reviewed kernel bodies and fusion boundaries

These findings use recovered container file bytes. V01's tag was not historically digest-pinned; V09–V11 were. Source facts establish possible launch boundaries; only the log observations above establish which diagnostic path was reached.

### Tiny-M activation quantization fused into block32 GEMM

**Before:** V09 has separate MXFP8 activation quantization and packed/tiled GEMM.

**After:** V11 _block32_packed_quant_kernel loads raw BF16/FP16 activations, calculates per-32 E8M0 scales/e4m3 data in registers and feeds tl.dot_scaled.

**Device language:** Triton, including tl.dot_scaled

**Code/body:** Adds quant_in_kernel dispatch and retuned fused-quant tiles while preserving stock K-accumulation structure. Exact HIT logs expose M/N/K, tile and k_pack. Source and logs establish the guarded producer/consumer fusion.

**Fusion boundaries:** Removes the standalone activation-quant launch and x_q/x_scale global-memory round trip. A separate split-K reducer can still be needed; the whole linear is not always one launch.

**Applicability / log limit:** Raw contiguous BF16/FP16, packed config, 0<M≤8. Client concurrency does not equal M. Most recorded HITs occur before measurement; every pair reports its own phase and shapes.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py` | `da6105b99593979c46ffc512860b183153299233e440633f3099254c7ebbb843` | `d0db2d8d69d5b8f4c45015a9b4de6f380ece0e04862b847479940c090929f926` | changed bytes |

### DSpark Markov sampling on gfx950

**Before:** V09 uses ordinary sampling; V10 contains a fused helper whose architecture guard admits gfx942 only.

**After:** V11 guard admits gfx950; exact endpoint logs report five steps, vocabulary 129280, rank 256, 505 blocks.

**Device language:** Triton sampling core; separate base-logit GEMM

**Code/body:** The per-step kernel loads the previous token via prior block maxima, computes Markov bias plus base logits and block Gumbel-max. Final kernel resolves the last token. The info message is emitted after eligibility checks and before launches.

**Fusion boundaries:** Sampling core uses five per-step kernels plus one final resolver. Base-logit computation, input gathers and scratch initialization are outside this count. There is no single launch covering all five sequential steps.

**Applicability / log limit:** gfx950 --monokernel, at most five request rows, supported sampler/dtypes. All observed first Markov messages precede E2E measurement; they do not prove sustained coverage at higher concurrency.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/ops/fused_markov.py` | `absent` | `28fb830c9bfa220baaa33f8f10a5db4faf7b7bdd654cd4ff917122c9729e6084` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/ops/__init__.py` | `absent` | `094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/dspark.py` | `b12b728bbdd1d77b2b3dfa7852631d539ad41ff2b312f59f596e7b20dce48d88` | `774266398fa2a49621441a9d36a6453ea7bed95f8a02deaa88aa741d9bf1cec9` | changed bytes |

### SWA metadata helper admitted on gfx950

**Before:** V09 has no helper; V10 helper is gfx942-gated.

**After:** V11 architecture predicate admits gfx950.

**Device language:** Triton device body; Python architecture/shape guards

**Code/body:** The existing Triton _decode_swa_meta_kernel writes validity, dense window data and ragged indices/indptr directly into graph buffers.

**Fusion boundaries:** One metadata-core launch replaces the prior validity/window/count/cumsum/compaction/copy sequence described in source. First-use allocation, token-to-request preparation and tile-scheduler work remain outside that core count.

**Applicability / log limit:** --monokernel, pure decode, ≤32 decode token rows, supported causal or DSpark non-causal metadata. No specific HIT message was found; source eligibility only, not measured launch elimination.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py` | `absent` | `2e8ec7dd48ded6d28156699a1473bc76beb6a597c04b59a31f67b31651750afe` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/ops/__init__.py` | `absent` | `094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/rocm.py` | `f121bd2ecf71dfc4074617b826415c8b81d18c93b07035254d083b2a23b63375` | `e5f8f25426f79a3d96a0e1050eb0e33d78f72588391406660277f6183718ff59` | changed bytes |

### TP2 ratio-2 internal mono tuning

**Before:** V09 uses generic persistent K1/K2 scheduling.

**After:** V11 MonoBuild.tuned selects target-only gfx950 TP2 ratio-2 specializations.

**Device language:** FlyDSL

**Code/body:** Seam reduction/slice work is combined; normalization publication and gates are rescheduled. MoE down stages pipeline loads/MFMA and cooperative shared-expert work. Same two external persistent-launch boundaries.

**Fusion boundaries:** Internal fusion/scheduling improvements inside K1/K2, not another layer-level launch collapse.

**Applicability / log limit:** TP2 target ratio2 only; excludes TP4, TP8 and draft routing. Generic active logs do not identify which layers instantiated the tuned build.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/mono/layer.py` | `c9e2791bce448a4041269053f97541b86aa085407e0d6e561eebb99acc20a5d6` | `ae6abc3c8c3824f985e341c7b5539d41273cafe88aebce6a75fc31a758f0fa5d` | changed bytes |
| `vllm/models/deepseek_v41/amd/mono/stages/seam.py` | `0e196c1e62a4d9a41e99ec551eefb9680c800326a91bbe978d7ed3163dd15643` | `3fac9a03558c18c29d6a4b4b4af16449002fafbf4ef3716eb085a30c3c094eb6` | changed bytes |
| `vllm/models/deepseek_v41/amd/mono/stages/moe.py` | `00476528913b1a046a65d1491d954020461c444dbb13a1c134ffe2a1624ee792` | `1075006608f00dd534432307a83ba3e9f15685960fa0d7ce5bf9c6bd3053b2f7` | changed bytes |
| `vllm/models/deepseek_v41/amd/mono/attention/plan.py` | `8ded6df37a0b6681040e7a9e5919b6582b0b104d9ba63b1e1234fe5aad541fda` | `9386470498616fbda768c08ff0a8091599cd32c433c74b04682bd9504fb7d57d` | changed bytes |

### Ordinary delayed mHC device bodies unchanged

**Before:** V09/V10 AITER delayed post/pre/RMSNorm wrapper and two Triton device bodies.

**After:** V11 retains byte-identical wrapper, device file and gfx950 delayed-seam tuning configuration.

**Device language:** Triton, unchanged device bodies

**Code/body:** Although AITER package versions change, the specific delayed mHC seam files are unchanged. JIT logs may differ because warmup coverage changes; that does not imply a different kernel body.

**Fusion boundaries:** Existing main+reduce fusion is retained. V11 separate mHC overlap path can replace the producer/consumer scheduling on FULL graphs; the unchanged AITER seam itself is not new fusion.

**Applicability / log limit:** Ordinary/fallback mHC path only; mono bypasses it. The added gfx942-only ASM seam in aiter/ops/mhc.py is not attributed to MI355X.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py` | `34003d031b0712e55b745c397e6c519b60ff32c2b6747024b42a7151592f949c` | `34003d031b0712e55b745c397e6c519b60ff32c2b6747024b42a7151592f949c` | identical bytes |
| `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py` | `d60eafdcddcc2fbb9888f0fd4350f06cdac8122207ba8fc72fd1a326834813b2` | `d60eafdcddcc2fbb9888f0fd4350f06cdac8122207ba8fc72fd1a326834813b2` | identical bytes |
| `aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json` | `309fd20e9aa660437123ae6d781932c1f7ea88ff2e59002967feac5796a589fc` | `309fd20e9aa660437123ae6d781932c1f7ea88ff2e59002967feac5796a589fc` | identical bytes |

### mHC overlap and Engram gate sharding

**Before:** Earlier ordinary model mHC gate work uses caller stream and common Engram implementation.

**After:** V11 adds an ordinary-path Triton post/collapse/norm producer with auxiliary-stream gate work; Engram gate work is TP-sharded.

**Device language:** Triton producers/Engram stages, AITER gate API, separate collective/stream operations

**Code/body:** mHC FULL graph capture creates stream dependencies, and Engram produces partial statistics/value slices, exchanges them, then finishes gating.

**Fusion boundaries:** mHC post/collapse/norm producer is fused; moving gate work to another stream is overlap, not single-kernel fusion. Engram retains partial-kernel, collective, final-kernel boundaries.

**Applicability / log limit:** gfx950 ordinary/fallback path; mHC ≤128 tokens, hc_mult4, no ubatching and FULL capture (not eager/piecewise). No direct overlap/Engram HIT log was found.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/mhc.py` | `absent` | `9be806b5fb84bee3ce3f874e118a9697e3b788e06f9ac5b91a48e4c90060cdfa` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/engram.py` | `absent` | `6a458dd68626126142fbf7f7deba5cdc2f66bc6489feee511fe903eca9bcdecc` | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/model.py` | `7cd417b4c36f9f5b32e8c82f68588c8df11d33e693d4e9f1d797642f62a546a3` | `5e86e7df41b4e41d6b173356f72750d50387ab7c6200a3f75c66d6a00708b120` | changed bytes |
| `vllm/model_executor/kernels/mhc/triton.py` | `a48f030e5df658416eb7d0abadad62c4742e4004fcee0f8f0514664e77965ef2` | `8f61199e55adf03d8072b7e51f76daf584a89ba0e7d6312fda254668f7f2370c` | changed bytes |

### MXFP4 Q/K producer and score bodies retained

**Before:** V09/V10 MXFP4 K-store, Q-RoPE quant and score implementations.

**After:** V11 retains these inspected producer and score files byte-for-byte.

**Device language:** Triton Q/K producers; Gluon score kernel

**Code/body:** Logs continue to expose _pa_mqa_logits_mxfp4_kernel and _q_rope_mxfp4_quant_kernel, though logging coverage differs. The score source is Gluon even when the generic JIT logger says Triton.

**Fusion boundaries:** Existing norm/RoPE/quant/cache boundaries remain. No new producer fusion is assigned to a package-version bump.

**Applicability / log limit:** Index-source work; still outside persistent mono launches. JIT warnings are observations of compilation, not every invocation.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py` | `0c824c1282c7348e43b954faf968eb78374e86684f03934541528a2fbbfe7a7c` | `0c824c1282c7348e43b954faf968eb78374e86684f03934541528a2fbbfe7a7c` | identical bytes |
| `aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py` | `481e185d288607b1ab89eb6a7d3aa10d13c1ef0468bddd5d547ee9e2789ccf28` | `481e185d288607b1ab89eb6a7d3aa10d13c1ef0468bddd5d547ee9e2789ccf28` | identical bytes |
| `aiter/ops/triton/rope/q_rope_mxfp4_quant.py` | `02a5ddb1835b90421e524b11ad7683da488cb18344c7306efdfe2dfb70ce7828` | `02a5ddb1835b90421e524b11ad7683da488cb18344c7306efdfe2dfb70ce7828` | identical bytes |
| `aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py` | `5fd9e06fbb9b8ead7496a2305a221f81577c90b8b0351e27a219eb07424809bc` | `5fd9e06fbb9b8ead7496a2305a221f81577c90b8b0351e27a219eb07424809bc` | identical bytes |
| `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py` | `f5380c2ecd995076feeca2823441b16b60e7e5f1a1c18125b512647df47243aa` | `f5380c2ecd995076feeca2823441b16b60e7e5f1a1c18125b512647df47243aa` | identical bytes |

### Ordinary sparse MLA body/dispatch changes

**Before:** Earlier gfx950 sparse main/reduce bodies.

**After:** Later shared Gluon body adds 64-bit row bases, explicit FMA inverse rotation, zero-denominator protection and BF16 gather-layout adjustment.

**Device language:** Gluon device bodies; Python launch/format dispatch

**Code/body:** The fp8_ds_mla route passes through pa_decode_sparse to this shared Gluon body. The new fp8_scalar fast path in sparse_mla.py has separate format guards and is not credited to the packed DSv4 cache.

**Fusion boundaries:** Existing inverse-RoPE/MXFP8 epilogue remains fused; these changes are body/layout/numerical robustness and dispatch changes, not proof of a new cross-operator fusion. BF16_GATHER tuning applies only when that gather format is selected.

**Applicability / log limit:** Ordinary/fallback sparse attention. Exact _sparse_mla/_sparse_mla_reduce JIT messages identify compiled candidates, not the full serving trace.

| Source path | Before SHA256 | After SHA256 | File relation |
| --- | --- | --- | --- |
| `aiter/ops/triton/attention/sparse_mla.py` | `91a6420d0c115eadb7a24d39d75412a7da7a3dba99e32163513ad3e899bdcf75` | `156bfbfd31eecd4a562250669942ba4a5474d4c06194f67b2900840edcaaea3e` | changed bytes |
| `aiter/ops/triton/attention/pa_decode_sparse.py` | `2e107a1e78f3645a159831a191391c7dd602f7e64ba3a39e22a7fcec016c452c` | `d290e3f45cb9daa7f5ece8c7449a19047b2f20422e7485186728ef2f89a851f1` | changed bytes |
| `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py` | `c71422552e29403d5a3b4863d63b937d36e84109bc319e6531134eed1adaeeb7` | `d7c7e731d5a037389d8b6eae785b323eac3ca02691b8a8afb3687cb5829dec4a` | changed bytes |

Pinned source evidence: [inventory](../evidence/recovered_source_inventory.json), [reviewed rules](../evidence/recovered_kernel_reviews.json), [V09→V11 source diff](../evidence/recovered_source_diffs/v09_to_v11.diff). Installation paths are preserved in the inventory; canonical comparison paths above normalize the earlier deepseek_v4_1 directory name.

## Named JIT kernels: observed sets and body comparisons

This table lists every symbol mentioned by the selected endpoints' JIT diagnostics. Missing messages do not imply missing execution: the monitor activates after server warmup and uses warning_once. A changed file can contain an unchanged symbol; AST comparisons below separate those cases. A name match is a source candidate, not proof of a unique binary implementation.

| Symbol | Before logged phases | After logged phases | Candidate source / symbol-body relation |
| --- | --- | --- | --- |
| `_block32_tiled_kernel` | benchmark_warmup | benchmark_warmup | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: identical_ast |
| `_compute_local_logits_stats_kernel` | e2e_measurement | not logged | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast |
| `_mask_candidates_kernel` | benchmark_warmup | not logged | `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`: identical_ast |
| `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel` | server_setup_or_graph_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast |
| `_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast |
| `_mxfp8_wo_a_bmm_kernel` | benchmark_warmup | benchmark_warmup | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`: identical_ast |
| `_pa_mqa_logits_mxfp4_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`: identical_ast |
| `_prepare_dflash_inputs_kernel` | benchmark_warmup | not logged | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`: changed_ast |
| `_sparse_mla` | not logged | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`: changed_ast |

The [structured pair record](../data/recovered_pair_records.json) retains complete generated kernel names (including FlyDSL variant parameters), candidate source paths, file hashes, symbol code/AST hashes and line ranges. Triton JIT log labeling can include Gluon implementations; source decorators distinguish them.

No GPU trace, kernel time ranking, per-step launch count, or causal kernel speedup is claimed. Admission, graph capture, batching, quick-reduce quantization and synthetic DSpark acceptance are retained in the original CI evidence and can affect E2E deltas.
