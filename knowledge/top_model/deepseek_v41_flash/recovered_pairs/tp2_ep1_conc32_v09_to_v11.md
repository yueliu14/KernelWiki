# TP2/EP1 conc32: V09 → V11

[Recovered-image comparisons](../recovered_image_history.md) · [Full matrix](../configuration_history.md)

**vLLM commits:** [e68efc685ab1](https://github.com/vllm-project/vllm/commit/e68efc685ab1e6f1eda004ced7157e3ca411da2f) → [4fab5a074758](https://github.com/vllm-project/vllm/commit/4fab5a07475822c6eaa971dceb2e2220d0446808). [Kernel implementation browser](../kernel_implementations.md). Container file identity is retained separately from the reported framework commit.

[Original pair and exact launch changes](../pairs/tp2_ep1_conc32_off_2026-10-20_to_2026-11-09.md).

| Endpoint | Docker image | Point / exact CI | (p90 interactivity, output/chip) | Server artifact |
| --- | --- | --- | ---: | --- |
| Before: 2026-10-20 | `vllm/vllm-openai-rocm:nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f` | [445388](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39295252770/job/118206655070) | (120.192308, 660.26131) | `12236676394` |
| After: 2026-11-09 | `vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808` | [447800](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42335637576/job/128309286703) | (186.915888, 645.73531) | `13564049818` |

Interactivity **+55.51%**; throughput/chip **-2.20%**. E2E measurements do not isolate kernel causality.

## Exact log evidence and measurement windows

Both artifacts are joined through their upload IDs in their own endpoint job logs. Server messages are aligned against that job's AIPerf/CI phase boundaries; timestamps are treated as UTC, and events within one second of a boundary are marked uncertain. AIPerf `profiling` means E2E measurement.

| Endpoint | E2E measurement start | E2E measurement end |
| --- | --- | --- |
| Before | 2026-10-20T01:24:29.041000 | 2026-10-20T02:24:52.485000 |
| After | 2026-11-07T05:41:46.903000 | 2026-11-07T06:41:57.211000 |

| Diagnostic | Before observed phases | After observed phases | Interpretation |
| --- | --- | --- | --- |
| mono_dispatch | server_setup_or_graph_warmup: 2 diagnostic messages | server_setup_or_graph_warmup: 2 diagnostic messages | Reached guarded mono path; no sustained coverage fraction |
| markov_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 1 diagnostic messages | Reached guarded sampling path; first message only |
| quant_gemm_dispatch | No matching message (not proof of non-use) | server_setup_or_graph_warmup: 42 diagnostic messages | Explicit HIT includes row/weight/tile shape; logs once per shape |
| jit_compile | benchmark_warmup: 7 diagnostic messages; e2e_measurement: 1 diagnostic messages | benchmark_warmup: 7 diagnostic messages; e2e_measurement: 2 diagnostic messages | Named compilation events; not launches or total GPU time |

Selected exact log lines (all retained events and source mappings are in [runtime facts](../evidence/recovered_runtime_facts.json)):

- **Before, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g11_agg_w0.out:421`: `(Worker_TP0 pid=3768703) INFO 10-20 01:03:11 [mono_decode.py:435] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`
- **Before, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g11_agg_w0.out:1158`: `(Worker_TP0 pid=3768703) WARNING 10-20 01:24:32 [jit_monitor.py:140] Triton kernel JIT compilation during inference: _compute_local_logits_stats_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g19_agg_w0.out:331`: `(Worker_TP0 pid=1477829) INFO 11-07 05:19:01 [mono_decode.py:693] DSv4.1 mono decode layer active (decode steps of <= 48 rows)`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g19_agg_w0.out:342`: `(Worker_TP0 pid=1477829) INFO 11-07 05:19:56 [fused_markov.py:274] DSpark fused Markov sampling: 5 steps, vocabulary 129280, rank 256, 505 blocks`
- **After, server_setup_or_graph_warmup**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g19_agg_w0.out:332`: `(Worker_TP0 pid=1477829) WARNING 11-07 05:19:07 [rocm_block32_gemm.py:989] tiny-M fused MXFP8 quant+GEMM HIT (M=2, N=12800, K=6144, packed tile 8x16x256 k_pack=2)`
- **After, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g19_agg_w0.out:703`: `(Worker_TP0 pid=1477829) WARNING 11-07 05:41:50 [jit_monitor.py:143] Triton kernel JIT compilation during inference: _mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- **After, e2e_measurement**, `srt-single-node-logs.tar.gz:./logs/mia1-p01-g19_agg_w0.out:1224`: `(Worker_TP0 pid=1477829) WARNING 11-07 06:22:33 [jit_monitor.py:143] Triton kernel JIT compilation during inference: _sparse_mla_reduce. This causes a latency spike; consider extending warmup to cover this shape/config.`

## Reviewed kernel bodies and fusion boundaries

These findings use recovered container file bytes. V01's tag was not historically digest-pinned; V09–V11 were. Source facts establish possible launch boundaries; only the log observations above establish which diagnostic path was reached.

### Tiny-M activation quantization fused into block32 GEMM

**Before:** V09 has separate MXFP8 activation quantization and packed/tiled GEMM.

**After:** V11 _block32_packed_quant_kernel loads raw BF16/FP16 activations, calculates per-32 E8M0 scales/e4m3 data in registers and feeds tl.dot_scaled.

**Device language:** Triton, including tl.dot_scaled

**Code/body:** Adds quant_in_kernel dispatch and retuned fused-quant tiles while preserving stock K-accumulation structure. Exact HIT logs expose M/N/K, tile and k_pack. Source and logs establish the guarded producer/consumer fusion.

**Fusion boundaries:** Removes the standalone activation-quant launch and x_q/x_scale global-memory round trip. A separate split-K reducer can still be needed; the whole linear is not always one launch.

**Applicability / log limit:** Raw contiguous BF16/FP16, packed config, 0<M≤8. Client concurrency does not equal M. Most recorded HITs occur before measurement; every pair reports its own phase and shapes.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py` | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`<br>`da6105b99593979c46ffc512860b183153299233e440633f3099254c7ebbb843`<br>[Complete implementation](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py)<br>[_block32_tiled_kernel](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L101)<br>[_block32_packed_kernel](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L182)<br>[_block32_splitk_reduce_kernel](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L276)<br>[_split_tile](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L67) | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`<br>`d0db2d8d69d5b8f4c45015a9b4de6f380ece0e04862b847479940c090929f926`<br>[Complete implementation](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py)<br>[_block32_tiled_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L107)<br>[_block32_packed_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L188)<br>[_block32_splitk_reduce_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L400)<br>[_split_tile](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L73) | changed bytes |

### DSpark Markov sampling on gfx950

**Before:** V09 uses ordinary sampling; V10 contains a fused helper whose architecture guard admits gfx942 only.

**After:** V11 guard admits gfx950; exact endpoint logs report five steps, vocabulary 129280, rank 256, 505 blocks.

**Device language:** Triton sampling core; separate base-logit GEMM

**Code/body:** The per-step kernel loads the previous token via prior block maxima, computes Markov bias plus base logits and block Gumbel-max. Final kernel resolves the last token. The info message is emitted after eligibility checks and before launches.

**Fusion boundaries:** Sampling core uses five per-step kernels plus one final resolver. Base-logit computation, input gathers and scratch initialization are outside this count. There is no single launch covering all five sequential steps.

**Applicability / log limit:** gfx950 --monokernel, at most five request rows, supported sampler/dtypes. All observed first Markov messages precede E2E measurement; they do not prove sustained coverage at higher concurrency.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/ops/fused_markov.py` | Absent from captured source selection | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/fused_markov.py`<br>`28fb830c9bfa220baaa33f8f10a5db4faf7b7bdd654cd4ff917122c9729e6084`<br>[Complete implementation](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py)<br>[_markov_step_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L58)<br>[_markov_last_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L161)<br>[_scratch](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L187)<br>[_why_not](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L205) | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/ops/__init__.py` | Absent from captured source selection | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py`<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[Complete implementation](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/dspark.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py`<br>`b12b728bbdd1d77b2b3dfa7852631d539ad41ff2b312f59f596e7b20dce48d88`<br>[Complete implementation](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py)<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py#L60)<br>[_insert_context_kv](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py#L223)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py#L292) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py`<br>`774266398fa2a49621441a9d36a6453ea7bed95f8a02deaa88aa741d9bf1cec9`<br>[Complete implementation](../kernel_sources/blobs/774266398fa2a496/dspark.py)<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/774266398fa2a496/dspark.py#L68)<br>[_insert_context_kv](../kernel_sources/blobs/774266398fa2a496/dspark.py#L231)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/774266398fa2a496/dspark.py#L300) | changed bytes |

### SWA metadata helper admitted on gfx950

**Before:** V09 has no helper; V10 helper is gfx942-gated.

**After:** V11 architecture predicate admits gfx950.

**Device language:** Triton device body; Python architecture/shape guards

**Code/body:** The existing Triton _decode_swa_meta_kernel writes validity, dense window data and ragged indices/indptr directly into graph buffers.

**Fusion boundaries:** One metadata-core launch replaces the prior validity/window/count/cumsum/compaction/copy sequence described in source. First-use allocation, token-to-request preparation and tile-scheduler work remain outside that core count.

**Applicability / log limit:** --monokernel, pure decode, ≤32 decode token rows, supported causal or DSpark non-causal metadata. No specific HIT message was found; source eligibility only, not measured launch elimination.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py` | Absent from captured source selection | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py`<br>`2e8ec7dd48ded6d28156699a1473bc76beb6a597c04b59a31f67b31651750afe`<br>[Complete implementation](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py)<br>[_decode_swa_meta_kernel](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L45)<br>[launch](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L134)<br>[build](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L182) | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/ops/__init__.py` | Absent from captured source selection | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py`<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[Complete implementation](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/rocm.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/rocm.py`<br>`f121bd2ecf71dfc4074617b826415c8b81d18c93b07035254d083b2a23b63375`<br>[Complete implementation](../kernel_sources/blobs/f121bd2ecf71dfc4/rocm.py)<br>[_compute_topk_lens_kernel](../kernel_sources/blobs/f121bd2ecf71dfc4/rocm.py#L139)<br>[_pack_global_topk_ragged_kernel](../kernel_sources/blobs/f121bd2ecf71dfc4/rocm.py#L165)<br>[_trust_dsv4_extra_cache_nan_free](../kernel_sources/blobs/f121bd2ecf71dfc4/rocm.py#L72)<br>[_aiter_sparse_mla_enabled](../kernel_sources/blobs/f121bd2ecf71dfc4/rocm.py#L85) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/rocm.py`<br>`e5f8f25426f79a3d96a0e1050eb0e33d78f72588391406660277f6183718ff59`<br>[Complete implementation](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py)<br>[_compute_topk_lens_kernel](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L140)<br>[_pack_global_topk_ragged_kernel](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L166)<br>[_trust_dsv4_extra_cache_nan_free](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L73)<br>[_aiter_sparse_mla_enabled](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L86) | changed bytes |

### TP2 ratio-2 internal mono tuning

**Before:** V09 uses generic persistent K1/K2 scheduling.

**After:** V11 MonoBuild.tuned selects target-only gfx950 TP2 ratio-2 specializations.

**Device language:** FlyDSL

**Code/body:** Seam reduction/slice work is combined; normalization publication and gates are rescheduled. MoE down stages pipeline loads/MFMA and cooperative shared-expert work. Same two external persistent-launch boundaries.

**Fusion boundaries:** Internal fusion/scheduling improvements inside K1/K2, not another layer-level launch collapse.

**Applicability / log limit:** TP2 target ratio2 only; excludes TP4, TP8 and draft routing. Generic active logs do not identify which layers instantiated the tuned build.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/mono/layer.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/layer.py`<br>`c9e2791bce448a4041269053f97541b86aa085407e0d6e561eebb99acc20a5d6`<br>[Complete implementation](../kernel_sources/blobs/c9e2791bce448a40/layer.py)<br>[build_mono_k1.k1](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L230)<br>[build_mono_k1.launch](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L369)<br>[build_mono_k2.k2](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L621)<br>[build_mono_k2.launch](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L782) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/layer.py`<br>`ae6abc3c8c3824f985e341c7b5539d41273cafe88aebce6a75fc31a758f0fa5d`<br>[Complete implementation](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py)<br>[build_mono_k1.k1](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L315)<br>[build_mono_k1.launch](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L460)<br>[build_mono_k2.k2](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L797)<br>[build_mono_k2.launch](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L963) | changed bytes |
| `vllm/models/deepseek_v41/amd/mono/stages/seam.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/seam.py`<br>`0e196c1e62a4d9a41e99ec551eefb9680c800326a91bbe978d7ed3163dd15643`<br>[Complete implementation](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py)<br>[scratch_layout](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L74)<br>[attn_peer_bytes](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L83)<br>[attn_region_pair](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L88)<br>[fma](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L92) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/seam.py`<br>`3fac9a03558c18c29d6a4b4b4af16449002fafbf4ef3716eb085a30c3c094eb6`<br>[Complete implementation](../kernel_sources/blobs/3fac9a03558c18c2/seam.py)<br>[scratch_layout](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L77)<br>[attn_peer_bytes](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L86)<br>[attn_region_pair](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L91)<br>[fma](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L95) | changed bytes |
| `vllm/models/deepseek_v41/amd/mono/stages/moe.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/moe.py`<br>`00476528913b1a046a65d1491d954020461c444dbb13a1c134ffe2a1624ee792`<br>[Complete implementation](../kernel_sources/blobs/00476528913b1a04/moe.py)<br>[scratch_layout](../kernel_sources/blobs/00476528913b1a04/moe.py#L175)<br>[peer_bytes](../kernel_sources/blobs/00476528913b1a04/moe.py#L208)<br>[ffn_offset](../kernel_sources/blobs/00476528913b1a04/moe.py#L213)<br>[region_pair](../kernel_sources/blobs/00476528913b1a04/moe.py#L218) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/moe.py`<br>`1075006608f00dd534432307a83ba3e9f15685960fa0d7ce5bf9c6bd3053b2f7`<br>[Complete implementation](../kernel_sources/blobs/1075006608f00dd5/moe.py)<br>[scratch_layout](../kernel_sources/blobs/1075006608f00dd5/moe.py#L213)<br>[peer_bytes](../kernel_sources/blobs/1075006608f00dd5/moe.py#L251)<br>[ffn_offset](../kernel_sources/blobs/1075006608f00dd5/moe.py#L256)<br>[region_pair](../kernel_sources/blobs/1075006608f00dd5/moe.py#L261) | changed bytes |
| `vllm/models/deepseek_v41/amd/mono/attention/plan.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/attention/plan.py`<br>`8ded6df37a0b6681040e7a9e5919b6582b0b104d9ba63b1e1234fe5aad541fda`<br>[Complete implementation](../kernel_sources/blobs/8ded6df37a0b6681/plan.py)<br>[cdiv](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L43)<br>[round_up](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L47)<br>[Dims](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L52)<br>[token_tiles](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L101) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/attention/plan.py`<br>`9386470498616fbda768c08ff0a8091599cd32c433c74b04682bd9504fb7d57d`<br>[Complete implementation](../kernel_sources/blobs/9386470498616fbd/plan.py)<br>[flag_stride](../kernel_sources/blobs/9386470498616fbd/plan.py#L55)<br>[flag_copy](../kernel_sources/blobs/9386470498616fbd/plan.py#L60)<br>[on_copy](../kernel_sources/blobs/9386470498616fbd/plan.py#L65)<br>[ardy_tiles](../kernel_sources/blobs/9386470498616fbd/plan.py#L70) | changed bytes |

### Ordinary delayed mHC device bodies unchanged

**Before:** V09/V10 AITER delayed post/pre/RMSNorm wrapper and two Triton device bodies.

**After:** V11 retains byte-identical wrapper, device file and gfx950 delayed-seam tuning configuration.

**Device language:** Triton, unchanged device bodies

**Code/body:** Although AITER package versions change, the specific delayed mHC seam files are unchanged. JIT logs may differ because warmup coverage changes; that does not imply a different kernel body.

**Fusion boundaries:** Existing main+reduce fusion is retained. V11 separate mHC overlap path can replace the producer/consumer scheduling on FULL graphs; the unchanged AITER seam itself is not new fusion.

**Applicability / log limit:** Ordinary/fallback mHC path only; mono bypasses it. The added gfx942-only ASM seam in aiter/ops/mhc.py is not attributed to MI355X.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`<br>`34003d031b0712e55b745c397e6c519b60ff32c2b6747024b42a7151592f949c`<br>[Complete implementation](../kernel_sources/blobs/34003d031b0712e5/mhc_fused_post_pre_delayed_rmsnorm.py)<br>[mhc_fused_post_pre_delayed_rmsnorm](../kernel_sources/blobs/34003d031b0712e5/mhc_fused_post_pre_delayed_rmsnorm.py#L20) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`<br>`34003d031b0712e55b745c397e6c519b60ff32c2b6747024b42a7151592f949c`<br>[Complete implementation](../kernel_sources/blobs/34003d031b0712e5/mhc_fused_post_pre_delayed_rmsnorm.py)<br>[mhc_fused_post_pre_delayed_rmsnorm](../kernel_sources/blobs/34003d031b0712e5/mhc_fused_post_pre_delayed_rmsnorm.py#L20) | identical bytes |
| `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`<br>`d60eafdcddcc2fbb9888f0fd4350f06cdac8122207ba8fc72fd1a326834813b2`<br>[Complete implementation](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py)<br>[_mhc_fused_post_pre_delayed_rmsnorm_main_kernel](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53)<br>[_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230)<br>[_split4](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L38) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`<br>`d60eafdcddcc2fbb9888f0fd4350f06cdac8122207ba8fc72fd1a326834813b2`<br>[Complete implementation](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py)<br>[_mhc_fused_post_pre_delayed_rmsnorm_main_kernel](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53)<br>[_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230)<br>[_split4](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L38) | identical bytes |
| `aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json`<br>`309fd20e9aa660437123ae6d781932c1f7ea88ff2e59002967feac5796a589fc`<br>[Complete implementation](../kernel_sources/blobs/309fd20e9aa66043/DEFAULT.json) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/mhc/mhc_fused_post_pre_delayed_rmsnorm/DEFAULT.json`<br>`309fd20e9aa660437123ae6d781932c1f7ea88ff2e59002967feac5796a589fc`<br>[Complete implementation](../kernel_sources/blobs/309fd20e9aa66043/DEFAULT.json) | identical bytes |

### mHC overlap and Engram gate sharding

**Before:** Earlier ordinary model mHC gate work uses caller stream and common Engram implementation.

**After:** V11 adds an ordinary-path Triton post/collapse/norm producer with auxiliary-stream gate work; Engram gate work is TP-sharded.

**Device language:** Triton producers/Engram stages, AITER gate API, separate collective/stream operations

**Code/body:** mHC FULL graph capture creates stream dependencies, and Engram produces partial statistics/value slices, exchanges them, then finishes gating.

**Fusion boundaries:** mHC post/collapse/norm producer is fused; moving gate work to another stream is overlap, not single-kernel fusion. Engram retains partial-kernel, collective, final-kernel boundaries.

**Applicability / log limit:** gfx950 ordinary/fallback path; mHC ≤128 tokens, hc_mult4, no ubatching and FULL capture (not eager/piecewise). No direct overlap/Engram HIT log was found.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/mhc.py` | Absent from captured source selection | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mhc.py`<br>`9be806b5fb84bee3ce3f874e118a9697e3b788e06f9ac5b91a48e4c90060cdfa`<br>[Complete implementation](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py)<br>[supports_mhc_overlap](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py#L19)<br>[MHCOverlapStream](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py#L29)<br>[mhc_pre_delayed_overlap](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py#L52) | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/engram.py` | Absent from captured source selection | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/engram.py`<br>`6a458dd68626126142fbf7f7deba5cdc2f66bc6489feee511fe903eca9bcdecc`<br>[Complete implementation](../kernel_sources/blobs/6a458dd686261261/engram.py)<br>[_engram_gate_partials_kernel](../kernel_sources/blobs/6a458dd686261261/engram.py#L28)<br>[_engram_gate_finish_kernel](../kernel_sources/blobs/6a458dd686261261/engram.py#L99)<br>[_shards_engram_gate](../kernel_sources/blobs/6a458dd686261261/engram.py#L22)<br>[RocmEngram](../kernel_sources/blobs/6a458dd686261261/engram.py#L180) | added/absent in snapshot |
| `vllm/models/deepseek_v41/amd/model.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py`<br>`7cd417b4c36f9f5b32e8c82f68588c8df11d33e693d4e9f1d797642f62a546a3`<br>[Complete implementation](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py)<br>[DeepseekV4MoE](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L79)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L145)<br>[_use_sequence_parallel](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L170)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L175) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py`<br>`5e86e7df41b4e41d6b173356f72750d50387ab7c6200a3f75c66d6a00708b120`<br>[Complete implementation](../kernel_sources/blobs/5e86e7df41b4e41d/model.py)<br>[DeepseekV4MoE](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L99)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L165)<br>[_use_sequence_parallel](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L190)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L195) | changed bytes |
| `vllm/model_executor/kernels/mhc/triton.py` | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/triton.py`<br>`a48f030e5df658416eb7d0abadad62c4742e4004fcee0f8f0514664e77965ef2`<br>[Complete implementation](../kernel_sources/blobs/a48f030e5df65841/triton.py)<br>[_rmsnorm_nw_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L12)<br>[_hc_head_reduce_store_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L66)<br>[_mhc_pre_mix_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L145)<br>[rmsnorm_nw](../kernel_sources/blobs/a48f030e5df65841/triton.py#L39) | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/triton.py`<br>`8f61199e55adf03d8072b7e51f76daf584a89ba0e7d6312fda254668f7f2370c`<br>[Complete implementation](../kernel_sources/blobs/8f61199e55adf03d/triton.py)<br>[_rmsnorm_nw_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L22)<br>[_hc_head_reduce_store_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L76)<br>[_mhc_post_stream](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L155)<br>[_mhc_post_collapse_rms_norm_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L167) | changed bytes |

### MXFP4 Q/K producer and score bodies retained

**Before:** V09/V10 MXFP4 K-store, Q-RoPE quant and score implementations.

**After:** V11 retains these inspected producer and score files byte-for-byte.

**Device language:** Triton Q/K producers; Gluon score kernel

**Code/body:** Logs continue to expose _pa_mqa_logits_mxfp4_kernel and _q_rope_mxfp4_quant_kernel, though logging coverage differs. The score source is Gluon even when the generic JIT logger says Triton.

**Fusion boundaries:** Existing norm/RoPE/quant/cache boundaries remain. No new producer fusion is assigned to a package-version bump.

**Applicability / log limit:** Index-source work; still outside persistent mono launches. JIT warnings are observations of compilation, not every invocation.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py`<br>`0c824c1282c7348e43b954faf968eb78374e86684f03934541528a2fbbfe7a7c`<br>[Complete implementation](../kernel_sources/blobs/0c824c1282c7348e/k_norm_rope_mxfp4_cache.py)<br>[k_norm_rope_mxfp4_cache](../kernel_sources/blobs/0c824c1282c7348e/k_norm_rope_mxfp4_cache.py#L16) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py`<br>`0c824c1282c7348e43b954faf968eb78374e86684f03934541528a2fbbfe7a7c`<br>[Complete implementation](../kernel_sources/blobs/0c824c1282c7348e/k_norm_rope_mxfp4_cache.py)<br>[k_norm_rope_mxfp4_cache](../kernel_sources/blobs/0c824c1282c7348e/k_norm_rope_mxfp4_cache.py#L16) | identical bytes |
| `aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py`<br>`481e185d288607b1ab89eb6a7d3aa10d13c1ef0468bddd5d547ee9e2789ccf28`<br>[Complete implementation](../kernel_sources/blobs/481e185d288607b1/k_norm_rope_mxfp4_cache.py)<br>[_k_norm_rope_mxfp4_cache_kernel](../kernel_sources/blobs/481e185d288607b1/k_norm_rope_mxfp4_cache.py#L20) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py`<br>`481e185d288607b1ab89eb6a7d3aa10d13c1ef0468bddd5d547ee9e2789ccf28`<br>[Complete implementation](../kernel_sources/blobs/481e185d288607b1/k_norm_rope_mxfp4_cache.py)<br>[_k_norm_rope_mxfp4_cache_kernel](../kernel_sources/blobs/481e185d288607b1/k_norm_rope_mxfp4_cache.py#L20) | identical bytes |
| `aiter/ops/triton/rope/q_rope_mxfp4_quant.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/rope/q_rope_mxfp4_quant.py`<br>`02a5ddb1835b90421e524b11ad7683da488cb18344c7306efdfe2dfb70ce7828`<br>[Complete implementation](../kernel_sources/blobs/02a5ddb1835b9042/q_rope_mxfp4_quant.py)<br>[_launch_config](../kernel_sources/blobs/02a5ddb1835b9042/q_rope_mxfp4_quant.py#L17)<br>[q_rope_mxfp4_quant](../kernel_sources/blobs/02a5ddb1835b9042/q_rope_mxfp4_quant.py#L30) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/rope/q_rope_mxfp4_quant.py`<br>`02a5ddb1835b90421e524b11ad7683da488cb18344c7306efdfe2dfb70ce7828`<br>[Complete implementation](../kernel_sources/blobs/02a5ddb1835b9042/q_rope_mxfp4_quant.py)<br>[_launch_config](../kernel_sources/blobs/02a5ddb1835b9042/q_rope_mxfp4_quant.py#L17)<br>[q_rope_mxfp4_quant](../kernel_sources/blobs/02a5ddb1835b9042/q_rope_mxfp4_quant.py#L30) | identical bytes |
| `aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py`<br>`5fd9e06fbb9b8ead7496a2305a221f81577c90b8b0351e27a219eb07424809bc`<br>[Complete implementation](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py)<br>[_q_rope_mxfp4_quant_kernel](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py#L20) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py`<br>`5fd9e06fbb9b8ead7496a2305a221f81577c90b8b0351e27a219eb07424809bc`<br>[Complete implementation](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py)<br>[_q_rope_mxfp4_quant_kernel](../kernel_sources/blobs/5fd9e06fbb9b8ead/q_rope_mxfp4_quant.py#L20) | identical bytes |
| `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`<br>`f5380c2ecd995076feeca2823441b16b60e7e5f1a1c18125b512647df47243aa`<br>[Complete implementation](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py)<br>[_pa_mqa_logits_mxfp4_kernel](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392)<br>[_shuffled_offsets](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L113)<br>[_shuffled_scale_offsets](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L132)<br>[_token_major_offsets](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L145) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`<br>`f5380c2ecd995076feeca2823441b16b60e7e5f1a1c18125b512647df47243aa`<br>[Complete implementation](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py)<br>[_pa_mqa_logits_mxfp4_kernel](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392)<br>[_shuffled_offsets](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L113)<br>[_shuffled_scale_offsets](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L132)<br>[_token_major_offsets](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L145) | identical bytes |

### Ordinary sparse MLA body/dispatch changes

**Before:** Earlier gfx950 sparse main/reduce bodies.

**After:** Later shared Gluon body adds 64-bit row bases, explicit FMA inverse rotation, zero-denominator protection and BF16 gather-layout adjustment.

**Device language:** Gluon device bodies; Python launch/format dispatch

**Code/body:** The fp8_ds_mla route passes through pa_decode_sparse to this shared Gluon body. The new fp8_scalar fast path in sparse_mla.py has separate format guards and is not credited to the packed DSv4 cache.

**Fusion boundaries:** Existing inverse-RoPE/MXFP8 epilogue remains fused; these changes are body/layout/numerical robustness and dispatch changes, not proof of a new cross-operator fusion. BF16_GATHER tuning applies only when that gather format is selected.

**Applicability / log limit:** Ordinary/fallback sparse attention. Exact _sparse_mla/_sparse_mla_reduce JIT messages identify compiled candidates, not the full serving trace.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `aiter/ops/triton/attention/sparse_mla.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/sparse_mla.py`<br>`91a6420d0c115eadb7a24d39d75412a7da7a3dba99e32163513ad3e899bdcf75`<br>[Complete implementation](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py)<br>[_check_out](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L29)<br>[_cache_pointers](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L43)<br>[_get_config](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L88)<br>[_check_packed_arch](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L99) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/sparse_mla.py`<br>`156bfbfd31eecd4a562250669942ba4a5474d4c06194f67b2900840edcaaea3e`<br>[Complete implementation](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py)<br>[_check_out](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L34)<br>[_cache_pointers](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L48)<br>[_get_config](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L93)<br>[_check_packed_arch](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L104) | changed bytes |
| `aiter/ops/triton/attention/pa_decode_sparse.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/pa_decode_sparse.py`<br>`2e107a1e78f3645a159831a191391c7dd602f7e64ba3a39e22a7fcec016c452c`<br>[Complete implementation](../kernel_sources/blobs/2e107a1e78f3645a/pa_decode_sparse.py)<br>[_check_out](../kernel_sources/blobs/2e107a1e78f3645a/pa_decode_sparse.py#L63)<br>[pa_decode_sparse](../kernel_sources/blobs/2e107a1e78f3645a/pa_decode_sparse.py#L74)<br>[_as_int32_contiguous_1d](../kernel_sources/blobs/2e107a1e78f3645a/pa_decode_sparse.py#L455)<br>[_decode_num_splits_occ](../kernel_sources/blobs/2e107a1e78f3645a/pa_decode_sparse.py#L461) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/pa_decode_sparse.py`<br>`d290e3f45cb9daa7f5ece8c7449a19047b2f20422e7485186728ef2f89a851f1`<br>[Complete implementation](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py)<br>[_check_out](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L69)<br>[pa_decode_sparse](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L90)<br>[_as_int32_contiguous_1d](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L619)<br>[_decode_num_splits_occ](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L625) | changed bytes |
| `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`<br>`c71422552e29403d5a3b4863d63b937d36e84109bc319e6531134eed1adaeeb7`<br>[Complete implementation](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py)<br>[_sparse_mla](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L2160)<br>[_sparse_mla_reduce](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L2794)<br>[_max2](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L66)<br>[_rmax](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L71) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`<br>`d7c7e731d5a037389d8b6eae785b323eac3ca02691b8a8afb3687cb5829dec4a`<br>[Complete implementation](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py)<br>[_sparse_mla](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2163)<br>[_sparse_mla_reduce](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2798)<br>[_max2](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L66)<br>[_rmax](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L71) | changed bytes |

Pinned source evidence: [inventory](../evidence/recovered_source_inventory.json), [reviewed rules](../evidence/recovered_kernel_reviews.json), [V09→V11 source diff](../evidence/recovered_source_diffs/v09_to_v11.diff). Installation paths are preserved in the inventory; canonical comparison paths above normalize the earlier deepseek_v4_1 directory name.

## Named JIT kernels: observed sets and body comparisons

This table lists every symbol mentioned by the selected endpoints' JIT diagnostics. Missing messages do not imply missing execution: the monitor activates after server warmup and uses warning_once. A changed file can contain an unchanged symbol; AST comparisons below separate those cases. A name match is a source candidate, not proof of a unique binary implementation.

| Symbol | Before logged phases | After logged phases | Candidate source / symbol-body relation |
| --- | --- | --- | --- |
| `_block32_tiled_kernel` | benchmark_warmup | benchmark_warmup | `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`: identical_ast [before source](../kernel_sources/blobs/da6105b99593979c/rocm_block32_gemm.py#L101) [after source](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L107) |
| `_compute_local_logits_stats_kernel` | e2e_measurement | not logged | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: changed_ast [before source](../kernel_sources/blobs/44de886702efc818/rejection_sampler_utils.py#L208) [after source](../kernel_sources/blobs/5fdc5e618efb25be/rejection_sampler_utils.py#L218) |
| `_mask_candidates_kernel` | benchmark_warmup | not logged | `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`: identical_ast [before source](../kernel_sources/blobs/604f588fcf96d546/candidate_blocks.py#L125) [after source](../kernel_sources/blobs/0990a1da8a18d9e5/candidate_blocks.py#L125) |
| `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast [before source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53) [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L53) |
| `_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel` | benchmark_warmup | e2e_measurement | `aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py`: identical_ast [before source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230) [after source](../kernel_sources/blobs/d60eafdcddcc2fbb/mhc_fused_post_pre_delayed_rmsnorm.py#L230) |
| `_mxfp8_wo_a_bmm_kernel` | benchmark_warmup | benchmark_warmup | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`: identical_ast [before source](../kernel_sources/blobs/fa6a9c47537abdc1/rocm_aiter_mla_sparse.py#L1743) [after source](../kernel_sources/blobs/8b5e4ad1de1d9eaa/rocm_aiter_mla_sparse.py#L1965) |
| `_pa_mqa_logits_mxfp4_kernel` | benchmark_warmup | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_mqa_logits_mxfp4.py`: identical_ast [before source](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392) [after source](../kernel_sources/blobs/f5380c2ecd995076/pa_mqa_logits_mxfp4.py#L1392) |
| `_prepare_dflash_inputs_kernel` | benchmark_warmup | not logged | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`: changed_ast [before source](../kernel_sources/blobs/28fb34eedddf62ac/speculator.py#L624) [after source](../kernel_sources/blobs/61be7bf7dd274a71/speculator.py#L706) |
| `_sparse_mla` | not logged | benchmark_warmup | `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`: changed_ast [before source](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L2160) [after source](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2163) |
| `_sparse_mla_reduce` | not logged | e2e_measurement | `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`: changed_ast [before source](../kernel_sources/blobs/c71422552e29403d/sparse_mla.py#L2794) [after source](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2798) |
| `launch_radix_topk_one_block` | not logged | benchmark_warmup | `aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py`: identical_ast [before source](../kernel_sources/blobs/b58fab682f7e4a85/radix_topk_one_block.py#L1400) [after source](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py#L1469) |

The [structured pair record](../data/recovered_pair_records.json) retains complete generated kernel names (including FlyDSL variant parameters), candidate source paths, file hashes, symbol code/AST hashes and line ranges. Triton JIT log labeling can include Gluon implementations; source decorators distinguish them.

No GPU trace, kernel time ranking, per-step launch count, or causal kernel speedup is claimed. Admission, graph capture, batching, quick-reduce quantization and synthetic DSpark acceptance are retained in the original CI evidence and can affect E2E deltas.
