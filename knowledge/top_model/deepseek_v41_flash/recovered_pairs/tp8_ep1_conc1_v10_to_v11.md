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

## Reviewed kernel bodies and fusion boundaries

These findings use recovered container file bytes. V01's tag was not historically digest-pinned; V09–V11 were. Source facts establish possible launch boundaries; only the log observations above establish which diagnostic path was reached.

### Tiny-M activation quantization fused into block32 GEMM

**Before:** V10 already contains the fused tiny-M block32 quant+GEMM path.

**After:** V11 retains the entire rocm_block32_gemm.py file byte-for-byte.

**Device language:** Triton, including tl.dot_scaled

**Code/body:** Both TP8 c1 endpoints have 41 HIT log lines with the same set of M/N/K/tile signatures; all are before measurement. These diagnostic line counts are not launch counts.

**Fusion boundaries:** Existing quant+GEMM fusion is retained; it cannot be claimed as newly introduced fusion for V10→V11.

**Applicability / log limit:** Same tiny-M/packed/contiguous activation guards; no source-level change in this file. Logging once per shape prevents using the HIT count as call frequency.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py` | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`<br>`d0db2d8d69d5b8f4c45015a9b4de6f380ece0e04862b847479940c090929f926`<br>[Complete implementation](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py)<br>[_block32_tiled_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L107)<br>[_block32_packed_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L188)<br>[_block32_splitk_reduce_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L400)<br>[_split_tile](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L73) | `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`<br>`d0db2d8d69d5b8f4c45015a9b4de6f380ece0e04862b847479940c090929f926`<br>[Complete implementation](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py)<br>[_block32_tiled_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L107)<br>[_block32_packed_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L188)<br>[_block32_splitk_reduce_kernel](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L400)<br>[_split_tile](../kernel_sources/blobs/d0db2d8d69d5b8f4/rocm_block32_gemm.py#L73) | identical bytes |

### DSpark Markov sampling on gfx950

**Before:** V09 uses ordinary sampling; V10 contains a fused helper whose architecture guard admits gfx942 only.

**After:** V11 guard admits gfx950; exact endpoint logs report five steps, vocabulary 129280, rank 256, 505 blocks.

**Device language:** Triton sampling core; separate base-logit GEMM

**Code/body:** The per-step kernel loads the previous token via prior block maxima, computes Markov bias plus base logits and block Gumbel-max. Final kernel resolves the last token. The info message is emitted after eligibility checks and before launches.

**Fusion boundaries:** Sampling core uses five per-step kernels plus one final resolver. Base-logit computation, input gathers and scratch initialization are outside this count. There is no single launch covering all five sequential steps.

**Applicability / log limit:** gfx950 --monokernel, at most five request rows, supported sampler/dtypes. All observed first Markov messages precede E2E measurement; they do not prove sustained coverage at higher concurrency.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/ops/fused_markov.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/fused_markov.py`<br>`3df910cf7e3cd06cfc15bf2578061addf97e80c50b897576299273211809c10f`<br>[Complete implementation](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py)<br>[_markov_step_kernel](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L56)<br>[_markov_last_kernel](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L159)<br>[_scratch](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L185)<br>[_why_not](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L203) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/fused_markov.py`<br>`28fb830c9bfa220baaa33f8f10a5db4faf7b7bdd654cd4ff917122c9729e6084`<br>[Complete implementation](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py)<br>[_markov_step_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L58)<br>[_markov_last_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L161)<br>[_scratch](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L187)<br>[_why_not](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L205) | changed bytes |
| `vllm/models/deepseek_v41/amd/ops/__init__.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py`<br>`3425a93ca2d9dba46db70dd112743d42816ff84e6ab06786fdd02da38f76129b`<br>[Complete implementation](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py)<br>[record_monokernel](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L26)<br>[enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L37)<br>[swa_decode_meta_enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L50) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py`<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[Complete implementation](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | changed bytes |
| `vllm/models/deepseek_v41/amd/dspark.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py`<br>`3ddc30bf30261b46d83b33588fc31bdbb355d1404bc9966c08637283c2270889`<br>[Complete implementation](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py)<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py#L65)<br>[_insert_context_kv](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py#L228)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py#L297) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py`<br>`774266398fa2a49621441a9d36a6453ea7bed95f8a02deaa88aa741d9bf1cec9`<br>[Complete implementation](../kernel_sources/blobs/774266398fa2a496/dspark.py)<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/774266398fa2a496/dspark.py#L68)<br>[_insert_context_kv](../kernel_sources/blobs/774266398fa2a496/dspark.py#L231)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/774266398fa2a496/dspark.py#L300) | changed bytes |

### SWA metadata helper admitted on gfx950

**Before:** V09 has no helper; V10 helper is gfx942-gated.

**After:** V11 architecture predicate admits gfx950.

**Device language:** Triton device body; Python architecture/shape guards

**Code/body:** The existing Triton _decode_swa_meta_kernel writes validity, dense window data and ragged indices/indptr directly into graph buffers.

**Fusion boundaries:** One metadata-core launch replaces the prior validity/window/count/cumsum/compaction/copy sequence described in source. First-use allocation, token-to-request preparation and tile-scheduler work remain outside that core count.

**Applicability / log limit:** --monokernel, pure decode, ≤32 decode token rows, supported causal or DSpark non-causal metadata. No specific HIT message was found; source eligibility only, not measured launch elimination.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py`<br>`e0e258cdd7142cab4db48ec1ed3e249890fa390f4d09da27a8ea11f17180a2be`<br>[Complete implementation](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py)<br>[_decode_swa_meta_kernel](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py#L45)<br>[launch](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py#L134)<br>[build](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py#L182) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py`<br>`2e8ec7dd48ded6d28156699a1473bc76beb6a597c04b59a31f67b31651750afe`<br>[Complete implementation](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py)<br>[_decode_swa_meta_kernel](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L45)<br>[launch](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L134)<br>[build](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L182) | changed bytes |
| `vllm/models/deepseek_v41/amd/ops/__init__.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py`<br>`3425a93ca2d9dba46db70dd112743d42816ff84e6ab06786fdd02da38f76129b`<br>[Complete implementation](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py)<br>[record_monokernel](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L26)<br>[enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L37)<br>[swa_decode_meta_enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L50) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py`<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[Complete implementation](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | changed bytes |
| `vllm/models/deepseek_v41/amd/rocm.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/rocm.py`<br>`e5f8f25426f79a3d96a0e1050eb0e33d78f72588391406660277f6183718ff59`<br>[Complete implementation](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py)<br>[_compute_topk_lens_kernel](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L140)<br>[_pack_global_topk_ragged_kernel](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L166)<br>[_trust_dsv4_extra_cache_nan_free](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L73)<br>[_aiter_sparse_mla_enabled](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L86) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/rocm.py`<br>`e5f8f25426f79a3d96a0e1050eb0e33d78f72588391406660277f6183718ff59`<br>[Complete implementation](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py)<br>[_compute_topk_lens_kernel](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L140)<br>[_pack_global_topk_ragged_kernel](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L166)<br>[_trust_dsv4_extra_cache_nan_free](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L73)<br>[_aiter_sparse_mla_enabled](../kernel_sources/blobs/e5f8f25426f79a3d/rocm.py#L86) | identical bytes |

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
| `vllm/models/deepseek_v41/amd/model.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py`<br>`ae488e391ed6d86ad5d20f92cdbf049cb52298d0d8fca1be3f5569516222f04b`<br>[Complete implementation](../kernel_sources/blobs/ae488e391ed6d86a/model.py)<br>[DeepseekV4MoE](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L86)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L152)<br>[_use_sequence_parallel](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L177)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L182) | `/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py`<br>`5e86e7df41b4e41d6b173356f72750d50387ab7c6200a3f75c66d6a00708b120`<br>[Complete implementation](../kernel_sources/blobs/5e86e7df41b4e41d/model.py)<br>[DeepseekV4MoE](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L99)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L165)<br>[_use_sequence_parallel](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L190)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L195) | changed bytes |
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

**After:** Later shared Gluon body adds 64-bit row bases, explicit FMA inverse rotation, zero-denominator protection.

**Device language:** Gluon device bodies; Python launch/format dispatch

**Code/body:** The fp8_ds_mla route passes through pa_decode_sparse to this shared Gluon body. The new fp8_scalar fast path in sparse_mla.py has separate format guards and is not credited to the packed DSv4 cache.

**Fusion boundaries:** Existing inverse-RoPE/MXFP8 epilogue remains fused; these changes are body/layout/numerical robustness and dispatch changes, not proof of a new cross-operator fusion. The BF16 gather-layout adjustment already existed in V10 and is not new in this pair.

**Applicability / log limit:** Ordinary/fallback sparse attention. Exact _sparse_mla/_sparse_mla_reduce JIT messages identify compiled candidates, not the full serving trace.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `aiter/ops/triton/attention/sparse_mla.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/sparse_mla.py`<br>`91a6420d0c115eadb7a24d39d75412a7da7a3dba99e32163513ad3e899bdcf75`<br>[Complete implementation](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py)<br>[_check_out](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L29)<br>[_cache_pointers](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L43)<br>[_get_config](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L88)<br>[_check_packed_arch](../kernel_sources/blobs/91a6420d0c115ead/sparse_mla.py#L99) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/sparse_mla.py`<br>`156bfbfd31eecd4a562250669942ba4a5474d4c06194f67b2900840edcaaea3e`<br>[Complete implementation](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py)<br>[_check_out](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L34)<br>[_cache_pointers](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L48)<br>[_get_config](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L93)<br>[_check_packed_arch](../kernel_sources/blobs/156bfbfd31eecd4a/sparse_mla.py#L104) | changed bytes |
| `aiter/ops/triton/attention/pa_decode_sparse.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/pa_decode_sparse.py`<br>`2f83e1433b6b39e53667b3637249761fbe02cf95887cb60153ebdf4e7e8bd893`<br>[Complete implementation](../kernel_sources/blobs/2f83e1433b6b39e5/pa_decode_sparse.py)<br>[_check_out](../kernel_sources/blobs/2f83e1433b6b39e5/pa_decode_sparse.py#L67)<br>[pa_decode_sparse](../kernel_sources/blobs/2f83e1433b6b39e5/pa_decode_sparse.py#L78)<br>[_as_int32_contiguous_1d](../kernel_sources/blobs/2f83e1433b6b39e5/pa_decode_sparse.py#L459)<br>[_decode_num_splits_occ](../kernel_sources/blobs/2f83e1433b6b39e5/pa_decode_sparse.py#L465) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/pa_decode_sparse.py`<br>`d290e3f45cb9daa7f5ece8c7449a19047b2f20422e7485186728ef2f89a851f1`<br>[Complete implementation](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py)<br>[_check_out](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L69)<br>[pa_decode_sparse](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L90)<br>[_as_int32_contiguous_1d](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L619)<br>[_decode_num_splits_occ](../kernel_sources/blobs/d290e3f45cb9daa7/pa_decode_sparse.py#L625) | changed bytes |
| `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`<br>`02c9ac2762fb8e61d8a8356098f4614d8a20346874842db4175d5facb150d5d6`<br>[Complete implementation](../kernel_sources/blobs/02c9ac2762fb8e61/sparse_mla.py)<br>[_sparse_mla](../kernel_sources/blobs/02c9ac2762fb8e61/sparse_mla.py#L2163)<br>[_sparse_mla_reduce](../kernel_sources/blobs/02c9ac2762fb8e61/sparse_mla.py#L2799)<br>[_max2](../kernel_sources/blobs/02c9ac2762fb8e61/sparse_mla.py#L66)<br>[_rmax](../kernel_sources/blobs/02c9ac2762fb8e61/sparse_mla.py#L71) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py`<br>`d7c7e731d5a037389d8b6eae785b323eac3ca02691b8a8afb3687cb5829dec4a`<br>[Complete implementation](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py)<br>[_sparse_mla](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2163)<br>[_sparse_mla_reduce](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L2798)<br>[_max2](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L66)<br>[_rmax](../kernel_sources/blobs/d7c7e731d5a03738/sparse_mla.py#L71) | changed bytes |

### FlyDSL radix top-k body and observed symbol

**Before:** V10 one-block radix selector.

**After:** V11 adds a full_rows factory option and updates FlyDSL vector construction APIs.

**Device language:** FlyDSL

**Code/body:** full_rows uses fixed width instead of row_ends and adds a _full symbol suffix. The TP8 V11 measured-phase log names radix_topk_one_block_decode_short_k512_b256_w64_v0_s0, with no _full suffix, so that new specialization is not credited to this observation.

**Fusion boundaries:** No new cross-operator fusion established. A later JIT warning is not proof that the earlier endpoint never ran a corresponding cached top-k kernel.

**Applicability / log limit:** TP8 c1 measured-phase JIT at 06:59:58 in the V11 endpoint; consult its exact phase window. Distinguish wrapper symbol and generated device symbol.

| Comparison path | Before installation / implementation | After installation / implementation | File relation |
| --- | --- | --- | --- |
| `aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py` | `/usr/local/lib/python3.12/dist-packages/aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py`<br>`3e9e7f4386532334c9b3fdaf304afa12fe0e82be040afe43ecf4f6b155b3ae4c`<br>[Complete implementation](../kernel_sources/blobs/3e9e7f4386532334/radix_topk_one_block.py)<br>[build_radix_topk_one_block_module.launch_radix_topk_one_block](../kernel_sources/blobs/3e9e7f4386532334/radix_topk_one_block.py#L1463)<br>[build_radix_topk_one_block_body.run_one_block_body](../kernel_sources/blobs/3e9e7f4386532334/radix_topk_one_block.py#L207)<br>[build_radix_topk_one_block_module.radix_topk_one_block_kernel](../kernel_sources/blobs/3e9e7f4386532334/radix_topk_one_block.py#L1434)<br>[_build_bitonic_schedule](../kernel_sources/blobs/3e9e7f4386532334/radix_topk_one_block.py#L65) | `/usr/local/lib/python3.12/dist-packages/aiter/ops/flydsl/kernels/topk/radix_topk_one_block.py`<br>`37360c50b7cfe5bf4cc0a52f74be37c94453a31e28803aeb17fbb71c7f682806`<br>[Complete implementation](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py)<br>[build_radix_topk_one_block_module.launch_radix_topk_one_block](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py#L1469)<br>[build_radix_topk_one_block_body.run_one_block_body](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py#L207)<br>[build_radix_topk_one_block_module.radix_topk_one_block_kernel](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py#L1440)<br>[_build_bitonic_schedule](../kernel_sources/blobs/37360c50b7cfe5bf/radix_topk_one_block.py#L65) | changed bytes |

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
