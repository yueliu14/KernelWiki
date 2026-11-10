# V10 → V11 tp8 ep1

[Image changes](../kernel_evolution.md) · [Matrix](../configuration_history.md) · [Evidence rules](../../README.md#evidence)

Before: `vllm/vllm-openai-rocm:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1`. After: `vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808`.

Configuration selector: `{"decode_ep": 1, "decode_tp": 8}`. Source-verified changes; GPU-time rank and causal contributions are unavailable.

Applicable measured pairs: [TP8/EP1 conc1 / GPU](../pairs/tp8_ep1_conc1_off_2026-11-03_to_2026-11-09.md).

Required required_cli: `{"after": {"moe-backend": "aiter", "monokernel": true}, "before": {"moe-backend": "aiter", "monokernel": true}}`.

<a id="kernel-1"></a>

## SWA decode metadata fusion enabled on MI355X

**Before:** V10 has one-launch SWA helper gated to gfx942; MI355X follows the ordinary metadata sequence.

**After:** monokernel_cdna3_or_4 admits gfx950: _decode_swa_meta_kernel writes dense indices/lens, validity, ragged indices/indptr and graph buffers.

**Device language:** Triton -> Triton; expanded architecture gate.

**Body change:** Helper body remains conceptually the single-program metadata builder; V11 broadens dispatch to CDNA4.

**Fusion:** One helper launch replaces the ordinary sequence of validity, zero-fill, window, counts/cumsum/compaction and graph copies. The source describes eight prior launches per build; no original CI trace verifies realized counts.

**Applicability:** --monokernel, pure decode and at most 32 decode token rows. Applies to causal target and supported non-causal DSpark metadata. Client concurrency alone cannot establish the 32-row guard.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py)<br>`e0e258cdd7142cab4db48ec1ed3e249890fa390f4d09da27a8ea11f17180a2be`<br>[_decode_swa_meta_kernel](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py#L45)<br>[launch](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py#L134)<br>[build](../kernel_sources/blobs/e0e258cdd7142cab/swa_decode_meta.py#L182) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py)<br>`2e8ec7dd48ded6d28156699a1473bc76beb6a597c04b59a31f67b31651750afe`<br>[_decode_swa_meta_kernel](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L45)<br>[launch](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L134)<br>[build](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L182) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py)<br>`3425a93ca2d9dba46db70dd112743d42816ff84e6ab06786fdd02da38f76129b`<br>[record_monokernel](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L26)<br>[enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L37)<br>[swa_decode_meta_enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L50) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | same_repository_path; different_file_bytes |

Evidence: [evidence/source_diffs/v10_to_v11.diff](../evidence/source_diffs/v10_to_v11.diff), [evidence/image_source_inventory.json](../evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](../evidence/ci_launch_configs.json).

<a id="kernel-2"></a>

## DSpark Markov sampling fusion enabled on MI355X

**Before:** V10 gfx950 uses the ordinary sequential Markov embedding/GEMV/add/Gumbel/argmax/gather/copy loop; fused helper is gfx942-only.

**After:** V11 permits gfx950 for at most five request rows. Each draft step computes vocabulary-block bias plus logits and block Gumbel-max in one kernel; next step resolves preceding block maxima.

**Device language:** Triton -> Triton.

**Body change:** Architecture gate expands and MAX_ROWS_GFX950=5 limits selection; helper preserves request/step/position/token-dependent noise and tie policy.

**Fusion:** Five drafts use one launch per step plus one final token-resolution launch, not one launch for all drafts. The helper stores draft logits; Markov GEMV summation order can differ in the last BF16 bit.

**Applicability:** --monokernel, supported sequential DSpark sampler and at most five request rows. Larger batches use the ordinary sampler. The benchmark uses synthetic acceptance length 3.51, which limits conclusions about real acceptance.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/fused_markov.py](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py)<br>`3df910cf7e3cd06cfc15bf2578061addf97e80c50b897576299273211809c10f`<br>[_markov_step_kernel](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L56)<br>[_markov_last_kernel](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L159)<br>[_scratch](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L185)<br>[_why_not](../kernel_sources/blobs/3df910cf7e3cd06c/fused_markov.py#L203) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/fused_markov.py](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py)<br>`28fb830c9bfa220baaa33f8f10a5db4faf7b7bdd654cd4ff917122c9729e6084`<br>[_markov_step_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L58)<br>[_markov_last_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L161)<br>[_scratch](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L187)<br>[_why_not](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L205) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py)<br>`3425a93ca2d9dba46db70dd112743d42816ff84e6ab06786fdd02da38f76129b`<br>[record_monokernel](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L26)<br>[enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L37)<br>[swa_decode_meta_enabled](../kernel_sources/blobs/3425a93ca2d9dba4/__init__.py#L50) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py)<br>`3ddc30bf30261b46d83b33588fc31bdbb355d1404bc9966c08637283c2270889`<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py#L65)<br>[_insert_context_kv](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py#L228)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/3ddc30bf30261b46/dspark.py#L297) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py](../kernel_sources/blobs/774266398fa2a496/dspark.py)<br>`774266398fa2a49621441a9d36a6453ea7bed95f8a02deaa88aa741d9bf1cec9`<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/774266398fa2a496/dspark.py#L68)<br>[_insert_context_kv](../kernel_sources/blobs/774266398fa2a496/dspark.py#L231)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/774266398fa2a496/dspark.py#L300) | same_repository_path; different_file_bytes |

Evidence: [evidence/source_diffs/v10_to_v11.diff](../evidence/source_diffs/v10_to_v11.diff), [evidence/image_source_inventory.json](../evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](../evidence/ci_launch_configs.json).

<a id="kernel-3"></a>

## Ordinary-path mHC overlap and Engram TP sharding

**Before:** mHC gate preparation is on the caller stream; Engram uses the common implementation.

**After:** New amd/mhc.py overlaps gate projection on an auxiliary stream after a Triton post/collapse/RMSNorm producer; amd/engram.py shards gate work across TP ranks.

**Device language:** Triton fused producer + AITER gate API; Triton Engram partial/final kernels with a separate collective.

**Body change:** Adds supports_mhc_overlap and mhc_pre_delayed_overlap; Engram creates gate partial sums, all-gathers value/partials, then finishes normalization/gating.

**Fusion:** mHC post+collapse+norm is producer fusion; auxiliary-stream gate work is overlap, not a whole-seam single kernel. Engram still has partial kernel, collective and finish kernel boundaries.

**Applicability:** gfx950 ordinary/fallback model path; mHC requires hc_mult=4, fused AITER support, no ubatching and at most 128 tokens. Mono-covered seams bypass this ordinary path. Engram injection remains outside mono K1/K2. mHC overlap is installed only while capturing a FULL graph (not piecewise or eager), so replay follows those graph edges.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| dispatch and body | Absent / separate existing operations | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mhc.py](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py)<br>`9be806b5fb84bee3ce3f874e118a9697e3b788e06f9ac5b91a48e4c90060cdfa`<br>[supports_mhc_overlap](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py#L19)<br>[MHCOverlapStream](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py#L29)<br>[mhc_pre_delayed_overlap](../kernel_sources/blobs/9be806b5fb84bee3/mhc.py#L52) | added; different_file_bytes |
| dispatch and body | Absent / separate existing operations | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/engram.py](../kernel_sources/blobs/6a458dd686261261/engram.py)<br>`6a458dd68626126142fbf7f7deba5cdc2f66bc6489feee511fe903eca9bcdecc`<br>[_engram_gate_partials_kernel](../kernel_sources/blobs/6a458dd686261261/engram.py#L28)<br>[_engram_gate_finish_kernel](../kernel_sources/blobs/6a458dd686261261/engram.py#L99)<br>[_shards_engram_gate](../kernel_sources/blobs/6a458dd686261261/engram.py#L22)<br>[RocmEngram](../kernel_sources/blobs/6a458dd686261261/engram.py#L180) | added; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py](../kernel_sources/blobs/ae488e391ed6d86a/model.py)<br>`ae488e391ed6d86ad5d20f92cdbf049cb52298d0d8fca1be3f5569516222f04b`<br>[DeepseekV4MoE](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L86)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L152)<br>[_use_sequence_parallel](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L177)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/ae488e391ed6d86a/model.py#L182) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py](../kernel_sources/blobs/5e86e7df41b4e41d/model.py)<br>`5e86e7df41b4e41d6b173356f72750d50387ab7c6200a3f75c66d6a00708b120`<br>[DeepseekV4MoE](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L99)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L165)<br>[_use_sequence_parallel](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L190)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L195) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/triton.py](../kernel_sources/blobs/a48f030e5df65841/triton.py)<br>`a48f030e5df658416eb7d0abadad62c4742e4004fcee0f8f0514664e77965ef2`<br>[_rmsnorm_nw_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L12)<br>[_hc_head_reduce_store_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L66)<br>[_mhc_pre_mix_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L145)<br>[rmsnorm_nw](../kernel_sources/blobs/a48f030e5df65841/triton.py#L39) | [/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/triton.py](../kernel_sources/blobs/8f61199e55adf03d/triton.py)<br>`8f61199e55adf03d8072b7e51f76daf584a89ba0e7d6312fda254668f7f2370c`<br>[_rmsnorm_nw_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L22)<br>[_hc_head_reduce_store_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L76)<br>[_mhc_post_stream](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L155)<br>[_mhc_post_collapse_rms_norm_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L167) | same_repository_path; different_file_bytes |

Evidence: [evidence/source_diffs/v10_to_v11.diff](../evidence/source_diffs/v10_to_v11.diff), [evidence/image_source_inventory.json](../evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](../evidence/ci_launch_configs.json).

## Excluded attributions

- TP8 support already existed in V10, so it is not new fusion for this TP8 pair.
- MonoBuild.tuned is TP2/ratio2-only and cannot explain TP8 conc1 gains. MI325X/gfx942-only kernels are excluded.
