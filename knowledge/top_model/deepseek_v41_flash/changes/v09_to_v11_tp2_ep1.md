# V09 → V11 tp2 ep1

[Image changes](../kernel_evolution.md) · [Matrix](../configuration_history.md) · [Evidence rules](../../README.md#evidence)

Before: `vllm/vllm-openai-rocm:nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f`. After: `vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808`.

Configuration selector: `{"decode_ep": 1, "decode_tp": 2}`. Source-verified changes; GPU-time rank and causal contributions are unavailable.

Applicable measured pairs: [TP2/EP1 conc1 / GPU](../pairs/tp2_ep1_conc1_off_2026-10-20_to_2026-11-09.md), [TP2/EP1 conc2 / GPU](../pairs/tp2_ep1_conc2_off_2026-10-20_to_2026-11-09.md), [TP2/EP1 conc4 / GPU](../pairs/tp2_ep1_conc4_off_2026-10-20_to_2026-11-09.md), [TP2/EP1 conc8 / GPU](../pairs/tp2_ep1_conc8_off_2026-10-20_to_2026-11-09.md), [TP2/EP1 conc16 / GPU](../pairs/tp2_ep1_conc16_off_2026-10-20_to_2026-11-09.md), [TP2/EP1 conc32 / GPU](../pairs/tp2_ep1_conc32_off_2026-10-20_to_2026-11-09.md), [TP2/EP1 conc64 / GPU](../pairs/tp2_ep1_conc64_off_2026-10-20_to_2026-11-09.md).

Required required_cli: `{"after": {"moe-backend": "aiter", "monokernel": true}, "before": {"moe-backend": "aiter", "monokernel": true}}`.

<a id="kernel-1"></a>

## SWA decode metadata fusion enabled on MI355X

**Before:** V09 has no amd/ops SWA helper; MI355X follows the ordinary metadata sequence.

**After:** monokernel_cdna3_or_4 admits gfx950: _decode_swa_meta_kernel writes dense indices/lens, validity, ragged indices/indptr and graph buffers.

**Device language:** Triton -> Triton; expanded architecture gate.

**Body change:** Helper body remains conceptually the single-program metadata builder; V11 broadens dispatch to CDNA4.

**Fusion:** One helper launch replaces the ordinary sequence of validity, zero-fill, window, counts/cumsum/compaction and graph copies. The source describes eight prior launches per build; no original CI trace verifies realized counts.

**Applicability:** --monokernel, pure decode and at most 32 decode token rows. Applies to causal target and supported non-causal DSpark metadata. Client concurrency alone cannot establish the 32-row guard.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| dispatch and body | Absent / separate existing operations | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py)<br>`2e8ec7dd48ded6d28156699a1473bc76beb6a597c04b59a31f67b31651750afe`<br>[_decode_swa_meta_kernel](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L45)<br>[launch](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L134)<br>[build](../kernel_sources/blobs/2e8ec7dd48ded6d2/swa_decode_meta.py#L182) | added; different_file_bytes |
| dispatch and body | Absent / separate existing operations | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | added; different_file_bytes |

Evidence: [evidence/source_diffs/v09_to_v11.diff](../evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](../evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](../evidence/ci_launch_configs.json).

<a id="kernel-2"></a>

## DSpark Markov sampling fusion enabled on MI355X

**Before:** V09 uses the ordinary sequential Markov embedding/GEMV/add/Gumbel/argmax/gather/copy loop.

**After:** V11 permits gfx950 for at most five request rows. Each draft step computes vocabulary-block bias plus logits and block Gumbel-max in one kernel; next step resolves preceding block maxima.

**Device language:** Triton -> Triton.

**Body change:** Architecture gate expands and MAX_ROWS_GFX950=5 limits selection; helper preserves request/step/position/token-dependent noise and tie policy.

**Fusion:** Five drafts use one launch per step plus one final token-resolution launch, not one launch for all drafts. The helper stores draft logits; Markov GEMV summation order can differ in the last BF16 bit.

**Applicability:** --monokernel, supported sequential DSpark sampler and at most five request rows. Larger batches use the ordinary sampler. The benchmark uses synthetic acceptance length 3.51, which limits conclusions about real acceptance.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| dispatch and body | Absent / separate existing operations | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/fused_markov.py](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py)<br>`28fb830c9bfa220baaa33f8f10a5db4faf7b7bdd654cd4ff917122c9729e6084`<br>[_markov_step_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L58)<br>[_markov_last_kernel](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L161)<br>[_scratch](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L187)<br>[_why_not](../kernel_sources/blobs/28fb830c9bfa220b/fused_markov.py#L205) | added; different_file_bytes |
| dispatch and body | Absent / separate existing operations | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/ops/__init__.py](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py)<br>`094744b5cdbe83eeefd0e5dd92df36094d85b22dc27251ab1dc0ffb3640cfa2b`<br>[record_monokernel](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L27)<br>[enabled](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L38)<br>[monokernel_cdna3_or_4](../kernel_sources/blobs/094744b5cdbe83ee/__init__.py#L51) | added; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py)<br>`b12b728bbdd1d77b2b3dfa7852631d539ad41ff2b312f59f596e7b20dce48d88`<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py#L60)<br>[_insert_context_kv](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py#L223)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/b12b728bbdd1d77b/dspark.py#L292) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/dspark.py](../kernel_sources/blobs/774266398fa2a496/dspark.py)<br>`774266398fa2a49621441a9d36a6453ea7bed95f8a02deaa88aa741d9bf1cec9`<br>[DSparkDeepseekV4Model](../kernel_sources/blobs/774266398fa2a496/dspark.py#L68)<br>[_insert_context_kv](../kernel_sources/blobs/774266398fa2a496/dspark.py#L231)<br>[DSparkDeepseekV4ForCausalLM](../kernel_sources/blobs/774266398fa2a496/dspark.py#L300) | same_repository_path; different_file_bytes |

Evidence: [evidence/source_diffs/v09_to_v11.diff](../evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](../evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](../evidence/ci_launch_configs.json).

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
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py)<br>`7cd417b4c36f9f5b32e8c82f68588c8df11d33e693d4e9f1d797642f62a546a3`<br>[DeepseekV4MoE](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L79)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L145)<br>[_use_sequence_parallel](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L170)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/7cd417b4c36f9f5b/model.py#L175) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/model.py](../kernel_sources/blobs/5e86e7df41b4e41d/model.py)<br>`5e86e7df41b4e41d6b173356f72750d50387ab7c6200a3f75c66d6a00708b120`<br>[DeepseekV4MoE](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L99)<br>[_select_dsv4_attn_cls](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L165)<br>[_use_sequence_parallel](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L190)<br>[DeepseekV4DecoderLayer](../kernel_sources/blobs/5e86e7df41b4e41d/model.py#L195) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/triton.py](../kernel_sources/blobs/a48f030e5df65841/triton.py)<br>`a48f030e5df658416eb7d0abadad62c4742e4004fcee0f8f0514664e77965ef2`<br>[_rmsnorm_nw_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L12)<br>[_hc_head_reduce_store_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L66)<br>[_mhc_pre_mix_kernel](../kernel_sources/blobs/a48f030e5df65841/triton.py#L145)<br>[rmsnorm_nw](../kernel_sources/blobs/a48f030e5df65841/triton.py#L39) | [/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/triton.py](../kernel_sources/blobs/8f61199e55adf03d/triton.py)<br>`8f61199e55adf03d8072b7e51f76daf584a89ba0e7d6312fda254668f7f2370c`<br>[_rmsnorm_nw_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L22)<br>[_hc_head_reduce_store_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L76)<br>[_mhc_post_stream](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L155)<br>[_mhc_post_collapse_rms_norm_kernel](../kernel_sources/blobs/8f61199e55adf03d/triton.py#L167) | same_repository_path; different_file_bytes |

Evidence: [evidence/source_diffs/v09_to_v11.diff](../evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](../evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](../evidence/ci_launch_configs.json).

<a id="kernel-4"></a>

## TP2 ratio-2 mono body and scheduling tuning

**Before:** Generic persistent K1/K2 scheduling and intermediate stores.

**After:** MonoBuild.tuned selects a gfx950 target-only TP2 compression-ratio-2 build; other shapes retain generic code.

**Device language:** FlyDSL -> FlyDSL.

**Body change:** Tuned seam combines TP reduction and residual slice, publishes normalized inputs before gates; MoE down stages pipeline weight loads/MFMA rounds and use cooperative shared-expert work. Attention readiness scratch/publication changes.

**Fusion:** Retains two persistent launches but fuses/reorders internal seam work and reduces internal handoff traffic; no new layer-wide launch collapse is claimed.

**Applicability:** Only TP2 target layers, ratio 2, ordinary target routing (384 experts/top-6) and supported row/residency guards. Excludes TP4, TP8 and DSpark routing.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/layer.py](../kernel_sources/blobs/c9e2791bce448a40/layer.py)<br>`c9e2791bce448a4041269053f97541b86aa085407e0d6e561eebb99acc20a5d6`<br>[build_mono_k1.k1](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L230)<br>[build_mono_k1.launch](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L369)<br>[build_mono_k2.k2](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L621)<br>[build_mono_k2.launch](../kernel_sources/blobs/c9e2791bce448a40/layer.py#L782) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/layer.py](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py)<br>`ae6abc3c8c3824f985e341c7b5539d41273cafe88aebce6a75fc31a758f0fa5d`<br>[build_mono_k1.k1](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L315)<br>[build_mono_k1.launch](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L460)<br>[build_mono_k2.k2](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L797)<br>[build_mono_k2.launch](../kernel_sources/blobs/ae6abc3c8c3824f9/layer.py#L963) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/seam.py](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py)<br>`0e196c1e62a4d9a41e99ec551eefb9680c800326a91bbe978d7ed3163dd15643`<br>[scratch_layout](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L74)<br>[attn_peer_bytes](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L83)<br>[attn_region_pair](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L88)<br>[fma](../kernel_sources/blobs/0e196c1e62a4d9a4/seam.py#L92) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/seam.py](../kernel_sources/blobs/3fac9a03558c18c2/seam.py)<br>`3fac9a03558c18c29d6a4b4b4af16449002fafbf4ef3716eb085a30c3c094eb6`<br>[scratch_layout](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L77)<br>[attn_peer_bytes](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L86)<br>[attn_region_pair](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L91)<br>[fma](../kernel_sources/blobs/3fac9a03558c18c2/seam.py#L95) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/moe.py](../kernel_sources/blobs/00476528913b1a04/moe.py)<br>`00476528913b1a046a65d1491d954020461c444dbb13a1c134ffe2a1624ee792`<br>[scratch_layout](../kernel_sources/blobs/00476528913b1a04/moe.py#L175)<br>[peer_bytes](../kernel_sources/blobs/00476528913b1a04/moe.py#L208)<br>[ffn_offset](../kernel_sources/blobs/00476528913b1a04/moe.py#L213)<br>[region_pair](../kernel_sources/blobs/00476528913b1a04/moe.py#L218) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/stages/moe.py](../kernel_sources/blobs/1075006608f00dd5/moe.py)<br>`1075006608f00dd534432307a83ba3e9f15685960fa0d7ce5bf9c6bd3053b2f7`<br>[scratch_layout](../kernel_sources/blobs/1075006608f00dd5/moe.py#L213)<br>[peer_bytes](../kernel_sources/blobs/1075006608f00dd5/moe.py#L251)<br>[ffn_offset](../kernel_sources/blobs/1075006608f00dd5/moe.py#L256)<br>[region_pair](../kernel_sources/blobs/1075006608f00dd5/moe.py#L261) | same_repository_path; different_file_bytes |
| dispatch and body | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/attention/plan.py](../kernel_sources/blobs/8ded6df37a0b6681/plan.py)<br>`8ded6df37a0b6681040e7a9e5919b6582b0b104d9ba63b1e1234fe5aad541fda`<br>[cdiv](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L43)<br>[round_up](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L47)<br>[Dims](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L52)<br>[token_tiles](../kernel_sources/blobs/8ded6df37a0b6681/plan.py#L101) | [/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono/attention/plan.py](../kernel_sources/blobs/9386470498616fbd/plan.py)<br>`9386470498616fbda768c08ff0a8091599cd32c433c74b04682bd9504fb7d57d`<br>[flag_stride](../kernel_sources/blobs/9386470498616fbd/plan.py#L55)<br>[flag_copy](../kernel_sources/blobs/9386470498616fbd/plan.py#L60)<br>[on_copy](../kernel_sources/blobs/9386470498616fbd/plan.py#L65)<br>[ardy_tiles](../kernel_sources/blobs/9386470498616fbd/plan.py#L70) | same_repository_path; different_file_bytes |

Evidence: [evidence/source_diffs/v09_to_v11.diff](../evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](../evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](../evidence/ci_launch_configs.json).

## Excluded attributions

- V10 contains only TP8 conc1; this measured TP2/TP4 delta spans a blank V10 column.
- New TP8 support is not a new fusion for this already supported TP2 pair.
