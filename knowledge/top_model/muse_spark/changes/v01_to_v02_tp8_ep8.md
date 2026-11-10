# V01 → V02 tp8 ep8

[Image changes](../kernel_evolution.md) · [Matrix](../configuration_history.md) · [Evidence rules](../../README.md#evidence)

Before: `lmsysorg/sglang-rocm:muse-spark-open`. After: `lmsysorg/sglang-rocm:muse-spark-open-1020`.

Configuration selector: `{"decode_ep": 8, "decode_tp": 8}`. Source-verified changes; GPU-time rank and causal contributions are unavailable.

Applicable measured pairs: [TEP8 conc1 / GPU](../pairs/tep8_conc1_off_2026-10-16_to_2026-10-22.md), [TEP8 conc4 / GPU](../pairs/tep8_conc4_off_2026-10-16_to_2026-10-22.md), [TEP8 conc8 / GPU](../pairs/tep8_conc8_off_2026-10-16_to_2026-10-22.md), [TEP8 conc16 / HiCache](../pairs/tep8_conc16_on_2026-10-16_to_2026-10-22.md).

Required required_cli: `{"after": {"monokernel-backend": null}, "before": {"monokernel-backend": null}}`.

<a id="kernel-1"></a>

## QK normalization, RoPE and FP8 KV write

**Before:** Existing _qk_norm_kernel combines Q/K RMSNorm and query scale; projection, RoPE and backend KV writing have separate boundaries.

**After:** _muse_qk_rope_cache or _qkv_epilogue writes normalized/rotated Q/K and FP8 KV; model attention bypasses its ordinary cache write.

**Device language:** Triton → Triton; the projection remains a separate split-K partial producer.

**Body change:** New cache-store epilogue folds normalization, scale, rotation and FP8 conversion; explicit BF16 round trips preserve the former sequence semantics.

**Fusion:** The QK route shares norm/RoPE/cache write in one launch. The QKV route is partial GEMM plus an epilogue launch; the whole projection is not one fused launch.

**Applicability:** TEP8, head_dim 64, local 16 Q / 2 KV heads, AITER NHD FP8 pages of size 16; decode or target verify; QKV at up to 32 rows, QK at up to 64. Actual row counts vary during the job.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| prior norm / new epilogue | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/muse_qk_norm.py](../kernel_sources/blobs/0e3246a8bbd8990f/muse_qk_norm.py)<br>`0e3246a8bbd8990f881aa38fa45e766249a1c6f109faf0b897fe8d86c2f24c6e`<br>[_qk_norm_kernel](../kernel_sources/blobs/0e3246a8bbd8990f/muse_qk_norm.py#L19)<br>[fused_qk_norm](../kernel_sources/blobs/0e3246a8bbd8990f/muse_qk_norm.py#L64) | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/qk_norm_rope_cache.py](../kernel_sources/blobs/df7233ecd09eea43/qk_norm_rope_cache.py)<br>`df7233ecd09eea43501bc03d36a247bd54cb9aef67b8865410449d877e761df3`<br>[_muse_qk_rope_cache](../kernel_sources/blobs/df7233ecd09eea43/qk_norm_rope_cache.py#L57)<br>[_qkv_epilogue](../kernel_sources/blobs/df7233ecd09eea43/qk_norm_rope_cache.py#L122)<br>[qkv_proj_qk_norm_rope_cache](../kernel_sources/blobs/df7233ecd09eea43/qk_norm_rope_cache.py#L233) | different_paths; different_file_bytes |
| dispatch | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/model.py](../kernel_sources/blobs/496fc8a155a50fd8/model.py)<br>`496fc8a155a50fd8fa885b70c579a065a2079bc3fad1e66e7ba5c65ca0d87b23`<br>[MuseSpark1xAttention.forward](../kernel_sources/blobs/496fc8a155a50fd8/model.py#L434)<br>[MuseSpark1xMoE.forward](../kernel_sources/blobs/496fc8a155a50fd8/model.py#L798) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/model.py](../kernel_sources/blobs/31520909318b5731/model.py)<br>`31520909318b5731f5f2dac1021c138e235b3f8a5655975e1765ba9e85e0c95f`<br>[MuseSpark1xAttention.forward](../kernel_sources/blobs/31520909318b5731/model.py#L567)<br>[MuseSpark1xMoE.forward](../kernel_sources/blobs/31520909318b5731/model.py#L934) | same_path; different_file_bytes |
| eligibility wrapper | Absent / separate existing operations | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/attention_kv_write.py](../kernel_sources/blobs/03180acecee59563/attention_kv_write.py)<br>`03180acecee59563eff94bed7d45cdced88f8b62c55baa2b5b38c9694576936b`<br>[_attention_shape_ok](../kernel_sources/blobs/03180acecee59563/attention_kv_write.py#L49)<br>[_kv_write_target](../kernel_sources/blobs/03180acecee59563/attention_kv_write.py#L65)<br>[qkv_proj_norm_rope_kv_write](../kernel_sources/blobs/03180acecee59563/attention_kv_write.py#L126)<br>[qk_norm_rope_kv_write](../kernel_sources/blobs/03180acecee59563/attention_kv_write.py#L166) | added; different_file_bytes |

Evidence: [evidence/source_diffs/01_to_02.diff](../evidence/source_diffs/01_to_02.diff), [evidence/source_excerpts.md#e1](../evidence/source_excerpts.md#e1).

<a id="kernel-2"></a>

## MoE router and pooler

**Before:** Router GEMM, fused top-k routing, pooler GEMM and pre-expert norm remain separate stages.

**After:** route_with_shared_pooler chooses a shared epilogue or a joint packed projection followed by _joint_epilogue.

**Device language:** Triton / GEMM library calls → Triton / GEMM library calls.

**Body change:** Packed pooler/router weights and shared reduction/top-k/norm epilogues avoid separate processing of completed router/pooler results.

**Fusion:** Shared router/top-k/pre-expert norm epilogue; selected widths also share a router/pooler GEMM. Expert GEMMs remain separate.

**Applicability:** Shape guards: hidden 8192, 256 experts, top-8 sigmoid, 4096-wide pooler, contiguous BF16 input and FP32 router; at most 64 rows, with joint widths 8/16/24 in this version.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| model routing | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/model.py](../kernel_sources/blobs/496fc8a155a50fd8/model.py)<br>`496fc8a155a50fd8fa885b70c579a065a2079bc3fad1e66e7ba5c65ca0d87b23`<br>[MuseSpark1xAttention.forward](../kernel_sources/blobs/496fc8a155a50fd8/model.py#L434)<br>[MuseSpark1xMoE.forward](../kernel_sources/blobs/496fc8a155a50fd8/model.py#L798) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/model.py](../kernel_sources/blobs/31520909318b5731/model.py)<br>`31520909318b5731f5f2dac1021c138e235b3f8a5655975e1765ba9e85e0c95f`<br>[MuseSpark1xAttention.forward](../kernel_sources/blobs/31520909318b5731/model.py#L567)<br>[MuseSpark1xMoE.forward](../kernel_sources/blobs/31520909318b5731/model.py#L934) | same_path; different_file_bytes |
| new routing dispatch | Absent / separate existing operations | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/router_pooler.py](../kernel_sources/blobs/5566d9752197d2b0/router_pooler.py)<br>`5566d9752197d2b0b9145a24497aaf220e502e2409cc5f9e454888e0e44c91b2`<br>[can_share_router_pooler](../kernel_sources/blobs/5566d9752197d2b0/router_pooler.py#L31)<br>[route_with_shared_pooler](../kernel_sources/blobs/5566d9752197d2b0/router_pooler.py#L59) | added; different_file_bytes |
| new epilogues | Absent / separate existing operations | [/sgl-workspace/sglang/python/sglang/kernels/ops/moe/router_pooler_topk.py](../kernel_sources/blobs/b82eeb4cc8a9f4a5/router_pooler_topk.py)<br>`b82eeb4cc8a9f4a52c65bb1d330358e26f3fa35ded3ae013b26d5026cbd2f10e`<br>[_joint_epilogue](../kernel_sources/blobs/b82eeb4cc8a9f4a5/router_pooler_topk.py#L107)<br>[joint_router_pooler_topk](../kernel_sources/blobs/b82eeb4cc8a9f4a5/router_pooler_topk.py#L296) | added; different_file_bytes |

Evidence: [evidence/source_diffs/01_to_02.diff](../evidence/source_diffs/01_to_02.diff).

<a id="kernel-3"></a>

## MXFP4 expert route

**Before:** Native-layout FlyDSL grouped experts and per-expert finalize; the MUSE_MOE_BACKEND=flydsl setting is present.

**After:** Compact A4 target-verify experts in Triton, or AITER FlyDSL A4W4 prefill stages, followed by per-expert finalize.

**Device language:** FlyDSL → shape-dependent Triton or FlyDSL; Python mxfp4_moe.py is dispatch glue.

**Body change:** Add per-32 MXFP4 activations, compact verify work and preshuffled A4W4 weights; prefill stage 1 emits quantized intermediates/scales.

**Fusion:** Activation and quantization staging change, while the required per-expert normalization and route-weighted reduction remain explicit. Do not call the entire MoE one kernel.

**Applicability:** Compact verify requires TEP8, 32 local experts, top-8, rows divisible by 8 and no more than 512; prefill A4W4 uses at least 256 rows. Native fallbacks remain.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| existing expert implementation | [/sgl-workspace/aiter/aiter/ops/flydsl/muse_moe_gemm.py](../kernel_sources/blobs/8bdebdca77615ff3/muse_moe_gemm.py)<br>`8bdebdca77615ff3c13ee79cdaf720f03921f13a9dd9f34f2b1684e1a6e65ed4`<br>[_launcher](../kernel_sources/blobs/8bdebdca77615ff3/muse_moe_gemm.py#L40)<br>[launch_muse_moe_gemm](../kernel_sources/blobs/8bdebdca77615ff3/muse_moe_gemm.py#L898) | [/sgl-workspace/aiter/aiter/ops/flydsl/muse_moe_gemm.py](../kernel_sources/blobs/8bdebdca77615ff3/muse_moe_gemm.py)<br>`8bdebdca77615ff3c13ee79cdaf720f03921f13a9dd9f34f2b1684e1a6e65ed4`<br>[_launcher](../kernel_sources/blobs/8bdebdca77615ff3/muse_moe_gemm.py#L40)<br>[launch_muse_moe_gemm](../kernel_sources/blobs/8bdebdca77615ff3/muse_moe_gemm.py#L898) | same_path; identical_file_bytes |
| new compact experts | Absent / separate existing operations | [/sgl-workspace/sglang/python/sglang/kernels/ops/moe/mxfp4_compact_a4_moe.py](../kernel_sources/blobs/6a2cd481da888c29/mxfp4_compact_a4_moe.py)<br>`6a2cd481da888c296c252e116cde2f26745e01a74858276cdc48e167d5031474`<br>[_prepare_a4](../kernel_sources/blobs/6a2cd481da888c29/mxfp4_compact_a4_moe.py#L26)<br>[_scatter_a4](../kernel_sources/blobs/6a2cd481da888c29/mxfp4_compact_a4_moe.py#L77)<br>[_expert_a4](../kernel_sources/blobs/6a2cd481da888c29/mxfp4_compact_a4_moe.py#L102)<br>[compact_a4_experts](../kernel_sources/blobs/6a2cd481da888c29/mxfp4_compact_a4_moe.py#L199) | added; different_file_bytes |
| new dispatch | Absent / separate existing operations | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/mxfp4_moe.py](../kernel_sources/blobs/39b054665c41c85e/mxfp4_moe.py)<br>`39b054665c41c85e6fc85a4fb9e1a706c240a6c90a9b6a5232eb9d229fe08c22`<br>[run_compact_a4](../kernel_sources/blobs/39b054665c41c85e/mxfp4_moe.py#L69)<br>[run_a4w4](../kernel_sources/blobs/39b054665c41c85e/mxfp4_moe.py#L114) | added; different_file_bytes |

Evidence: [evidence/source_diffs/01_to_02.diff](../evidence/source_diffs/01_to_02.diff).

## Excluded attributions

- The SGLang monokernel interface is present in 1020, but this pair does not pass --monokernel-backend amd and has no AMD plugin execution claim.
