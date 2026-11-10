# V03 → V04 tp4 ep4

[Image changes](../kernel_evolution.md) · [Matrix](../configuration_history.md) · [Evidence rules](../../README.md#evidence)

Before: `lmsysorg/sglang-rocm:muse-spark-open-1027`. After: `lmsysorg/sglang-rocm:muse-spark-open-mi355-1106`.

Configuration selector: `{"decode_ep": 4, "decode_tp": 4}`. Source-verified changes; GPU-time rank and causal contributions are unavailable.

Applicable measured pairs: [TEP4 conc28 / HiCache](../pairs/tep4_conc28_on_2026-10-30_to_2026-11-08.md), [TEP4 conc36 / HiCache](../pairs/tep4_conc36_on_2026-10-30_to_2026-11-08.md), [TEP4 conc40 / HiCache](../pairs/tep4_conc40_on_2026-10-30_to_2026-11-08.md).

Required required_cli: `{"after": {"monokernel-backend": null, "page-size": "128"}, "before": {"monokernel-backend": null, "page-size": "16"}}`.

<a id="kernel-1"></a>

## QK/RoPE/FP8 cache fusion eligibility

**Before:** The existing fused wrapper allows only page size 16 and at most 64 QK rows.

**After:** The same QK device file is retained; wrapper guards admit pages 16/32/64/128 and a default cap of 512 rows.

**Device language:** Triton → Triton; the change is in Python dispatch guards.

**Body change:** Wrapper changes page-size and row predicates. The recorded device source hash stays identical for qk_norm_rope_cache.py.

**Fusion:** Broader applicability of existing norm/RoPE/FP8-cache fusion; no new device body is inferred from the larger cap.

**Applicability:** This pair is TEP4 with page size 16 → 128, matching the expanded page guard; target-verify row count and other shape guards still determine use. The c40 endpoint server log records rows=512.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| wrapper guards | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/attention_kv_write.py](../kernel_sources/blobs/34933e2895490a37/attention_kv_write.py)<br>`34933e2895490a3719f918779df60c9575a7b05abccf3d51a03a396b31752adf`<br>[_attention_shape_ok](../kernel_sources/blobs/34933e2895490a37/attention_kv_write.py#L49)<br>[_kv_write_target](../kernel_sources/blobs/34933e2895490a37/attention_kv_write.py#L65)<br>[qkv_proj_norm_rope_kv_write](../kernel_sources/blobs/34933e2895490a37/attention_kv_write.py#L126)<br>[qk_norm_rope_kv_write](../kernel_sources/blobs/34933e2895490a37/attention_kv_write.py#L169) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/attention_kv_write.py](../kernel_sources/blobs/a21de58492ca4d4d/attention_kv_write.py)<br>`a21de58492ca4d4d59bbbb2d57d168aa5f77cb2a696e236277e1ce25d32548c0`<br>[_attention_shape_ok](../kernel_sources/blobs/a21de58492ca4d4d/attention_kv_write.py#L50)<br>[_kv_write_target](../kernel_sources/blobs/a21de58492ca4d4d/attention_kv_write.py#L66)<br>[qkv_proj_norm_rope_kv_write](../kernel_sources/blobs/a21de58492ca4d4d/attention_kv_write.py#L127)<br>[qk_norm_rope_kv_write](../kernel_sources/blobs/a21de58492ca4d4d/attention_kv_write.py#L170) | same_path; different_file_bytes |
| device body | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/qk_norm_rope_cache.py](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py)<br>`3fb15d9d004eeb88757b3ae2aa30497e4bc3e9ff7f01c47a9a2c3a1c9323eb8e`<br>[_muse_qk_rope_cache](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py#L57)<br>[_qkv_epilogue](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py#L122)<br>[qkv_proj_qk_norm_rope_cache](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py#L233) | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/qk_norm_rope_cache.py](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py)<br>`3fb15d9d004eeb88757b3ae2aa30497e4bc3e9ff7f01c47a9a2c3a1c9323eb8e`<br>[_muse_qk_rope_cache](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py#L57)<br>[_qkv_epilogue](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py#L122)<br>[qkv_proj_qk_norm_rope_cache](../kernel_sources/blobs/3fb15d9d004eeb88/qk_norm_rope_cache.py#L233) | same_path; identical_file_bytes |

Evidence: [evidence/source_diffs/03_to_04.diff](../evidence/source_diffs/03_to_04.diff), [evidence/source_excerpts.md#e5](../evidence/source_excerpts.md#e5).

<a id="kernel-2"></a>

## MoE finalize + all-reduce on TP4

**Before:** Already uses a one-launch TP4 HIP finalize/peer reduction on supported compact-A4 verify rows.

**After:** Retains TP4 fusion; source is generalized to NR=4 or 8 and gains a one-shot branch used for selected TP8 widths.

**Device language:** HIP/C++ → HIP/C++; Python/ctypes wrapper is unchanged in role.

**Body change:** The device symbol becomes muse_finalize_allreduce; peer storage/arguments grow for up to 8 ranks. This TEP4 pair still compiles NR=4 and selects the two-stage protocol.

**Fusion:** Existing TP4 fusion is retained. New TP8 eligibility and its one-shot branch are not an added fusion benefit for this TEP4 pair.

**Applicability:** Both endpoints are TEP4/EP4; the wrapper’s TP4 thresholds keep the two-stage route. Actual rows must satisfy compact-A4/finalize predicates.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| HIP device body | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/finalize_allreduce.hip](../kernel_sources/blobs/8ab37d684c03d806/finalize_allreduce.hip)<br>`8ab37d684c03d8063428854fd416da182280863f6b9cf83611d7f4fc4ba03e77`<br>[muse_finalize_allreduce_tp4](../kernel_sources/blobs/8ab37d684c03d806/finalize_allreduce.hip#L108)<br>[muse_finalize_allreduce_fill](../kernel_sources/blobs/8ab37d684c03d806/finalize_allreduce.hip#L236) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/finalize_allreduce.hip](../kernel_sources/blobs/20a0a111064e9d1c/finalize_allreduce.hip)<br>`20a0a111064e9d1c41ea5a10f0af60c3812d77a6b9631bef62fd4db0d2c3a11b`<br>[muse_finalize_allreduce](../kernel_sources/blobs/20a0a111064e9d1c/finalize_allreduce.hip#L111)<br>[muse_finalize_allreduce_fill](../kernel_sources/blobs/20a0a111064e9d1c/finalize_allreduce.hip#L263) | same_path; different_file_bytes |
| protocol dispatch | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/finalize_allreduce.py](../kernel_sources/blobs/1684029d6e761575/finalize_allreduce.py)<br>`1684029d6e76157557c3f044126e11e49a27f32ac1ed9b515ee92e1500f76b12`<br>[MuseFinalizeAllreduce](../kernel_sources/blobs/1684029d6e761575/finalize_allreduce.py#L139)<br>[run_compact_a4_allreduce](../kernel_sources/blobs/1684029d6e761575/finalize_allreduce.py#L260)<br>[ensure_finalize_allreduce](../kernel_sources/blobs/1684029d6e761575/finalize_allreduce.py#L249) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/finalize_allreduce.py](../kernel_sources/blobs/b27664445f8928ac/finalize_allreduce.py)<br>`b27664445f8928ac5b358ba15b82a0bfd3e3d113874200f40ffa922bf978bd92`<br>[MuseFinalizeAllreduce](../kernel_sources/blobs/b27664445f8928ac/finalize_allreduce.py#L139)<br>[run_compact_a4_allreduce](../kernel_sources/blobs/b27664445f8928ac/finalize_allreduce.py#L272)<br>[ensure_finalize_allreduce](../kernel_sources/blobs/b27664445f8928ac/finalize_allreduce.py#L259) | same_path; different_file_bytes |

Evidence: [evidence/source_diffs/03_to_04.diff](../evidence/source_diffs/03_to_04.diff), [evidence/source_excerpts.md#e6](../evidence/source_excerpts.md#e6).

<a id="kernel-3"></a>

## Joint router/pooler wider rows

**Before:** The existing shared routing front admits at most 64 rows and its established FP8 joint range.

**After:** Joint row-major routing can extend to default 128 rows, and model gating uses FP8_JOINT_MAX_M.

**Device language:** Triton → Triton.

**Body change:** Add wider eligibility and the row-major tile route; retain hidden/expert/weight guards and the separate shared fallback.

**Fusion:** Existing projection/epilogue fusion extends to additional shapes; not a new whole-MoE fusion.

**Applicability:** TEP4 is not rejected by _routing_front_matches: its tests are hidden/expert/weight shapes. Applicability remains conditional on packed weights and rows; client concurrency is not the kernel row count.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| routing wrapper | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/router_pooler.py](../kernel_sources/blobs/c570dfbdc9d85dbb/router_pooler.py)<br>`c570dfbdc9d85dbb99aced47e0cbc73f938468969acf1a3cc007816b61444166`<br>[can_share_router_pooler](../kernel_sources/blobs/c570dfbdc9d85dbb/router_pooler.py#L40)<br>[route_with_shared_pooler](../kernel_sources/blobs/c570dfbdc9d85dbb/router_pooler.py#L83) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/router_pooler.py](../kernel_sources/blobs/d1e1447d7980b697/router_pooler.py)<br>`d1e1447d7980b69734bc17e7f49fc106dafc2042adb2f5f6ff4f474d949362d5`<br>[can_share_router_pooler](../kernel_sources/blobs/d1e1447d7980b697/router_pooler.py#L42)<br>[route_with_shared_pooler](../kernel_sources/blobs/d1e1447d7980b697/router_pooler.py#L97) | same_path; different_file_bytes |
| router device epilogues | [/sgl-workspace/sglang/python/sglang/kernels/ops/moe/router_pooler_topk.py](../kernel_sources/blobs/6c876d2421932de4/router_pooler_topk.py)<br>`6c876d2421932de4d2abee1a27b32d04430ff354a8e0a7970407a83b99e7640e`<br>[_joint_epilogue](../kernel_sources/blobs/6c876d2421932de4/router_pooler_topk.py#L127)<br>[_joint_fp8_partial](../kernel_sources/blobs/6c876d2421932de4/router_pooler_topk.py#L426)<br>[joint_router_pooler_topk](../kernel_sources/blobs/6c876d2421932de4/router_pooler_topk.py#L356)<br>[joint_router_pooler_topk_fp8](../kernel_sources/blobs/6c876d2421932de4/router_pooler_topk.py#L494) | [/sgl-workspace/sglang/python/sglang/kernels/ops/moe/router_pooler_topk.py](../kernel_sources/blobs/8492b6be85065fe9/router_pooler_topk.py)<br>`8492b6be85065fe9dab1863e2624bf94583f328f513c65f7b018fe931b758131`<br>[_joint_epilogue](../kernel_sources/blobs/8492b6be85065fe9/router_pooler_topk.py#L127)<br>[_joint_fp8_partial](../kernel_sources/blobs/8492b6be85065fe9/router_pooler_topk.py#L431)<br>[joint_router_pooler_topk](../kernel_sources/blobs/8492b6be85065fe9/router_pooler_topk.py#L356)<br>[joint_router_pooler_topk_fp8](../kernel_sources/blobs/8492b6be85065fe9/router_pooler_topk.py#L499) | same_path; different_file_bytes |

Evidence: [evidence/source_diffs/03_to_04.diff](../evidence/source_diffs/03_to_04.diff).

<a id="kernel-4"></a>

## Full-attention prefill implementation

**Before:** Gluon muse_extend_attention serves the specialized ordinary full-attention prefill route.

**After:** AITER backend can select muse_extend_attention_fly; Gluon remains for supported fallback/sliding paths.

**Device language:** Gluon → FlyDSL for the full-attention route; Gluon remains available.

**Body change:** New FP8 MFMA kernel uses transposed fragments, two-stage LDS loading, matrix-core row sums and explicit paged addressing.

**Fusion:** A kernel implementation/language change; no additional cross-operator fusion is established.

**Applicability:** The new endpoint enables the default FlyDSL extend route; dispatch requires gfx950 and supported FP8 Q/KV shape/layout. These TEP4 jobs retain ordinary prefill, which is outside the Muse monokernel route.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| old/new attention bodies | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/muse_extend_gluon.py](../kernel_sources/blobs/73f8c5b89137e10e/muse_extend_gluon.py)<br>`73f8c5b89137e10eaeb8be67a6cf6686194af50a03300a22569764a0af58e471`<br>[_muse_extend_kernel](../kernel_sources/blobs/73f8c5b89137e10e/muse_extend_gluon.py#L204)<br>[muse_extend_attention](../kernel_sources/blobs/73f8c5b89137e10e/muse_extend_gluon.py#L360) | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/muse_extend_flydsl.py](../kernel_sources/blobs/cdfa8cadad2f2520/muse_extend_flydsl.py)<br>`cdfa8cadad2f252028875ef1b4e9506281161f7b974f7cfe47326bc49fcf882e`<br>[_launcher](../kernel_sources/blobs/cdfa8cadad2f2520/muse_extend_flydsl.py#L117)<br>[muse_extend_attention_fly](../kernel_sources/blobs/cdfa8cadad2f2520/muse_extend_flydsl.py#L628) | different_paths; different_file_bytes |
| backend dispatch | [/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py](../kernel_sources/blobs/76d2d1732b8e1c76/aiter_backend.py)<br>`76d2d1732b8e1c765d437504270ecfd885c28fec4db2c7f639c2cd2fbc698d2b`<br>[_muse_verify_attention](../kernel_sources/blobs/76d2d1732b8e1c76/aiter_backend.py#L224)<br>[AiterAttnBackend.forward_extend](../kernel_sources/blobs/76d2d1732b8e1c76/aiter_backend.py#L4115)<br>[AiterAttnBackend.forward_decode](../kernel_sources/blobs/76d2d1732b8e1c76/aiter_backend.py#L5130) | [/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py](../kernel_sources/blobs/e8053575975ff7d2/aiter_backend.py)<br>`e8053575975ff7d2de8971c4c47de19699b0251b3f6a60458f8011a5e609db33`<br>[_muse_verify_attention](../kernel_sources/blobs/e8053575975ff7d2/aiter_backend.py#L237)<br>[AiterAttnBackend.forward_extend](../kernel_sources/blobs/e8053575975ff7d2/aiter_backend.py#L4165)<br>[AiterAttnBackend.forward_decode](../kernel_sources/blobs/e8053575975ff7d2/aiter_backend.py#L5180) | same_path; different_file_bytes |

Evidence: [evidence/source_diffs/03_to_04.diff](../evidence/source_diffs/03_to_04.diff), [evidence/source_excerpts.md#e7](../evidence/source_excerpts.md#e7).

<a id="kernel-5"></a>

## Length-balanced verify and segmented fused merge

**Before:** FlyDSL full-attention verify uses the former split scheduling; verify_reduce_norm_gate accepts four-element partial state.

**After:** A Triton planner builds a length-balanced segment list for FlyDSL partials; five-element state selects segmented_reduce_norm_gate.

**Device language:** FlyDSL + Triton → FlyDSL + Triton.

**Body change:** Distribute KV work by sequence lengths; add segment offsets/counts and teach the reducer dispatch to accept segmented state.

**Fusion:** Planning adds a launch. The segmented output reducer retains split-KV reduction + norm/gate fusion; this is not a reduction to a single verify launch.

**Applicability:** Ordinary full-attention target verify when use_balanced_full and backend flags/shape guards select it. Heterogeneous context lengths, not just client concurrency, determine applicability.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| new segmented planner/reducer | Absent / separate existing operations | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/muse_verify_balanced.py](../kernel_sources/blobs/7ab7b5084c44fd90/muse_verify_balanced.py)<br>`7ab7b5084c44fd90cd128ed4304d4e33939bb56b0f5590a9b5dc83563484b19a`<br>[_plan_kernel](../kernel_sources/blobs/7ab7b5084c44fd90/muse_verify_balanced.py#L33)<br>[_seg_reduce](../kernel_sources/blobs/7ab7b5084c44fd90/muse_verify_balanced.py#L113)<br>[segmented_reduce_norm_gate](../kernel_sources/blobs/7ab7b5084c44fd90/muse_verify_balanced.py#L229)<br>[gqa_verify_flydsl_balanced](../kernel_sources/blobs/7ab7b5084c44fd90/muse_verify_balanced.py#L243) | added; different_file_bytes |
| partial kernel | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/muse_verify_flydsl.py](../kernel_sources/blobs/72dc1e2cd6f57fc5/muse_verify_flydsl.py)<br>`72dc1e2cd6f57fc503904782e2c082745fbc93bd98c2576e53f2a99563b5a651`<br>[_launcher](../kernel_sources/blobs/72dc1e2cd6f57fc5/muse_verify_flydsl.py#L136)<br>[verify_fly_partials](../kernel_sources/blobs/72dc1e2cd6f57fc5/muse_verify_flydsl.py#L546)<br>[gqa_verify_flydsl](../kernel_sources/blobs/72dc1e2cd6f57fc5/muse_verify_flydsl.py#L618) | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/muse_verify_flydsl.py](../kernel_sources/blobs/3df4cf24bded6134/muse_verify_flydsl.py)<br>`3df4cf24bded61346b0dd4b57aa8c2a3ff4851e5a0a614da94131bc4ff771f5c`<br>[_launcher](../kernel_sources/blobs/3df4cf24bded6134/muse_verify_flydsl.py#L136)<br>[verify_fly_partials](../kernel_sources/blobs/3df4cf24bded6134/muse_verify_flydsl.py#L604)<br>[gqa_verify_flydsl](../kernel_sources/blobs/3df4cf24bded6134/muse_verify_flydsl.py#L679) | same_path; different_file_bytes |
| fused-reducer dispatch | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/verify_reduce_norm_gate.py](../kernel_sources/blobs/191dc497d3084ba2/verify_reduce_norm_gate.py)<br>`191dc497d3084ba24455abffae2d8445db5cb8526208ae3a78d218498f0903e3`<br>[_reduce_norm_gate](../kernel_sources/blobs/191dc497d3084ba2/verify_reduce_norm_gate.py#L32)<br>[verify_reduce_norm_gate](../kernel_sources/blobs/191dc497d3084ba2/verify_reduce_norm_gate.py#L90) | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/verify_reduce_norm_gate.py](../kernel_sources/blobs/6e1a6ca9b85aa6bd/verify_reduce_norm_gate.py)<br>`6e1a6ca9b85aa6bda3fbb18f630a4231ce809b3a9f76cdd35b202081fd8c8b69`<br>[_reduce_norm_gate](../kernel_sources/blobs/6e1a6ca9b85aa6bd/verify_reduce_norm_gate.py#L32)<br>[verify_reduce_norm_gate](../kernel_sources/blobs/6e1a6ca9b85aa6bd/verify_reduce_norm_gate.py#L90) | same_path; different_file_bytes |

Evidence: [evidence/source_diffs/03_to_04.diff](../evidence/source_diffs/03_to_04.diff), [evidence/source_excerpts.md#e8](../evidence/source_excerpts.md#e8).

## Excluded attributions

- Both endpoints are ordinary TEP4 serving: the AMD layer monokernel is not enabled by either CLI. Its captured Muse bodies are unchanged between these image digests.
- TP8 finalize/all-reduce eligibility is not a new TP4 optimization.
