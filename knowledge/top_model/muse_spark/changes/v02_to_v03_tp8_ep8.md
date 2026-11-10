# V02 → V03 tp8 ep8

[Image changes](../kernel_evolution.md) · [Matrix](../configuration_history.md) · [Evidence rules](../../README.md#evidence)

Before: `lmsysorg/sglang-rocm:muse-spark-open-1020`. After: `lmsysorg/sglang-rocm:muse-spark-open-1027`.

Configuration selector: `{"decode_ep": 8, "decode_tp": 8}`. Source-verified changes; GPU-time rank and causal contributions are unavailable.

Applicable measured pairs: [TEP8 conc1 / GPU](../pairs/tep8_conc1_off_2026-10-22_to_2026-10-30.md), [TEP8 conc4 / GPU](../pairs/tep8_conc4_off_2026-10-22_to_2026-10-30.md), [TEP8 conc8 / GPU](../pairs/tep8_conc8_off_2026-10-22_to_2026-10-30.md).

Required required_cli: `{"after": {"monokernel-backend": "amd"}, "before": {"monokernel-backend": null}}`.

<a id="kernel-1"></a>

## AMD Muse layer monokernel

**Before:** Ordinary SGLang attention, projection, routing, experts, residual/norm and collective launches. The monokernel interface alone does not activate the plugin.

**After:** runtime → Muse spec → MuseMonoRunner.launch → build_layer; eligible MXFP4 layers combine attention, K2, K3 and the next eligible layer K1.

**Device language:** Mixed Triton/FlyDSL/library path → FlyDSL layer kernel (the log/spec name muse_spark_hip does not identify the implementation language).

**Body change:** Device mailbox flags connect producer/consumer stages; one launch reuses stage LDS while retaining scratch regions. The first necessary K1 and BF16-expert layers keep separate execution.

**Fusion:** Merge attention, output norm/gate and projection/reduction, routing/pooler/top-k, experts/finalize/reduction, unpooler/boundary and next-layer QKV preparation across host launch boundaries. Scratch traffic is not assumed eliminated.

**Applicability:** Both endpoint CLI logs are checked: only the new endpoint passes monokernel-backend=amd. TEP8/EP8 with 8-token DFLASH; one admitted request corresponds to 8 verify rows and two to 16. Unsupported shapes, prefill and BF16-expert layers use fallback. The exact admission is shown in this pair’s CI table.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| runtime integration | [/sgl-workspace/sglang/python/sglang/srt/monokernels/runtime.py](../kernel_sources/blobs/914e7957ee99f22e/runtime.py)<br>`914e7957ee99f22eef5e1e11a1bf05a0cdc9ae90b6efe7a7ccbf810f5a8f03dc`<br>[install_monokernel_forward](../kernel_sources/blobs/914e7957ee99f22e/runtime.py#L103)<br>[_install_forward](../kernel_sources/blobs/914e7957ee99f22e/runtime.py#L306) | [/sgl-workspace/sglang/python/sglang/srt/monokernels/runtime.py](../kernel_sources/blobs/707226c7c205a14f/runtime.py)<br>`707226c7c205a14fb9af18fe9694f94fc815556b7a59d985e1334eca925dcf79`<br>[install_monokernel_forward](../kernel_sources/blobs/707226c7c205a14f/runtime.py#L112)<br>[_install_forward](../kernel_sources/blobs/707226c7c205a14f/runtime.py#L315) | same_path; different_file_bytes |
| new runner | Absent / separate existing operations | [/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/runner.py](../kernel_sources/blobs/267c5b33db7be6ce/runner.py)<br>`267c5b33db7be6ce439a33e417166d6d4b91a2c45e33457d7a84a15539f8ffeb`<br>[MuseMonoRunner.launch](../kernel_sources/blobs/267c5b33db7be6ce/runner.py#L504)<br>[MuseMonoRunner.layer_kernel](../kernel_sources/blobs/267c5b33db7be6ce/runner.py#L277)<br>[MuseMonoRunner.merged](../kernel_sources/blobs/267c5b33db7be6ce/runner.py#L335) | added; different_file_bytes |
| new device body | Absent / separate existing operations | [/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/layer.py](../kernel_sources/blobs/507604a48233c776/layer.py)<br>`507604a48233c776fad241df7b62ce5639bc862ecb53957f85d85675e82a2bed`<br>[build_layer](../kernel_sources/blobs/507604a48233c776/layer.py#L294)<br>[build_k1](../kernel_sources/blobs/507604a48233c776/layer.py#L70)<br>[build_kattn](../kernel_sources/blobs/507604a48233c776/layer.py#L193) | added; different_file_bytes |

Evidence: [evidence/source_diffs/02_to_03.diff](../evidence/source_diffs/02_to_03.diff), [evidence/source_excerpts.md#e2](../evidence/source_excerpts.md#e2), [evidence/source_excerpts.md#e3](../evidence/source_excerpts.md#e3).

<a id="kernel-2"></a>

## DFlash2 draft stage-2 fusion and sharded context projection

**Before:** The draft uses the ordinary SGLang model forward; fused KV materialization is already present and is not newly attributed to 1027.

**After:** The AMD plugin attaches DFlash2 at stage 2. Each of five draft layers has D-pre, D-attn, D-post and D-MLP launches; the context projection is TP-sharded (1024 output rows per rank) followed by all-gather.

**Device language:** Ordinary Triton/library operators → FlyDSL device kernels via Python launch wrappers. The _hip spec name is not the device language.

**Body change:** D-pre folds residual/input norm, convolution coefficients and conv0, QKV projection, per-head Q/K norm, RoPE and FP8 KV stores. D-post folds O projection, TP sum and conv1. D-MLP folds post norm/residual, convolutions, gate/up/down and TP sum. D-attn is a separate block-attention kernel.

**Fusion:** Four launches per eligible draft layer in the source call graph, not one whole-draft kernel; draft embedding/final norm, context projection/all-gather and token selection remain separate. Historical logs confirm attachment/build during setup; realized measurement launch counts remain unavailable.

**Applicability:** Only the new endpoint enables monokernel-backend=amd. Exact c1/c4/c8 server artifacts report stage 2; graph widths are [8] for c1 and [8,16] for c4/c8. Eight-token DFLASH block, TP8/EP8, matching draft architecture/cache guards. Do not extend to TEP4 endpoints without the plugin flag.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| draft routing and four launch boundaries | Absent / separate existing operations | [/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/draft.py](../kernel_sources/blobs/f1778a6cbe79a950/draft.py)<br>`f1778a6cbe79a9501ba2a1728eccf85cfa1d256ffde9218179086d3bfc5c17ca`<br>[DraftRunner.launch_attn](../kernel_sources/blobs/f1778a6cbe79a950/draft.py#L264)<br>[DraftRunner.launch](../kernel_sources/blobs/f1778a6cbe79a950/draft.py#L312)<br>[_Draft.forward](../kernel_sources/blobs/f1778a6cbe79a950/draft.py#L393)<br>[_shard_fc](../kernel_sources/blobs/f1778a6cbe79a950/draft.py#L475) | added; different_file_bytes |
| D-pre and D-post device bodies | Absent / separate existing operations | [/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/draft_attn_half.py](../kernel_sources/blobs/d7f558567b18a26b/draft_attn_half.py)<br>`d7f558567b18a26b295fec1fd360649e413742aacfe7d080365211f9035d5738`<br>[run_dpre](../kernel_sources/blobs/d7f558567b18a26b/draft_attn_half.py#L310)<br>[run_dpost](../kernel_sources/blobs/d7f558567b18a26b/draft_attn_half.py#L433)<br>[build_dpre](../kernel_sources/blobs/d7f558567b18a26b/draft_attn_half.py#L470)<br>[build_dpost](../kernel_sources/blobs/d7f558567b18a26b/draft_attn_half.py#L521) | added; different_file_bytes |
| D-attn device body | Absent / separate existing operations | [/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/draft_attn.py](../kernel_sources/blobs/4a7d829bf604e76d/draft_attn.py)<br>`4a7d829bf604e76d7d1647e9f45fdae42fbfbe4fb82fbf2ac813e003d15daa49`<br>[run_dattn](../kernel_sources/blobs/4a7d829bf604e76d/draft_attn.py#L129)<br>[build_dattn](../kernel_sources/blobs/4a7d829bf604e76d/draft_attn.py#L708) | added; different_file_bytes |
| D-MLP device body | Absent / separate existing operations | [/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/draft_mlp.py](../kernel_sources/blobs/9fe45862cef73b76/draft_mlp.py)<br>`9fe45862cef73b76ff51b66339411982bdb39e9cb6a76cbc3d5065d74d3bcbc7`<br>[run_dmlp](../kernel_sources/blobs/9fe45862cef73b76/draft_mlp.py#L500)<br>[build_dmlp](../kernel_sources/blobs/9fe45862cef73b76/draft_mlp.py#L577) | added; different_file_bytes |

Evidence: [runtime_verification.md#draft-fusion](../runtime_verification.md#draft-fusion), [evidence/verified_runtime_facts.json](../evidence/verified_runtime_facts.json), [evidence/verified_source_audit.json](../evidence/verified_source_audit.json), [evidence/verification_source_excerpts.md](../evidence/verification_source_excerpts.md).

<a id="kernel-3"></a>

## Ordinary verify merge with output norm/gate

**Before:** The attention partials are merged, then the existing output norm-gate kernel runs.

**After:** VerifyReduceDeferral hands partials to forward_verify_partials → _reduce_norm_gate.

**Device language:** Triton → Triton.

**Body change:** The new body reads partials/maxima/denominators, performs split-KV reduction, rounds to BF16, and applies per-head RMSNorm and sigmoid gate before the output store.

**Fusion:** One reduce+norm+gate launch replaces separate merge and norm-gate launches; removes the completed-attention intermediate between those launches.

**Applicability:** Ordinary target-verify path, head_dim 64, output normalization enabled, and backend deferral active. On monokernel-covered MXFP4 layers the layer kernel supersedes this path; retain this candidate only for ordinary/fallback layers.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| output consumer | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/fused_norm_gate.py](../kernel_sources/blobs/94d0b45392248de3/fused_norm_gate.py)<br>`94d0b45392248de3b71d846323a7961d0d9cc54755b5154035261a9349c8e266`<br>[_rms_norm_gate_fwd_kernel](../kernel_sources/blobs/94d0b45392248de3/fused_norm_gate.py#L43) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/fused_norm_gate.py](../kernel_sources/blobs/ada56887a3648995/fused_norm_gate.py)<br>`ada56887a364899592ecc263d2e9da82a1c1a976e416557e057ac78ea386bfe9`<br>[_rms_norm_gate_fwd_kernel](../kernel_sources/blobs/ada56887a3648995/fused_norm_gate.py#L46)<br>[FusedRMSNormGate.forward_verify_partials](../kernel_sources/blobs/ada56887a3648995/fused_norm_gate.py#L178) | same_path; different_file_bytes |
| new fused reducer | Absent / separate existing operations | [/sgl-workspace/sglang/python/sglang/kernels/ops/attention/verify_reduce_norm_gate.py](../kernel_sources/blobs/191dc497d3084ba2/verify_reduce_norm_gate.py)<br>`191dc497d3084ba24455abffae2d8445db5cb8526208ae3a78d218498f0903e3`<br>[_reduce_norm_gate](../kernel_sources/blobs/191dc497d3084ba2/verify_reduce_norm_gate.py#L32)<br>[verify_reduce_norm_gate](../kernel_sources/blobs/191dc497d3084ba2/verify_reduce_norm_gate.py#L90) | added; different_file_bytes |

Evidence: [evidence/source_diffs/02_to_03.diff](../evidence/source_diffs/02_to_03.diff).

<a id="kernel-4"></a>

## Residual/norm boundary with FP8 output

**Before:** The fused boundary yields a BF16 attention input and residual stream.

**After:** forward_fp8 can additionally write the next projection input as FP8 plus per-row scales.

**Device language:** Triton → Triton.

**Body change:** The boundary kernel gains output/scaling buffers and a guarded FP8 output path; native fallback can still perform separate quantization.

**Fusion:** For its fused Triton branch, quantization is part of the norm/boundary producer, saving a separate read/quantize launch.

**Applicability:** Ordinary path when the model requests fp8_out/forward_fp8; do not transfer this benefit to the monokernel-covered layers without a trace.

| Role | Before implementation / SHA256 | After implementation / SHA256 | Relation |
| --- | --- | --- | --- |
| boundary body | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/fused_norm_residual.py](../kernel_sources/blobs/e11bbcae7c1504a0/fused_norm_residual.py)<br>`e11bbcae7c1504a01dd346145d1e7af5f298580a4792c0fc117f50c18c908bfd`<br>[_fused_norm_residual_triton_kernel](../kernel_sources/blobs/e11bbcae7c1504a0/fused_norm_residual.py#L51) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/fused_norm_residual.py](../kernel_sources/blobs/16edaf5510160b30/fused_norm_residual.py)<br>`16edaf5510160b306632ff31ad2346ae4c369f14a8c4ea61205b32c81b5109b5`<br>[_fused_norm_residual_triton_kernel](../kernel_sources/blobs/16edaf5510160b30/fused_norm_residual.py#L51)<br>[FusedPostNormResidualPreNorm.forward_fp8](../kernel_sources/blobs/16edaf5510160b30/fused_norm_residual.py#L372) | same_path; different_file_bytes |
| requesting model | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/model.py](../kernel_sources/blobs/31520909318b5731/model.py)<br>`31520909318b5731f5f2dac1021c138e235b3f8a5655975e1765ba9e85e0c95f`<br>[MuseSpark1xAttention.forward](../kernel_sources/blobs/31520909318b5731/model.py#L567)<br>[MuseSpark1xMoE.forward](../kernel_sources/blobs/31520909318b5731/model.py#L934) | [/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/model.py](../kernel_sources/blobs/d7e4953c71a74b44/model.py)<br>`d7e4953c71a74b44905d34e3bb364facaf2df812fd1a04fd00e21bcf618f3242`<br>[MuseSpark1xAttention.forward](../kernel_sources/blobs/d7e4953c71a74b44/model.py#L638)<br>[MuseSpark1xMoE.forward](../kernel_sources/blobs/d7e4953c71a74b44/model.py#L1070) | same_path; different_file_bytes |

Evidence: [evidence/source_diffs/02_to_03.diff](../evidence/source_diffs/02_to_03.diff).

## Excluded attributions

- The new standalone TP4 finalize/all-reduce kernel is outside this TEP8 configuration; do not list it as this pair’s executed fusion.
