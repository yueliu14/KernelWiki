# DeepSeek Flash kernel and fusion evolution

[Full performance matrix](configuration_history.md) · [110 measured pair records](pair_changes.md) · [Source inventory](evidence/image_source_inventory.json)

This is a source-verified candidate-path history, not a measured GPU-time ranking. Source kinds and hashes are preserved per file. V02–V08 use exact upstream commits corroborated by CI; their historical container/AITER bytes are unavailable. V01 is a current tag snapshot, while V09–V11 are API-digest-matched containers. Each source edge can contain many unrelated changes; the reviewed pipelines below identify relevant changes and their applicability, without claiming exhaustive kernel coverage or causal E2E speedups.

The main fusion progression is ordinary staged operators → fused reduction/inverse RoPE → reduction/inverse RoPE/MXFP8 output → fused mHC API → two persistent FlyDSL launches → broader backbone coverage → draft-layer coverage → TP8 eligibility → MI355X metadata/Markov helper eligibility and TP2-specific internal tuning. Compressor/indexer and Engram are not silently included in the two persistent launches.

## V01 → V02

6 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### mHC dispatch and stream collapse

**Before:** AMD model directly calls mhc_pre_delayed_torch and mhc_post_torch; final/cross-layer HC collapse uses PyTorch multiply/reduce.

**After:** Model calls MHCPreDelayedOp and MHCPostOp; supported ROCm shapes dispatch to AITER. hc_collapse_triton replaces explicit HC multiply/reduce.

**Language:** PyTorch/reference operations -> Triton HC collapse plus AITER calls; TileLang/reference fallbacks remain. Historical AITER device bytes for V02 were not recovered.

**Body:** Replace reference call sites with platform CustomOps. Delayed AITER pre requires x=None, no fused norm, hc_mult=4 and hidden width divisible by 256; entry broadcast can fall back.

**Fusion:** HC weighted collapse has a fused Triton implementation. Post and delayed-pre are still separate call sites, and following RMSNorm remains separate; do not claim the entire mHC seam became one launch.

**Guards:** MI355X/gfx950 ordinary vLLM path. No mono decode is enabled in these endpoint commands. Actual token rows and backend dispatch still determine the kernels executed.

Source paths: `vllm/models/deepseek_v41/amd/model.py`, `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/triton.py`, `vllm/model_executor/kernels/mhc/aiter.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v01_to_v02.diff](evidence/source_diffs/v01_to_v02.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### MXFP4 MoE backend selection

**Before:** CLI explicitly selects aiter_triton_mxfp4_bf16.

**After:** CLI selects aiter; vLLM MXFP4 oracle selects its AITER backend and prepares the corresponding weight/scaling layout.

**Language:** Triton backend -> AITER-selected implementation; exact historical AITER device language/body unavailable for V02.

**Body:** Backend and weight-layout dispatch change; the wrapper is Python and is not itself a device kernel.

**Fusion:** No single-launch whole-MoE fusion is established by this CLI change. Historical dependency internals are an evidence gap.

**Guards:** MI355X/gfx950 ordinary vLLM path. No mono decode is enabled in these endpoint commands. Actual token rows and backend dispatch still determine the kernels executed.

Source paths: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v01_to_v02.diff](evidence/source_diffs/v01_to_v02.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- V01 is a current tag snapshot whose vLLM commit matches CI, not a historical-digest-verified container. V02 is commit source corroborated by CI.
- Server admission and CUDA graph capture limits also changed. The large E2E gain is not a measured causal effect of mHC or MoE alone.

## V02 → V03

6 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### Shared sparse top-k metadata

**Before:** Every compressed-attention consumer builds ragged top-k indices/indptr from the shared dense index buffer.

**After:** The source layer caches one ragged tuple per compression ratio and consumers reuse it; cache is reset when the source publishes new indices.

**Language:** Python dispatch around existing Triton metadata kernels.

**Body:** Adds _topk_ragged_cache, _index_source_prefix and _decode_topk_ragged; model directory changes deepseek_v4_1 -> deepseek_v41.

**Fusion:** Removes redundant metadata producer launches for consumers sharing a source/ratio. This is reuse of an existing result, not fusion of attention and metadata into a new device kernel.

**Guards:** MI355X/gfx950 ordinary vLLM path. No mono decode is enabled in these endpoint commands. Actual token rows and backend dispatch still determine the kernels executed. Applies to compressed decode layers with an index source.

Source paths: `vllm/models/deepseek_v41/amd/rocm.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v02_to_v03.diff](evidence/source_diffs/v02_to_v03.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- New common MXFP8/NVFP4 cache layouts and NVIDIA fused-attention paths also appear in this diff; their presence alone does not establish use on MI355X. ROCm decode still uses its fp8_ds_mla path.
- Engram placement and launch recipes change across this range; keep per-endpoint flags as confounders.

## V03 → V04

16 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### Sparse decode reduction + inverse RoPE

**Before:** Split-K sparse attention reduce writes BF16; a standalone inverse GPT-J RoPE pass precedes wo_a.

**After:** _sparse_attn_decode_reduce_kernel receives positions/cos-sin and rotates register-resident output under FUSE_INV_ROPE before storing.

**Language:** Triton -> Triton.

**Body:** Adds even/odd lane-pair inverse rotation in the reduction body; forward_mqa tracks the count of rotated decode rows and separately settles remaining rows.

**Fusion:** Decode split reduction + inverse RoPE become one launch. Prefill/unfused rows keep a standalone rotation; wo_a remains separate.

**Guards:** MI355X/gfx950 ordinary vLLM path. No mono decode is enabled in these endpoint commands. Actual token rows and backend dispatch still determine the kernels executed.

Source paths: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/models/deepseek_v41/amd/rocm.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v03_to_v04.diff](evidence/source_diffs/v03_to_v04.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- CuTe DSL wo_b + reduce-scatter support in shared attention code is not evidence of this ROCm path using it.

## V04 → V05

16 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### Sparse decode output directly in MXFP8

**Before:** Reduction already fuses inverse RoPE but stores BF16; wo_a uses cached BF16 weights and einsum.

**After:** gfx950 reduction additionally writes e4m3 data and one E8M0 scale per 32 lanes; wo_a consumes QuantizedActivation through a grouped tl.dot_scaled kernel.

**Language:** Triton + BF16 library contraction -> Triton reduction/quantization and Triton MXFP8 grouped BMM.

**Body:** QUANT_OUT invokes _mxfp8_quantize_rows in the reduce epilogue; new _mxfp8_wo_a_bmm_kernel selects tiles by token-count times local groups.

**Fusion:** Decode reduction + inverse RoPE + MXFP8 quantization share one launch, eliminating the completed BF16 output/re-quantization boundary. wo_a is still a separate launch. Prefill uses BF16 attention followed by one fused inverse-RoPE/quant pass.

**Guards:** MI355X/gfx950 ordinary vLLM path. No mono decode is enabled in these endpoint commands. Actual token rows and backend dispatch still determine the kernels executed. MXFP8 output is gfx950-only.

Source paths: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/models/deepseek_v41/amd/rocm.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v04_to_v05.diff](evidence/source_diffs/v04_to_v05.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

## V05 → V06

14 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### mHC seam with following RMSNorm

**Before:** mHC delayed-pre/post processing leaves an explicit attn_norm/ffn_norm call.

**After:** Model checks HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM, supplies norm_weight, and skips the corresponding standalone norm on supported seams.

**Language:** Python vLLM dispatch -> AITER fused-seam API. V06 historical AITER body and device language unavailable.

**Body:** MHCPreDelayedOp can fold the preceding post block and following RMSNorm through mhc_fused_post_pre_delayed_rms_norm_aiter; model-entry broadcast remains an exception.

**Fusion:** Source verifies delegation of post/pre/norm to the fused AITER API and removal of the model-level norm call. Exact AITER internal launch count cannot be established from this upstream-only snapshot.

**Guards:** MI355X/gfx950 ordinary vLLM path. No mono decode is enabled in these endpoint commands. Actual token rows and backend dispatch still determine the kernels executed. Requires available fused AITER seam and its shape guards; first 2-D entry broadcast does not fuse attn norm.

Source paths: `vllm/models/deepseek_v41/amd/model.py`, `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/kernels/mhc/aiter.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v05_to_v06.diff](evidence/source_diffs/v05_to_v06.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### MXFP4 sparse indexer and AITER sparse MLA

**Before:** Prior ROCm indexer/attention route and launch recipe.

**After:** CLI explicitly requests indexer_kv_dtype=mxfp4 and indexer_sparse_logits=true; ROCm MXFP4 indexer dispatches to AITER cache/query producers.

**Language:** Python wrappers calling AITER Triton cache/RoPE/quantization ops; historical dependency bodies unavailable.

**Body:** Adds DeepseekV41RocmMxfp4Indexer, rocm_mxfp4_indexer_k_store and rocm_mxfp4_indexer_q_quant; uses k_norm_rope_mxfp4_cache and q_rope_mxfp4_quant APIs.

**Fusion:** Indexer norm/RoPE/quant/cache contracts are fused producer APIs; sparse scoring, top-k, attention and wo_a remain separate stages. Do not infer a whole indexer/attention fusion.

**Guards:** MI355X/gfx950 ordinary vLLM path. No mono decode is enabled in these endpoint commands. Actual token rows and backend dispatch still determine the kernels executed. New endpoint attention-config and VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA enable the relevant selection; KV transfer/other guards can change fallback.

Source paths: `vllm/models/deepseek_v41/amd/rocm.py`, `vllm/models/deepseek_v41/attention.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v05_to_v06.diff](evidence/source_diffs/v05_to_v06.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- Source review confirms the vLLM call sites and API contracts, not historical V06 AITER internals. Engram sharing/prefetch and recipe changes are additional contributors.

## V06 → V07

14 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### Persistent mono decode introduced

**Before:** Ordinary vLLM attention, mHC/FFN seam, MoE and TP collective boundaries.

**After:** VLLM_ROCM_MONO_DECODE=1 selects two persistent FlyDSL launches for eligible standard layers; other layers use ordinary attention plus a fused FFN launch.

**Language:** Triton/AITER/library stages -> persistent FlyDSL K1/K2 with Python dispatch.

**Body:** New mono_decode.py and mono/{layer,runner,attention,stages} implement peer-memory handshakes, register/LDS staging and in-kernel TP reductions on loaded MXFP8/MXFP4 weights.

**Fusion:** K1 fuses attention seam (slice/gate/norm-to-MXFP8) with attention front. K2 fuses attention back/wo_b peer push, TP reduction + FFN seam, MoE and its TP reduction. It is two launches. First/Engram/compressor/indexer layers retain ordinary attention and take one FFN-only launch.

**Guards:** MI355X/gfx950, TP2/TP4, EP1, decode-only, at most 48 active/padded token rows, with AITER A8W4-compatible weights and supported SWA/cache metadata. Five drafts can yield six rows per request; client concurrency is not the runtime row count. Larger/mixed steps take ordinary vLLM paths. Initial whole-layer path is standard seam, compression ratio 1/2, no local compressor/indexer; no EP/DP/EPLB or compiled path.

Source paths: `vllm/models/deepseek_v41/amd/model.py`, `vllm/models/deepseek_v41/amd/mono_decode.py`, `vllm/models/deepseek_v41/amd/mono/layer.py`, `vllm/models/deepseek_v41/amd/mono/stages/seam.py`, `vllm/models/deepseek_v41/amd/mono/stages/moe.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v06_to_v07.diff](evidence/source_diffs/v06_to_v07.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- No GPU trace establishes what fraction of a high-concurrency run falls within the 48-row guard. New quick-reduce INT4 setting can affect ordinary fallback collectives separately.

## V07 → V08

14 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### Whole-backbone mono coverage

**Before:** Only standard seam compressed layers run the complete K1/K2 path; first/Engram/index-source layers can take FFN-only. Enablement uses an environment flag.

**After:** --monokernel replaces the environment switch; whole path expands to compression ratios 0/1/2, settled first/Engram residuals and index-source layers.

**Language:** FlyDSL -> FlyDSL; Python dispatch changes.

**Body:** MonoBuild/runner gains fold/index-aware branches; K1 publishes normed inputs and qr when the indexer needs them; K2 consumes the resulting indices.

**Fusion:** Extends existing two-launch backbone fusion to more layers. At index sources, vLLM compressor and indexer still execute between K1 and K2. Engram injection is outside the two launches. DSpark draft layers remain excluded at V08.

**Guards:** MI355X/gfx950, TP2/TP4, EP1, decode-only, at most 48 active/padded token rows, with AITER A8W4-compatible weights and supported SWA/cache metadata. Five drafts can yield six rows per request; client concurrency is not the runtime row count. Larger/mixed steps take ordinary vLLM paths.

Source paths: `vllm/models/deepseek_v41/amd/mono_decode.py`, `vllm/models/deepseek_v41/amd/mono/layer.py`, `vllm/models/deepseek_v41/amd/mono/attention/front.py`, `vllm/models/deepseek_v41/amd/mono/attention/back.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v07_to_v08.diff](evidence/source_diffs/v07_to_v08.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

## V08 → V09

14 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### DSpark draft layers join mono decode

**Before:** Whole-backbone path exists, but draft layer IDs are excluded.

**After:** Draft layers can use a dedicated runner for their 128-expert/top-3 routing and non-causal SWA block; target remains 384-expert/top-6.

**Language:** FlyDSL -> FlyDSL.

**Body:** Parameterizes MonoBuild/scratch/runner by experts, topk, ug_groups and swa_width; draft runner has its own peer handshake and routing shape.

**Fusion:** Existing attention/FFN/MoE/collective fusion now also covers supported draft layers. Compressor/indexer remain external; this does not fuse five sequential draft sampling steps into one launch.

**Guards:** MI355X/gfx950, TP2/TP4, EP1, decode-only, at most 48 active/padded token rows, with AITER A8W4-compatible weights and supported SWA/cache metadata. Five drafts can yield six rows per request; client concurrency is not the runtime row count. Larger/mixed steps take ordinary vLLM paths. Draft-specific non-causal metadata and routing/weight guards apply.

Source paths: `vllm/models/deepseek_v41/amd/mono_decode.py`, `vllm/models/deepseek_v41/amd/mono/layer.py`, `vllm/models/deepseek_v41/amd/mono/runner.py`, `vllm/models/deepseek_v41/amd/dspark.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v08_to_v09.diff](evidence/source_diffs/v08_to_v09.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- NVIDIA sharded wqa/wkv code in the shared model is not credited to MI355X. New source support does not by itself establish benchmark-phase execution counts.

## V09 → V10

0 measured same-configuration pairs. This is an image-source transition only; V10 has only TP8 c1, absent from V09.

### TP8 mono attention support

**Before:** Mono dimension guards allow TP2/TP4; each rank has at least one full 16-head MFMA tile.

**After:** Dimension guards admit TP8. Its eight real heads occupy half a 16-head tile, with padded heads kept out of live output.

**Language:** FlyDSL -> FlyDSL.

**Body:** Dims separates heads and heads_pad; front/back schedules and addressing support padded head tiles.

**Fusion:** Existing two-launch fusion is enabled at TP8; it is not an additional launch reduction for already supported TP2/TP4.

**Guards:** Image-level source edge only: V09 has no TP8 observation and V10 only has TP8 conc1, so there is no same-configuration measured V09 -> V10 pair.

Source paths: `vllm/models/deepseek_v41/amd/mono_decode.py`, `vllm/models/deepseek_v41/amd/mono/attention/plan.py`, `vllm/models/deepseek_v41/amd/mono/attention/front.py`, `vllm/models/deepseek_v41/amd/mono/attention/back.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v10.diff](evidence/source_diffs/v09_to_v10.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- gfx942 helper additions in V10 do not imply MI355X eligibility. No TP2/TP4 V10 measurement is invented.

## V10 → V11 (TP8)

1 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### SWA decode metadata fusion enabled on MI355X

**Before:** V10 has one-launch SWA helper gated to gfx942; MI355X follows the ordinary metadata sequence.

**After:** monokernel_cdna3_or_4 admits gfx950: _decode_swa_meta_kernel writes dense indices/lens, validity, ragged indices/indptr and graph buffers.

**Language:** Triton -> Triton; expanded architecture gate.

**Body:** Helper body remains conceptually the single-program metadata builder; V11 broadens dispatch to CDNA4.

**Fusion:** One helper launch replaces the ordinary sequence of validity, zero-fill, window, counts/cumsum/compaction and graph copies. The source describes eight prior launches per build; no original CI trace verifies realized counts.

**Guards:** --monokernel, pure decode and at most 32 decode token rows. Applies to causal target and supported non-causal DSpark metadata. Client concurrency alone cannot establish the 32-row guard.

Source paths: `vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py`, `vllm/models/deepseek_v41/amd/ops/__init__.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v10_to_v11.diff](evidence/source_diffs/v10_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### DSpark Markov sampling fusion enabled on MI355X

**Before:** V10 gfx950 uses the ordinary sequential Markov embedding/GEMV/add/Gumbel/argmax/gather/copy loop; fused helper is gfx942-only.

**After:** V11 permits gfx950 for at most five request rows. Each draft step computes vocabulary-block bias plus logits and block Gumbel-max in one kernel; next step resolves preceding block maxima.

**Language:** Triton -> Triton.

**Body:** Architecture gate expands and MAX_ROWS_GFX950=5 limits selection; helper preserves request/step/position/token-dependent noise and tie policy.

**Fusion:** Five drafts use one launch per step plus one final token-resolution launch, not one launch for all drafts. The helper stores draft logits; Markov GEMV summation order can differ in the last BF16 bit.

**Guards:** --monokernel, supported sequential DSpark sampler and at most five request rows. Larger batches use the ordinary sampler. The benchmark uses synthetic acceptance length 3.51, which limits conclusions about real acceptance.

Source paths: `vllm/models/deepseek_v41/amd/ops/fused_markov.py`, `vllm/models/deepseek_v41/amd/ops/__init__.py`, `vllm/models/deepseek_v41/amd/dspark.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v10_to_v11.diff](evidence/source_diffs/v10_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### Ordinary-path mHC overlap and Engram TP sharding

**Before:** mHC gate preparation is on the caller stream; Engram uses the common implementation.

**After:** New amd/mhc.py overlaps gate projection on an auxiliary stream after a Triton post/collapse/RMSNorm producer; amd/engram.py shards gate work across TP ranks.

**Language:** Triton fused producer + AITER gate API; Triton Engram partial/final kernels with a separate collective.

**Body:** Adds supports_mhc_overlap and mhc_pre_delayed_overlap; Engram creates gate partial sums, all-gathers value/partials, then finishes normalization/gating.

**Fusion:** mHC post+collapse+norm is producer fusion; auxiliary-stream gate work is overlap, not a whole-seam single kernel. Engram still has partial kernel, collective and finish kernel boundaries.

**Guards:** gfx950 ordinary/fallback model path; mHC requires hc_mult=4, fused AITER support, no ubatching and at most 128 tokens. Mono-covered seams bypass this ordinary path. Engram injection remains outside mono K1/K2. mHC overlap is installed only while capturing a FULL graph (not piecewise or eager), so replay follows those graph edges.

Source paths: `vllm/models/deepseek_v41/amd/mhc.py`, `vllm/models/deepseek_v41/amd/engram.py`, `vllm/models/deepseek_v41/amd/model.py`, `vllm/model_executor/kernels/mhc/triton.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v10_to_v11.diff](evidence/source_diffs/v10_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- TP8 support already existed in V10, so it is not new fusion for this TP8 pair.
- MonoBuild.tuned is TP2/ratio2-only and cannot explain TP8 conc1 gains. MI325X/gfx942-only kernels are excluded.

## V09 → V11 (TP2)

7 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### SWA decode metadata fusion enabled on MI355X

**Before:** V09 has no amd/ops SWA helper; MI355X follows the ordinary metadata sequence.

**After:** monokernel_cdna3_or_4 admits gfx950: _decode_swa_meta_kernel writes dense indices/lens, validity, ragged indices/indptr and graph buffers.

**Language:** Triton -> Triton; expanded architecture gate.

**Body:** Helper body remains conceptually the single-program metadata builder; V11 broadens dispatch to CDNA4.

**Fusion:** One helper launch replaces the ordinary sequence of validity, zero-fill, window, counts/cumsum/compaction and graph copies. The source describes eight prior launches per build; no original CI trace verifies realized counts.

**Guards:** --monokernel, pure decode and at most 32 decode token rows. Applies to causal target and supported non-causal DSpark metadata. Client concurrency alone cannot establish the 32-row guard.

Source paths: `vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py`, `vllm/models/deepseek_v41/amd/ops/__init__.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v11.diff](evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### DSpark Markov sampling fusion enabled on MI355X

**Before:** V09 uses the ordinary sequential Markov embedding/GEMV/add/Gumbel/argmax/gather/copy loop.

**After:** V11 permits gfx950 for at most five request rows. Each draft step computes vocabulary-block bias plus logits and block Gumbel-max in one kernel; next step resolves preceding block maxima.

**Language:** Triton -> Triton.

**Body:** Architecture gate expands and MAX_ROWS_GFX950=5 limits selection; helper preserves request/step/position/token-dependent noise and tie policy.

**Fusion:** Five drafts use one launch per step plus one final token-resolution launch, not one launch for all drafts. The helper stores draft logits; Markov GEMV summation order can differ in the last BF16 bit.

**Guards:** --monokernel, supported sequential DSpark sampler and at most five request rows. Larger batches use the ordinary sampler. The benchmark uses synthetic acceptance length 3.51, which limits conclusions about real acceptance.

Source paths: `vllm/models/deepseek_v41/amd/ops/fused_markov.py`, `vllm/models/deepseek_v41/amd/ops/__init__.py`, `vllm/models/deepseek_v41/amd/dspark.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v11.diff](evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### Ordinary-path mHC overlap and Engram TP sharding

**Before:** mHC gate preparation is on the caller stream; Engram uses the common implementation.

**After:** New amd/mhc.py overlaps gate projection on an auxiliary stream after a Triton post/collapse/RMSNorm producer; amd/engram.py shards gate work across TP ranks.

**Language:** Triton fused producer + AITER gate API; Triton Engram partial/final kernels with a separate collective.

**Body:** Adds supports_mhc_overlap and mhc_pre_delayed_overlap; Engram creates gate partial sums, all-gathers value/partials, then finishes normalization/gating.

**Fusion:** mHC post+collapse+norm is producer fusion; auxiliary-stream gate work is overlap, not a whole-seam single kernel. Engram still has partial kernel, collective and finish kernel boundaries.

**Guards:** gfx950 ordinary/fallback model path; mHC requires hc_mult=4, fused AITER support, no ubatching and at most 128 tokens. Mono-covered seams bypass this ordinary path. Engram injection remains outside mono K1/K2. mHC overlap is installed only while capturing a FULL graph (not piecewise or eager), so replay follows those graph edges.

Source paths: `vllm/models/deepseek_v41/amd/mhc.py`, `vllm/models/deepseek_v41/amd/engram.py`, `vllm/models/deepseek_v41/amd/model.py`, `vllm/model_executor/kernels/mhc/triton.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v11.diff](evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### TP2 ratio-2 mono body and scheduling tuning

**Before:** Generic persistent K1/K2 scheduling and intermediate stores.

**After:** MonoBuild.tuned selects a gfx950 target-only TP2 compression-ratio-2 build; other shapes retain generic code.

**Language:** FlyDSL -> FlyDSL.

**Body:** Tuned seam combines TP reduction and residual slice, publishes normalized inputs before gates; MoE down stages pipeline weight loads/MFMA rounds and use cooperative shared-expert work. Attention readiness scratch/publication changes.

**Fusion:** Retains two persistent launches but fuses/reorders internal seam work and reduces internal handoff traffic; no new layer-wide launch collapse is claimed.

**Guards:** Only TP2 target layers, ratio 2, ordinary target routing (384 experts/top-6) and supported row/residency guards. Excludes TP4, TP8 and DSpark routing.

Source paths: `vllm/models/deepseek_v41/amd/mono/layer.py`, `vllm/models/deepseek_v41/amd/mono/stages/seam.py`, `vllm/models/deepseek_v41/amd/mono/stages/moe.py`, `vllm/models/deepseek_v41/amd/mono/attention/plan.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v11.diff](evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- V10 contains only TP8 conc1; this measured TP2/TP4 delta spans a blank V10 column.
- New TP8 support is not a new fusion for this already supported TP2 pair.

## V09 → V11 (TP4)

2 measured same-configuration pairs. See the pair index for exact dates, both axes and endpoint-specific commands.

### SWA decode metadata fusion enabled on MI355X

**Before:** V09 has no amd/ops SWA helper; MI355X follows the ordinary metadata sequence.

**After:** monokernel_cdna3_or_4 admits gfx950: _decode_swa_meta_kernel writes dense indices/lens, validity, ragged indices/indptr and graph buffers.

**Language:** Triton -> Triton; expanded architecture gate.

**Body:** Helper body remains conceptually the single-program metadata builder; V11 broadens dispatch to CDNA4.

**Fusion:** One helper launch replaces the ordinary sequence of validity, zero-fill, window, counts/cumsum/compaction and graph copies. The source describes eight prior launches per build; no original CI trace verifies realized counts.

**Guards:** --monokernel, pure decode and at most 32 decode token rows. Applies to causal target and supported non-causal DSpark metadata. Client concurrency alone cannot establish the 32-row guard.

Source paths: `vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py`, `vllm/models/deepseek_v41/amd/ops/__init__.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v11.diff](evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### DSpark Markov sampling fusion enabled on MI355X

**Before:** V09 uses the ordinary sequential Markov embedding/GEMV/add/Gumbel/argmax/gather/copy loop.

**After:** V11 permits gfx950 for at most five request rows. Each draft step computes vocabulary-block bias plus logits and block Gumbel-max in one kernel; next step resolves preceding block maxima.

**Language:** Triton -> Triton.

**Body:** Architecture gate expands and MAX_ROWS_GFX950=5 limits selection; helper preserves request/step/position/token-dependent noise and tie policy.

**Fusion:** Five drafts use one launch per step plus one final token-resolution launch, not one launch for all drafts. The helper stores draft logits; Markov GEMV summation order can differ in the last BF16 bit.

**Guards:** --monokernel, supported sequential DSpark sampler and at most five request rows. Larger batches use the ordinary sampler. The benchmark uses synthetic acceptance length 3.51, which limits conclusions about real acceptance.

Source paths: `vllm/models/deepseek_v41/amd/ops/fused_markov.py`, `vllm/models/deepseek_v41/amd/ops/__init__.py`, `vllm/models/deepseek_v41/amd/dspark.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v11.diff](evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

### Ordinary-path mHC overlap and Engram TP sharding

**Before:** mHC gate preparation is on the caller stream; Engram uses the common implementation.

**After:** New amd/mhc.py overlaps gate projection on an auxiliary stream after a Triton post/collapse/RMSNorm producer; amd/engram.py shards gate work across TP ranks.

**Language:** Triton fused producer + AITER gate API; Triton Engram partial/final kernels with a separate collective.

**Body:** Adds supports_mhc_overlap and mhc_pre_delayed_overlap; Engram creates gate partial sums, all-gathers value/partials, then finishes normalization/gating.

**Fusion:** mHC post+collapse+norm is producer fusion; auxiliary-stream gate work is overlap, not a whole-seam single kernel. Engram still has partial kernel, collective and finish kernel boundaries.

**Guards:** gfx950 ordinary/fallback model path; mHC requires hc_mult=4, fused AITER support, no ubatching and at most 128 tokens. Mono-covered seams bypass this ordinary path. Engram injection remains outside mono K1/K2. mHC overlap is installed only while capturing a FULL graph (not piecewise or eager), so replay follows those graph edges.

Source paths: `vllm/models/deepseek_v41/amd/mhc.py`, `vllm/models/deepseek_v41/amd/engram.py`, `vllm/models/deepseek_v41/amd/model.py`, `vllm/model_executor/kernels/mhc/triton.py`. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.

Evidence: [evidence/source_diffs/v09_to_v11.diff](evidence/source_diffs/v09_to_v11.diff), [evidence/image_source_inventory.json](evidence/image_source_inventory.json), [evidence/ci_launch_configs.json](evidence/ci_launch_configs.json).

Excluded attributions:

- V10 contains only TP8 conc1; this measured TP2/TP4 delta spans a blank V10 column.
- New TP8 support is not a new fusion for this already supported TP4 pair.
- TP2 ratio-2 tuned schedule is not selected at TP4.
