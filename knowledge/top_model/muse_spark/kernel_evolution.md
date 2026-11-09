# Muse Spark kernel evolution: MI355X / SGLang

This pilot compares actual text/source bytes exported from four Docker images. All four pulls and CPU source exports succeeded on 2026-11-09. [Image inventories](evidence/image_source_inventory.json) identify exact digests, paths, file SHA256 values, packages, dirty Git trees, and external source archives. [Source changes](evidence/source_changes.json) and [selected code excerpts](evidence/source_excerpts.md) support the observations below.

For configuration-specific applicability, use the [matrix](configuration_history.md) and [11 individual pair records](pair_changes.md). The version-level summary below groups source changes; a finding here is not automatically applicable to every TP/concurrency point. Each pair now has both endpoint launch logs and a separate guard/flag analysis.

These are source-verified implementation changes, with selected CI server logs confirming configured/initialized paths. They are not a measured top-kernel ranking or an attribution of an E2E gain to one kernel. No matching GPU execution trace has been established in this pilot. The inspected artifacts include request-level AgentX results and server logs; `agentx profile` is not a GPU kernel profiler. The corpus keeps this distinction explicit.

## 2026-10-16 → 2026-10-22: baseline → 1020

Images: `muse-spark-open@sha256:72065dfc…` → `muse-spark-open-1020@sha256:7343002d…`. See the [exact source diff](evidence/source_diffs/01_to_02.diff).

The baseline already contains fused Q/K normalization, output RMSNorm plus sigmoid gate, residual/norm boundaries, a fused router, and FlyDSL expert kernels. The change is an expansion and reorganization of fusion, not the first use of fused kernels.

| Pipeline | Before | After and implementation language | Evidence and boundary |
| --- | --- | --- | --- |
| Attention Q/K and cache | Projection → Q/K norm → RoPE → attention backend cache store | `srt/rocm_muse_spark/attention_kv_write.py` dispatches to Triton `kernels/ops/attention/qk_norm_rope_cache.py`; Q/K normalization, query scaling, RoPE, FP8 conversion and KV store share an epilogue | `model.py` sets `save_kv_cache=False` after a successful fused write. The QKV variant consumes split-K projection partials; it still has a partial-GEMM launch followed by an epilogue launch. It is not one monolithic QKV kernel. |
| Router and pooler | Separate routing and pooler operations | New `router_pooler.py` uses `router_pooler_topk.py`; router top-k and pre-expert RMSNorm share an epilogue, with a joint projection option | Shape and weight guards determine eligibility. The wrapper name alone does not imply the whole MoE is one launch. |
| MoE expert numerics | Existing native expert path | New `mxfp4_moe.py`: compact A4 verify experts in Triton, plus AITER FlyDSL A4W4 prefill stages; stage 1 can write MXFP4 values and sorted scales directly | Per-expert normalization before route-weighted reduction is preserved in a separate FlyDSL finalize. Quantization/numerics change is an additional explanation variable. |
| Unpooler / layer boundary | Unpooler output followed by FFN boundary operations | Fused unpooler/boundary path and sequence-parallel prefill wiring appear in `model.py` and `sequence_parallel.py` | Elimination of intermediate processing is conditional on dispatch; sequence parallelism and Ulysses communication are distinct from GPU-kernel fusion. |

The same TEP8 c8 dashboard point changes from **(116.822, 43.152)** to **(223.214, 52.504)**: interactivity **+91.07%**, output throughput/chip **+21.67%**. The launch recipe also adds a 2048-token draft window, prefill/decode interval 20, scheduler receive interval 20, explicit seed, tokenizer workers and SWA allocation controls. These prevent a kernel-only causal interpretation.

The 1020 image contains the SGLang monokernel interface, but this is not evidence that the AMD Muse monokernel ran. Its representative CI launch has no `--monokernel-backend amd`, and the AMD plugin package first appears in the next image. Server artifact `12384514175` records specialized verify paths and fused QKV/cache paths; see [server-log extracts](evidence/server_log_extracts.json).

## 2026-10-22 → 2026-10-30: 1020 → 1027

Images: `muse-spark-open-1020@sha256:7343002d…` → `muse-spark-open-1027@sha256:7184d3ae…`. See [source diff](evidence/source_diffs/02_to_03.diff) and excerpts E2–E4.

**The central fusion change is the AMD Muse Spark monokernel.** The installed package is `sgl_amd_mono`, also present under `/opt/sgl-amd-mono`; its Muse implementation is FlyDSL, although the spec/log identifier is `muse_spark_hip`.

The actual call chain is SGLang monokernel runtime → `sgl_amd_mono/muse_spark/spec.py` → `gfx950/runner.py` → `gfx950/layer.py:build_layer`. Under the default `SGLANG_AMD_MONO_MUSE_LAYER=1`, eligible MXFP4 layers merge these stages into a device launch:

1. Attention over paged KV.
2. Output norm/gate, O projection and attention all-reduce, attention boundary, router/pooler, top-8 and pre-expert norm.
3. Experts, finalize plus EP all-reduce, unpooler and FFN boundary.
4. The next eligible layer's QKV projection, Q/K norm, RoPE and FP8 cache write.

The first needed K1 runs separately; subsequent eligible layers can inherit K1 from the previous merged launch. BF16-expert layers 0 and 61 use the ordinary SGLang layer. Prefill and unsupported shapes also fall back. The implementation uses device mailbox flags and shared-memory reuse across stages; scratch regions still exist. Therefore fewer host launch boundaries do **not** mean all intermediate global-memory traffic disappears.

For the representative TEP8 c8 run, server logs explicitly announce `muse_spark_hip on gfx950, widths [8, 16] (8 tokens a request)` and successful builds. The client still has concurrency 8, while `max-running-requests` changes from **128 to 2** and `cuda-graph-max-bs-decode=2` is set. That admission cap and `prefill-decode-interval=64` are major co-changes. The API point moves from **(223.214, 52.504)** to **(363.636, 54.006)**: **+62.91%** interactivity and **+2.86%** throughput/chip. A trace would be needed to quantify which phases or launches account for that movement.

A separate, ordinary-path fusion also appears: `rocm_muse_spark/finalize_allreduce.hip` and its Python/ctypes launcher replace FlyDSL MoE finalize plus the AITER all-reduce with **one HIP/C++ launch** for supported TP4 compact-A4 verify rows. The device body performs per-expert normalization, route-weighted summation and peer exchange/reduction. This differs from the FlyDSL monokernel. Server artifact `12862296844` confirms TP4 finalize/all-reduce initialization. New FP8 small-M projection paths and their guards are recorded in the source diff as additional changes.

The per-pair source review also identifies two ordinary-path fusions: `verify_reduce_norm_gate.py` merges split-KV reduction with the existing output RMSNorm/sigmoid gate, and `fused_norm_residual.py` can emit FP8 activations/scales directly from the boundary kernel. Both are conditional on their callers requesting those paths. On MXFP4 layers covered by the monokernel, its layer body supersedes the ordinary path; the pair records preserve that distinction.

## 2026-10-30 → 2026-11-08: 1027 → 1106

Images: `muse-spark-open-1027@sha256:7184d3ae…` → `muse-spark-open-mi355-1106@sha256:d231181d…`. See [source diff](evidence/source_diffs/03_to_04.diff) and excerpts E5–E8.

**All captured Muse Spark source files in the AMD monokernel package have identical content hashes between these two images.** The package paths and bodies are stable. The improvement on the latest curve cannot be described as a newly introduced or rewritten Muse monokernel. The changes observed here are in SGLang's ordinary paths, eligibility guards, attention implementations and launch configuration.

| Change | Source-level finding | Fusion interpretation |
| --- | --- | --- |
| QK norm/RoPE/FP8 cache | Same wrapper path; allowed page sizes expand from 16 to 16/32/64/128, and the row cap moves from 64 to a configurable default of 512 | Broader applicability of existing fusion, not a new QK kernel body. Relevant to the latest TP4 page-size 128 recipe. |
| MoE finalize + all-reduce | Same HIP source path; TP4-only body becomes compile-time TP4/TP8, adds an all-peer one-shot protocol, and changes the device symbol from `muse_finalize_allreduce_tp4` to `muse_finalize_allreduce` | Fusion expands to TP8. The wrapper selects one-shot below 16 rows and two-stage from 128 rows for TP8; intermediate sizes fall back. Server logs confirm TP8 initialization, not a per-call timing gain. |
| Joint router/pooler | The joint path expands beyond 64 rows up to default 128 with a row-major tile, and the model consults `FP8_JOINT_MAX_M` | Existing joint projection/epilogue fusion reaches additional shapes. Guards and the actual tile matter. |
| Full-attention prefill | New `muse_extend_flydsl.py` is selected by the AITER backend when applicable; Gluon remains available, including sliding-window paths | Language/path change: a FlyDSL implementation with FP8 MFMA, LDS staging, transposed fragments and matrix-core row sums. It is not evidence of a new cross-operator fusion. |
| Full-attention verify | New `muse_verify_balanced.py`: Triton planning creates a length-balanced split-KV work list; FlyDSL computes partials and a merge combines them | Work balancing adds a planning launch. Calling this “more fusion” would be inaccurate. |

The balanced route also supplies a five-element partial state to `verify_reduce_norm_gate.py`, which dispatches to `segmented_reduce_norm_gate`. That segmented reducer preserves the existing reduction + output norm/gate fusion while adapting to the new segment layout. The planner itself remains a separate launch.

The directly comparable latest points are **TEP4 c28/c36/c40 with HiCache**, not TEP8 c10. Their throughput changes are **+24.49%, +29.62%, +29.02%**. Interactivity changes are **+19.17%, +78.52%, −24.42%**. The c40 result is a throughput/latency tradeoff.

For TEP4 c40, the CI command also changes page size **16 → 128**, prefill/decode interval **80 → 24**, SWA/full token ratio **0.1 → 0.2**, adds `stream-interval=20` and an explicit target revision, while preserving client concurrency 40, admission 80 and 310 GB/rank HiCache. These settings can affect both metric axes. The latest TEP8 c10 point is **(373.134 tok/s/user, 64.960 tok/s/chip)** and has no same-configuration predecessor in the retained history.

## Evidence limits and next measurement

- Performance values are upstream reported and recalculated from raw API fields, not locally reproduced GPU benchmarks. All 24 inspected performance jobs use DFLASH acceptance simulation with expected length 4.05, `match-expected`, and `real-draft-token`; the recorded public dataset is `semianalysis_cc_traces_weka_062126`.
- `head_sha` from run metadata differs from the effective checkout in all six sampled run groups. [CI source manifests](evidence/ci_source_manifest.json) pin the actual recipe revisions. [CI log extracts](evidence/ci_log_extracts.json) preserve command lines and line numbers.
- The image exporter captures selected source/text extensions. It omits compiled binaries, large files, caches and tuning CSVs; conclusions above are limited to captured bodies and observed dispatch. A package version or Git HEAD is not a source digest.
- GPU-time top kernels, call counts, time shares, launch counts during benchmark phases, and causal fusion speedups remain **unavailable**. A follow-on measurement should replay matched recipes on MI355X, preserve acceptance simulation and trace distribution, profile prefill/verify separately, and compare both the ordinary and monokernel routes. It must label new profiles as reproduced measurements rather than historical CI traces.
