# Muse Spark endpoint recheck: 2026-11-09

The performance history remains **54 observations, four Docker names, 27 configuration rows, 38 representatives and 11 pairs**. This recheck retrieves the server artifacts for **all 19 distinct pair endpoints**, aligns their diagnostics to each endpoint's actual AgentX phase windows, and rechecks container source bytes. It adds draft-fusion evidence and distinguishes changed dispatch guards from unchanged device bodies. **No GPU-time top-kernel ranking or causal kernel speedup has been established.** The [GPU allocation attempt](gpu_validation/README.md) obtained no node.

The [configuration matrix](configuration_history.md) and [pair index](pair_changes.md) remain the main performance view. TEP8 c2 and latest c10 still have no earlier same-configuration partner; this recheck does not manufacture pairs for them.

## Coverage and evidence

Each artifact was selected from the `/artifacts/<id>` upload link in that endpoint's own GitHub job log, rather than matching a run-level filename or another concurrency. All 19 archives were recovered, totaling 345,937,623 bytes; nested server stdout was unpacked and hashed. All 19 endpoints have warmup start/end and E2E measurement start/end markers. [Archive manifests](evidence/verified_endpoint_logs.json) retain links, members, hashes and external locations. Full logs remain outside Git.

[Phase-aligned facts](evidence/verified_runtime_facts.json) retain exact lines, first/last examples, logged row sets and file hashes. [Pair runtime comparisons](evidence/verified_pair_runtime.json) join these facts to all 11 existing edges. Each [pair report](pair_changes.md) now includes its own endpoint artifact IDs and phase-specific dispatch table, replacing the earlier incomplete artifact coverage statement.

The selected diagnostics comprise 143 initialization lines, 34,332 setup/graph-warmup dispatch lines, 520 benchmark-warmup dispatch lines, 488 boundary-uncertain dispatch lines and 5,692 dispatch lines inside E2E measurement. These are **log lines across ranks and layers, not kernel invocations**. No `JIT compilation during inference:` warning was found in the inspected server stdout. Neither absence nor the differing diagnostic counts establishes non-use, top-kernel status, or execution frequency.

The `attention_kv_write._note` logger emits once per path name; `router_pooler.route_with_shared_pooler` emits once per `(path, layer, rows)` or shared-path category. Graph capture can therefore consume a diagnostic before the benchmark. The timestamps use the recorded server/client clocks; server events within one second of the measurement start are kept uncertain. AIPerf's `Profiling` and `profile_export` refer to request measurement, not GPU profiling.

## What the pair logs establish

| Image transition / configuration | Earlier endpoint | Later endpoint | Interpretation |
| --- | --- | --- | --- |
| Baseline → 1020, TEP8 c1/c4/c8 and c16 HiCache | No selected Muse dispatch diagnostics; existing DFLASH KV materialization initializes | QKV/QK cache and router/pooler paths reached in server setup | The new guarded source paths are reached during setup; sustained E2E coverage is unknown |
| Baseline → 1027, TEP4 c16 HiCache (gap pair) | Existing DFLASH KV materialization initializes | Ordinary fast paths and TP4 finalize/all-reduce initialize during setup; no monokernel CLI | Preserve the missing 1020 point; do not attribute target/draft monokernels to this TEP4 pair |
| 1020 → 1027, TEP8 c1 | Ordinary path diagnostics during setup | Target width 8 and draft stage 2 built/attached during setup; router/pooler rows 16 occur at the measurement boundary | Boundary events remain uncertain; no measured mono share or launch count |
| 1020 → 1027, TEP8 c4 | Ordinary path diagnostics during setup | Target/draft widths 8 and 16 initialize; ordinary router/pooler rows 8 also logged inside measurement | Ordinary work coexists with configured monokernel routing; logs cannot quantify its fraction |
| 1020 → 1027, TEP8 c8 | Ordinary path diagnostics during setup | Target/draft widths 8 and 16 initialize; selected dispatch messages are setup/warmup only | Missing measurement-phase messages are not proof of non-use |
| 1027 → 1106, TEP4 c28 HiCache | Ordinary paths reached during setup; TP4 finalize/all-reduce already enabled | Ordinary joint router/pooler rows **76, 77, 78, 82, 85** logged inside measurement | Broadened joint routing is reached above the old 64-row boundary |
| 1027 → 1106, TEP4 c36 HiCache | Same evidence class as c28 | Ordinary joint router/pooler rows **36, 77, 80, 83** logged inside measurement | Both existing small shapes and newly admitted larger shapes occur |
| 1027 → 1106, TEP4 c40 HiCache | Same evidence class as c28 | Joint router/pooler rows **39, 63, 74, 80, 81, 90, 91, 101, 102, 113, 114, 121**; shared router/norm + hipBLASLt pooler rows **28** logged inside measurement | Joint and shared fallback dispatches both occur; throughput +29.02% / interactivity −24.42% remains an unexplained deployment tradeoff |

The latest TEP4 artifacts also show the FP8-pooler joint path during setup, including larger row counts. They do not show that marker inside the measurement window. The ordinary joint marker and the **FP8 pooler** marker are separate branches and must not be conflated. A log row count is the tensor rows at that logging site, not necessarily client concurrency multiplied by draft width.

## Source bodies and applicability

The [source audit](evidence/verified_source_audit.json) verifies the four archive hashes and **all 231 existing relevant-file entries**, then records additional function-source and AST hashes for the dispatch/device modules inspected here. AST equality is a syntax comparison, not proof of identical runtime behavior: globals, callees, launch grids and shapes may change. [Supplemental diffs](evidence/verification_diffs/03_to_04.diff) and [pinned excerpts](evidence/verification_source_excerpts.md) show the exact changes.

| 1027 → 1106 component | Body/path result | Fusion interpretation |
| --- | --- | --- |
| `/opt/sgl-amd-mono/sgl_amd_mono/` | All **97 captured package files** have the same paths and SHA256 values | No new target or draft monokernel body in this transition; TEP4 matched jobs do not enable it |
| `kernels/ops/attention/qk_norm_rope_cache.py` | All six function ASTs are identical | Existing fused QK/RoPE/cache body remains; `attention_kv_write.py` broadens page-size/row guards |
| `rocm_muse_spark/router_pooler.py` | `route_with_shared_pooler` unchanged; `can_share_router_pooler` and a module-level limit change | Ordinary joint routing can now admit supported shapes above 64 rows; this is confirmed by the latest endpoint logs |
| `kernels/ops/moe/router_pooler_topk.py` | `_joint_epilogue`, ordinary `joint_router_pooler_topk` and `_joint_finish` unchanged; `_joint_fp8_partial` and its wrapper change | Preserve the distinction between unchanged ordinary fusion and updated FP8 tiling/coverage |
| FP8 joint partial producer | Adds a third grid axis for row tiles, caps tile M at 128, retunes configurations, extends supported M from 32 to 512 | Partial GEMM and epilogue remain separate launches; no new single-launch whole-pipeline fusion is claimed |
| TP4 finalize + all-reduce | Both endpoints' logs report it enabled during setup | Already introduced by 1027; extending TP8 support in 1106 is not a newly introduced TP4 fusion |

The ordinary joint router/pooler consists of a packed projection partial producer followed by `_joint_epilogue` through `_joint_finish`. The epilogue combines reduction/routing/top-k with pooler normalization and optional expert-input quantization. The log message does not mean that the GEMM and epilogue are one kernel. The new observed >64-row routes establish guard coverage, not their share of the measured speed change.

## Draft fusion

The earlier pair records emphasized the target layer monokernel but omitted an additional change on **1020 → 1027 TEP8 c1/c4/c8**: the exact server artifacts report `muse_spark DFlash2 draft attached at stage 2`, plus a TP-sharded context projection with 1024 output rows per rank. These initialization messages occur before measurement. Their source is `sgl_amd_mono/muse_spark/draft.py`; the device implementations import FlyDSL. The `_hip` spec names do not identify the implementation language.

The source launch boundaries for each of the five eligible draft layers are:

| Launch | Combined operations | Device body |
| --- | --- | --- |
| D-pre | Residual/input norm, convolution coefficients and conv0, QKV projection, per-head Q/K norm, table RoPE, Q output and FP8 K/V stores | `gfx950/draft_attn_half.py` |
| D-attn | Block attention over the compact windowed KV cache | `gfx950/draft_attn.py` |
| D-post | O projection, TP sum and conv1 | `gfx950/draft_attn_half.py` |
| D-MLP | Post-norm/residual, convolutions, gate/up/down and TP sum | `gfx950/draft_mlp.py` |

`DraftRunner.launch_attn` explicitly invokes pre, attention and post, and `_Draft.forward` then invokes the MLP launch. This is **four source-level launch boundaries per eligible layer**, not one fused draft or one fused model. Context projection/all-gather, embedding/final norm and token selection remain outside those layer launches. Existing DFLASH fused KV materialization also appears in baseline artifacts and must not be credited as a new 1027 optimization.

The three TEP8 pair records now include these paths, file hashes, languages, semantics and guards. The launch-count statement is `source_verified`; actual graph-replay counts, draft/target time shares and historical GPU-time leaders remain `unavailable`. Stage 0/1 and unsupported-shape fallback behavior must remain separate if later profiling uses different settings.

## Reproduce the recheck

Run from the repository root, with the original external cache available:

```bash
python3 tools/inferencex/muse_spark/collect_endpoint_logs.py \
  --cache /shared_nfs/yueliu14/inferencex_muse_spark_research \
  --knowledge knowledge/top_model/muse_spark
python3 tools/inferencex/muse_spark/analyze_endpoint_logs.py \
  --cache /shared_nfs/yueliu14/inferencex_muse_spark_research \
  --knowledge knowledge/top_model/muse_spark
python3 tools/inferencex/muse_spark/audit_sources.py \
  --knowledge knowledge/top_model/muse_spark
python3 tools/inferencex/skills/inferencex-kernel-evolution/scripts/build_pair_history.py \
  --knowledge knowledge/top_model/muse_spark
```

The collector reuses cached archives or downloads missing ones through `gh api`. Processing time and cache mtime are recorded separately; cached artifacts are not assigned a fabricated fresh retrieval timestamp. Full phase events are stored externally with hashes. No scripts in this sequence submit a GPU job.

Validation: all endpoint dates/metrics match the previously committed pairs, delta formulas and exact artifact membership pass, all 19 phase windows are complete, report links resolve, and pair/matrix regeneration is idempotent. Repository checks pass: **222 tests**, corpus generated-page check, and workload generated-page check. These checks do not validate GPU execution.
