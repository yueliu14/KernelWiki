# DeepSeek-V4.1-Flash: MI355X / vLLM evolution

Collected **2026-11-09 UTC** from the [InferenceX dashboard](https://inferencex.semianalysis.com/infx/inference/deepseek-v41-flash). The requested “DeepSeek v4 flash” resolves to the catalog's **DeepSeek-V4.1-Flash** (`dsv41flash`); no different model version is merged into this history. [Reusable workflow skill](../../../tools/inferencex/skills/inferencex-kernel-evolution/SKILL.md).

The primary [configuration × image matrix](configuration_history.md) covers **2026-09-12 → 2026-11-09**, **11 Docker image names**, **154 observations**, **17 configuration rows**, and **127 image/configuration representatives**. All observations have unique CI job matches. Every one of the **110 consecutive-observed-image pairs** has its own kernel record; **9** span a missing V10 column. The narrower adjacent-image-only CSV contains 101 pairs.

- [Full configuration matrix](configuration_history.md): all observed TP2/TP4/TP8 + concurrency points, with blanks and per-pair links.
- [Pair index](pair_changes.md): both dates, performance changes, source paths/hashes, implementation languages, bodies, fusion and exact launch changes.
- [All 154 observations](performance_points.md): repeats are preserved; deduplication chooses one representative per image/configuration.
- [Kernel/fusion evolution](kernel_evolution.md): image-to-image source chain, including transitions with no measured matching point.
- [Machine-readable pairs](data/pair_records.json), [raw selected history](data/raw_selected_history.json), [coverage](data/coverage.json), [source inventory](evidence/image_source_inventory.json), [source changes](evidence/source_changes.json), [reviewed applicability rules](evidence/kernel_pair_rules.json).

**Container evidence is incomplete for historical tags.** CPU-only job 275610 successfully pulled/exported V01, V09, V10 and V11; V02–V08 tags are unavailable. Those seven snapshots are pinned upstream vLLM commits corroborated by the actual CI startup version, not recovered container bytes. V01's current tag matches the logged vLLM commit but lacks a historical digest. V09–V11 match the API's immutable digests. Historical AITER dependency internals for V02–V08 remain unavailable. These distinctions are carried into every pair record. [Direct Docker Hub/Registry verification](evidence/dockerhub_verification.md) confirms V02–V08 tag 404s; V09 also has a missing tag but its historical digest remains addressable. Missing tags do not prove deleted image bytes.

## Scope and comparison semantics

Hardware is fixed to **MI355X**, and the framework is **vLLM**, as explicitly selected by the user. [Framework discovery](data/framework_discovery.json) records all MI355X coverage: vLLM 154 rows, SGLang 56, ATOM 65 and TileRT 1. The reusable workflow now inspects actual coverage to select vLLM or SGLang; its automatic policy also chooses vLLM here. It never combines different frameworks into one same-stack pair.

These are aggregate AgentX replay points with FP4 model precision and **EP1**, so label them **TP2/EP1, TP4/EP1, TP8/EP1**, not TEP. The matrix's `GPU` means **KV offload is off**. It does not mean all model state is on GPU: the launch's Engram CPU-offload setting is separately retained (often on for TP2). All inspected representative commands use DSpark, **five speculative tokens**, probabilistic draft sampling and **synthetic acceptance length 3.51**. That protocol is not a measurement of natural model acceptance. ISL/OSL vary over the replay dataset.

**x = p90 interactivity = 1 / p90 full-response ITL in seconds (tok/s/user)**. **y = output_tput_per_gpu (output tok/s/chip)**. Both axes are better when larger. Throughput is already per chip: TP4/EP1 uses four chips, not TP×EP or eight chips. The collector uses legacy p90 ITL only when the full-response schema is absent. The inspected chart implementation is [benchmark-transform.ts](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d475545122661430a1f6de280e81418b0ba2f198/packages/app/src/lib/benchmarks/benchmark-transform.ts); this inspected revision does not prove the live deployment SHA.

Version nodes use the full Docker name/tag. Representatives are the earliest valid point for each image and full available dashboard configuration. Repeated same-image observations, their recipe fingerprints and spreads remain in [images.json](data/images.json). Pairing connects consecutive **observed** image versions within a row; it never interpolates a blank or compares different TP/concurrency values. Dates are dashboard dates, distinct from CI creation/execution and image creation times.

Same dashboard configuration does not guarantee identical execution: model/dataset revisions are not fully pinned at all endpoints; Engram placement, server admission, graph capture, batched-token limits, quick-reduce quantization, ROCm/AITER versions and recipes can change. Each pair includes its own two launch commands and environment delta. These are deployment-evolution measurements, not isolated causal kernel speedups.

## Image versions

All tags below have prefix `vllm/vllm-openai-rocm:`. A source commit corroborated by CI is evidence of vLLM revision only, not of all historical image bytes.

| Version | First → last dashboard date | Image tag | Representatives / observations | Source recovery |
| --- | --- | --- | ---: | --- |
| V01 | 2026-09-12 → 2026-09-12 | `deepseekv41-flash-0909` | 6 / 6 | Current tag; historical digest unknown |
| V02 | 2026-09-13 → 2026-09-13 | `nightly-eed1f3d0c6043bd494424a22443ee198dd56f657` | 6 / 6 | CI-corroborated upstream commit; tag unavailable |
| V03 | 2026-09-22 → 2026-09-22 | `nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984` | 16 / 16 | CI-corroborated upstream commit; tag unavailable |
| V04 | 2026-09-24 → 2026-09-24 | `nightly-rocm100-7f1a5398e9610d96c473931a26c0e12bbe0d0423` | 16 / 16 | CI-corroborated upstream commit; tag unavailable |
| V05 | 2026-09-25 → 2026-09-25 | `nightly-rocm100-29468dde8b515031dc6d4d9d06bf0a2fa0442098` | 16 / 16 | CI-corroborated upstream commit; tag unavailable |
| V06 | 2026-10-02 → 2026-10-02 | `nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516` | 14 / 14 | CI-corroborated upstream commit; tag unavailable |
| V07 | 2026-10-12 → 2026-10-12 | `nightly-rocm100-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe` | 14 / 14 | CI-corroborated upstream commit; tag unavailable |
| V08 | 2026-10-13 → 2026-10-13 | `nightly-rocm100-5555a2d8f7116b2b4877f66516f4128176cf722a` | 14 / 14 | CI-corroborated upstream commit; tag unavailable |
| V09 | 2026-10-20 → 2026-11-03 | `nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f` | 14 / 41 | Container; API digest verified |
| V10 | 2026-11-03 → 2026-11-03 | `nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1` | 1 / 1 | Container; API digest verified |
| V11 | 2026-11-09 → 2026-11-09 | `nightly-4fab5a07475822c6eaa971dceb2e2220d0446808` | 10 / 10 | Container; API digest verified |

## Selected point-to-point examples

Cells are `(x, y)` in the units above; the full matrix preserves every configuration, including points without a later version.

| Configuration | Dates / versions | Before (x, y) | After (x, y) | Δ x | Δ y | Kernel record |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TP4/EP1 c1 | 2026-09-12 → 2026-09-13 (V01→V02) | (35.727, 8.052) | (245.700, 33.438) | +587.71% | +315.25% | [Changes](pairs/tp4_ep1_conc1_off_2026-09-12_to_2026-09-13.md) |
| TP4/EP1 c1 | 2026-10-02 → 2026-10-12 (V06→V07) | (347.222, 39.714) | (543.478, 45.129) | +56.52% | +13.63% | [Changes](pairs/tp4_ep1_conc1_off_2026-10-02_to_2026-10-12.md) |
| TP4/EP1 c1 | 2026-10-12 → 2026-10-13 (V07→V08) | (543.478, 45.129) | (568.182, 45.477) | +4.55% | +0.77% | [Changes](pairs/tp4_ep1_conc1_off_2026-10-12_to_2026-10-13.md) |
| TP4/EP1 c1 | 2026-10-13 → 2026-10-20 (V08→V09) | (568.182, 45.477) | (588.235, 45.710) | +3.53% | +0.51% | [Changes](pairs/tp4_ep1_conc1_off_2026-10-13_to_2026-10-20.md) |
| TP4/EP1 c2 | 2026-10-20 → 2026-11-09 (V09→V11) | (526.316, 51.941) | (666.667, 53.374) | +26.67% | +2.76% | [Changes](pairs/tp4_ep1_conc2_off_2026-10-20_to_2026-11-09.md) |
| TP2/EP1 c1 | 2026-10-20 → 2026-11-09 (V09→V11) | (465.116, 86.818) | (452.489, 84.948) | -2.71% | -2.15% | [Changes](pairs/tp2_ep1_conc1_off_2026-10-20_to_2026-11-09.md) |
| TP2/EP1 c32 | 2026-10-20 → 2026-11-09 (V09→V11) | (120.192, 660.261) | (186.916, 645.735) | +55.51% | -2.20% | [Changes](pairs/tp2_ep1_conc32_off_2026-10-20_to_2026-11-09.md) |
| TP2/EP1 c64 | 2026-10-20 → 2026-11-09 (V09→V11) | (64.392, 1341.846) | (62.696, 1349.502) | -2.63% | +0.57% | [Changes](pairs/tp2_ep1_conc64_off_2026-10-20_to_2026-11-09.md) |
| TP8/EP1 c1 | 2026-11-03 → 2026-11-09 (V10→V11) | (684.932, 23.654) | (704.225, 24.005) | +2.82% | +1.48% | [Changes](pairs/tp8_ep1_conc1_off_2026-11-03_to_2026-11-09.md) |

## Evidence and reproduction

Performance values are **measured** E2E CI results. Kernel facts are **source_verified** candidate paths, with historical-container limitations above. **GPU-time top-kernel ranks, benchmark-phase launch counts, and per-kernel causal contributions are unavailable**: no compatible original trace was recovered. Server startup/mono messages establish initialization eligibility and may include warmup; they do not rank hot kernels.

The [127 endpoint CI extracts](evidence/ci_log_extracts.json), [launch configurations](evidence/ci_launch_configs.json), [13-run metadata](data/ci_manifest.json), [pinned recipes](evidence/ci_recipe_sources.json) and [server artifacts](evidence/server_log_extracts.json) preserve provenance. Eleven selected server artifacts have an exact upload link in the endpoint's own job log. Two older V08 attempt artifacts are retained as inspected candidates but are explicitly not matched to the representative endpoint. Canary and rerun jobs are retained where the dashboard used them. Early log environments without a resolved environment line are marked unknown.

[CPU receipt](evidence/cpu_job_receipt.txt) records a completed **1 CPU / 0 GPU** allocation. [Docker receipts](evidence/docker_receipts.json) retain successes and failed historical pulls. Full images, raw logs, artifact ZIPs and source bundles remain outside Git in `/shared_nfs/yueliu14/inferencex_deepseek_flash_research/`; archive/file hashes and compact diffs are committed. [Retrieval manifest](data/retrieval_manifest.json) records API URLs and response identities.

From the repository root:

```bash
python3 tools/inferencex/skills/inferencex-kernel-evolution/scripts/collect_history.py \
  --model DeepSeek-V4.1-Flash --framework auto \
  --cache /shared_nfs/yueliu14/inferencex_deepseek_flash_research/raw \
  --output knowledge/top_model/deepseek_v41_flash/data --ci --offline
python3 tools/inferencex/deepseek_flash/prepare_sources.py \
  --cache /shared_nfs/yueliu14/inferencex_deepseek_flash_research \
  --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/skills/inferencex-kernel-evolution/scripts/build_pair_history.py \
  --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/deepseek_flash/render_summary.py \
  --knowledge knowledge/top_model/deepseek_v41_flash
```

The cached run is deterministic for performance/pair selection. For fresh data, use a fresh raw cache and omit `--offline`; review new source rules before rendering new transitions. [Collection tools](../../../tools/inferencex/deepseek_flash/README.md) describe evidence refresh. Coverage is the currently retained published history, not every CI run ever executed or every historical deployed frontier.
