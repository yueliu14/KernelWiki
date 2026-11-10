# Muse Spark Open: MI355X / SGLang evolution

Initial collection: **2026-11-09 UTC**. [Dashboard](https://inferencex.semianalysis.com/infx/inference/muse-spark-open) · [Workflow skill](../../../tools/inferencex/skills/inferencex-kernel-evolution/SKILL.md)

The retained public AgentX history contains **54 measured points, 6 CI runs and 4 distinct Docker image names**, covering dashboard dates **2026-10-16 through 2026-11-08**. Docker pulls and CPU source exports succeeded for all four immutable digests. The primary [configuration matrix](configuration_history.md) contains **27 configuration rows and 38 representatives**, with **11 pair-specific kernel records**: ten adjacent-image pairs and one TEP4 c16 HiCache pair spanning a missing version. GPU-time top-kernel rankings remain unmeasured.

- [History by configuration](configuration_history.md): the full matrix, including singletons such as TEP8 c2, missing observations, and links to each pair's kernel changes.
- [Point-to-point kernel records](pair_changes.md): paths, languages, body/fusion changes, applicability, exact CI differences and evidence for all 11 pairs.
- [Kernel implementation browser](kernel_implementations.md): direct source and function-line links for each image version, using byte-exact container snapshots.
- [Endpoint runtime recheck](runtime_verification.md): all 19 pair endpoint artifacts, measurement-window dispatch evidence, function-body checks and DFlash2 draft fusion. The [bounded GPU attempt](gpu_validation/README.md) obtained no allocation and produced no trace.
- [Every measured point](performance_points.md), including repeat observations of the first image.
- [Kernel and fusion evolution](kernel_evolution.md): three adjacent transitions, paths, languages, changed bodies, and evidence limits.
- [Structured evolution edges](evidence/evolution_edges.json): adjacent image nodes, performance pairs, fusion findings and evidence status.
- [Normalized points](data/points.json), [image occurrences](data/images.json), [comparison CSV](data/point_comparisons.csv), and [transition membership](data/transitions.json).
- [Image source inventory](evidence/image_source_inventory.json), [source diffs](evidence/source_changes.json), [CI metadata](data/ci_manifest.json), [launch configurations](evidence/ci_launch_configs.json), and [server-log extracts](evidence/server_log_extracts.json).

## Metric and comparison rules

**x = p90 interactivity (tok/s/user)**, the reciprocal of p90 full-response ITL in seconds. **y = Output Token Throughput per Chip (tok/s/chip)**, the API field `output_tput_per_gpu`. Both axes are better when larger. This is not the percentile of token rates; the similarly named producer `p90_full_response_intvty` is not used.

All selected points are aggregate AgentX replay, FP4 model precision, DFLASH draft-model speculation, and TP=EP. Here `TEP8` means TP8/EP8 on 8 chips, not 64. AgentX has variable input/output lengths, so ISL and OSL are null. The public dataset in sampled logs is `semianalysis_cc_traces_weka_062126`. The model path is `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4`; the draft is `RadixArk/Muse-Spark-Open-DFlash2`.

Version nodes are deduplicated by the full Docker name/tag; immutable digests are retained. Representatives are the earliest valid observation **per image and matching dashboard configuration**, including TP/EP, concurrency, offload mode and available protocol fields. Other observations remain in the point table and per-image spread records. Same-image recipe changes are preserved as observations rather than new image versions. These spreads are not a statistical noise estimate.

Every matched pair changes its recipe fingerprint. The comparisons below therefore describe deployment evolution at the same available workload configuration; they do not isolate kernel improvements. Actual model/dataset revisions, admission settings, quantization paths, simulation and cache policy need the linked CI/source context. All 24 inspected performance job logs use expected DFLASH acceptance length **4.05**, `match-expected`, and `real-draft-token`.

## Image timeline

All image names below have prefix `lmsysorg/sglang-rocm:`. Dates are API/dashboard dates, not image creation dates or job execution timestamps; those are stored separately. For example, the latest dashboard date is November 8 but the example job ran on November 7. October 19 reuses the first image.

| First dashboard date | Last occurrence | Image tag | Digest prefix | Measured points | CI runs |
| --- | --- | --- | --- | ---: | ---: |
| 2026-10-16 | 2026-10-19 | `muse-spark-open` | `72065dfc0b87` | 30 | 3 |
| 2026-10-22 | 2026-10-22 | `muse-spark-open-1020` | `7343002d0dc2` | 8 | 1 |
| 2026-10-30 | 2026-10-30 | `muse-spark-open-1027` | `7184d3ae2bfd` | 10 | 1 |
| 2026-11-08 | 2026-11-08 | `muse-spark-open-mi355-1106` | `d231181df079` | 6 | 1 |

## Adjacent-image point comparisons

Each `(x, y)` pair is `(p90 interactivity, output tok/s/chip)`. `HiCache` means DRAM offload; `GPU` means no offload. Percentages use `100 × (new / old − 1)`. Point IDs link back to exact raw records and CI jobs through the normalized data.

| Dates | Configuration | Before (x, y) | After (x, y) | Δ interactivity | Δ throughput/chip | Point IDs |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 10-16 → 10-22 | TEP8 c1 / GPU | (184.162, 17.808) | (316.456, 18.770) | +71.84% | +5.40% | 445154 → 446033 |
| 10-16 → 10-22 | TEP8 c4 / GPU | (164.204, 25.608) | (277.778, 28.073) | +69.17% | +9.62% | 445155 → 446037 |
| 10-16 → 10-22 | TEP8 c8 / GPU | (116.822, 43.152) | (223.214, 52.504) | +91.07% | +21.67% | 445156 → 446039 |
| 10-16 → 10-22 | TEP8 c16 / HiCache | (66.181, 80.089) | (152.672, 85.642) | +130.69% | +6.93% | 445172 → 446032 |
| 10-22 → 10-30 | TEP8 c1 / GPU | (316.456, 18.770) | (469.484, 20.888) | +48.36% | +11.29% | 446033 → 446944 |
| 10-22 → 10-30 | TEP8 c4 / GPU | (277.778, 28.073) | (373.134, 31.544) | +34.33% | +12.37% | 446037 → 446945 |
| 10-22 → 10-30 | TEP8 c8 / GPU | (223.214, 52.504) | (363.636, 54.006) | +62.91% | +2.86% | 446039 → 446946 |
| 10-30 → 11-08 | TEP4 c28 / HiCache | (105.152, 212.740) | (125.313, 264.845) | +19.17% | +24.49% | 446940 → 447695 |
| 10-30 → 11-08 | TEP4 c36 / HiCache | (54.437, 237.046) | (97.182, 307.267) | +78.52% | +29.62% | 446941 → 447696 |
| 10-30 → 11-08 | TEP4 c40 / HiCache | (95.602, 258.197) | (72.254, 333.119) | -24.42% | +29.02% | 446942 → 447697 |

TEP8 c10 has only one retained observation: **2026-11-08, x=373.134, y=64.960**, [job 128469428179](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42394597374/job/128469428179). It has no same-configuration earlier point. The 1027 → 1106 matched set is TEP4 c28/c36/c40 with HiCache. At c40, throughput rises **29.02%** while interactivity falls **24.42%**.

## What the container inspection established

- **Baseline → 1020:** additional QKV/QK norm/RoPE/FP8 cache epilogues, router/pooler and FFN-boundary fusion, and new A4 expert paths.
- **1020 → 1027:** the AMD FlyDSL Muse target monokernel, DFlash2 stage-2 draft fusion, and a separate HIP MoE-finalize/all-reduce fusion. The target monokernel combines attention, projections, routing, expert work, communication and the next eligible layer's QKV preparation across former launch boundaries. The draft uses four launches per eligible layer, not one whole-draft kernel.
- **1027 → 1106:** the captured Muse monokernel package bodies are unchanged. Ordinary paths add full-attention FlyDSL prefill, balanced verify scheduling and broader existing fusion support, including TP8 finalize/all-reduce and larger QK/cache shapes.

All images report the same SGLang Git HEAD `c722c0a160981f21c81cffaab7bbcc117089a2a6`, while source hashes differ substantially. The digest and exported file hashes are the source identity used for this report. A server “enabled/built” message confirms initialization; GPU traces are still needed for time shares and realized launch counts.

The follow-up verified all 231 relevant source-file entries and found all 97 captured monokernel package files identical from 1027 to 1106. Exact latest TEP4 c28/c36/c40 logs show joint router/pooler dispatch during measurement, including shapes above the former 64-row guard. The underlying ordinary joint epilogue remains unchanged; expanded dispatch coverage and FP8 producer changes are recorded separately in the [runtime recheck](runtime_verification.md).

## Coverage, provenance and reproduction

This covers the history **currently retained by the public published-curve API**, not every historical CI execution or a reconstruction of what each historical website deployment displayed. The first available MI355X/SGLang record is October 16; prior unpublished or removed curves are unknown. The latest snapshot's six points are all contained in the retrieved history. All 54 measured points have unique matching CI jobs.

The [retrieval manifest](data/retrieval_manifest.json) records URLs, sizes and SHA256 values. The initial page/API exploratory downloads have cache timestamps instead of verified retrieval timestamps, and this distinction is preserved. The initial raw cache, full job logs, nested server archives and full source bundles are outside Git at `/shared_nfs/yueliu14/inferencex_muse_spark_research/`. Compact source/CI excerpts, manifests, selected diffs and performance data are kept here. The [CPU allocation receipt](evidence/cpu_job_receipt.txt) records job 275462 with **1 CPU and no GPU**, completed successfully.

The inspected dashboard source is [InferenceX-app benchmark-transform.ts](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d475545122661430a1f6de280e81418b0ba2f198/packages/app/src/lib/benchmarks/benchmark-transform.ts) and [historical query implementation](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d475545122661430a1f6de280e81418b0ba2f198/packages/db/src/queries/benchmarks.ts). This commit is the inspected upstream source revision; the live deployed-build SHA was not verified. Full-response ITL aliases and chip normalization were checked against every selected raw point.

From the KernelWiki root, replay the cached collection with:

```bash
python3 tools/inferencex/skills/inferencex-kernel-evolution/scripts/collect_history.py \
  --cache /shared_nfs/yueliu14/inferencex_muse_spark_research/raw \
  --output knowledge/top_model/muse_spark/data --ci --offline
```

Use a fresh cache directory and omit `--offline` to refresh. The [skill reference](../../../tools/inferencex/skills/inferencex-kernel-evolution/references/muse-spark-pilot.md) documents CPU image inspection and source comparison. Kernel top-time tables and causal attribution remain the next measurement stage, with missing trace evidence explicitly marked.
