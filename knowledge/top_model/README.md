# Dated model histories

These pages reconstruct published **deployment evolution**, scoped to a model, hardware and framework. They complement operator implementation contracts and workload cases; their E2E deltas are not universal kernel recommendations.

| Model / stack | Start here | Captured history |
| --- | --- | --- |
| Muse Spark Open / MI355X / SGLang | [Model guide](muse_spark/README.md) → [matrix](muse_spark/configuration_history.md) | 2026-10-16 → 2026-11-08; 54 observations, 11 pairs |
| DeepSeek V4.1 Flash / MI355X / vLLM | [Model guide](deepseek_v41_flash/README.md) → [matrix](deepseek_v41_flash/configuration_history.md) | 2026-09-12 → 2026-11-09; 154 observations, 110 pairs |

Both were collected on 2026-11-09. “Latest” means the latest retained point in that collection, not a live assertion about today's upstream performance.

## Read only what the question needs

1. Select a matrix row with the requested TP/EP, concurrency and offload mode. Blank cells remain absent; singleton rows do not have a performance pair.
2. Open its pair for both metric axes, dates, CI jobs, launch differences and applicability. Follow the linked shared image-change record for body/fusion details, then the source/function link when inspecting implementation.
3. Open runtime evidence for observed dispatch. Audit metadata, full kernel snapshots and source diffs are supporting data, not another narrative to ingest by default. Each model's `evidence/README.md` identifies the relevant records and historical subsets.

## Evidence

The curve values are upstream-reported CI measurements. They do not imply a local benchmark reproduction by KernelWiki.

| Evidence | Supports | Does not establish |
| --- | --- | --- |
| Published CI measurements | The recorded E2E point for its workload/recipe | An individual kernel's causal speedup |
| Inspected source / digest / function hash | Bodies, guards and possible launch boundaries | Executed paths or GPU-time shares |
| Server active/HIT/JIT diagnostics | The logged event, shape and phase | Complete execution sets, invocation counts or top kernels |
| Compatible GPU trace | Executed kernels and timings in its capture window | Historical CI execution when the trace is a new reproduction |

Neither current model history has a compatible GPU-time ranking. Muse's bounded GPU attempt obtained no allocation; its prepared commands are unexecuted reproduction inputs. AIPerf `Profiling` and `profile_export` are request-level measurement outputs. Initialization before measurement is not proof of sustained coverage, and missing once-only diagnostics do not prove non-use.

Framework commits, dependency commits and container identity are separate. A stable function name need not mean a stable body; identical bodies can behave differently under changed guards, tuning tables, inputs or callees. Distinguish a fused kernel from a graph containing multiple kernels and from communication overlap.

## Metrics and pairing

- **x:** p90 interactivity, `1 / p90_full_response_itl` in seconds, in tok/s/user. Use legacy p90 ITL only when the full-response schema is absent. Do not substitute a percentile of token rates.
- **y:** `output_tput_per_gpu`, already output tok/s/chip. Both axes are better when larger. TP×EP is not automatically the physical chip count.
- Version nodes are Docker names/tags; digests identify source snapshots. Use the earliest valid observation per image and full available configuration, retaining repeats separately. Connect consecutive **observed** versions and record gaps; never interpolate missing configurations.
- Dates are dashboard dates, separate from job timestamps. Model/data revisions, speculation policy, admission, cache placement, graph settings and other launch changes remain explanatory variables.

Metric implementation: [pinned dashboard transform](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d475545122661430a1f6de280e81418b0ba2f198/packages/app/src/lib/benchmarks/benchmark-transform.ts). This inspected commit is not a verified live-deployment revision. Collection and maintenance use the [InferenceX skill](../../tools/inferencex/skills/inferencex-kernel-evolution/SKILL.md).
