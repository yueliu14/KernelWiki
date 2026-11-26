# Kernel tuning recipes — AMD Instinct

Per-language procedures for tuning a kernel's configuration on gfx942 (MI300X) and gfx950 (MI355X). Everything here
is method: tuned parameters do not transfer across shapes, architectures or library versions; the procedure does.

> A tuning run that reports a speedup has proven nothing until you have shown that the artifact it produced is the
> code the machine actually runs.

Almost every way tuning fails on this platform fails silently: the config lands where nothing reads it, a lookup
misses on an architecture key, a stale config loads via a fallback, the benchmark measures cache instead of compute.
Every recipe therefore ends with an engagement check, not a timing.

## Start here

| You want to … | Read |
| --- | --- |
| the loop every recipe specializes | [tuning-core/SKILL.md](tuning-core/SKILL.md) |
| a timing protocol that survives gfx950 drift (back-to-back A/B drifts 20-67%; interleave) | [tuning-core/measurement.md](tuning-core/measurement.md) |
| time a graph-captured (decode) kernel | [tuning-core/graph_captured_benchmarking.md](tuning-core/graph_captured_benchmarking.md) |
| prove the tuned kernel is the one that runs | [tuning-core/engagement_verification.md](tuning-core/engagement_verification.md) |
| correctness gates for a tuned or rewritten kernel | [tuning-core/correctness_gates.md](tuning-core/correctness_gates.md) |
| search a config space efficiently | [tuning-core/search_strategy.md](tuning-core/search_strategy.md) |
| gfx942 → gfx950 constants (LDS 64 KB → 160 KB, dispatch floor, FP8 dialect) | [tuning-core/arch_migration.md](tuning-core/arch_migration.md) |
| clocks, power and their effect on timing | [tuning-core/clocks_and_power.md](tuning-core/clocks_and_power.md) |
| a graph-captured benchmark harness and shape lists | [benchmark/README.md](benchmark/README.md) |

## By kernel language

| Language | Recipe | Tuning is … |
| --- | --- | --- |
| Triton / Gluon | [tuning-triton](tuning-triton/SKILL.md) | authoring a config space; the autotuner only races what you supply |
| FlyDSL | [tuning-flydsl](tuning-flydsl/SKILL.md) | same model as Triton, different `Config` semantics |
| Composable Kernel | [tuning-ck](tuning-ck/SKILL.md) | selecting among pre-compiled instances |
| raw HIP | [tuning-hip](tuning-hip/SKILL.md) | choosing launch geometry; the profiler is the only feedback (also the `rocprofv3` reference) |
| AITER dispatch tables | [tuning-aiter](tuning-aiter/SKILL.md) | racing backends on your shapes and recording the winner in a CSV row |

Two divergences between gfx942 and gfx950 need explicit care: the FP8 dialect (gfx942 computes FNUZ, gfx950 OCP;
same 8-bit layout, different bias, so a mismatch corrupts numerics silently) and gfx950-only features (MXFP4/MXFP8,
scaled MFMA, the Gluon `cdna4` dialect), which have no gfx942 configuration to carry over.

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/expert_skills/tuning/README.md)
and cut to its kernel-level part; numbers keep their original dates and environments.
