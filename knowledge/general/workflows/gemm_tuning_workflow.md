---
title: GEMM tuning through the active dispatch path
kind: workflow
operator: dense_gemm
backend: aiter
status: reference
updated: 2026-11-02
---

# GEMM tuning workflow

Tune the configuration that the measured caller actually consumes. When dense GEMM dispatches
through `aiter.tuned_gemm.gemm_a16w16`, AITER's per-shape database is one tuning surface. An
explicit backend replacement or a change to the active kernel can also be a candidate. Confirm the
runtime path before choosing a tool: PyTorch TunableOp settings cannot change a call that bypasses
PyTorch BLAS dispatch, and a different caller may use another selector entirely.

The [dense GEMM corpus](../../operator/dense_gemm/corpus/decisions.md#gemm-shape-regime-routing)
maintains conditioned dispatch conclusions and reported experience. The
[AITER library reference](../libraries/aiter/tuned_gemm.md) describes a pinned implementation.

## Preconditions

Identify the GPU/CU count, software revisions, phase, shapes, layouts, dtypes, bias/scales,
quantization and graph mode. Inspect the caller and selected backend. Use the
[paired-measurement method](../corpus/decisions.md#method-paired-measurement-controls) to establish
what improvement can be resolved within the experiment budget.

For quantized or grouped GEMM, locate that operation's tuner and database: a dense bf16 recipe does
not establish the interface of an FP8, MXFP4 or MoE tuner. See the
[AITER tuning procedure](../expert_recipes/tuning/tuning-aiter/SKILL.md).

## 1. Capture the real lookup inputs

On a revision supporting them, `AITER_TUNE_GEMM=1` records untuned shapes and
`AITER_LOG_TUNED_CONFIG=1` exposes database hits. Check the installed source for output paths and
capture semantics before changing configuration. Preserve any existing capture data and keep the
new workload's capture in private scratch storage.

Read the lookup key from the actual wrapper. At the referenced AITER snapshot it includes CU count,
padded M, N, K, bias, input/output dtype, scale flags and preshuffle state. Capture each field from
the live path; do not substitute a guessed bias flag or treat a single observed shape as coverage
of all prefill/decode branches. Observe every relevant worker/rank.

## 2. Generate and race candidates

Use the pinned tuner's supported CLI and the GPUs allocated to this experiment. Respect other
users and concurrent measurements; increasing process count can add contention and timing variance.
Start with shapes whose runtime share can justify the tuning cost, then expand coverage deliberately.

The tuner may race library solutions or backend implementations and write results incrementally.
Uncovered shapes can retain their existing fallback, but a partial database can still regress
covered shapes. A microbenchmark selection must be rechecked under the actual workload, cache
state and graph mode. Preserve the current accepted implementation as the control.

Treat the tuner's error threshold as a filter on its tested inputs. It does not establish full
model accuracy, tail coverage, state/aliasing correctness or the numerical contract of a different
kernel. Use the workload's accepted correctness criteria.

## 3. Deploy a reversible candidate and verify engagement

If supported by the installed revision, point `AITER_CONFIG_GEMM_BF16` at the private candidate CSV.
Inspect merge order and cache behavior before overlaying it with other configuration files. Kernel
names and solution indices depend on the exact library/compiler build; do not copy them across
revisions without regenerating or validating the selection.

Restart processes and recapture graphs when they retain the old configuration. Invalidate only the
relevant derived caches. Verify the expected selected implementation and its complete parameters at
the intended call sites, as well as the expected fallbacks for untouched cases. A positive hit count
alone does not prove full coverage or that the wrapper honored every configuration field.

## 4. Gate correctness and end-to-end behavior

Compare the candidate with the current accepted configuration using interleaved baseline/candidate
runs and comparable warmup. Include correctness checks and, where applicable, the serving accuracy
gate. Derive an acceptance threshold from the target environment's uncertainty; a historical fixed
percentage or non-overlapping extrema is not a universal test.

Measure the full intended workload as well as important cases and fallback branches. Retain the
candidate only if it passes the correctness and performance criteria for that workload. Re-profile
after an accepted change because the dominant bottleneck and the value of the next change may move.
Combined improvements need separate ablations before assigning a gain to each component.

## 5. Update knowledge

Keep captures, tuned CSVs, logs and run histories private. Update the relevant operator's corpus
with the reusable conclusion, applicability, contrary outcomes and scoped validation summary. Put
cross-operator measurement or integration lessons in the general corpus. See the
[contribution contract](../../README.md#contributing-knowledge).

## Historical sources

The June 2026 Qwen3.5-27B reports motivated the dispatch-engagement and bias-capture lessons. Their
measured outcomes remain attributed in the [GEMM decision](../../operator/dense_gemm/corpus/decisions.md#gemm-shape-regime-routing).
The original [workflow](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/workflows/gemm_tuning_workflow.md)
and [run-specific recipe](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md)
are archived at their source revision. KernelWiki adapted the procedure and narrowed its claims;
this update adds no benchmark result or current-best certification.
