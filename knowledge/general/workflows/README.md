---
title: Workflows — end-to-end optimization playbooks (AMD MI-series)
kind: workflow
gens: [gfx942, gfx950]
status: sota
updated: 2026-06-08
sources:
  - GEAK/e2e_workflow/roles/op_benchmarker.md
  - GEAK/e2e_workflow/roles/e2e_integrator.md
  - GEAK/perf_knowledge/index/decision_trees.md
---

# Workflows

Reusable procedures for authoring, tuning, profiling and integrating kernels. They connect
[operator knowledge](../../operator/README.md) with backend, library and measurement guidance.
GEMM tuning and attention backend selection belong here: the procedure can be reused across
models and variants. Concrete configurations, implementation constraints and measured outcomes
belong with the relevant operator.

These playbooks were imported from GEAK. Their source dates, command-line interfaces and historical
measurements remain attached; check the installed stack before applying them.

## The two altitudes
- **Single kernel** — make one operator (a GEMM, an attention, a norm) faster in
  isolation, gated by an immutable unittest. → [`optimize_single_kernel.md`](optimize_single_kernel.md)
- **End-to-end model** — raise serving throughput of a whole model on a live
  sglang/vLLM server, gated by a measured e2e delta. → [`optimize_e2e_model.md`](optimize_e2e_model.md)

The e2e layer **recursively calls** the single-kernel layer: it extracts a hot kernel
into an immutable task dir, hands it down, and overlays the verified result back. So the
single-kernel playbook is the inner loop of the e2e playbook.

## Index

| Workflow | Use when | Gate |
|---|---|---|
| [`optimize_single_kernel.md`](optimize_single_kernel.md) | One operator, isolated | unittest correctness + isolated ms |
| [`optimize_e2e_model.md`](optimize_e2e_model.md) | Whole model serving throughput | e2e Δ > 0.5%, parity holds |
| [`gemm_tuning_workflow.md`](gemm_tuning_workflow.md) | Dense bf16/fp16 GEMM dominates the profile | aiter DB engages (`is tuned on cu_num`>0) + e2e Δ |
| [`attention_backend_selection.md`](attention_backend_selection.md) | Pick prefill/decode attention backend | isolated ms + e2e flag swap |
| [`integrating_a_new_kernel.md`](integrating_a_new_kernel.md) | A faster kernel exists, server doesn't use it | engagement proof on live path |
| [`authoring_a_kernel_with_geak.md`](authoring_a_kernel_with_geak.md) | No editable backend impl exists | unittest passes + beats `best_known_ms` |
| [`choosing_a_backend.md`](choosing_a_backend.md) | "Which backend for operator family X?" | per-family decision table |
| [`model_bringup_checklist.md`](model_bringup_checklist.md) | New model, first time on MI300X/MI350X | model runs + each lever banked |
| [AITER corpus refresh](../../../tools/corpus/skills/aiter-corpus-refresh/SKILL.md) | Maintain knowledge as upstream inference kernels change | pinned source audit + corpus validation |

## How to read a workflow
Each file: **TL;DR → preconditions → numbered steps (with the exact env/flags/commands)
→ verification gate → pitfalls → cross-links → Sources**. Numbers are only quoted when
they come from a cited source or a validated run; never invented. The performance-number
format is the one in [`../index/conventions.md`](../reference/geak_index/conventions.md).

## Shared checks and historical run conventions
1. **Amdahl first** — `pct_gpu_time × achievable_speedup` decides what to touch. A 5× on
   a 2%-of-time kernel is invisible. See [`optimize_e2e_model.md`](optimize_e2e_model.md).
2. **Engagement proof, not inference** — prove the change is on the live path
   (`grep -c 'is tuned on cu_num'` > 0, or a load banner). A throughput wiggle is not
   proof. This is the **TunableOp lesson**.
3. **Measure the noise floor** — the imported GEAK runs used a 0.5% band and
   `cand_min > ref_max`. Treat these as the recorded protocol, not universal significance thresholds.
   Choose and report a protocol appropriate to the actual workload and variability.
4. **Match the deployment topology** — TP=1 was the convention for the cited single-GPU runs.
   Distributed workloads need their actual TP/EP configuration, communication and synchronization
   in the validation boundary.
5. **Immutable oracle** — the unittest + `reference_io.pt` are never edited
   (anti-cheating); re-hash before trusting any result.

## Sources
- e2e doctrine, Amdahl gate, three verdicts: `GEAK/e2e_workflow/roles/e2e_integrator.md`, `GEAK/e2e_workflow/README.md`.
- The cheapest-first ladder: `GEAK/e2e_workflow/roles/op_benchmarker.md`.
- Routing priors: [`../index/decision_trees.md`](../reference/geak_index/decision_trees.md).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/workflows/README.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
