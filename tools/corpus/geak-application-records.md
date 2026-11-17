# Proposed GEAK knowledge application records

This is a follow-up design, not an implemented GEAK integration. The first step is
the offline [evidence inventory](statistics.md). This proposal connects knowledge use
to independent measurements without changing current knowledge ranking or curation.

## Existing integration points

Source review is pinned to GEAK `a86bbc90f2ed941a781ab982b35990b28069b396`:

- [`kernel_lane.js`](https://github.com/AMD-AGI/GEAK/blob/a86bbc90f2ed941a781ab982b35990b28069b396/kernel_workflow/kernel_lane.js#L3620)
  joins planned `learned_refs` to independently verified directions, records
  `cited_then_verified` against the frozen baseline and `became_winner`, and returns
  `learned_citations`. Its round history contains directions, verification and integration.
- [Citation filing](https://github.com/AMD-AGI/GEAK/blob/a86bbc90f2ed941a781ab982b35990b28069b396/kernel_workflow/kernel_lane.js#L3779)
  is already outside successful-run curation. Held-out runs do not vote on the KB.
  Preserve both behaviors; evaluation logging should still record held-out outcomes
  in an isolated, non-learning store.
- [`kb.py cite/drain`](https://github.com/AMD-AGI/GEAK/blob/a86bbc90f2ed941a781ab982b35990b28069b396/kernel_workflow/scripts/kb.py#L538)
  persists citation feedback and updates existing learned-card counters. These cards
  are not automatically KernelWiki canonical cards; they need explicit identity mapping.
- [`inv_record.py`](https://github.com/AMD-AGI/GEAK/blob/a86bbc90f2ed941a781ab982b35990b28069b396/kernel_workflow/scripts/inv_record.py#L1)
  already records invocation starts/completions, declared source-domain hashes, harness
  identity-file hashes and parsed per-case timing. It can join actual GPUs from the
  GPU-lock use-log. It explicitly does not establish comparability by itself.
- The lane's `llm_timeline` and `reports/trace/llm_calls.jsonl` provide existing cost
  accounting. Join those records instead of estimating tokens from summary text.

## Minimal first integration

Use each private `EVAL_DIR` as the run container. Add `knowledge_usage/run.json` and
an append-only `knowledge_usage/events.jsonl`, or immutable per-event files if several
processes write concurrently. Allocate a UUID for a new run; preserve it on resume and
record a separate resume segment. A kernel name is a task label, not a unique run ID.
Use a controller/locked writer or per-worker event files, never unprotected concurrent
JSONL writes. Each event has a stable `event_id` for idempotent replay.

The run manifest records `run_id`, `task_id`, `geak_commit`, `kb_commit`, experiment
arm, held-out flag, seed, budget, hardware/software identity and the frozen baseline.
Keep legacy card path/digest references namespaced until an explicit mapping to
`{repository, kb_commit, card_id}` exists. Do not infer identity from a title or filename.

Record events at four existing boundaries:

| Boundary | Record | Authoritative producer |
|---|---|---|
| Context assembled | Cards retrieved and cards actually included in the agent context, query/filter, KB revision, rank | Reader/context builder |
| Candidate planned/applied | Direction and attempt ID, parent attempt, cited cards, changed source digest, invocation refs, declared action/parameters | Planner declaration plus engineer/controller evidence |
| Verification completed | Build/correctness status, per-case baseline/candidate timing, full protocol/contract, metric uncertainty and evidence refs | Harness/verifier |
| Round integrated | Accepted/rejected result, incumbent before/after, integration measurements and combined card set | Integration controller |

Retrieval, exposure, planned use and observed application are different events. A
planner citation is a declaration, not proof that the edit implemented that technique.
Store `application_status: declared|observed|unresolved` with change/evidence refs;
only verified mappings support application-level statistics. Never infer application
from a net source hash change alone. Log every direction, including uncited, failed,
cancelled and incomplete directions. Flush events as they happen so an abort cannot
erase failed attempts before the end-of-run ledger is written.

Suggested application record (schema sketch; null means unavailable):

```yaml
schema_version: 1
event_id: <stable UUID>
run_id: <run UUID>
attempt_id: <direction/candidate UUID>
parent_attempt_id: null
round: 1
direction_id: <planner direction ID>
knowledge_refs:
  - repository: https://github.com/yueliu14/KernelWiki
    kb_commit: <full commit>
    card_id: <canonical card ID>
application_status: declared
action: {technique: split_k, parameters: {}}
invocation_refs: []
baseline_refs: {frozen: <source identity>, incumbent: <source identity>}
candidate_ref: <source identity>
context_ref: <hardware/software/cases/contracts/timing manifest>
state_before: {bottleneck: unknown, profile_ref: null}
state_after: {bottleneck: unknown, profile_ref: null}
verification: {build: pending, correctness: pending, measurement: pending}
comparisons: []
attribution: {scope: bundle, ablation_refs: []}
cost_refs: []
```

Each comparison needs `comparison_id`, `case_id`, `scope`, metric kind/unit,
baseline/candidate invocation refs, timing summaries, repetition count, uncertainty,
numerical validation and a versioned decision rule. Keep correctness, measurement
availability and performance outcome separate. Hardware/compiler failures do not
become zero-speedup observations. Profile-derived explanations are hypotheses unless
supported by counters or isolated experiments.

## Denominators and credit

Keep three comparisons separate:

1. Candidate versus frozen baseline: final implementation improvement.
2. Candidate versus the pre-action incumbent: incremental effect of this action/bundle.
3. Integrated workload versus its prior version: deployment/subgraph/end-to-end effect.

Existing `cited_then_verified > 1` only addresses the first. Existing `became_winner`
also reflects competition with other directions. A nonwinning direction may still
improve the incumbent, so adoption and beneficial effect must be separate fields.
Use same-context paired measurements; subtracting or dividing unrelated headline
geomeans is not an independent measurement of the action's contribution.

One direction applying several cards yields one bundle comparison linked to several
cards. Preserve the parent/action order; allocate component effects only when a valid
isolated comparison or ablation exists. Repeated timings are repetitions, not new
applications. Report attempt-level and independent-task-level statistics separately.
Do not infer agent improvement from observational card usage; selection is biased.

Initially write this ledger in observation mode. Keep `cite/drain` and ranking semantics
unchanged until identities, missing results and replay behavior have been audited.
Held-out measurements must remain available for evaluation but never update cards,
ranking counters or the learned memory used to solve the held-out tasks.

## Publication and pilot acceptance

An offline exporter can validate and join the private records, then prepare a reviewed
public evaluation summary with context, per-case metrics, coverage, negative results
and knowledge associations. Raw traces and private paths stay private. A later corpus
schema revision can add `experience` references to those canonical summaries; do not
populate today's deliberately empty `measurement_evidence` with private run dumps.

Pilot on GEMM, attention indexer and MoE. Before using any success-rate dashboard,
check that uncited/failed/incomplete directions appear; duplicate delivery and resume
do not double count; every numerical result resolves to exact source and comparison
identities; held-out runs never vote; and grouped cards retain bundle attribution.
Then run fixed-budget, paired multi-seed comparisons of no KB, frozen KB and updating
KB on separate held-out tasks. Measure both final latency and cost to a fixed target.
