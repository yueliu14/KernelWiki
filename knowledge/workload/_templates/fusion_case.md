# <Semantic chain>: <model or stack> <regime>

Save as `<semantic_chain>/cases/<model_or_stack>_<regime>.md`. Replace the placeholders and remove guidance that does not apply. Link the completed case from its semantic-chain README. Keep reusable contracts in that README and model-specific details here.

Add front matter to the completed case:

```yaml
semantic_chain: <parent directory name>
model: <model name>
phase: <prefill, decode, decode_target_verify, or both>
evidence: <source_inspection, source_inspection_and_local_reports, reported_measurement, reproduced_measurement, or hypothesis>
source_set: ../../_sources/<source_set>.yaml
```

## Context and evidence

- Model and revision; engine/backend source commits or image digest:
- Hardware; TP/EP; prefill, decode or speculative verification; graph/eager mode:
- Shapes, dtypes/layouts, routing/context distribution and branch coverage:
- Evidence: source inspection / reported measurement / reproduced measurement / hypothesis.
- Source entry points and artifact links; digest for local-only evidence:

## Semantic chain and implemented boundaries

Show the operations in execution order and name their input/output tensors. Mark a decomposed reference as illustrative unless it is an actual measured baseline.

```text
Semantic chain: <producer> -> <transform> -> <consumer>
Implementation: [launch A: ...] -> <materialized tensor> -> [launch B: ...]
```

| Caller / branch | Device launches | Intermediates retained | Boundaries removed |
| --- | --- | --- | --- |
| <symbol, source, dispatch predicate> | <symbols in execution order> | <shape, dtype, producer, consumers> | <specific operations or materializations> |

Explain the fusion mechanism and why the remaining boundary exists. Include companion kernels even if they fall below a profiling selection threshold. Keep separate call sites for a shared kernel identity.

## Applicability and correctness

Record dispatch predicates and fallback paths, reduction/rounding points, mutation/aliasing, cache indices, padding, scale semantics, collective placement and synchronization. State the oracle and tolerance, plus unsupported or untested shapes, values and execution modes.

## Outcome

If only source has been inspected, say so and leave performance unclaimed. Otherwise record:

| Comparison | Complete-chain latency | Launches | Intermediate bytes / measured traffic | Correctness | Evidence |
| --- | --- | --- | --- | --- | --- |
| Baseline revision | <timing and protocol> | <observed> | <label logical vs measured> | <result> | <artifact> |
| Candidate revision | <same protocol> | <observed> | <same accounting> | <result> | <artifact> |

Record repetitions/noise, cache policy, weights and speedup aggregation. Separate fusion ablation from tuning an existing fused implementation. Report end-to-end evidence independently, including engagement, parity, workload coverage and remaining bottlenecks. Explain rejected approaches and unresolved limitations.

## Transferable lesson and related operators

State what can be reused, the conditions under which it applies and what remains a hypothesis. Link participating operator pages, relevant general methods and exact source/evidence records.
