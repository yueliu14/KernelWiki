# <Semantic chain>: <model or stack> <topic>

Save as `<semantic_chain>/cases/<model_or_stack>_<regime>[_<topic>].md` and index the case in its semantic-chain README. Replace placeholders and retain only relevant sections. Keep reusable contracts in the chain's topic pages and source-specific details here.

```yaml
semantic_chain: <parent directory name>
model: <model name>
phase: <prefill, decode, decode_target_verify, or both>
topics: [dataflow, scheduling, dispatch, memory, communication, measurement, fusion]
evidence: <source_inspection, source_inspection_and_local_reports, reported_measurement, reproduced_measurement, or hypothesis>
source_set: ../../_sources/<source_set>.yaml
```

Select the topics covered by the case. Cite source IDs/symbols for observations, label arithmetic as a derivation and label proposed changes as hypotheses. A useful case can document existing behavior without proposing a change or claiming a speedup.

## Question, scope and conclusion

What does this case establish about the complete subgraph? Identify its external inputs, outputs and state mutations. Record the model/image/backend revisions, hardware, phase, graph mode, TP/EP and shape/routing distribution, or link a shared source context.

## Dataflow and execution contract

Show operations and dependencies, including branches/joins and collective boundaries. Distinguish the semantic computation from physical launches.

| Tensor / state | Representation | Producer | Consumers / last use | Ownership and numerical constraints |
| --- | --- | --- | --- | --- |
| <name> | <shape, layout, dtype, scales> | <stage> | <stages> | <rank, aliasing, masks, rounding> |

Record any reused representation, intermediate storage, initialization requirements and externally visible mutations. Label logical allocation/traffic calculations separately from measured memory traffic.

## Scheduling, dispatch and communication

Record ordering and stream/event dependencies, independent work and resource contention. State whether overlap is implemented, observed or proposed. Include graph address/lifetime requirements for scratch reuse.

Separate implementation eligibility, caller dispatch and observed coverage. List fallback paths and their performance weight. State the output's rank scope and where downstream completion/collectives are required; cite the caller before claiming an implemented communication schedule.

## Evidence or proposed experiment

For source-only cases, state what was inspected and what remains unknown. For an experiment, identify exact baseline/candidate revisions, the variable changed, expected effect, semantic checks and cases likely to regress. Launch counts may remain unchanged.

Measure the same complete subgraph boundary under comparable inputs, layouts, routing, graph mode and cache policy. Report correctness (including state mutations), latency/noise, workload weights, scratch footprint and relevant stage/communication timing. A single-kernel speedup or a logical byte reduction does not establish a subgraph or serving improvement.

## Transferable lesson and limits

State the reusable conclusion, conditions under which it applies and unresolved questions. Link the relevant topic pages, constituent operators and source/measurement artifacts. For fusion-specific comparisons, also use the [fusion supplement](fusion_case.md).
