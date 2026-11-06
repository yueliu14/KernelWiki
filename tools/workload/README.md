# Workload knowledge maintenance

Workloads own composed computation and execution contracts; operators own constituent kernels.
`knowledge/workload/catalog.yaml` indexes semantic chains and canonical cases. Each case remains a
reviewed Markdown document. A chain with reusable decisions owns `corpus/decisions.yaml` and its
generated reading page; `knowledge/workload/CORPUS.md` indexes every registered case and decision.

## Build and verify

Python 3.10+ and the repository's PyYAML dependency are sufficient; no GPU stack is imported.

```bash
python3 tools/workload/build.py --emit
python3 tools/workload/build.py --check
python3 tools/workload/build.py --verify-source /path/to/sgl-amd-mono
python3 -m pytest -q
```

CI runs both the operator and workload generated-page checks and the repository tests. The optional
Git audit reads the full commits and file/line references through `git show`; it never checks out,
imports or executes upstream kernels. It matches sources to the checkout's official HTTPS origin and
reports sources in other repositories that were not audited. Repeat against each repository needed
by an expanded corpus. This location audit does not prove prose interpretations, benchmark results
or runtime engagement.

## Catalog schema 1

- `source_sets` maps stable source-set IDs to `_sources/<id>.yaml`. A new reviewed snapshot should get
  a new ID; do not repoint historically cited references to a new commit.
- `chains` declares semantic owners, directories and constituent operator IDs. Existing chains can
  have cases without a decision corpus.
- `cases` declares ID, owner, path, model, phases, actual implementation backends, execution forms,
  a scoped launch summary and qualified `source_refs` (`source_set:local_id`).
- Phases are `prefill`, `decode`, `target_verify`, `draft_decode`, `draft_extend`. Legacy case front
  matter `decode_target_verify` normalizes to `target_verify`. The model, owner, phases and source
  set must agree with the case Markdown. An empty backend list means unspecified, not a new backend.
- Forms are nonexclusive `fusion`, `staged`, `persistent`, `megakernel`, `monokernel`. These labels do
  not establish launch counts. Record conditional branches and excluded work in the launch summary
  and the case's actual execution graph.

Every `*/cases/*.md` belongs in the catalog; duplicate paths, unknown owners/operators/sources and
unregistered decision corpora fail validation. Generated files are reproducible from canonical inputs.
Fully public source sets also get a generated `_sources/<id>.md` reading page. Existing private source
context pages stay manually curated and are not overwritten.

## Decision schema 1

Use the checked-in decoder-layer, post-attention and decoder-stack records as examples. Each decision
contains a stable ID, question, axes, `candidate_only` status, conditions, actions, rationale,
alternatives, case/source references, effect, assessment, validation plan and limitations. It has one
semantic owner and must cite a case in that owner's chain. Sources must be among the cited cases'
registered evidence. Cross-chain cases can support the same decision without duplicating its body.

This initial schema is deliberately **source-only**:

- `effect.outcome` is `not_assessed`, with a scoped explanation. It means the pass has not evaluated a
  comparable measurement; it does not claim no upstream report exists or no gain is possible.
- `assessment.implementation` grades the implementation association, with reasons and evidence gaps.
- `assessment.effect` and `assessment.attribution` stay `unknown`, with explicit missing evidence.
- Grades are ordinal curator judgments, not calibrated probabilities. High implementation confidence
  does not establish target-model engagement, correctness or a speedup.

Measured effects need a deliberate schema extension with baseline/candidate revisions, environment,
workload, timing/statistic, correctness and source provenance. Do not insert a speedup into this
source-only schema or assign a whole-layer gain to each component. Existing AITER PR reports keep their
own canonical schema; this addition does not reclassify sgl-amd-mono as AITER.

## Reviewing a new implementation

Read the model caller, dispatch/spec, actual device body, scratch/state layout, synchronization and
relevant test definitions at one fixed public revision. Distinguish documentation claims from code
observations. State installed dependency revisions as unknown when the review does not establish them.
Keep device launch count separate from kernel stage calls, CUDA graph submissions and surrounding
framework operations. Draw the semantic graph separately from physical launches and branch conditions.

Review the next consumer of residuals, KV and speculative state. For a persistent implementation,
record residency/progress assumptions and actual epoch/memory-scope rules. A one-launch implementation
may still materialize many global scratch values. Use operator links for instruction/tile details;
use workload decisions for cross-stage choices. Put new model/stack cases under a matching semantic
boundary rather than creating a category from a kernel nickname.

Public source indexes retain their existing schema 1 availability/verification contract;
`kind: documentation` identifies upstream guides separately from executable source. Test references identify
validation routes, not successful test runs. Raw profiles, private experiments and artifact locators
remain in a private ledger. Update third-party attribution when adding another source provider.
