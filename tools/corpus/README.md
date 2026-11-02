# Corpus production and validation

Knowledge lives in `knowledge/operator/<operator>/corpus/`. The
[`catalog`](../../knowledge/operator/catalog.yaml) declares operator directories, aliases, workload
groups and source snapshots; the generated [corpus index](../../knowledge/operator/CORPUS.md)
lives beside it. The catalog uses schema version 3 and repository-relative paths. Reusable method
cards live in `knowledge/general/corpus/`; both use decision schema version 5.

## Commands

```bash
python3 tools/corpus/build.py --emit
python3 tools/corpus/build.py --check
python3 tools/corpus/validate.py
python3 tools/corpus/validate.py --verify-source /path/to/aiter
python3 tools/corpus/extract.py --operator attention_indexer --aiter /path/to/aiter --check
python3 tools/corpus/extract.py --operator attention_indexer --aiter /path/to/aiter --emit
python3 -m pytest tests/corpus -q
```

For manual or recurring upstream maintenance, use the
[AITER corpus refresh skill](skills/aiter-corpus-refresh/SKILL.md). Its portable read-only planner
finds new unscoped kernels and maps source/test/configuration changes to affected corpus consumers;
the skill then guides source review, citation migration and a validated knowledge update.

The build validates the whole reference graph and renders all operators by default. `--operator`
or `--family` limits the selected pages or source audit. Source extraction reads the exact git
objects declared by the scope: the checkout must hold those revisions, but its HEAD and working
files need not match them. No GPU, GEAK checkout or live workflow is required.

Extraction writes source evidence only. Curated cards, axes, upstream-test references and the
preserved shipped configuration inventories are separate inputs. Source refresh is not a benchmark
run or a re-tuning of those inventories. Private underscore modules contain shared parsing,
normalization and schema helpers; the three scripts above are the maintenance entry points.

## Ownership and references

- `scope.yaml` owns explicit `(snapshot, file)` bindings, their backend context, optional Python
  function extents, named interface anchors and upstream correctness-test paths. A file at one
  revision has one extraction owner; other operators reference its observations.
- `source_evidence.yaml` stores the owner's observations, coverage and pinned extraction gaps.
  `symbol_declaration` records a function extent; its content ID binds the excerpt, not the whole
  function body. `interface_parameter` records literal parameter occurrences, not their semantics.
- `decisions.yaml` stores complete owned cards and a `shared_cards` list of canonical card IDs.
  A shared reference never duplicates a card body. Each referenced card must include the importing
  operator in its applicability list; its other conditions still need review.
  A decision can also carry scoped `experience` summaries. Keep one decision per question and
  add supporting or contrary experience to it instead of creating a second learned-card collection.
- Source and seed references are snapshot-qualified, for example
  `aiter_b3ad6c46:src_0123456789abcdef`. This preserves the original content-ID suffix and keeps
  identical observations at different revisions distinguishable. Owners may move without changing
  reference identity. Unqualified or unresolved references fail validation.
- `performance_axes.yaml` is the local model of that operator's questions and dependencies.
  Dependencies can cite canonical cards owned elsewhere. Shared cards retain their owner's model.
- `tuned_configs.yaml`, where present, owns upstream configuration groups and backend selections.
  They retain source revisions and configuration IDs, exclude timing columns, and generate a local
  `seeds.md`. Counts are source selections, not speedups or a ranking of kernels.

## Adding or extending an operator

1. Add the operator overview and its catalog directory/group entry. Existing operators should be
   extended in place; an empty corpus directory is not coverage.
2. Pin an upstream snapshot in the catalog and put the relevant files in the operator's scope.
   Check whether another operator already owns each file/revision before adding it.
3. Extract from the declared git objects. Inspect the actual implementations before writing a card:
   each card needs conditions, actions, rationale, alternatives, limitations and qualified evidence.
   Add cross-operator `shared_cards` references when an existing card supplies the same knowledge.
4. Define the operator's axes and cross-axis dependencies. Keep layout, masks, scale formats,
   routing weights, state updates and communication ownership as explicit contracts. Source presence
   and caller eligibility do not prove hardware support or performance.
5. Add pinned upstream test references, regenerate the reading pages, run the corpus tests and audit
   the source citations. Commit inputs and generated pages together. Update the snapshot prefix and
   review affected references when moving upstream revisions.

`build.py --check` verifies the published pages. `validate.py --verify-source` additionally checks
quoted lines, source extents, upstream-test paths and seed-source paths through `git show`. Offline
CI validates IDs, ownership, schemas, dependencies, references, links and generated artifacts.
No knowledge-production command executes an upstream correctness test or records GPU timings.

## Updating conclusions from private runs

Run histories, traces, candidates, raw measurements and their internal locations stay in a private
store. Publish a reviewed conclusion in the relevant decision's `experience` list. Each entry needs
an outcome (`beneficial`, `no_gain`, `regressed`, `mixed` or `constraint`), explicit conditions,
a concise summary and provenance. Keep a private mapping from the public card ID to its run artifacts.

Imported summaries use `kind: historical_summary`, pinned public summary references,
`artifacts: not_published` and `verification: not_reproduced`. These links attribute already-public
historical explanations; they do not make the original runs independently reproducible. The migration
publishes qualitative findings rather than incomplete benchmark rankings or isolated leaderboard numbers.

New results may use `kind: private_summary`, `artifacts: not_published` and
`verification: internally_validated`; public references may be empty. They also require a public
`context` with `gfx`, `software`, `dtype`, `shape_layout`, `baseline`, `protocol` and `correctness`.
Internal validation is an attribution claim, not public reproduction. Do not publish a result that
cannot disclose enough context to bound its applicability.

Source observations stay content-bound and separate from experience. A new card without source
observations uses `evidence_level: reported_experience`; existing `source_observed` cards keep their
source references when experience is added. All remain `candidate_only` for a new target workload.
The `measurement_evidence` field remains empty: raw runs are not embedded into the public schema.

Do not turn a run-specific negative into a permanent rule. Keep contradictory outcomes with their
conditions, revisit them when the implementation changes, and attribute a combined improvement to the
combination unless independent ablations establish component effects. Preserve the accepted numerical
contract; a diagnostic control that omits work is not a valid implementation.
