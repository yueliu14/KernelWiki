# Corpus production and validation

Knowledge lives in `knowledge/operator/<operator>/corpus/`. The root
[`catalog`](../../corpus/catalog.yaml) declares operator directories, aliases, workload groups and
source snapshots. It is schema version 2; the former family-level file layout is retired.

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
