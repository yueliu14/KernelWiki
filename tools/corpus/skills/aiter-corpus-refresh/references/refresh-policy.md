# Safe source and decision refresh

Read this before applying the scan report. Use the target checkout's maintenance guide and validators
for the exact schema; the rules below explain what a successful refresh must preserve.

## Canonical owners and immutable snapshots

`knowledge/operator/catalog.yaml` maps operators to local corpus directories and snapshot names to
full upstream commits. Source bindings live in each operator's `corpus/scope.yaml`; a `(snapshot,
file)` pair has one owner. Shared cards contain references to their owner, not copied bodies. On
schema 3, reusable methods also have a general corpus. Keep general tuning/authoring skills there;
an AITER inventory update does not relocate or rewrite those skills.

Choose a new snapshot name such as `aiter_<unique-commit-prefix>` and add the full commit and source
repository. If the prefix collides, lengthen it. An existing name is immutable. Old cards and shipped
configuration groups may deliberately use an older snapshot even after a new implementation is added.

For each reviewed file, find any existing target-snapshot owner before adding a binding. Reuse it.
When a shared helper affects multiple operators, extract it once and reference it from the consumers.
Read actual decorators/implementation language: an ASM launcher is not an inspected binary kernel,
OPUS is not automatically the authoring language, and mixed Python files can contain Triton and
FlyDSL kernels in the same module.

## Add evidence before retiring references

A refresh can retain an old binding and add a new binding for the same path at a different snapshot.
This is useful when a historical decision or shipped seed still needs the old evidence. For renamed
files, retain the old path at its old snapshot until all references have been accounted for.

A practical order:

1. Record the target snapshot, add reviewed scope bindings and pinned test sources. Keep currently
   referenced old evidence while preparing the replacement.
2. Run `extract.py --emit` for the owners. It reads the declared git objects and writes source
   evidence only; it does not curate decisions or refresh tuning inventories.
3. Read the new observations, symbol extents and actual source bodies. Resolve IDs from the new YAML.
   A source ID binds the recorded observation/excerpt, not every statement in the whole function.
4. Review every affected card, its solutions and shared/dependency consumers. Update citations only
   when the new observation supports that claim. A filename, same symbol or similar hash is not
   enough to equate two observations.
5. Regenerate reading pages and validate the entire reference graph. Retire unused old evidence only
   after checking all source, solution, shared-card, dependency and seed references. Historical
   observations that still support scoped conclusions can remain indefinitely.

Do not replace `aiter_old:src_…` with `aiter_new:src_…` by string substitution. Line numbers, excerpts,
symbol extents, backend classification and file paths contribute to identity. Even an unchanged ID
suffix at the new revision must come from extraction, not an assumption.

## What a new kernel contributes

A useful decision states a question and a conditioned conclusion. Review at least the evidence
needed for that question: caller eligibility, shapes/strides, data and scale formats, mask/selection
semantics, buffer/state ownership, reduction order, synchronization and test coverage where relevant.
Examples of substantive updates include:

- a new MLA/paged-cache layout or a change to split/reduce validity;
- an MoE preparation path that changes expert-row padding, scale ownership or routed-weight placement;
- GEMM dispatch selecting a new instance only for a restricted dtype/layout;
- an indexer implementation changing score precision, top-k bounds or distributed merge ownership.

The source can establish an implementation precedent or a constraint. It cannot by itself establish
an isolated speedup, deployment engagement, complete architecture support or end-to-end superiority.
New test files are correctness references, not executed test results. Binary-only additions can be
recorded as inventory/launcher contracts with their visibility limit.

When a current conclusion no longer applies, narrow or revise its conditions while retaining the
historical scope of the old evidence. Keep one canonical decision for the same question when the
alternatives can be expressed clearly. Separate decisions only when contracts or questions differ.
Do not add a card for every regex observation or every new file.

## Shipped configurations and performance experience

The source extractor does **not** rebuild `tuned_configs.yaml`. If configuration files changed,
inspect the existing parsers in `tools/corpus/_extract_impl_facts.py` and the local seed schema before
regenerating reviewed groups. Preserve source filenames, grouping/ID rules, inventory and stated
gaps. Do not copy a new snapshot prefix onto an old group or overwrite a historically cited group.
Configuration counts and selected backend rows are starting points; do not import timing columns as
source evidence or call a shipped selection a measured winner.

With schema 3 / decision schema 5, cards can contain `experience`. Preserve provenance and comparison
context. Imported historical summaries stay `not_reproduced`; an upstream code diff cannot turn them
into internal validation. New private validation summaries require the public context accepted by
`tools/corpus/experience.py`; private raw runs and their locations remain outside KernelWiki. Do not
invent such a summary during a source-only refresh. For older schemas, follow the available contract
and retain existing experience rather than silently converting the repository to a newer schema.

If the source suggests a promising optimization, express it as a candidate with explicit conditions
and validation needs. A contrary old result is not disproven merely because the technique now appears
upstream. Re-test triggers include a changed dtype, compiler, tile, scale contract or limiting resource.

## Review coverage and checkpointing

The incremental report uses `--since` to enumerate all changed paths, including unscoped additions,
deletions, exact renames, helpers, tests and configuration files. On a first review, `--full-inventory`
also exposes candidates already present at the latest catalog snapshot. Lexical family/training hints
never justify silently excluding an otherwise relevant forward implementation. For edited renames,
compare the paired additions/deletions and callers manually.

Keep the target commit, completed interval, excluded/deferred candidates and pending branch/PR in a
private state file. A no-change outcome can advance the reviewed interval after all candidates have
been triaged. A partial review keeps a backlog and the old completed checkpoint. If a pending PR is
closed or its branch disappears, verify which changes reached the next checkout before advancing.

The public PR should explain knowledge changes and coverage limits, not attach the raw scan report
or a run diary. Pinned public source citations and required upstream license notices belong in the
repository; private paths, traces and benchmark artifacts do not.
