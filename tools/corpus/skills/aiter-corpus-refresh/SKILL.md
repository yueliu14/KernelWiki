---
name: aiter-corpus-refresh
description: Refresh KernelWiki corpus from new ROCm/aiter kernels, wrappers, configurations and tests. Use for manual or recurring source-knowledge maintenance, including discovery of unscoped inference kernels and citation-safe updates.
---

# AITER corpus refresh

Update the knowledge, not merely the source pin. Produce reviewed operator-local decisions and
reproducible evidence for a fixed AITER commit. Existing private-run experience remains separate from
source observations; a new upstream implementation is not evidence of a performance win.

## Establish the working set

Locate the KernelWiki checkout and the AITER cache from the request or workspace. Read the checkout's
`tools/corpus/README.md`, `knowledge/README.md` and `knowledge/operator/catalog.yaml`. The repository's
current schema and commands are authoritative. This skill supports the operator-local layouts in
catalog schemas 2 and 3; do not upgrade a schema merely to perform a refresh.

Use a clean branch/worktree for edits, preserving unrelated work. For recurring runs, first look for
an existing refresh branch or draft PR and resume it rather than duplicating it. Keep the scan report,
review backlog and checkpoint outside the public repository.

Fetch the official `https://github.com/ROCm/aiter.git` default branch into an AITER cache. Verify the
cache's origin; if it belongs to another project, use a separate clone. Discover the default branch
with `git ls-remote --symref <upstream-url> HEAD` instead of assuming `main`. A partial clone with
`--filter=blob:none --no-checkout` is sufficient. Resolve the fetched target to a full commit once;
use that commit throughout the pass. Fetch any older catalog commits required for comparison/audit.
If the cache is shallow, deepen the relevant history until the comparison ancestry is known;
having two commit objects alone does not prove their relationship. Do not execute upstream code
during source inventory.

## Discover and prioritize

Run the helper relative to this skill directory, substituting the actual checkout paths and pinned
commit. It requires Git, Python >= 3.10 and PyYAML:

```bash
python3 <skill-dir>/scripts/plan_refresh.py \
  --kernelwiki /path/to/KernelWiki --aiter /path/to/aiter \
  --target <full-target-commit> --since <last-reviewed-commit> \
  --output /private/scratch/aiter-refresh.json
```

On the first pass, omit `--since` and add `--full-inventory`: a catalog snapshot is not proof that
all kernels at that snapshot were already reviewed. Later passes use the private checkpoint. The
helper reads git objects only and never fetches, changes scopes, regenerates evidence or executes
kernels. If required objects are absent, fetch them and retry; do not silently compare working files.

Review `changes`, `pinned_differences` and, when requested, `unscoped_inventory`. On a first pass
(no `--since`), or any `--full-inventory` pass, the helper also compares every existing binding with
the target. This includes older snapshots whose changes predate the newest catalog snapshot.
Differences are review candidates: retain historical evidence deliberately, rather than assuming
every old binding must migrate. Later incremental passes use the completed private checkpoint;
historical bindings already reviewed and retained do not make every invocation repeat the same work.
The report links changed files to
source owners, test/configuration owners, cited cards and shared/dependency consumers. It detects
exact renames; edited renames can appear as deletion plus addition. Family and training-name hints
are triage aids, not classifications to accept blindly. Review unrecognized paths and shared helpers
as well as obvious kernel names.
Inventory counts include per-shape configuration files and binary objects, not distinct kernels.
Group related files by implementation/caller before deciding whether they contribute new knowledge.

Prioritize semantic changes and correctness fixes, new inference implementations, dispatch/layout
changes, configuration coverage and tests. Cover the existing GEMM, attention, MoE and indexer
families and their variants. Read kernel bodies together with their callers, layouts, configuration
and tests. Exclude clearly training-only material after checking for shared forward use. Put small
supporting operations with their consuming workload; do not expand the operator taxonomy based on
filenames alone.

## Update canonical knowledge

Before changing source bindings or citations, read [refresh-policy.md](references/refresh-policy.md).
It describes the multi-snapshot ownership and reference migration that makes this update fragile.

- Add a fresh snapshot entry; never change the commit behind an existing snapshot name.
- Extend the canonical owner's scope for reviewed files, including relevant validation sources.
  Retain historical bindings needed by old source or configuration references.
- Extract the selected owners, then resolve new evidence IDs from the generated YAML. Update
  cross-operator citations and shared/dependency links deliberately; do not mass-rewrite prefixes.
- Update an existing decision when it answers the same question. Add a new decision only for a
  distinct, useful contract or choice. Capture applicability, action, rationale, alternatives and
  limits; cite the implementation that actually supports the statement.
- Keep historical experience scoped to its tested context. Source changes alone cannot promote it
  to internally validated, refresh its benchmark result or erase a contrary outcome.

When the checkout supports decision schema 6, also read its `tools/corpus/upstream-evidence.md`.
Trace the changed construct to its introducing/modifying commits and associated PRs; a corpus
snapshot commit is not necessarily the optimization commit. Use `tools/corpus/collect_pr.py` to
capture PR text, comments, reviews and commit/file inventories outside the public repository.
Review the diff and comparison context before linking a PR to a card's specific backend/axes.
Preserve merged versus closed/unmerged status, measured revisions versus PR metadata, separate
operator and end-to-end results, and contrary reports. Record absent effects as `not_reported`;
keep unresolved associations explicit. Do not distribute a bundle gain across individual axes.
Importing an upstream report never establishes reproduction. Preserve the source/result IDs and
capture timestamps of historically cited reports when adding new evidence.

If a file moves or an implementation disappears, follow the call path and explain the resulting
scope change. Old evidence remains true at its original commit; it must not masquerade as support
for the new implementation. Do not delete a card solely because an upstream file was renamed.

## Verify and deliver

From the KernelWiki root, use its existing tools:

```bash
python3 tools/corpus/extract.py --aiter /path/to/aiter --operator <owner> --emit
python3 tools/corpus/seeds.py --aiter /path/to/aiter --operator <owner> --emit
python3 tools/corpus/build.py --emit
python3 tools/corpus/validate.py
python3 tools/corpus/extract.py --aiter /path/to/aiter --check
python3 tools/corpus/seeds.py --aiter /path/to/aiter --check
python3 tools/corpus/validate.py --verify-source /path/to/aiter
python3 tools/corpus/build.py --check
python3 -m pytest -q
git diff --check
```

Repeat extraction for each affected owner, or omit `--operator` when refreshing all scopes. Run
`seeds.py --emit` for changed tuning scopes; it rebuilds groups, file inventory and gaps from their
pinned inputs. On older checkouts without this command, use that checkout's maintenance procedure
and report any missing configuration-value audit. Include
borrowed references and configuration/test paths in the citation audit. Check changed local links,
new source licenses and the final diff; stage new generated files before the final whitespace check.
These checks do not run upstream correctness tests or GPU benchmarks, and must not be reported as
such. Resolve failures introduced by the refresh. If missing objects or upstream/schema ambiguity
prevents a valid result, retain the unfinished work and report the concrete blocker without advancing
the checkpoint.

For changed PR evidence, also check its quotations against the saved API capture with
`collect_pr.py --check-evidence <report.yaml> --capture <private-capture.json>` and run
`validate.py --verify-upstream /path/to/aiter`. The latter audits actual change-parent links and
before/after code excerpts, not benchmark results. Update confidence only from reviewed evidence.

Deliver the reviewed change on a branch and, when repository publication is authorized, update or
create a draft PR. Follow session authorization for any merge; refreshing knowledge does not itself
merge a PR or enable a scheduled job. Report the AITER commit/range, operator coverage, changed
conclusions, unresolved gaps, validation and PR. If review yields no knowledge change, report a no-op
without manufacturing a pin-only PR.

## Recurring invocation

A scheduler can invoke this skill with a stable KernelWiki path, AITER cache and private checkpoint.
For example: “Use $aiter-corpus-refresh to review new AITER inference changes for KernelWiki, update
its corpus, validate citations, and prepare a draft PR. Resume any pending refresh; keep run histories
and scan reports private.” The skill itself installs no timer.

Advance `reviewed_through` only after the whole interval has been reviewed and the accepted changes
are present in the branch that the next pass will read. For an unmerged PR, keep its branch/commit/URL
as pending work and resume it on the next invocation; do not skip its interval when reading `main`.
Record remaining candidates as a private backlog. A partial review or failing validation is not a
completed checkpoint. Repeated runs against the same reviewed state should be no-ops.

Track PR evidence review separately from `reviewed_through`: comments, benchmark tables and PR
lifecycle can change without a new source commit. Revisit pending or updated reports, preserving
their older cited excerpts/results. A failed API capture is not proof that no report exists and
must not advance that PR's evidence checkpoint.
