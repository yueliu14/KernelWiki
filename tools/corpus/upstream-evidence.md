# PR effects and confidence

Decision schema 6 adds `upstream_review` to every canonical card and allows
`experience.provenance.kind: upstream_report`. Catalog schema 3, source evidence IDs,
historical experience and private validation retain their existing meanings.

PR reports live once in `knowledge/_sources/aiter/pull_requests/<number>.yaml` (report
schema 1). `build.py` produces adjacent reading pages and adds the relevant effects to
decision pages. A shared card links to its canonical owner; it does not duplicate a PR
or count its results twice. The first reviewed examples cover PRs #1909, #3231, #4538
and #4963. Other cards explicitly start as `unreviewed`; this migration is not a claim
that their upstream histories have been searched.

The next backfill covers A8 GEMM incorporation (#2546), architecture-aware LDS checks
(#2732), the small-M GEMM family (#2775), MoE sorting (#3266), MLA reduce (#3901), guarded
prefill mask elision (#4761), DCP top-k merge (#5123), and HERD routing (#5693). See the
[generated coverage index](../../knowledge/operator/CORPUS.md) for current counts.

Further backfill covers the split-K HGEMM introduction and tail-row correctness fix
(#2536/#2567), MXFP8 GEMM comparison populations (#5896), compressor constraints
(#3357), route-map and quant/scatter incorporation (#3460/#3851), and GDN MTP kernel
results alongside an early public-entry regression and its later fix (#4907).
The later operand cache-policy and XCD additions are traced to #2581 and #2726;
shared-expert readiness and padded-layout handling are traced to #5907, with its
operator/serving comparisons and cross-rank reuse constraints kept separate.
Historical GEAK summaries without a uniquely identified AITER change remain
`unresolved`, including accounts explicitly attributed to vLLM or SGLang. Their
documentation pins and existing run conclusions are retained.

## Three independent questions

1. **Association:** Does the cited change implement this particular direction? Verify
   the changed construct, its callers and the selected backend, not just a filename or
   PR title. GitHub commit associations and `git blame` produce candidates.
   A file-creation PR need not introduce a later optimization in that file. Follow the
   specific construct through subsequent commits before associating its effect.
2. **Effect:** What did the upstream source actually report, with what comparison
   context? Each result belongs to one kernel, operator or end-to-end comparison.
3. **Independent attribution:** Does the comparison isolate this direction? A PR
   changing tiling, LDS layout and dispatch may establish a bundle result while leaving
   each component's contribution unknown.

Grades are ordinal curator assessments, not calibrated probabilities. Each contains
`level`, nonempty `reasons`, and `missing_evidence` (possibly empty). PR approval, merge
status, popularity and author identity are not evidence of a speedup.

| Assessment | Unknown / low | Medium | High |
|---|---|---|---|
| Association | Unresolved candidates remain in the review backlog; low means a tentative but documented mechanism match. | A supported match with remaining uncertainty. | Reviewed before/after constructs establish the relationship. |
| Effect | Unknown when no effect is reported; low for qualitative, incomplete or contradictory reports. | Quantitative comparison with hardware, dtype, shape, baseline/candidate, scope, protocol and correctness context. | Not assignable to an imported PR report in this version. Reproduction is separate validated experience. |
| Independent attribution | Unknown or low for bundle/unclear effects. | Explicit isolated comparison, still reported rather than reproduced. | Requires explicit isolation evidence; it does not promote effect confidence or claim reproduction. |

Missing software versions or measured commits remain `null`, even when the PR's
base/head/merge commits are known. Partial information in a context string must say
what is unknown. List all null context fields in `missing_evidence`, plus unresolved
issues such as repetitions or independent reproduction. These grades apply only to
the recorded comparison; every card remains `candidate_only` for a new workload.

## Canonical report

Use the checked-in examples as complete templates:

- [Selector, operator and serving results](../../knowledge/_sources/aiter/pull_requests/4963.yaml)
- [Different baselines and contrary outcomes](../../knowledge/_sources/aiter/pull_requests/4538.yaml)
- [Implementation change without a performance report](../../knowledge/_sources/aiter/pull_requests/3231.yaml)
- [Closed, unmerged proposal and conflicting reported extrema](../../knowledge/_sources/aiter/pull_requests/1909.yaml)
- [A validity change with no assigned speedup](../../knowledge/_sources/aiter/pull_requests/2732.yaml)
- [Serving gains and per-shape gains, regressions and equal displayed timings](../../knowledge/_sources/aiter/pull_requests/3266.yaml)
- [Measured candidate revisions distinct from PR metadata](../../knowledge/_sources/aiter/pull_requests/4761.yaml)
- [A qualitative report with image-only numerical results left unencoded](../../knowledge/_sources/aiter/pull_requests/5693.yaml)
- [Kernel gains, an early public-entry regression and its subsequent fix](../../knowledge/_sources/aiter/pull_requests/4907.yaml)
- [Shared-expert contracts within a communication-fusion bundle](../../knowledge/_sources/aiter/pull_requests/5907.yaml)

Each report contains:

- PR identity, lifecycle (`open`, `closed`, `merged`), full base/head/merge commits,
  capture time and upstream update time. A closed unmerged PR has a null merge commit,
  even if the GitHub API exposes a temporary test-merge SHA.
- `sources`: exact excerpts with CRLF normalized to LF, a PR/comment URL, SHA-256,
  and per-source capture/update times. A fingerprint binds the saved excerpt; it does
  not authenticate the author or prove a benchmark. Save small relevant excerpts;
  omit signed image URLs, access tokens, unrelated discussion and raw private runs.
- `changes`: stable local IDs, a change commit and its actual parent, file, before/after
  excerpts and a mechanism summary. An empty `before` means the file was added. These
  are reviewed historical changes, not replacements for existing source snapshots.
- `results`: stable local IDs, outcome, summary, source IDs, comparison context,
  metrics, effect confidence and limitations. Different shapes, baselines, scopes and
  source revisions can have separate results. Preserve contradictory results with
  their contexts rather than selecting only the strongest positive number.

The comparison context declares `hardware`, `gfx`, `dtype`, `shape_layout`, `software`,
`baseline`, `candidate`, `baseline_commit`, `candidate_commit`, `scope`, `protocol` and
`correctness`. Unknown values are null; `scope` uses `unknown` if absent. A source-only
implementation observation cannot fill in an absent benchmark protocol or test result.

Outcomes are `beneficial`, `no_gain`, `regressed`, `mixed`, `constraint`, and
`not_reported`. The last requires no metrics and unknown effect confidence. Capability
extensions and correctness fixes need not carry speedup numbers. An invalid baseline
(for example, a dtype it does not support correctly) cannot support a performance comparison.

An empty metric list is not synonymous with an absent report. `constraint` records can
describe a validity or interface change without assigning speedup. A qualitative
`beneficial`, `regressed` or `mixed` report can remain low-confidence with no encoded
metrics, including when its numeric results are only in untranscribed image attachments.
State that gap explicitly; do not relabel existing image results as `not_reported`,
invent numbers from a PR title, or treat equal rounded timings as statistical equivalence.

### Metrics

| Kind | Stored fields | Unit | Derived speedup |
|---|---|---|---|
| `latency` | `before`, `after` | ns/us/ms/s, same for both | before / after |
| `throughput` | `before`, `after` | TFLOP/s, GB/s, tokens/s | after / before |
| `speedup` | `value` | x | value |
| `latency_reduction` | `value` | percent | 1 / (1 - value/100) |
| `throughput_increase` | `value` | percent | 1 + value/100 |

Every metric retains its reported statistic: `single_case`, `mean`, `median`, `min`,
`max`, `geomean` or `unspecified`. A single-case row does not imply any particular
repetition count or timing estimator; disclose those gaps in the context. A reported
maximum stays a maximum. Rendering derives ratios only within that metric; it never
averages across shapes or multiplies improvements across axes. Preserve an upstream
aggregate as an attributed aggregate even if a table disagrees, and document the
disagreement rather than silently correcting or endorsing it.

Each stored metric also carries `source: {ref, quote}` pointing to the specific table
row or claim within a saved excerpt. Validation checks the quotation and the presence
of its numeric values, catching transcription errors. This does not interpret arbitrary
benchmark tables: metric meaning, units, before/after roles and aggregation still need
review against the original headers and context.

## Decision links and review coverage

`upstream_review` has a `status`:

- `unreviewed`: no completed provenance review for the direction.
- `reviewed`: at least one PR is linked; `summary` describes the exact reviewed scope.
- `unresolved`: the scoped search did not establish a PR relationship; this is not proof
  that no relationship exists.
- `blocked`: review is incomplete because evidence or access is unavailable. Previously
  accepted links can remain while the unresolved work stays explicit.

All states except `unreviewed` require a timezone-qualified `checked_at` and a summary.
The review status describes the stated scope, not every backend or every historical
version mentioned by a card. Known PRs without performance data use a `not_reported`
result, so readers can distinguish missing association from missing benchmark data.

An upstream experience entry retains `outcome`, `conditions`, `summary`, and:

```yaml
provenance:
  kind: upstream_report
  references: []
  artifacts: not_published
  verification: not_reproduced
```

Its `upstream` mapping names `pr`, local `change_refs`, local `result_refs`, and an
`assessment` with association confidence, independent attribution and `applies_to`.
The last names the card's specific backend and axes, allowing one solution within a
broader card to receive a scoped assessment. Result confidence lives on the canonical
result; it is not copied onto every consuming card. The entry's outcome must agree
with the referenced results, or use `mixed` for differing outcomes.

Attribution declares `scope: isolated | bundle | unclear`, a confidence assessment,
and `isolation_source_refs`. An isolated claim needs explicit source support. Bundle
and unclear claims cannot have medium/high independent-attribution confidence.
Historical and private experience continue through the existing validator; importing
a PR never upgrades them to `internally_validated` or inserts raw `measurement_evidence`.

## Collection, review and verification

The collector requires Python 3.10+, PyYAML and public GitHub API access. It fetches the
PR body, issue comments, reviews, inline review comments, commits and files, paginating
collections and rejecting incomplete file/commit inventories. It executes no upstream
code and makes no corpus edits. API failures remain failures, not empty successful scans.

```bash
# Discover candidate PRs for a full change commit.
python3 tools/corpus/collect_pr.py --commit <full-commit> --output /private/scratch/associations.json

# Preserve review inputs outside the public repository.
python3 tools/corpus/collect_pr.py --pr 4963 --output /private/scratch/pr-4963.json

# After curating YAML, check its quotations, source times, PR metadata and membership.
python3 tools/corpus/collect_pr.py \
  --check-evidence knowledge/_sources/aiter/pull_requests/4963.yaml \
  --capture /private/scratch/pr-4963.json

python3 tools/corpus/validate.py --verify-upstream /path/to/aiter
python3 tools/corpus/build.py --emit
python3 tools/corpus/build.py --check
python3 -m pytest -q
```

The Git audit reads full commit objects to check actual parent links, confirms excerpts
in parent/candidate files, and rejects unchanged constructs. Fetch missing revisions
before auditing. The capture audit checks quotations against the actual saved response;
schema/fingerprint checks alone cannot prove a quote came from GitHub. Neither audit
reproduces a benchmark, proves causal attribution or establishes that an unmerged
proposal reached the catalog snapshot.

Repeat `--capture` when a report retains sources from multiple reviews. The current PR
metadata is checked against its matching capture; each historical excerpt is checked
against its own capture/update time. Missing historical captures fail the audit rather
than silently checking old quotations against newly edited PR text.

## Recurring maintenance

Review changed code and affected card links during `aiter-corpus-refresh`. Also revisit
PRs with pending questions or changed discussion independently of source commits: a
benchmark comment can change without a code commit. Preserve private API captures and
maintain a PR evidence checkpoint separate from the source `reviewed_through` checkpoint.
Rate limits, partial collection, missing objects and unfinished attribution leave the
corresponding review incomplete.

Treat cited source/result IDs as immutable. Add a new source excerpt and result ID for
new or corrected evidence; keep the old result and its conditions where still cited.
Record the new source's own timestamps instead of relabeling the old capture. A reverted,
closed or superseded PR remains historical evidence and must not appear as the current
implementation. The tools do not install a timer, auto-merge changes or promote kernels.
