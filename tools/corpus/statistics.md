# Evidence statistics contract

`stats.py` reads the validated `Catalog` and generates:

- `knowledge/statistics/README.md`: inventory, evidence coverage, outcome distributions,
  missing context, owner coverage and an index of every canonical card.
- `knowledge/statistics/reported_effects.md`: individual reported metrics grouped by
  kernel, operator, end-to-end or unknown measurement scope.
- `knowledge/statistics/statistics.json`: schema 1 export containing the summary,
  owner groups, cards, canonical results, full comparison contexts and associations.

Run `python3 tools/corpus/stats.py --emit` after editing inputs, or use the existing
`python3 tools/corpus/build.py --emit` to regenerate all corpus pages. Both commands
support `--check`; the existing build CI gate checks these artifacts too. Output is
deterministic, contains no timestamps or local absolute paths, and needs no GPU or
network. Git records the version of both the inputs and generated output.

## Population and counting units

The population is the operator/general cards in `knowledge/operator/catalog.yaml`
and the canonical AITER reports in `knowledge/_sources/aiter/pull_requests/`.
Workload cases, model histories and unstructured Markdown are excluded, rather than
interpreted as compatible experiments. This is a curated evidence inventory, not a
complete or random sample of upstream optimization attempts.

Cards count once by canonical `card_id`; shared references do not add cards. Results
count once by `(pr_id, result_id)`, even when several cards or experience entries cite
them. Distinct result IDs remain distinct reported claims; the corpus does not encode
enough information to deduplicate shared underlying experiments or sample populations.
A result containing several metrics counts as one result and several metric rows.
Metrics are not independent trials. Experience-entry counts describe links/summaries,
not distinct runs or successes.

Cards have one canonical owner. Within each owner, result IDs are deduplicated;
result totals across owners overlap and must not be summed. JSON preserves every
association's conditions, axes, backend and attribution assessment alongside a unique
`linked_card_ids` list. An association with an improved bundle does not establish an
improvement from the individual card.

Coverage records expose `numerator`, `denominator` and `fraction`. A zero denominator
produces null, not zero or a fabricated 100%. All card coverage denominators include
semantic and constraint cards as well as performance candidates. Therefore numerical
coverage is an inventory measure, not a quality ranking: useful constraints need not
produce a speedup. The card index and JSON `type` allow inspection by knowledge type.

## Effects and missing evidence

Upstream metrics retain kind, unit, original statistic and source quotation. The
existing `upstream.speedup()` normalizes each metric separately. Reported maxima,
means, medians and geometric means never become interchangeable observations, and
the generator does not calculate an average across those rows. Even equal scope,
hardware and dtype do not establish compatible baselines or measurement protocols.

No-metric results retain their reported outcome: a qualitative gain, constraint or
untranscribed figure is different from `not_reported`. No metric means unknown, not
1x. Reported regressions and closed/unmerged proposals remain visible, with lifecycle
and confidence; a merge does not certify deployment, correctness or reproduction.
The report does not assign current best-kernel status.

Null-field counts identify literally null context fields only. A non-null description
may still contain unknown versions, mixed shapes or missing repetitions. Consult the
full `context`, `confidence.missing_evidence` and `limitations`; no completeness score
or comparability certificate is inferred from non-null strings. Unknown scope is its
own group. Effect confidence is the existing curator grade, not a probability.

Internal-validation coverage counts cards containing `private_summary` experience
with `verification: internally_validated`. It does not prove public reproduction or
that a numerical result has been published. Upstream and historical experience never
count as internal validation. Raw/private artifacts are not copied into the export.

The corpus currently lacks a complete attempt ledger. Success rate, correctness rate,
agent benefit, uncertainty intervals, independent-task counts and token/GPU cost
cannot be recovered from this inventory. Those require the GEAK application records
described in [the integration proposal](geak-application-records.md).
