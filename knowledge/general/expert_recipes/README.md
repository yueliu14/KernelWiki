# Reusable expert recipes

This collection belongs in `general/` because it contains reusable procedures, including GEMM
tuning and backend authoring. An operator/scenario selector scopes a recipe's applicability; it does
not turn the recipe into a record of one operator's result. The specific evidence and limitations
remain attached to each recipe.

These are imported GEAK resources. The selection and validation behavior described below belongs
to the original GEAK workflows; KernelWiki does not automatically execute it or refresh its status.

> **Contract (read this first).** An expert skill is an **advisory prior**, not a mandate. It packages a
> human expert's *proven, reusable optimization recipe* (the regulated steps + knobs + pitfalls) for a
> specific `operator × scenario`. A skill that has passed validation carries a high prior and a
> reproducible procedure — but it **never overrides on-box measurement**. The consuming workflow
> (`e2e_workflow` / `kernel_workflow`) treats a matched skill as a *candidate to reproduce first*, then
> decides the winner by its own A/B gate. When a skill conflicts with the live measurement, the
> measurement wins and the skill is flagged `stale` for re-review.
>
> This is the same discipline as the sibling `perf_knowledge/` base — *seed/locate candidates faster,
> never reduce a result below its measured baseline* — just with stronger, validated, opinionated recipes.

## Why this exists

`perf_knowledge/` holds *facts* (APIs, knobs, which backends exist). The `e2e_workflow/knowledge/`
ledgers hold the *agent's own rolling experience*. Neither captures a **human expert's end-to-end
recipe** — e.g. "port the MLA decode core from TileLang to Triton on gfx942", or "the FlyDSL fp8 a8w8
blockscale down-proj playbook that won +67% e2e". `expert_skills/` is that third tier: contributed,
indexed, and **validated to actually move e2e (or an isolated kernel) without regressing the baseline**.

## Layout

```
expert_skills/
├── README.md                  # this file (the contract)
├── index.yaml                 # machine-queryable selector + validation status (AUTO-MAINTAINED)
├── tuning_index.yaml          # selector metadata for tuning/ (lives OUT here — see below)
├── skills/<id>/               # one SUBDIRECTORY per skill
│   ├── skill.md               #   the main recipe (frontmatter selector + body) — REQUIRED
│   └── ...                    #   optional extra files: reference kernels, configs, validation manifest
├── tuning/                    # locally maintained copy of imported tuning methods
│   ├── tuning-<backend>/SKILL.md   #   one skill per backend, retaining source attribution
│   └── validate/claims.py     #   its own executable validator (drives validation_status)
├── _template/                 # SKILL_TEMPLATE.md + validation_manifest.yaml
└── _contribute/               # the "add a skill to GEAK" skill: scaffold / validate / make_pr / SKILL.md
```

Each skill lives in its own directory `skills/<id>/` so a skill that needs more than prose — a reference
kernel, tuned config JSONs, a custom validation manifest — can carry those files alongside its
`skill.md`. The selector (`index.yaml`) always points at `skills/<id>/skill.md`.

`tuning/` retains the imported per-backend layout. Its selector metadata lives in `tuning_index.yaml`
and is merged into `index.yaml` by `scaffold.py --reindex`. Edit the source metadata and regenerate
the index. Local changes to the methods are allowed through review; retain attribution and revisit
any validation claims affected by the change.

## How a skill is selected by the workflows

Each skill's frontmatter has a `match:` block. Expert-skill selection is **opt-in** — the workflows do
not inject or query this registry unless the run passes `use_expert_skills=true` (default OFF). When
OFF, the base role prompt remains active, including general `perf_knowledge` guidance that is independent
of expert-skill selection. When enabled, the workflow filters `index.yaml` by the current bottleneck:

```
(match.operator is scalar AND match.operator == bottleneck.operator)
  OR (match.operator is a list AND bottleneck.operator ∈ match.operator)
AND gen ∈ match.gens
AND model_arch_class ∈ match.arch_class   (or match.arch_class contains '*')
AND (migration skills) from_backend/to_backend fit the live path
AND scope matches the querying layer     (a kernel lane ignores scope: tuning, and vice versa)
AND validation.status == validated        (stale/draft/failed are NOT auto-applied)
```

There is **no ranking** here (same as `capability_index.yaml`). Every match enters the candidate set;
the on-box A/B picks the winner.

## scope: kernel vs e2e vs tuning

A skill declares `scope: kernel | e2e | tuning | dependency`. It controls which harness validates it and
which workflow layer consumes it:

| scope  | validated by      | consumed by                          | pass criteria |
|--------|-------------------|--------------------------------------|---------------|
| kernel | `kernel_workflow` (isolated A/B vs the immutable unittest oracle) | `kernel_workflow` author/optimize | `isolated_speedup ≥ expects.isolated_speedup_min` + parity |
| e2e    | `e2e_workflow` (Director same-session A/B) | `e2e_workflow` routing/integration | `e2e_delta ≥ expects.e2e_delta_min_pct` + parity + non-trigger inertness |
| tuning | the skillset's own `tuning/validate/claims.py` (one executable check per load-bearing claim) | `e2e_workflow` TuningSkillset phase | every applicable claim PASS, none FAIL — see below |

### How the tuning scope gets its status

Not by hand. `scaffold.py --reindex` reads `tuning/validate/claims.py` output — the four reports the
skillset ships (both images x both gens) by default, or `--claims-report` from a fresh in-container
run — and derives: any `FAIL` → `draft`, at least one `PASS` and no `FAIL` → `validated`, nothing but
`N/A` → `unvalidated`. `N/A` means the precondition is absent (tool not installed, framework not in
this image), so the image did not answer the question; promoting that to `validated` would be exactly
the failure the validator exists to catch. Re-run the reindex after an image or upstream bump.

## Validation status lifecycle

`draft` → (validate passes) → `validated` → (staleness: age / aiter·triton version drift) → `stale`
→ (re-validate) → `validated`. A failed validation sets `failed` and the PR cannot land as `validated`.
Only `validated` skills are auto-applied; `stale` is demoted to a plain reference (re-measure forced).

## Contributing

See [`_contribute/SKILL.md`](_contribute/SKILL.md). Short version:

```
python _contribute/scaffold.py --id <slug> --operator <op> --scope <kernel|e2e>   # make skeleton + register
$EDITOR skills/<slug>/skill.md                                                           # fill Procedure/Mechanism/Do-no-harm
bash   _contribute/make_pr.sh <slug>                                               # validate (by scope) → set status → open PR
```

## Canonical copy

KernelWiki maintains this imported copy with Git. The local Git revision identifies the maintained
files. [`tuning_skillset.manifest.sha256`](../workflows/geak/tuning_skillset.manifest.sha256) is an
archived upstream manifest: it is provenance, not a checksum contract for this modified copy.
It intentionally includes historical files that are no longer maintained here. Do not overwrite
local changes to satisfy that historical manifest or run GEAK-specific sync commands in KernelWiki.

The selector metadata and validation reports describe their recorded revisions. Re-run applicable
checks after changing executable methods; a historical `validated` label does not validate the
current environment. Reusable outcomes go into corpus, and run histories and deployable run
artifacts stay private. There is no automatic sync or validation-status refresh.

## Sources
- Design discussion 2026-06-17 (this repo).
- Sibling contract: `perf_knowledge/README.md` (reference-only doctrine).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/expert_skills/README.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
