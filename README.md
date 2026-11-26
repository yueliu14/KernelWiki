# KernelWiki — kernel-level edition

A **kernel-level GPU kernel knowledge base for LLM inference on AMD Instinct**: how GEMM, MoE, attention and
attention-indexer kernels and their fusions are implemented, tuned, measured and validated.

This branch (`agent/kernel-level-lean`) is derived from `main` for agents that optimize a single kernel. It keeps the
kernel-level knowledge and drops everything above the kernel; see [What changed](#what-changed-from-main).

**Agents: start at [AGENTS.md](AGENTS.md).** It routes from "what my kernel computes", "which language" and "what
limits it" to a handful of pages.

## Layout

```text
AGENTS.md                         # router: operator × language × bottleneck -> pages
knowledge/
  operator/<op>/                  # 23 operators: GEMM, MoE, attention, indexer and their fusions
    README.md                     # overview, regimes, TL;DR
    cards.md                      # decision cards (rendered from corpus/decisions.yaml)
    tuning.md numerics.md fusion.md
    backend/<lang>.md             # GEAK implementation notes (dated)
    backend/<lang>_aiter.md       # pinned AITER source walkthroughs (reference)
    corpus/*.yaml                 # machine-readable source of cards.md (edited on main)
  general/
    cards.md                      # cross-operator method cards
    backend/<lang>/               # authoring guides: triton_amd, gluon, flydsl, hip_cpp, composable_kernel, asm_mfma, tilelang, ...
    hardware/                     # cdna4_mi350 (gfx950), cdna3_mi300 (gfx942), shared CDNA facts
    optimization/                 # methods by bottleneck
    profile_bottleneck/           # profiling, timing protocol, engagement checks
    quantization/                 # formats, scaling, kernel-level accuracy gates
    expert_recipes/               # authoring skills and per-language tuning recipes
    case_studies/by_kernel/
    workflows/                    # optimize_single_kernel, choosing_a_backend, roofline skill
tools/agent/                      # prune.sh, build_cards.py, slim.py, lint.py
```

## What changed from main

| Change | Why |
| --- | --- |
| Removed `top_model/` (model CI histories, 25 MB of data and kernel snapshots), `workload/`, `_sources/`, `general/{landscape,reference,libraries}`, model case studies, serving/integration workflows and tuning recipes, model-level quantization deployment, pre-CDNA3 and client GPUs | Not kernel-level knowledge |
| Generated `corpus/decisions.md`, `source_evidence.md`, `seeds.md`, `CORPUS.md` replaced by one compact `cards.md` per operator | Same cards without source-id lists, review boilerplate and HTML escapes; sorted so reported outcomes and constraints come first; each card states the languages it applies to |
| Backend source walkthroughs keep symbol names but drop the per-symbol pinned URLs | Those link lists were ~40% of the pages' bytes; the file-level source link still pins the revision |
| Imported GEAK metadata blocks removed from operator READMEs; links to removed material reduced to text | No dead ends |
| Added `AGENTS.md` router | One entry page that routes by operator, language and bottleneck instead of a tree to explore |
| Corpus maintenance tools and tests stay on `main` | They need the full tree (PR records, generated pages); edit `corpus/*.yaml` there |

Re-derive after `main` changes:

```bash
git merge main                       # or check out main's files
bash tools/agent/prune.sh            # remove non-kernel-level material and generated corpus pages
python3 tools/agent/build_cards.py   # cards.md from corpus/decisions.yaml
python3 tools/agent/slim.py          # symbol-link slimming, README cleanup, dead-link removal
python3 tools/agent/lint.py          # links resolve, cards fresh, router size, no HTML escapes
```

## Evidence

Nothing here is a measured winner for a particular kernel. Cards separate `reported` upstream outcomes (not
reproduced here) from `source-only` observations of shipped code; pages pin their source revision and date. Keep
independently measured results, upstream-reported results and source-only observations distinct when adding
knowledge, and name hardware, software revision, shapes, dtypes and timing protocol with every measured result.

See [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
