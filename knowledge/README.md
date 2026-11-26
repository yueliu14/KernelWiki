# Kernel knowledge

Agents: start at [AGENTS.md](../AGENTS.md).

- [Operators](operator/README.md): GEMM, MoE, attention and indexer kernels, their variants and fusions. Each operator
  has an overview, decision cards (`cards.md`), short tuning/numerics/fusion notes and per-language backend pages.
- [General knowledge](general/README.md): kernel authoring by language, hardware, optimization methods, profiling and
  measurement, quantization formats and reusable recipes.

## Reading a backend page

`<backend>_aiter.md` describes the pinned AITER source snapshot: entry points, dispatch and shape contracts,
configuration references and related tests. `<backend>_notes.md` explains selected implementation decisions in detail.
`<backend>.md` preserves older GEAK material, including dated measurements and upstream reports; read the source date
before applying a historical recommendation.

## Performance evidence

Keep independently measured results, upstream-reported results and source-only observations distinct. A source
change or a tuned configuration does not itself establish a speedup. Every new performance result should name
hardware, software/source revisions, shapes, layouts/dtypes, timing protocol and correctness gate.

## Contributing knowledge

Keep entries at kernel level: implementation, tuning, numerics, measurement and validation of GEMM, MoE, attention
and indexer kernels and their fusions. Operator-specific knowledge goes under `operator/<op>/` (cards are edited in
`corpus/decisions.yaml` on `main`, then rendered here with `python3 tools/agent/build_cards.py`), transferable methods
under `general/`. Model histories, serving and workload material belong on `main`. All documentation uses English.
