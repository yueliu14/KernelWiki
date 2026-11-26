# General kernel knowledge

Methods and reference material shared across operators. Agents: [AGENTS.md](../../AGENTS.md) routes by language,
GPU and bottleneck; the [operator collection](../operator/README.md) holds operator-specific knowledge.

- [backend](backend/README.md) — authoring guides per kernel language (Triton, Gluon, FlyDSL, HIP/C++, CK, assembly, TileLang, …)
- [hardware](hardware/README.md) — CDNA4 / MI350 (gfx950), CDNA3 / MI300 (gfx942) and shared CDNA facts
- [optimization](optimization/README.md) — methods by bottleneck: memory, grid sizing, MFMA scheduling, LDS, occupancy, fusion, numerics
- [profile_bottleneck](profile_bottleneck/README.md) — profilers, timing protocol, reading a bottleneck, engagement checks
- [quantization](quantization/README.md) — formats, FNUZ vs OCP, block/MX scaling, kernel-level accuracy gates
- [expert_recipes](expert_recipes/README.md) — authoring skills (FlyDSL, Gluon, …) and per-language tuning recipes
- [case_studies](case_studies/README.md) — kernel-level case studies
- [workflows](workflows/README.md) — optimizing a single kernel, choosing a backend, roofline analysis
- [cards.md](cards.md) — cross-operator method cards (rendered from `corpus/decisions.yaml`)

Imported documents retain their source dates and evidence. Historical status, validation results and performance
thresholds describe those sources; they do not certify the current stack.
