# General kernel knowledge

Methods and reference material shared across operators. Start with the [GEMM, MoE, attention and indexer collection](../operator/README.md) when optimizing an LLM inference workload. Imported library and taxonomy documents can describe a broader historical ecosystem; they do not define the active operator scope.

Multi-operator execution chains live in [workload knowledge](../workload/README.md), organized by semantic composition with model-specific implementation cases. Historical run reports remain in [case studies](case_studies/README.md).

- [backend](backend/README.md)
- [case_studies](case_studies/README.md)
- [expert_recipes](expert_recipes/README.md)
- [hardware](hardware/README.md)
- [landscape](landscape/README.md)
- [Corpus: reusable development decisions](corpus/decisions.md)
- [libraries](libraries/README.md)
- [optimization](optimization/README.md)
- [profile_bottleneck](profile_bottleneck/README.md)
- [quantization](quantization/README.md)
- [reference](reference/README.md)
- [workflows](workflows/README.md)

## Reusable procedures

Reusable methods, skills, tools and reference material. Classify a document by what the reader can
reuse, not by whether its title names an operator. For example,
[GEMM tuning](workflows/gemm_tuning_workflow.md), [attention backend selection](workflows/attention_backend_selection.md),
backend authoring guides and [tuning skills](expert_recipes/tuning/README.md) stay here because their
procedures apply across models, shapes and implementations.

## Content boundaries

| Content | Canonical location | Example |
| --- | --- | --- |
| Reusable workflow, skill or library/tool procedure | `general/workflows/`, `expert_recipes/`, `backend/`, `libraries/` | Capture shapes, tune GEMM, deploy configuration and verify engagement |
| Hardware, numerics, profiling and optimization methods | Corresponding `general/` section | LDS layout, roofline analysis, FP8 formats, paired A/B measurement |
| An operator's implementation contract and source observations | `operator/<op>/backend/` and `corpus/` | MLA cache ABI, MoE routing weights, supported GEMM layouts |
| An outcome for a particular operator, dtype, shape and environment | `operator/<op>/corpus/decisions.yaml` | A gfx950 FP8 blockscale tuning result |
| Whole-model reports and examples teaching a transferable method | `general/case_studies/` and `general/corpus/` | End-to-end Amdahl accounting, measurement controls |

Start with the [operator collection](../operator/README.md) for implementation-specific knowledge.
General guides link to that knowledge instead of maintaining another operator overview or repeating
its measurements. A reusable skill can still declare an operator-specific applicability selector.

Imported documents retain their source dates and evidence. Historical `sota`, validation statuses,
serving flags and performance thresholds describe those sources; they do not certify the current
stack. Broader ecosystem references do not expand the active operator scope.

Reusable conclusions from the former learned cards are maintained in the [general corpus](corpus/decisions.md).
The public corpus stores scoped findings and evidence summaries. Complete run histories, logs,
candidate artifacts and private locations stay outside KernelWiki.
