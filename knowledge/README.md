# Kernel knowledge

- [Operators](operator/README.md): GEMM, MoE, attention and indexer implementations, variants, fusions and lessons for LLM inference.
- [Workloads](workload/README.md): semantic execution chains spanning multiple operators, with fusion boundaries, dispatch conditions and model/stack-specific evidence.
- [General knowledge](general/README.md): profiling, optimization, backend programming, hardware and reusable recipes.
- [Kernel source collection](../kernels/README.md): complete implementations with entry points, dependencies, tests and benchmarks.

## Reading a backend page

`<backend>_aiter.md` describes the pinned AITER source snapshot with entry points, dispatch/shape contracts, configuration references and related tests. Some pages explain a workload variant of a shared implementation; others list source files and symbols. Symbol lists include support functions; they are not counts of independently launchable kernels. Native `.cuh` pipelines belong with their wrappers, instance traits and launch plans. ASM code objects are grouped by family, and may include compiled DSL output.

`<backend>_notes.md` explains selected implementation decisions in detail. `<backend>.md` preserves older GEAK material, including dated measurements and upstream reports. Read the source date before applying a historical recommendation.

## Performance evidence

Keep independently measured results, upstream-reported results and source-only observations distinct. A source change or a tuned configuration does not itself establish a speedup. Every new performance result should name hardware, software/source revisions, workload, layouts/dtypes, timing protocol, baseline, correctness check and evidence location. Rank implementations only for compatible cases.

## Contributing knowledge

Keep new operator entries within GEMM, MoE, attention or attention indexers and their variants/fusions. Document small supporting operations with the main workload: packing and quantization with GEMM, routing and dispatch with MoE, cache writes and positional encoding with attention, and causal state mixing with gated-delta attention. Do not add separate collections for generic small operators, vision convolutions or training-only kernels.

Put operator-specific implementation knowledge under `operator/<op>/backend/`; put transferable run-derived lessons under that operator's `learned_knowledge/`. Organize multi-operator execution chains under `workload/<semantic_chain>/`, with a model-independent contract and links to the participating operators. Store model/stack-specific implementations under that chain's `cases/`, with model and phase in the case metadata; shared source context belongs in `workload/_sources/`. Record actual kernel boundaries, intermediate tensors, dispatch conditions and evidence using the [workload case template](workload/_templates/fusion_case.md). A joint benchmark or adjacent launches alone do not establish fusion. Put methods that apply across operators under `general/`; the workload [fusion pattern index](workload/fusion_patterns.md) connects those methods to concrete chains. Add a local navigation link and cite the relevant source or measurement. AITER implementation links should include a commit.

Store reusable code as a complete source bundle under `kernels/<operator>/<backend>/<implementation>/`. Link its implementation ID from the corresponding knowledge page. Include the exact source digest with measured results; historical patch references remain historical evidence, not the current storage format. All repository documentation and comments use English.

The AITER references prioritize inference-relevant files changed from 2026-07-21 through 2026-10-21. Backend pages can link shared source files or test suites containing both forward and backward code; the documented target is inference, with gradients and dropout disabled. Older GEAK evidence retains its original context. There is no scheduled update process or automatic best-kernel promotion.
