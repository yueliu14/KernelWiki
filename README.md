# KernelWiki

An **operator- and workload-level GPU kernel knowledge base for LLM inference**, focused on **GEMM, MoE, attention and attention indexers**, including their variants and fusions. It documents backend implementations, multi-operator dataflow, scheduling, dispatch, memory, communication and performance evidence to help developers and GEAK write, select and improve kernels and execution paths.

## Content

| Content | Start here | Purpose |
| --- | --- | --- |
| Written knowledge | [Operators](knowledge/operator/README.md) · [Workloads](knowledge/workload/README.md) · [General](knowledge/general/README.md) | Implementations, multi-operator execution contracts and optimization, source entry points, applicability and evidence |
| Dated model history | [Model index](knowledge/top_model/README.md) | Configuration-matched CI evolution, shared kernel changes and exact implementation links |
| Structured corpus | [Operator catalog](knowledge/operator/catalog.yaml) · [Corpus guide](knowledge/operator/CORPUS.md) · [General decisions](knowledge/general/corpus/decisions.md) | Source observations, conditioned decisions, experience summaries and upstream configuration seeds |
| Corpus maintenance | [Tools](tools/corpus/README.md) · [AITER refresh skill](tools/corpus/skills/aiter-corpus-refresh/SKILL.md) | Extract pinned source evidence, update decisions, validate references and generate reading pages |

```text
knowledge/
  top_model/
    README.md                  # reading order and evidence semantics
    <model>/
      README.md
      configuration_history.md
      pairs/                   # endpoints, applicability, links
      changes/                 # shared kernel/body/fusion explanations
      kernel_implementations.md
      data/                    # reproducible machine-readable exports
      evidence/                # dated provenance and runtime records
      kernel_sources/          # selected exact source snapshots
  operator/
    README.md
    catalog.yaml
    CORPUS.md                   # generated navigation
    <operator>/
      README.md
      backend/<backend>_aiter.md
      backend/<backend>_notes.md
      corpus/
        scope.yaml
        source_evidence.yaml
        decisions.yaml
        performance_axes.yaml
        source_evidence.md      # generated
        decisions.md            # generated
        tuned_configs.yaml      # where upstream seeds are available
        seeds.md                # generated where applicable
  workload/
    README.md
    fusion_patterns.md
    _templates/workload_case.md
    _templates/fusion_case.md
    _sources/<source_set>.md
    _sources/<source_set>.yaml
    <semantic_chain>/
      README.md
      dataflow.md
      scheduling.md
      dispatch.md
      measurement.md
      cases/<case>.md
  general/
    corpus/
      decisions.yaml
      performance_axes.yaml
      decisions.md             # generated
    profile_bottleneck/
    optimization/
    backend/
    hardware/
    quantization/
    ...
tools/corpus/
  extract.py
  seeds.py
  validate.py
  build.py
  skills/aiter-corpus-refresh/
tests/corpus/
```

## Getting started

- [Dense GEMM](knowledge/operator/dense_gemm/README.md) · [Quantized GEMM](knowledge/operator/scaled_quant_gemm/README.md) · [Grouped GEMM / MoE](knowledge/operator/grouped_gemm_moe/README.md)
- [MLA](knowledge/operator/mla_attention/README.md) · [Paged decode attention](knowledge/operator/attention_decode_paged/README.md) · [Attention indexer](knowledge/operator/attention_indexer/README.md)
- [GEMM fusions](knowledge/operator/gemm_epilogue_fused/README.md) · [Fused MoE](knowledge/operator/fused_moe_grouped_gemm/README.md) · [Q/K preparation fusions](knowledge/operator/qk_norm_rope_cache/README.md)
- [Workload semantic chains](knowledge/workload/README.md) · [MoE dataflow example](knowledge/workload/moe_expert_pipeline/cases/muse_spark_decode_dataflow.md) · [Fusion patterns](knowledge/workload/fusion_patterns.md) · [QKV preparation chain](knowledge/workload/qkv_projection_norm_rope_cache/README.md)
- [Bottleneck analysis](knowledge/general/profile_bottleneck/README.md) · [Optimization methods](knowledge/general/optimization/README.md) · [Writing knowledge and recording evidence](knowledge/README.md)

## Operator scope and evidence

For source-grounded implementation decisions, start with the [GEMM](knowledge/operator/dense_gemm/corpus/decisions.md),
[attention](knowledge/operator/attention_prefill_fmha/corpus/decisions.md), [MoE](knowledge/operator/grouped_gemm_moe/corpus/decisions.md) or
[indexer](knowledge/operator/attention_indexer/corpus/decisions.md) corpus. All **23 operators** own a local
`corpus/` with source evidence, development cards, extraction scope and performance axes.
The collection contains **107 canonical cards (including 8 general-method cards), 4,481 source observations and 359 upstream
configuration/selection groups**. Cross-operator knowledge uses shared references to a single owner;
the catalog under `knowledge/operator/` provides navigation and snapshot metadata. See the [maintenance guide](tools/corpus/README.md).

The corpus preserves the AITER `a6bb49937584` GEMM evidence and adds source from `b3ad6c461703`
across GEMM, attention, MoE and indexers. References include a snapshot prefix and stable content ID;
source links use the full commit. Each operator records its coverage limits and pinned upstream
correctness-test sources. Cards distinguish source precedents from reported experience; historical
reports have not been reproduced for this corpus. Complete run histories and raw artifacts remain private.

- **23 operator directories**, grouped by GEMM, MoE, attention and indexer workloads, with **92 AITER backend/provider reference pages**, **13 implementation-note pages**; reusable conclusions and reported experience are maintained in corpus decisions.
- Standalone small operators, spatial convolutions, training losses, recommendation attention and general solvers are excluded from the operator collection. Quantization, activation, cache layout, normalization and communication are documented where they affect a main kernel or fusion. MoE routing/dispatch and Q/K preparation fusions remain part of those workloads.
- AITER references primarily use [`e9995854abbd`](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca), with a recent-source window of 2026-07-21 through 2026-10-21. Additional native-pipeline and workload-variant reviews, including [Split-K](knowledge/operator/splitk_streamk_gemm/backend/opus_aiter.md), [shared experts](knowledge/operator/shared_expert_fusion/backend/flydsl_aiter.md) and [FP4 GEMM](knowledge/operator/scaled_quant_gemm/backend/asm_notes.md), use [`b3ad6c461703`](https://github.com/ROCm/aiter/tree/b3ad6c461703110dc36286265c16e453ef0e2dd2). Explicit gradient-only implementations and tests are excluded from current operator reference pages.
- Historical explanations and the summaries distilled into corpus come from GEAK [`02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/tree/02bfbb0b72746941be3ab3ad48fb21a284ae2fde). General methods remain under `knowledge/general/`; citations to excluded operators can point to pinned archived documents.

AITER is the main source of implementation knowledge. This is a curated workload collection, not a claim to cover every AITER kernel. Documents include pinned links to source, configuration and tests. Choosing the best implementation requires a specific GPU, dtype, shape/layout and comparable measurements. Source coverage adds no GPU benchmark results and does not certify historical SOTA labels as current winners.

## Reading and searching

Browse the operator, workload and general indexes, or search the Markdown and YAML files directly.
For model history, use the [model index](knowledge/top_model/README.md) and follow the requested
configuration's pair. Default `rg` discovery omits large model audit/source payloads via `.rgignore`;
use a linked file or `rg --no-ignore` in the selected directory when inspecting evidence.
Agents can use the catalog to locate an operator's corpus, read its decisions and follow the cited
evidence to pinned upstream source. For example, from the repository root:

```bash
rg -n -i 'split.?k' knowledge/operator/dense_gemm knowledge/operator/splitk_streamk_gemm
rg -n 'source_evidence:' knowledge/operator/dense_gemm/corpus/decisions.yaml
```

Use the [AITER corpus refresh skill](tools/corpus/skills/aiter-corpus-refresh/SKILL.md) to review
upstream changes and maintain the corpus. Keep run histories and raw artifacts in your private store.

## Development checks

Maintenance tools require Python 3.10+ and Git; discovery tests also require `ripgrep` (`rg`).
CI installs it explicitly. Install the Python dependencies in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
python3 tools/corpus/build.py --check
```

## Attribution and licenses

Imported GEAK content retains its attribution and Apache-2.0 license. AITER excerpts retain upstream notices, including MIT and per-file Apache-2.0 notices. See [LICENSE](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
