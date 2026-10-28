# KernelWiki

An **operator-centered GPU kernel knowledge base for LLM inference**, focused on **GEMM, MoE, attention and attention indexers**, including their variants and fusions. It documents backend implementations, applicability, performance evidence and optimization experience to help developers and GEAK write, select and improve kernels.

## Content

| Content | Start here | Purpose |
| --- | --- | --- |
| Written knowledge | [Operators](knowledge/operator/README.md) · [General](knowledge/general/README.md) | Implementation details, source entry points, performance and applicability, tuning methods and lessons from previous runs |
| Structured corpus | [Operator catalog](corpus/catalog.yaml) · [Corpus guide](corpus/README.md) | Operator-local source observations, conditioned development cards and upstream configuration seeds |
| Kernel implementations | [Source collection](kernels/README.md) · [Put, search and fetch API](kernelwiki/README.md) | Reuse complete source bundles with entry points, dependencies, correctness checks and benchmarks |

```text
knowledge/
  operator/<operator>/
    README.md
    backend/<backend>_aiter.md
    backend/<backend>_notes.md
    learned_knowledge/
    corpus/
      scope.yaml
      source_evidence.yaml
      decisions.yaml
      performance_axes.yaml
      source_evidence.md       # generated
      decisions.md             # generated
      tuned_configs.yaml       # where upstream seeds are available
      seeds.md                 # generated where applicable
  general/
    profile_bottleneck/
    optimization/
    backend/
    hardware/
    quantization/
    ...
kernels/<operator>/<backend>/<implementation>/
  kernel.py
  entry.py
  test.py
  benchmark.py
  metadata.yaml
  README.md
kernelwiki/
  client.py
corpus/
  catalog.yaml
  README.md                    # generated navigation
tools/corpus/
  extract.py
  validate.py
  build.py
```

## Getting started

- [Dense GEMM](knowledge/operator/dense_gemm/README.md) · [Quantized GEMM](knowledge/operator/scaled_quant_gemm/README.md) · [Grouped GEMM / MoE](knowledge/operator/grouped_gemm_moe/README.md)
- [MLA](knowledge/operator/mla_attention/README.md) · [Paged decode attention](knowledge/operator/attention_decode_paged/README.md) · [Attention indexer](knowledge/operator/attention_indexer/README.md)
- [GEMM fusions](knowledge/operator/gemm_epilogue_fused/README.md) · [Fused MoE](knowledge/operator/fused_moe_grouped_gemm/README.md) · [Q/K preparation fusions](knowledge/operator/qk_norm_rope_cache/README.md)
- [Bottleneck analysis](knowledge/general/profile_bottleneck/README.md) · [Optimization methods](knowledge/general/optimization/README.md) · [Writing knowledge and recording evidence](knowledge/README.md)

## Operator scope and evidence

For source-grounded implementation decisions, start with the [GEMM](knowledge/operator/dense_gemm/corpus/decisions.md),
[attention](knowledge/operator/attention_prefill_fmha/corpus/decisions.md), [MoE](knowledge/operator/grouped_gemm_moe/corpus/decisions.md) or
[indexer](knowledge/operator/attention_indexer/corpus/decisions.md) corpus. All **23 operators** own a local
`corpus/` with source evidence, development cards, extraction scope and performance axes.
The collection contains **69 canonical cards, 4,481 source observations and 359 upstream
configuration/selection groups**. Cross-operator knowledge uses shared references to a single owner;
the root catalog provides navigation and snapshot metadata. See the [maintenance guide](tools/corpus/README.md).

The corpus preserves the AITER `a6bb49937584` GEMM evidence and adds source from `b3ad6c461703`
across GEMM, attention, MoE and indexers. References include a snapshot prefix and stable content ID;
source links use the full commit. Each operator records its coverage limits and pinned upstream
correctness-test sources. Cards describe source precedents, constraints and unmeasured candidates.

- **23 operator directories**, grouped by GEMM, MoE, attention and indexer workloads, with **92 AITER backend/provider reference pages**, **13 implementation-note pages** and **122 operator experience cards**.
- Standalone small operators, spatial convolutions, training losses, recommendation attention and general solvers are excluded from the operator collection. Quantization, activation, cache layout, normalization and communication are documented where they affect a main kernel or fusion. MoE routing/dispatch and Q/K preparation fusions remain part of those workloads.
- AITER references primarily use [`e9995854abbd`](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca), with a recent-source window of 2026-07-21 through 2026-10-21. Additional native-pipeline and workload-variant reviews, including [Split-K](knowledge/operator/splitk_streamk_gemm/backend/opus_aiter.md), [shared experts](knowledge/operator/shared_expert_fusion/backend/flydsl_aiter.md) and [FP4 GEMM](knowledge/operator/scaled_quant_gemm/backend/asm_notes.md), use [`b3ad6c461703`](https://github.com/ROCm/aiter/tree/b3ad6c461703110dc36286265c16e453ef0e2dd2). Explicit gradient-only implementations and tests are excluded from current operator reference pages.
- Historical explanations and experience cards come from GEAK [`02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/tree/02bfbb0b72746941be3ab3ad48fb21a284ae2fde). General methods remain under `knowledge/general/`; citations to excluded operators can point to pinned archived documents.

AITER is the main source of implementation knowledge. This is a curated workload collection, not a claim to cover every AITER kernel. Documents include pinned links to source, configuration and tests. Choosing the best implementation requires a specific GPU, dtype, shape/layout and comparable measurements. Source coverage adds no GPU benchmark results and does not certify historical SOTA labels as current winners.

## Kernel source collection

The repository now stores complete kernel implementations alongside their explanatory knowledge. The first source bundle is [AITER's row-tiled Triton RMSNorm](kernels/rmsnorm/triton/aiter_row_tiled/README.md), with its kernel body, standalone entry point, launch configurations, numerical reference, correctness checks and benchmark program. It retains AITER's MIT license and pinned source attribution.

This initial bundle is a source reference with upstream-reported performance. It has not been remeasured on the packaging host and is not labeled `best_known`. A best-known entry must identify exact measured cases and attach correctness/performance evidence to its source digest. See the [source-bundle contract](kernels/README.md).

## Using the source API

```bash
python -m pip install -e .
```

```python
from kernelwiki import KernelClient

client = KernelClient.local("kernels")
result = client.search(
    "rmsnorm", backend="triton", gfx="gfx950", dtype="bf16",
    shape={"M": 16384, "N": 128},
)
item = result["implementations"][0]
client.fetch(
    item["implementation_id"], revision=item["revision"],
    destination="downloaded_rmsnorm",
)
```

For remote candidates, set `GEAK_KB_URL` and use `KernelClient.from_env()`. The existing `GEAK_KB_STORE_URL`, `KB_STORE_URL` and token variables remain supported. `put()` uploads a complete source bundle; `fetch()` returns the same runnable directory layout. New source records use `geak:kernel_code:...` identities. The previous patch-specific Python API has been removed in version 0.2. Historical remote patch records are not modified or presented as complete source implementations.

Repository entries are curated with Git. The remote service can retain multiple candidate revisions and measurement artifacts. Both use the same implementation ID, source digest and bundle revision. Read text knowledge directly from Markdown. See the [API guide](kernelwiki/README.md) for publication, search and retrieval details.

## Development checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python tools/corpus/build.py --check
```

## Attribution and licenses

Imported GEAK content retains its attribution and Apache-2.0 license. AITER excerpts retain upstream notices, including MIT and per-file Apache-2.0 notices. See [LICENSE](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
