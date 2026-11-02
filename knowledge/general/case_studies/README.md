# General case studies

Whole-model serving reports and worked examples of reusable optimization methods. Operator-specific
results live with their operator; use their recorded model, stack, shapes, baseline and date when
interpreting the outcome. Imported measurements were reported by GEAK or upstream authors, not
newly reproduced by KernelWiki.

The reusable procedures remain in [end-to-end optimization](../workflows/optimize_e2e_model.md),
[single-kernel optimization](../workflows/optimize_single_kernel.md) and
[GEMM tuning](../workflows/gemm_tuning_workflow.md). A tuning skill explains how to run an experiment;
a case records what happened in one environment.

## Whole-model reports

| Report | Context |
| --- | --- |
| [Qwen3.5-27B](by_model/qwen3.5-27b_sglang_e2e.md) | SGLang attention and GEMM changes, including rejected attempts |
| [DeepSeek MLA](by_model/deepseek_mla_mi300x.md) | Upstream-reported MLA serving results |
| [DeepSeek V3/V4 attention](by_model/deepseek_v3v4_attention.md) | Model and attention-backend context |
| [Kimi K2.6 INT4 MoE](by_model/kimi_k2.6_int4_moe_mi300x.md) | Model deployment and MoE precision context |
| [Llama FP8 serving](by_model/llama_fp8_serving.md) | Quantized serving and integration |

## Reusable method examples

- [MFMA tile selection](by_kernel/mfma_tile_selection.md): a historical worked example of instruction
  shape, register pressure and occupancy. Its GEMM examples illustrate a method; compare candidates
  for the actual workload instead of treating its old default as a universal ranking.

## Operator case records

- [Dense GEMM DB tuning](../../operator/dense_gemm/case_studies/gemm_aiter_db_tuning.md): captured
  lookup keys, a bias mismatch and the measured Qwen3.5-27B outcome.
- [Hybrid attention backend swap](../../operator/linear_attention_gated_delta/case_studies/gated_delta_backend_swap.md):
  Qwen3.5-27B's full-attention path within a gated-delta model.
- [Norm + quant preparation for scaled GEMM](../../operator/scaled_quant_gemm/case_studies/fused_norm_quant_win.md):
  consumer scale contracts and upstream-reported fusion results.

## Evidence and protocol

Keep measured results, vendor reports and source-only observations distinct. A case should record
context, baseline, attempted changes, unsuccessful attempts, results, limitations and source evidence.
The imported GEAK cases often used a 0.5% band and non-overlapping repeat ranges; those are historical
protocol choices. A new experiment must report its own variability and justified acceptance rule,
with correctness and engagement checked for its actual TP/EP configuration.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/case_studies/README.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
