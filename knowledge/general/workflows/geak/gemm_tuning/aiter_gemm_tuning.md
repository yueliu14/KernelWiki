# Historical AITER GEMM tuning recipe

Use the maintained [GEMM tuning workflow](../../gemm_tuning_workflow.md) for capture, candidate
selection, deployment and validation. The [dense GEMM corpus](../../../../operator/dense_gemm/corpus/decisions.md#gemm-shape-regime-routing)
retains the reusable dispatch and bias-key lessons and their scoped historical experience.

The original recipe described Qwen3.5-27B with SGLang 0.5.11 on gfx942 in June 2026. Its observed
caller used AITER's database and bypassed PyTorch BLAS dispatch. That identifies the tuning surface
for that caller; it does not make the database the only possible optimization on other stacks.
A partial database preserves untouched fallback cases but can regress the shapes it replaces.

The [original GEAK document](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md)
contains the archived measurements and run-specific commands. This page preserves attribution and
navigation; new conclusions go into corpus and new run artifacts remain private.
