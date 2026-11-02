# attention_prefill_fmha: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [vLLM V1 ROCM_ATTN chunked-prefill — the live path IS the editable Triton `_fwd_kernel`](e2e-prefill-attn-prefix-triton-gfx950-vllm.md)

- [`--attention-backend triton` — a cheap server-flag win on hybrid models](e2e-attention-backend-triton-gfx942.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
