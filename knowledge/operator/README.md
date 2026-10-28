# GEMM, MoE, attention and indexer knowledge

The operator collection focuses on the main LLM inference workloads: GEMM, MoE, attention and attention indexers, including their variants and fusions. Start with a workload family, then follow its backend implementations and run-derived lessons.

For execution chains spanning these operators, see [workload knowledge](../workload/README.md), including [QKV projection and preparation](../workload/qkv_projection_norm_rope_cache/README.md) and the [MoE expert pipeline](../workload/moe_expert_pipeline/README.md). Workload directories are named by semantic composition; their cases record model-specific inputs, outputs, dispatch and serving behavior.

Standalone normalization, activation, softmax, positional encoding, quantization, sampling, copy/transpose, scans and generic collectives are outside this operator index. Their relevant contracts belong with the main kernel that consumes them. MoE routing/dispatch and Q/K preparation fusions remain because they define the corresponding workload.

AITER is the primary implementation source. Provider pages describe dispatch; backend pages cover Triton, Gluon, FlyDSL, HIP, CK, ASM or OPUS where an applicable source reference exists. A source reference is not a measured winner. Historical GEAK notes may cite excluded operators through pinned archived documents.

All 23 workload families have pinned AITER implementation references. A single upstream kernel can serve several variants, such as grouped-query, chunked-prefill and sliding-window attention; those pages explain different contracts rather than count the shared source as separate kernels. Native pipeline notes distinguish registered public routes from source-only bodies.

## GEMM and projection fusions

| Operator | Inference focus | AITER references | Learned cards |
| --- | --- | --- | ---: |
| [dense_gemm](dense_gemm/README.md) | Dense attention and FFN projections; prefill versus decode tiling. | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 16 |
| [scaled_quant_gemm](scaled_quant_gemm/README.md) | FP8, INT8, FP4 and microscaled projections, including packing and scale contracts. | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 13 |
| [batched_gemm](batched_gemm/README.md) | Batched matrix products and head-wise projection variants. | aiter, ck, flydsl, gluon, opus, triton | 0 |
| [skinny_gemv_decode](skinny_gemv_decode/README.md) | Small-M decode projections and GEMV-like regimes. | hip | 0 |
| [splitk_streamk_gemm](splitk_streamk_gemm/README.md) | Split-K and Stream-K schedules for poorly occupied GEMMs. | asm, flydsl, opus, triton | 0 |
| [gemm_epilogue_fused](gemm_epilogue_fused/README.md) | GEMM with activation, residual, normalization/quantization and communication boundaries. | aiter, triton | 3 |

## MoE and expert parallelism

| Operator | Inference focus | AITER references | Learned cards |
| --- | --- | --- | ---: |
| [grouped_gemm_moe](grouped_gemm_moe/README.md) | Expert grouped GEMMs, token imbalance and low-bit expert weights. | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 37 |
| [fused_moe_grouped_gemm](fused_moe_grouped_gemm/README.md) | Fused expert execution and compute/communication overlap. | aiter, flydsl | 0 |
| [moe_routing_topk](moe_routing_topk/README.md) | Expert selection, routing weights, token sorting and alignment for MoE GEMMs. | aiter, flydsl, hip, opus, triton | 4 |
| [moe_dispatch_combine](moe_dispatch_combine/README.md) | Expert-parallel dispatch, gather, reduction and combine. | aiter, flydsl, hip | 0 |
| [shared_expert_fusion](shared_expert_fusion/README.md) | Shared-expert computation and its overlap or fusion with routed experts. | flydsl | 0 |

## Attention and its fusions

| Operator | Inference focus | AITER references | Learned cards |
| --- | --- | --- | ---: |
| [attention_prefill_fmha](attention_prefill_fmha/README.md) | Forward FMHA for prompt processing, including variable-length causal attention. | aiter, asm, flydsl, gluon, hip, triton | 1 |
| [attention_decode_paged](attention_decode_paged/README.md) | Decode attention over paged KV caches, split/reduce and mixed precision. | asm, flydsl, gluon, hip, triton | 35 |
| [mla_attention](mla_attention/README.md) | Multi-head latent attention, compressed KV and decode/prefill variants. | aiter, asm, flydsl, gluon, hip, opus, triton | 0 |
| [gqa_mqa_attention](gqa_mqa_attention/README.md) | Grouped-query and multi-query head sharing. | triton | 5 |
| [sparse_attention_nsa](sparse_attention_nsa/README.md) | Selected-block and sparse MLA/NSA attention execution. | aiter, gluon, hip, opus, triton | 2 |
| [compressed_attention](compressed_attention/README.md) | Compression stages and fused attention over compressed representations. | flydsl | 0 |
| [linear_attention_gated_delta](linear_attention_gated_delta/README.md) | Recurrent and chunked gated-delta attention, including its causal-conv front end. | aiter, flydsl, gluon, hip, triton | 6 |
| [chunked_prefill](chunked_prefill/README.md) | Prompt chunks attending to cached prefixes. | triton | 0 |
| [sliding_window_attention](sliding_window_attention/README.md) | Window-limited attention and local/global layer combinations. | triton | 0 |
| [context_parallel_attention](context_parallel_attention/README.md) | Attention split across context-parallel ranks and output merging. | aiter | 0 |
| [qk_norm_rope_cache](qk_norm_rope_cache/README.md) | Fused Q/K normalization, positional encoding, quantization and cache writes. | aiter, flydsl, gluon, hip, triton | 0 |

## Attention indexer

| Operator | Inference focus | AITER references | Learned cards |
| --- | --- | --- | ---: |
| [attention_indexer](attention_indexer/README.md) | MQA logits, index scores, block selection and top-k for sparse attention. | aiter, flydsl, gluon, hip, opus, triton | 0 |
