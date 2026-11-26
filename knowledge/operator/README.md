# GEMM, MoE, attention and indexer kernels

Agents: [AGENTS.md](../../AGENTS.md) maps a kernel to its operator directory.

Each operator directory has `README.md` (overview, regimes), `cards.md` (decision cards rendered from
`corpus/decisions.yaml`; open the ones matching your kernel language), `tuning.md`, `numerics.md`, `fusion.md`, and
`backend/` pages per language: `<lang>.md` holds dated GEAK notes, `<lang>_aiter.md` walks the pinned AITER source.
The last column counts the cards an operator owns; shared cards are listed at the end of each `cards.md`.

Standalone normalization, activation, softmax, positional encoding, quantization and copy kernels have no operator
of their own: their contracts live with the kernel that consumes them (see the fusion operators below), and the
methods in `general/optimization/` apply directly.

AITER is the primary implementation source. A source reference is not a measured winner.

## GEMM and projection fusions

| Operator | Inference focus | AITER references | Cards |
| --- | --- | --- | ---: |
| [dense_gemm](dense_gemm/README.md) | Dense attention and FFN projections; prefill versus decode tiling. | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 19 |
| [scaled_quant_gemm](scaled_quant_gemm/README.md) | FP8, INT8, FP4 and microscaled projections, including packing and scale contracts. | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 12 |
| [batched_gemm](batched_gemm/README.md) | Batched matrix products and head-wise projection variants. | aiter, ck, flydsl, gluon, opus, triton | 2 |
| [skinny_gemv_decode](skinny_gemv_decode/README.md) | Small-M decode projections and GEMV-like regimes. | hip | 2 |
| [splitk_streamk_gemm](splitk_streamk_gemm/README.md) | Split-K and Stream-K schedules for poorly occupied GEMMs. | asm, flydsl, opus, triton | 3 |
| [gemm_epilogue_fused](gemm_epilogue_fused/README.md) | GEMM with activation, residual, normalization/quantization and communication boundaries. | aiter, triton | 4 |

## MoE and expert parallelism

| Operator | Inference focus | AITER references | Cards |
| --- | --- | --- | ---: |
| [grouped_gemm_moe](grouped_gemm_moe/README.md) | Expert grouped GEMMs, token imbalance and low-bit expert weights. | aiter, asm, ck, flydsl, gluon, hip, opus, triton | 8 |
| [fused_moe_grouped_gemm](fused_moe_grouped_gemm/README.md) | Fused expert execution and compute/communication overlap. | aiter, flydsl | 1 |
| [moe_routing_topk](moe_routing_topk/README.md) | Expert selection, routing weights, token sorting and alignment for MoE GEMMs. | aiter, flydsl, hip, opus, triton | 4 |
| [moe_dispatch_combine](moe_dispatch_combine/README.md) | Expert-parallel dispatch, gather, reduction and combine. | aiter, flydsl, hip | 4 |
| [shared_expert_fusion](shared_expert_fusion/README.md) | Shared-expert computation and its overlap or fusion with routed experts. | flydsl | 2 |

## Attention and its fusions

| Operator | Inference focus | AITER references | Cards |
| --- | --- | --- | ---: |
| [attention_prefill_fmha](attention_prefill_fmha/README.md) | Forward FMHA for prompt processing, including variable-length causal attention. | aiter, asm, flydsl, gluon, hip, triton | 3 |
| [attention_decode_paged](attention_decode_paged/README.md) | Decode attention over paged KV caches, split/reduce and mixed precision. | asm, flydsl, gluon, hip, triton | 7 |
| [mla_attention](mla_attention/README.md) | Multi-head latent attention, compressed KV and decode/prefill variants. | aiter, asm, flydsl, gluon, hip, opus, triton | 2 |
| [gqa_mqa_attention](gqa_mqa_attention/README.md) | Grouped-query and multi-query head sharing. | triton | 3 |
| [sparse_attention_nsa](sparse_attention_nsa/README.md) | Selected-block and sparse MLA/NSA attention execution. | aiter, gluon, hip, opus, triton | 2 |
| [compressed_attention](compressed_attention/README.md) | Compression stages and fused attention over compressed representations. | flydsl | 1 |
| [linear_attention_gated_delta](linear_attention_gated_delta/README.md) | Recurrent and chunked gated-delta attention, including its causal-conv front end. | aiter, flydsl, gluon, hip, triton | 6 |
| [chunked_prefill](chunked_prefill/README.md) | Prompt chunks attending to cached prefixes. | triton | 1 |
| [sliding_window_attention](sliding_window_attention/README.md) | Window-limited attention and local/global layer combinations. | triton | 3 |
| [context_parallel_attention](context_parallel_attention/README.md) | Attention split across context-parallel ranks and output merging. | aiter | 1 |
| [qk_norm_rope_cache](qk_norm_rope_cache/README.md) | Fused Q/K normalization, positional encoding, quantization and cache writes. | aiter, flydsl, gluon, hip, triton | 2 |

## Attention indexer

| Operator | Inference focus | AITER references | Cards |
| --- | --- | --- | ---: |
| [attention_indexer](attention_indexer/README.md) | MQA logits, index scores, block selection and top-k for sparse attention. | aiter, flydsl, gluon, hip, opus, triton | 9 |
