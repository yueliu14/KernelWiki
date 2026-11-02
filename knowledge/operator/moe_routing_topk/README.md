# moe_routing_topk

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Expert selection, routing weights, token sorting and alignment for MoE GEMMs.

5 AITER implementation references; 4 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [MoE routing, sorting and token assignment — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [MoE routing, sorting and token assignment — AITER flydsl](backend/flydsl_aiter.md) |
| hip | [MoE routing, sorting and token assignment — AITER hip](backend/hip_aiter.md) |
| opus | [MoE routing, sorting and token assignment — AITER opus](backend/opus_aiter.md) |
| triton | [MoE routing, sorting and token assignment — AITER triton](backend/triton_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: moe_routing_topk — overview
kind: operator_overview
operator: moe_routing_topk
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/topk.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/topk_softmax_kernels_group.cu
  - https://github.com/vllm-project/vllm/pull/17955
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
```

</details>

# moe_routing_topk  (router gating → top-k expert select → expert histogram/sort)

## TL;DR
The MoE router turns the gate logits `[T, E]` into, per token, the **top-k expert ids + normalized
weights**, then builds the **expert histogram + sorted/padded token→expert layout** the grouped GEMM
consumes. It is tiny in FLOPs but **latency-critical and on the decode critical path** (it serializes
before the expensive grouped GEMM, can't overlap it). The single most important fact: the **align&sort**
step (`moe_align_block_size`), not the softmax, is the kernel that historically dominated router time —
AMD/SGLang's multi-block rewrite got **7× on MI300X** there. On MI300X the whole router is memory-bound,
and its multi-die (XCD) behavior makes naive grid sizing actively harmful.

## Math contract
Two sub-ops, run back-to-back:

1. **gating + top-k select** (`[T,E] → topk_ids[T,k] int32, topk_weights[T,k] fp32`)
   - score: `softmax(logits)` (Mixtral/Qwen-MoE) **or** `sigmoid(logits)` (DeepSeek-V3/R1, Kimi-K2).
   - **grouped/biased top-k** (DeepSeek): add a per-expert `correction_bias`, pick `topk_group` of
     `num_expert_group` groups by their group score (sum of top-2 in group), then top-k experts within
     the chosen groups; optionally `renorm` the k weights to sum 1 and multiply by `routed_scaling_factor`.
   - dtype: logits bf16/fp16/fp32 in; reduction in **fp32**; ids int32, weights fp32.
2. **expert histogram + align&sort** (`topk_ids → sorted_token_ids, expert_ids, num_tokens_post_pad`)
   - count tokens per expert, then produce a token permutation padded so each expert's run is a multiple
     of `BLOCK_M` (so the grouped GEMM sees contiguous, aligned per-expert tiles). This is
     `moe_align_block_size` (SGLang/vLLM) / `moe_sorting` (aiter). It is the seam to
     [moe_dispatch_combine](../moe_dispatch_combine/README.md) (EP) and [fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md) (single-GPU).

## Shape regimes
- `T` = tokens in the batch (prefill: chunk×batch, 1k–16k; decode: running batch 1–256).
- `E` ∈ {8 (Mixtral), 128 (Qwen3-MoE), 256 (DeepSeek-V3/R1, Kimi-K2)}; `k` ∈ {2, 6, 8}.
- The kernels must scale to **E=256, k=8** — the DPP softmax/sigmoid kernels were specifically extended
  past CK's old 192-expert cap (aiter #1909). align&sort is tuned for `MAX_EXPERT_NUMBER=256`.

## Where it matters (Amdahl)
Router FLOPs are negligible, but on **decode** the router + align&sort is a **serial latency tax** before
the grouped GEMM — it can't hide behind compute. Pre-optimization, `moe_align_block_size` alone was a
visible % of MoE-layer time; the SGLang multi-block rewrite cut it **~7× on MI300X / 10× on MI100**
(memory-bound op — fewer XCD-crossing syncs win). On prefill the router is amortized into the large
grouped GEMM and matters less.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (biased_grouped_topk HIP + DPP, moe_sorting) | [backends/aiter.md](backend/aiter.md) |
| hip | 🟢 sota (the actual kernels: `moe_align_block_size_kernels.cu`, `moe_fused_gate.cu`, `topk_softmax_kernels_group.cu`) | [backends/hip.md](backend/hip.md) |
| triton | 🟡 competitive (vLLM/sglang fused-MoE Triton routing; editable fallback) | [backends/triton.md](backend/triton.md) |
| vllm_kernels | 🟢 (registers aiter routing ops + has its own Triton/CUDA `topk_softmax`/`grouped_topk`) | [backends/vllm_kernels.md](backend/vllm_kernels.md) |

## Fusion neighbors
- **softmax/sigmoid + top-k + bias** fused into one kernel (`moe_fused_gate` / `biased_grouped_topk`).
- **shared-expert into routing**: treat shared experts as synthetic always-selected experts that get
  top-k slots → one fused dispatch for shared+routed (see [shared_expert_fusion](../shared_expert_fusion/README.md) and
  [fusion.md](fusion.md)).
- **routed-weight multiply pushed downstream** into the grouped-GEMM epilogue (`MulRoutedWeight1`) or the
  combine kernel (`doweight_stage1`) — the router only emits weights; the multiply lands later.
- align&sort output **directly feeds** the dispatch ([moe_dispatch_combine](../moe_dispatch_combine/README.md)) or grouped GEMM
  ([fused_moe_grouped_gemm](../fused_moe_grouped_gemm/README.md)).

## Numerics
fp32 reduction for the gate; **argmax tie-breaks** can flip an expert id between backends (benign
numerical-equivalence-class, gate with a parity probe, not byte parity). **The trap (aiter #2153):** the
biased-grouped HIP kernel **hardcodes sigmoid** — using it for a softmax-scored model gives wrong weights.
See [numerics.md](numerics.md).

## How to bench
Isolated: aiter `op_tests/test_moeTopkSoftmax.py` (`biased_grouped_topk` vs torch ref) and
`test_moe_topk_sigmoid.py` (E=64/128/256). align&sort: time `moe_align_block_size` standalone at your
(T, E, k, BLOCK_M). e2e: a MoE model (DeepSeek/Qwen-MoE) tok/s with `VLLM_ROCM_USE_AITER=1` vs `=0`
(falls back to Triton routing). Oracle = greedy/temp=0 expert-id + weight match within tolerance.

## Sources
- aiter routing entrypoints (`biased_grouped_topk`, `grouped_topk`, `moe_fused_gate`, `top_k_per_row_*`):
  `ROCm/aiter@a6bb49937:aiter/ops/topk.py`; HIP kernels `csrc/kernels/topk_softmax_kernels_group.cu`,
  `moe_align_block_size_kernels.cu`, `moe_fused_gate.cu` (on-box `/sgl-workspace/aiter`).
- Biased group top-k for DeepSeek-V3 in vLLM: https://github.com/vllm-project/vllm/pull/17955
- align&sort multi-block rewrite (7× MI300X, XCD/multi-die design): https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang ; https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- DPP topk_sigmoid (256 experts, fp32, 1.66×): https://github.com/ROCm/aiter/pull/1909

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/moe_routing_topk/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
