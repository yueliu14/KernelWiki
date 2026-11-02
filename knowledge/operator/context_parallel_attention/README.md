# context_parallel_attention

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Attention split across context-parallel ranks and output merging.

1 AITER implementation references; 0 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Round-robin context-parallel MLA and index selection](backend/aiter_aiter.md) |

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: context_parallel_attention — overview
kind: operator_overview
operator: context_parallel_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
```

</details>

# context_parallel_attention  (ring / all-to-all context parallel)

## TL;DR
Context (sequence) parallelism splits a **single long sequence across GPUs** so that the KV cache and the
O(N²) attention compute are sharded — essential when one device's HBM can't hold a 256K+ context.
**Ring attention** keeps Q local and **rotates K/V blocks around a GPU ring** (overlapping the
point-to-point KV send with the local FA tile), accumulating with online-softmax LSE-merge.
**All-to-all (Ulysses)** instead shards the sequence and does an all-to-all to switch to head-parallel for
the attention, then all-to-all back. On AMD this is **two layers**: the local attention tile is the same
FA kernel ([attention_prefill_fmha](../attention_prefill_fmha/README.md) via Triton/aiter), and the **partial-result merge across ranks**
(LSE rescale) + the **collective** (P2P ring / all-to-all). The single most important fact on MI300X: the
collective rides **Infinity Fabric / XGMI (~448 GB/s/dir)**, slower than NVLink, so **stay within one
8-GPU XGMI island (TP) before reaching for CP**, and use `allgather_reducescatter`, not NCCL all-to-all.

## Math contract
Standard attention `O = softmax(QKᵀ·scale + mask)V`, computed with the sequence sharded across `cp` ranks.
- **Ring**: rank `r` holds `Q_r, K_r, V_r`. Over `cp` steps each rank computes a local FA tile against the
  currently-held K/V block, **merges** into a running `(O, m, l)` via online-softmax LSE, then **sends**
  its K/V block to the next rank and receives the previous. Causal load-balancing uses a **zigzag**
  assignment so every rank does equal work.
- **All-to-all (Ulysses)**: all-to-all on the head dimension → each rank gets all tokens for a subset of
  heads → local full-seq FA → all-to-all back. One collective per direction, larger messages.
- The cross-rank merge is the FA **LSE-rescale** (`merge_state` / `merge_attn_states`): combine partial
  outputs with their log-sum-exp, fp32.

## Shape regimes
- **Prefill only** (TTFT for very long context, 128K–1M). Decode generally uses TP/paged KV, not CP.
- The local tile is a normal FA prefill at `seq/cp`; the operator's new cost is the **ring P2P / all-to-all
  + LSE merge** overlapped with compute. CP shines when `seq/cp` is still large (compute hides comm).

## Where it matters (Amdahl)
Long-context prefill is O(N²); CP is what makes 256K+ TTFT feasible when KV exhausts HBM. On MI300X the
limiter is interconnect: a poorly-overlapped ring exposes XGMI latency and tanks scaling. The win comes
from (a) overlapping the KV rotation with the local FA tile, (b) zigzag causal balancing, (c) keeping CP
inside the 8-GPU island. ROCm long-seq FA already lags CUDA ~20–25% at 32K+, so the local kernel must be
tuned too ([attention_prefill_fmha](../attention_prefill_fmha/README.md)).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 (local FA tile + LSE merge; the portable CP kernel layer) | [backends/triton.md](backend/triton.md) |
| aiter | 🟢 (local FA + `merge_attn_states` + Iris GPU-initiated comm primitives) | [backends/aiter.md](backend/aiter.md) |
| sglang_kernels | 🟡 (zigzag ring-attention + split-KV transfer; PD-disagg, maturing on ROCm) | [backends/sglang_kernels.md](backend/sglang_kernels.md) |

## Fusion neighbors
LSE-merge of partial outputs ([reduction](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/reduction/README.md)), the collective ([all_to_all_dispatch_combine](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/all_to_all_dispatch_combine/README.md) / allgather /
reduce-scatter), KV transfer for PD-disaggregation, fp8 KV. See [fusion.md](fusion.md).

## Numerics
LSE-merge must be associative & fp32; ring vs all-to-all must match a single-GPU reference. See
[numerics.md](numerics.md).

## How to bench
Long-context prefill (e.g. 128K, cp=8 on one node); measure TTFT scaling vs cp and comm/compute overlap;
oracle = single-GPU full attention at the same seq (if it fits) or a CP-disabled reference at shorter seq.
See [tuning.md](tuning.md).

## Sources
- SGLang prefill CP / zigzag ring attention + split-KV transfer (256K+, PD-disagg): https://github.com/sgl-project/sglang/issues/22223
- ROCm: stay within 8-GPU XGMI island (TP), `allgather_reducescatter` not NCCL all-to-all: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; vLLM ROCm guidance
- aiter Iris GPU-initiated comm (reduce-scatter/all-gather), long-context up to 2×: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- vLLM ROCm extend/chunked-context path, long-seq gap: https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/context_parallel_attention/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
