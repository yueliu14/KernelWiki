# linear_attention_gated_delta

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Recurrent and chunked gated-delta attention, including its causal-conv front end.

5 AITER implementation references; 6 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Linear attention and gated delta rules — AITER aiter](backend/aiter_aiter.md) |
| flydsl | [Linear attention and gated delta rules — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Linear attention and gated delta rules — AITER gluon](backend/gluon_aiter.md) |
| hip | [Linear attention and gated delta rules — AITER hip](backend/hip_aiter.md) |
| triton | [Linear attention and gated delta rules — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [Causal state mixing inside gated-delta attention](backend/aiter_notes.md)

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: linear_attention_gated_delta — overview
kind: operator_overview
operator: linear_attention_gated_delta
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/flash-linear-attention
  - https://github.com/NVlabs/GatedDeltaNet
  - https://arxiv.org/abs/2412.06464
  - https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
```

</details>

# linear_attention_gated_delta  (Gated DeltaNet / chunked linear-attention scan)

## TL;DR
Gated DeltaNet (Qwen3-Next / Qwen3.5, Kimi-Linear; "Gated Delta Networks: Improving Mamba2 with Delta
Rule", arXiv 2412.06464) is a **linear-attention** layer carrying a fixed-size recurrent **state matrix**
`S ∈ R^{d_k×d_v}` per head, updated with a **gated delta rule** (Mamba2-style exponential decay + DeltaNet
error-correcting update). It is NOT softmax attention — there is no seq² score matrix. The kernel comes in
two regimes: a **chunked scan** for prefill (process chunks of C tokens with intra-chunk parallelism,
O(T/C) sequential steps) and a **fused recurrent** kernel for decode (one state update per token). On AMD
this is **Triton-portable**: the reference is `fla-org/flash-linear-attention` (FLA, used by HF
transformers), and aiter ships an on-box Triton port (`aiter/ops/triton/gated_delta_net/`,
prefill `chunk*` + decode `fused_recurrent*`). The single most important fact: it is **memory-bandwidth
bound on the state**, so the win is **keeping S in registers/LDS** across the update — a decomposed path
that round-trips S to HBM is estimated **10–50× slower**.

## Math contract
Per head, per token `t` (DeltaNet + gating):
```
S_t = (diag(α_t) ⊙ S_{t-1}) + β_t · k_t (v_t - S_{t-1}^T k_t)^T     # gated delta update
o_t = S_t^T q_t                                                       # readout
```
- `α_t` = exponential decay gate (Mamba2-style, per-channel), `β_t` = delta write strength.
- Pre-steps: **causal conv1d** on Q/K/V (local mixing) and **L2-norm on Q/K** (fused in the kernel).
- **Chunked form** (prefill): within a chunk, compute intra-chunk contributions in parallel via a
  WY-representation / `solve_tril` (lower-triangular solve) + cumulative decay (`cumsum`), carry the state
  across chunks. This is what makes prefill O(T/C) not O(T).
- Shapes (aiter `fused_recurrent_gated_delta_rule`): `q,k [B,T,H,K]`, `v [B,T,HV,V]` (GVA if `HV>H`),
  `g [B,T,HV]` decays, `beta`, optional `initial_state`/`output_final_state`, `cu_seqlens` for varlen.
- bf16/fp16 in, fp32 state accumulate.

## Shape regimes
- **Prefill**: chunked scan; chunk size C (commonly 64) trades parallelism vs state-carry cost.
- **Decode**: fused recurrent — one `S` update per token; pure memory-bound on `S` (d_k×d_v per head).
- Hybrid models (Qwen3-Next/3.5, Kimi-Linear): **3:1** ratio — 3 Gated-DeltaNet layers per 1 full-attn
  layer. When 75% of layers are GDN, this kernel **dominates** inference time.

## Where it matters (Amdahl)
On Qwen3-Next-class hybrids GDN is the **majority of layers** → the dominant attention-side cost. SGLang
auto-detects the hybrid layers and uses the optimized GDN kernels; the launch command on AMD uses
`--attention-backend triton`. A decomposed (non-fused) GDN path is estimated 10–50× slower, so the fused
kernel is not optional for serving.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (FLA reference + aiter on-box port; sglang default GDN path) | [backends/triton.md](backend/triton.md) |
| hip | 🟡 (causal-conv1d + state-update glue; no hand-asm GDN scan) | [backends/hip.md](backend/hip.md) |
| tilelang | 🧪 (expressible; no published AMD-tuned GDN kernel) | [backends/tilelang.md](backend/tilelang.md) |
| flydsl | 🧪 experimental (`flydsl_gdr_decode`; import commented out, gfx950-tuned) | [backends/flydsl.md](backend/flydsl.md) |

## Fusion neighbors
Causal conv1d ([causal_conv1d](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/causal_conv1d/README.md)), L2-norm + RoPE-free QK, sigmoid gating, the chunked cumsum/solve_tril,
fp8 state quant (rare). See [fusion.md](fusion.md). The full-attn layers it interleaves with are
[attention_prefill_fmha](../attention_prefill_fmha/README.md) / [gqa_mqa_attention](../gqa_mqa_attention/README.md).

## Numerics
fp32 state accumulate; the recurrent scan accumulates error over T — chunk boundaries and the L2-norm/gate
order matter. See [numerics.md](numerics.md).

## How to bench
Reference (FLA/aiter): B,T,H,K,V with chunk C=64; bench chunked prefill + fused-recurrent decode
separately; oracle = the FLA reference scan in fp32. See [tuning.md](tuning.md).

## Sources
- Gated DeltaNet paper (Mamba2 + delta rule, gating): https://arxiv.org/abs/2412.06464
- FLA (chunked Triton kernels for GDN/DeltaNet/GLA; HF uses it for Qwen3.5): https://github.com/fla-org/flash-linear-attention
- NVlabs GatedDeltaNet (official; optimized FLA kernels): https://github.com/NVlabs/GatedDeltaNet
- AMD Qwen3.5 day-0 (SGLang auto-detects hybrid, optimized GDN kernels, `--attention-backend triton`): https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html
- aiter on-box GDN kernels: `ROCm/aiter@a6bb49937:aiter/ops/triton/gated_delta_net/`, `.../_triton_kernels/gated_delta_rule/{prefill,decode,utils}/`.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/linear_attention_gated_delta/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
