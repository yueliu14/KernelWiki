# dense_gemm

[Corpus decisions](corpus/decisions.md) · [Source evidence](corpus/source_evidence.md) · [Extraction scope](corpus/scope.yaml)


Dense attention and FFN projections; prefill versus decode tiling.

8 AITER implementation references; 16 historical experience cards.

## AITER implementations

Each page pins its source revision. Source references describe implementations and validation routes; they do not establish a measured performance ranking.

| Backend / provider entry | Reference |
| --- | --- |
| aiter | [Dense matrix multiplication — AITER aiter](backend/aiter_aiter.md) |
| asm | [Dense matrix multiplication — AITER asm](backend/asm_aiter.md) |
| ck | [Dense matrix multiplication — AITER ck](backend/ck_aiter.md) |
| flydsl | [Dense matrix multiplication — AITER flydsl](backend/flydsl_aiter.md) |
| gluon | [Dense matrix multiplication — AITER gluon](backend/gluon_aiter.md) |
| hip | [Dense matrix multiplication — AITER hip](backend/hip_aiter.md) |
| opus | [Dense matrix multiplication — AITER opus](backend/opus_aiter.md) |
| triton | [Dense matrix multiplication — AITER triton](backend/triton_aiter.md) |

### Implementation notes

- [AITER FlyDSL A16W16: useful tiles and occupancy-aware policy](backend/flydsl_notes.md)
- [AITER OPUS: exact kernel ids and shape-driven dispatch](backend/opus_notes.md)

## Optimization experience

[Learned knowledge](learned_knowledge/README.md) — applicability, outcomes and original evidence.

## GEAK reference material

The material below is imported from GEAK. Its backend status and performance statements belong to the original source dates. Use the pinned AITER references where available to check current applicability.

<details>
<summary>Original GEAK metadata</summary>

```yaml
title: dense_gemm — overview
kind: operator_overview
operator: dense_gemm
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - ROCm/aiter@HEAD:aiter/tuned_gemm.py
```

</details>

# dense_gemm  (`C = A · Bᵀ [+ bias] [+ act]`)

## TL;DR
Dense GEMM is the Amdahl head of LLM prefill (~78–81% of GPU time). On MI300X the **executed** kernels
are hipBLASLt Tensile `Cijk_*`, but on sglang/vllm they are **dispatched by aiter** — so the live tuning
lever is aiter's per-shape DB, *not* PyTorch TunableOp. Use `mfma_16x16`, ≥1024 workgroups, 8-multiple
tiles; expect ~45–55% of peak, not peak.

## Math contract
`D[M,N] = A[M,K] · B[N,K]ᵀ + bias[N]` (sglang `nn.Linear` → `transpose_b=true`). dtype: bf16/fp16 in,
fp32 accumulate, bf16 out; quant variants take fp8/fp4 inputs + scales. **The `bias` flag is part of the
lookup key** for aiter's tuned DB (most live GEMMs are `bias=false`).

## Shape regimes (typical LLM, hidden 5120 / inter 17408)
- **prefill (large M ~ chunk×batch, 1k–16k)**: up/gate `K=5120, N∈{14336,16384,34816}`; down/qkv
  `N=5120, K∈{17408,6144}`. This is where the 78–81% lives.
- **decode (M = running batch ≤ conc, 1..256)**: same N/K, skinny M → split-K / skinny kernels matter.

## Where it matters (Amdahl)
A 1.15× on the ~80% GEMM mass ≈ +10% e2e ceiling. Even a blended 1.03× clears a 0.5% gate. This is the
single biggest lever — but the library is already near-optimal, so real wins are modest (measured
**+2.23% e2e** from a full bias-correct aiter tune on Qwen3.5-27B; see backends/aiter.md).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live path) | [backends/aiter.md](backend/aiter.md) |
| hipblaslt | 🟢 sota (executed kernels) | [backends/hipblaslt.md](backend/hipblaslt.md) |
| flydsl | 🟢 sota (authorable, mixed-precision/MoE) | [backends/flydsl.md](backend/flydsl.md) |
| triton | 🟡 competitive (loses to tuned hipBLASLt on plain GEMM) | [backends/triton.md](backend/triton.md) |
| tilelang | 🟡 (~0.94–1.05× of Triton GEMM on MI300X) | backends/tilelang.md (P2) |
| ck / asm | 🟢 (asm = peak; CK templates) | backends/ck.md, backends/asm.md (P2) |
| rocblas | 🟡 (small/odd M; raced by TunableOp) | backends/rocblas.md (P2) |

## Fusion neighbors
`+bias`, `+silu/gelu act_and_mul` (up/gate epilogue), `+residual add`, `+fp8 quant` epilogue → see
[fusion.md](fusion.md). Fusing the up/gate GEMM with `act_and_mul` removes a separate [M,34816] pass.

## Numerics
hipBLASLt solution swaps are same-math bf16 (parity-safe, tuner err_ratio<0.05). Quant variants need a
task-accuracy gate, not byte parity. See [numerics.md](numerics.md). **fp8 a8w8 block-scale
(per-`[128,128]`, arbitrary fp32) has a *representational* parity trap on gfx950 — the E8M0 scaled-MFMA
can't hold an arbitrary fp32 scale, so it needs a software fp32 post-MFMA scale ([numerics.md](numerics.md)).**

## How to bench
Isolated: `op_bench.py`/gradlib on the exact (M,N,K,bias,dtype). e2e: same-session 2-launch A/B at the
target ISL/OSL/conc, gate on delta>0.5% AND non-overlapping AND engagement proven.

## Sources
- MI300X tuning levers (mfma_16x16, ≥1024 WGs, 8-multiple tiles, 512B Tagram hotspot): ROCm workload guide.
- Live GEMM dispatch through aiter: `ROCm/aiter@HEAD:aiter/tuned_gemm.py` (on-box `/sgl-workspace/aiter`).
- +2.23% measured: perf_knowledge e2e validation run 2026-06-08 (see backends/aiter.md).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/dense_gemm/overview.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
