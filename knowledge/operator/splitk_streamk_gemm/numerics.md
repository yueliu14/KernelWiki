---
title: splitk_streamk_gemm — numerics
kind: technique
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://arxiv.org/abs/2301.03598
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
---
# splitk_streamk_gemm — numerics

## TL;DR
> The decomposition is mathematically identical to dense GEMM, but **the reduction across K-partitions
> changes summation order**, so atomic-add results are non-deterministic and can differ run-to-run; use a
> workspace reduction when you need bitwise reproducibility.

## Considerations
- **Accumulate in fp32** per partition; combine partials in fp32 before casting to out dtype.
- **Atomic reduction**: fast but non-associative across the differing add orders → results vary slightly
  run-to-run and differ from the single-pass dense result within fp rounding. Avoid bf16/fp16 atomics
  (large rounding) — accumulate/atomic in fp32.
- **Workspace reduction**: deterministic given a fixed partition→workspace layout; one extra read/write
  pass. Prefer for reproducibility-gated tests.
- **Stream-K fix-up**: partial output tiles owned by multiple workgroups must be combined exactly once;
  a fix-up bug double-counts or drops K-iterations (correctness, not just precision).
- For fp8/mxfp inputs, scales apply per dense rules before the cross-partition reduction →
  [../scaled_quant_gemm/numerics.md](../scaled_quant_gemm/numerics.md).

### Keeping bitwise parity when changing the split (recorded: small-M decode GEMMs)
- **fp32 is not enough; the combine tree must match.** In a bf16 decode GEMM whose reference used 4-way
  split-K, full-K tiles stayed bit-exact (M=1, 3, 8, 16, 17, 33, 64) only when each workgroup kept 4
  accumulators over the same K/4 slices and summed them as (p0+p1)+(p2+p3), the reference's order.
  A sequential p0+p1+p2+p3 sum, a contiguous full-K loop (0.68x of the tolerance) and a batched 3D
  `tl.dot` over the slices were not bit-exact.
- **Re-tiling is safe if each tile's K chain is unchanged.** BN/BK/warps/stages and grid-order changes
  (e.g. XCD-contiguous tile order) stayed bitwise identical in these runs; each output tile still ran
  its MFMA chain over K in ascending order.

### Deterministic single-launch reduction: correct, not automatically faster
- A fixed-order in-kernel reduction (the last CTA to take an atomic ticket reduces the fp32 partials in
  slice order) was bitwise-exact, but in decode-sized batched/skinny GEMMs it was slower than the
  baseline in every variant tried.
- The engineers attribute the loss to the serialized tail (partial stores, ticket atomic, partial
  loads): about 1–1.7 µs more than the K-work saved in one decode BMM, about 1.3–1.5 µs in another
  decode GEMM. acq_rel ticket ordering was slowest (an L2 writeback and invalidate per CTA); the relaxed
  variant was correct only by hardware behaviour, not by the memory model.
- Scope: microsecond kernels at small M; larger shapes were not measured here.

## Parity gate
- Oracle = single-pass dense fp32 reference. Expect agreement within fp accumulation tolerance; for the
  deterministic gate, use workspace reduction and require bitwise/near-bitwise match across repeats.
- If the reference is itself a split-K kernel, bitwise parity against it requires matching its K-slice
  boundaries and its partial-combine order (see above), not just fp32 accumulation.

## Sources
- Stream-K (reduction/fix-up semantics): https://arxiv.org/abs/2301.03598
- Triton split-K/stream-K reduction modes: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/splitk_streamk_gemm/numerics.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
