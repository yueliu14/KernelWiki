---
title: splitk_streamk_gemm — tuning
kind: technique
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - https://arxiv.org/abs/2301.03598
---
# splitk_streamk_gemm — tuning

## TL;DR
> The decision tree: compute `output_tiles = ceil(M/BM)*ceil(N/BN)`; if it's far below CU count, enable
> stream-K (or split-K for large-K) and size the persistent grid ≈ CU count (304 MI300X / 256 MI350X).
> Otherwise leave it off.

## The levers
- **SPLIT_K** (split-K): number of K-partitions. Larger SPLIT_K → more parallelism but more reduction
  cost; tune 2/4/8/16 against large-K small-M shapes.
- **stream-K grid / `NUM_SMS`**: persistent workgroup count, set ≈ CU count so the flattened work divides
  evenly; this is what kills wave-quantization tails.
- **BLOCK_M/N/K**: smaller tiles raise the tile count (may avoid needing split-K) but lower MFMA
  efficiency; co-tune with the split/stream decision.
- **Reduction mode**: `atomic` (in-place add, lower latency, non-deterministic) vs `workspace +
  reduction kernel` (deterministic, extra pass). See [numerics.md](numerics.md).
- **mfma_16x16 vs 32x32**: 16x16 for small-M skinny tiles (fewer wasted lanes), 32x32 for throughput.
- **num_stages / num_warps / waves_per_eu** (triton): standard pipeline/occupancy.

## Heuristic
- Split-K when K is large relative to M·N and tiles ≪ CUs.
- Stream-K when tiles are slightly above or below a CU-count multiple (wave quantization) — it balances
  the remainder.
- Neither when tiles ≫ CUs (plain dense GEMM, [../dense_gemm/overview.md](../dense_gemm/README.md)).

## Recorded results (small-M decode)
- **M tile covers M**: in a Triton bf16 split-K GEMM (SK=8), one BM=32 tile for 16<M≤32 instead of two
  BM=16 tiles: M=24 12.0→9.9 us, M=32 12.5→10.8 us (engineer: weight streamed once). Retuning at BM=32
  did not help.
- **Waves per CTA**: in a bf16 split-4 BM=32 decode tile (K=6144, N=2624), 4→8 waves (N-split) measured
  +2.6% weighted, mostly at the larger decode batches; the smallest were flat.
- **Join critical path**: in the same GEMM, two join changes measured +2.5–4.6% at the smallest M and
  were built on: reading the split-K signal before the on-chip reduction, and moving the arrival-counter
  increment and reset after the output atomics. An early semaphore increment was flat (engineer's ISA:
  the compiler still waited on it at once). Narrowing join stores from system to agent scope gave no gain.
- **Tuned tiers for untuned keys**: adding split-K tiers (split 2–3, existing counter protocol; tile,
  layout and warps chosen together) to two mxfp8 block-32 GEMM keys on the default config (N=5120 K=15360;
  N=12800 K=6144) gave site-level decode speedups of 1.906x and 1.127x: tuned vs untuned, not split-K vs
  none. The official weighted result was 1.0137, below its 1.015 target.

## Pitfalls
- Over-splitting K turns a compute-bound GEMM into a reduction-bound one — net slowdown.
- Atomic reduction on bf16/fp16 output risks precision loss and contention; prefer fp32 atomics or
  workspace.
- At bf16 M=3 and M=6 (K=6144, N=2624), the aiter skinny GEMV kernels screened 2.19x slower or worse
  than a tuned split-K tile.
- A Triton intra-CTA K split via 3D batched `tl.dot` (bf16, K=2048, N=6144) measured 0.72–0.74x at M=12
  across KSPLIT variants and 0.708x at M=9; BK256 was 0.55x, a 2D-load reshape 0.49x. The engineer blamed a
  swizzled rather than padded LDS layout, from a KSPLIT=1 control of the same 3D kernel (7.11 vs 4.07 us,
  identical MFMA/VMEM counts).

## Verify
- Bench split/stream on vs off for each target shape; accept only on a non-overlapping win vs plain dense.
- Alternate same-window A/B pairs against the parent; confirm with a dispatch probe that the new config
  runs. Noise is setup-specific: ~1–2 points in the bf16 BM=32 split-4 decode A/B runs, 0.984–0.998
  identical-code band in the mxfp8 run.

## Sources
- Triton stream-K / persistent matmul: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- Stream-K paper: https://arxiv.org/abs/2301.03598

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/splitk_streamk_gemm/tuning.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
