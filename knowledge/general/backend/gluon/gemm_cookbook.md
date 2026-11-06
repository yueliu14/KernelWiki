---
title: Gluon near-peak GEMM cookbook (CDNA4) — reference
kind: language
gens: [gfx950]
dtypes: [fp16, fp8_e5m2, fp4_e2m1]
regimes: [prefill, training, both]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
---
# Gluon GEMM cookbook (CDNA4, gfx950)

## TL;DR
AMD's `gfx950-gluon-tutorials` walks a single FP16 GEMM from **naive 520 TFLOPS @ 25%** (v0) to
**near-peak 1489 TFLOPS @ 98.75% MFMA eff** (v9) in nine steps, then reuses the same design for BF8 and
MXFP4. The recipe = explicit LDS layout → software pipeline → register budgeting → MFMA scheduling
(interleave) → post-hot-loop cleanup. The measured ceilings below are the **practical low-precision limits**
on MI355X.

## The recipe (a16w16 FP16, v0 → v9)
The tutorial's FP16 kernel introduces every technique; BF8 and MXFP4 reuse the same skeleton with a larger
`BLOCK_K` and the scaled-MFMA op. Progression (v0→v9, `kernels/gemm/a16w16/`):
1. **v0 naive** — straight tiled GEMM, 520 TFLOPS @ 25%. Matrix core idle most of the time.
2. **Explicit LDS layouts** — swizzled/padded shared layouts to remove bank conflicts; `ds_read_tr` to feed
   MFMA operands transposed without an extra pass.
3. **Software pipeline** — explicit 2-/3-stage pipeline (global-read → LDS-write → LDS-read → MFMA),
   authored by hand instead of via Triton `num_stages`.
4. **Register budgeting** — keep live values inside the 512-VGPR budget so occupancy/no-spill holds.
5. **MFMA scheduling** — interleave MFMA with memory so the matrix core never stalls (the CDNA
   ping-pong / 4-wave-interleave patterns; see [optimization/mfma_scheduling](../../optimization/mfma_scheduling.md)).
6. **Beyond the hot loop (v9)** — `llirSched` IR interleave + `amdgcnas` post-assembly peephole, enabled
   with `TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1`.

Tile shapes per dtype (from the tutorial repo):
| kernel | dtype | tile (M×N×K) | notes |
|---|---|---|---|
| a16w16 | FP16 | 256×256×64 | foundation; all techniques introduced here |
| a8w8 | BF8 | 256×256×128 | same design, larger `BLOCK_K`, 32-cycle scaled MFMA |
| a4w4 | MXFP4 | 256×256×256 | adds scale pipeline GR→LW→LR, 16-cycle MFMA, LDS port contention |

Benchmark invocation:
```bash
cd kernels/gemm/a16w16
python bench.py --version 0 --K 8192 --dtype fp16     # naive: 520 TFLOPS @ 25%
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 9 --K 8192 --dtype fp16  # near-peak: 1489 TFLOPS @ 98.75%
```

## Measured ceilings (AMD-measured, single MI355, ROCm 7.0, gfx950, 2026)
| kernel | dtype | shape (M×N×K) | TFLOPS | MFMA eff |
|---|---|---|---|---|
| a16w16 | FP16 | 4096×4096×8192 | **1489** | 98.75% |
| a8w8 | BF8 | 4096×4096×16384 | **3257** | 99.72% |
| a4w4 | MXFP4 | 4096×4096×32768 | **5255** | 92.41% |
| v0 baseline | FP16 | (naive) | 520 | 25% |

FP16 v0→v9 is ~3× to peak. BF8 is essentially peak (99.72%). MXFP4 caps lower (92.41%) because of the extra
**scale pipeline (GR→LW→LR)** and LDS port contention from the 8-bit-per-32-element block scale.

## Numerics
FP16/BF8/FP4 operands, **FP32 accumulate**. MXFP4 = two 4-bit values per byte + an 8-bit per-group scale
(every 32 elements), consumed by `v_mfma_scale_f32_16x16x128_f8f6f4` in one instruction.

## Pitfalls
- The ceilings are **gfx950 headline** — CDNA3 (gfx942) lacks native scaled-MFMA, so the MXFP4 5255 number
  does not transfer. FP16/BF8 manual pipelines do run on gfx942.
- These are **large-K, square-ish shapes** (K=8192/16384/32768). Skinny/decode GEMM is a different regime —
  the near-peak ceilings assume compute-bound dimensions; the decode runs below are a different regime with mixed results.
- Manual scheduling is the cost: you author the layout, pipeline, register budget, and schedule by hand —
  far more effort than Triton's autoscheduler for the last ~20–75% of peak.

## Recorded gfx950 runs outside large-K GEMM
Directions that consulted this cookbook on decode, indexer and MoE kernels gained about as often as they lost; each note is one run.
- **LDS-DMA rings did not gain in the recorded decode cases.** A 2-stage `buffer_load_to_shared` K prefetch in a sparse-attention indexer decode ran 0.9% slower than its parent. An LDS-DMA W ring in an MXFP4 MoE decode GEMM at bs2 (HBM-bound per its engineer) lost 0.6 to 3us. An X-resident persistent tier for a latency-bound bf16 GEMM (M=12, K=2048, N=6144) peaked at 0.888x; its engineer attributed it to occupancy (1 CTA per CU; more slabs per CTA was slower). In the indexer decode, `async_copy.wait_group` lowered to a vmcnt that broke parity until replaced by inline-asm `s_waitcnt`.
- **NUM_CHAINS 4→2 gained in one indexer prefill; the engineer attributed it to liveness.** It cut instructions 1.1% and measured 1.0553x; `NUM_CHAINS=1` spilled (79→203).
- **Indexer prefill store relayout stayed scalar.** VGPRs rose 233→243 but the ISA kept `buffer_store_dword`; per the engineer, Triton could not prove 4-column alignment of a per-row causal start.
- **MFMA priority hints, indexer prefill split arm (BLOCK_M=4, 2 warps).** Per the engineer, Gluon (Triton 3.7) has no `s_setprio`/`sched_barrier`/`sched_group_barrier` call; inline-asm `s_setprio` ended with no MFMA in its window (the compiler had spread MFMAs among the VALU fold), and `schedule_hint="memory-bound-attention"` raised scratch 80→600 B.
- **A Gluon twin that staged KV once for both QK and PV gained.** In an fp8 sparse-MLA decode split-K body (bs1/bs2) it read one LDS KV tile as the B operand of QK (scaled MFMA 16x16x128) and PV (MFMA 16x16x32), among other layout changes: +3.9% call-weighted (bs1 8.7%, bs2 10.8%).

## Cross-links
[languages/gluon](overview.md) · [languages/triton_amd](../triton_amd/overview.md) · [optimization/mfma_scheduling](../../optimization/mfma_scheduling.md) ·
[operators/dense_gemm/backends/gluon](../../../operator/dense_gemm/backend/gluon.md) · [operators/scaled_quant_gemm/backends/gluon](../../../operator/scaled_quant_gemm/backend/gluon.md) ·
[operators/quant_fp4_mxfp/backends/gluon](https://github.com/yueliu14/KernelWiki/blob/45cd3f5b51d16322db8dc301fa783148d323d27b/knowledge/operator/quant_fp4_mxfp/backend/gluon.md)

## Sources
- Gluon GEMM tutorial (v0→v9, ceilings, scale pipeline, env vars): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Tutorial code + per-dtype tile sizes: https://github.com/ROCm/gfx950-gluon-tutorials
- Scheduling patterns (ping-pong / interleave): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html ; https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/languages/gluon/gemm_cookbook.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
