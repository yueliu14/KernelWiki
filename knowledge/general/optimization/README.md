---
title: optimization — index
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
---

# optimization — cross-operator performance techniques (MI300X / MI350X)

Hardware-grounded optimization levers that apply across operators. Each file is one technique,
tied to CDNA3 (gfx942 / MI300X) and CDNA4 (gfx950 / MI350X·MI355X) hardware facts and to AMD docs.
For per-operator application see [operators/dense_gemm/tuning.md](../../operator/dense_gemm/tuning.md) (the canonical style example) and the
operator cards; for raw hardware numbers see [hardware/cdna3_mi300/arch.md](../hardware/cdna3_mi300/arch.md) and [hardware/cdna4_mi350/arch.md](../hardware/cdna4_mi350/arch.md).

## The hierarchy of levers (apply top-down)

1. **Pick the right algorithm / fusion** — [optimization/kernel_fusion_strategy.md](kernel_fusion_strategy.md),
   [optimization/roofline_and_bottlenecks.md](roofline_and_bottlenecks.md) (classify the kernel first; do not tune a bandwidth-bound
   kernel as if it were compute-bound).
2. **Feed the matrix cores** — [optimization/mfma_scheduling.md](mfma_scheduling.md),
   [optimization/memory_pipelining.md](memory_pipelining.md), [optimization/lds_and_bank_conflicts.md](lds_and_bank_conflicts.md).
3. **Get parallelism right** — [optimization/wave_and_grid_sizing.md](wave_and_grid_sizing.md),
   [optimization/occupancy_and_registers.md](occupancy_and_registers.md), [optimization/xcd_l2_locality.md](xcd_l2_locality.md).
4. **Get the memory subsystem right** — [optimization/vectorization_and_coalescing.md](vectorization_and_coalescing.md),
   [optimization/memory_pipelining.md](memory_pipelining.md), [optimization/xcd_l2_locality.md](xcd_l2_locality.md).
5. **Search the residual** — [optimization/autotuning_methodology.md](autotuning_methodology.md) (choose the tuner for the dispatch path actually observed; see
   [GEMM tuning](../workflows/gemm_tuning_workflow.md) for a reusable capture/tune/deploy procedure).
6. **Keep it correct** — [optimization/numerical_stability.md](numerical_stability.md).

## Files in this section

| file | what it covers |
|---|---|
| [optimization/occupancy_and_registers.md](occupancy_and_registers.md) | 512 VGPR/EU, 16-granule alloc, AGPR pool, occupancy vs spilling, waves/EU, `num_warps`/`waves_per_eu` |
| [optimization/lds_and_bank_conflicts.md](lds_and_bank_conflicts.md) | 64KB (CDNA3) / 160KB (CDNA4) LDS, 32 banks, padding, XOR swizzle, double-buffer |
| [optimization/mfma_scheduling.md](mfma_scheduling.md) | 16×16 vs 32×32 MFMA, AGPR accumulators, issue cadence, latency hiding, `OPTIMIZE_EPILOGUE`, 512B Tagram |
| [optimization/memory_pipelining.md](memory_pipelining.md) | `global_load_lds` / async copy, software pipelining, `num_stages`, prefetch, ds_read/ds_write overlap, 128-bit GLOBAL_LOAD_LDS on CDNA4 |
| [optimization/vectorization_and_coalescing.md](vectorization_and_coalescing.md) | 128-bit `dwordx4` loads, alignment, grid-stride, coalesced access |
| [optimization/xcd_l2_locality.md](xcd_l2_locality.md) | 8-XCD MI300X, XCD-aware grid, L2 partition, ≥1024 workgroups, 8-multiple tiles, swizzled CTA order |
| [optimization/wave_and_grid_sizing.md](wave_and_grid_sizing.md) | wave64, workgroup size, `__launch_bounds__`, persistent kernels, CU=304 (MI300X) / 256 (MI350X) |
| [optimization/autotuning_methodology.md](autotuning_methodology.md) | triton autotune, hipBLASLt bench, aiter `AITER_TUNE_GEMM`→gradlib `err_ratio<0.05`→`AITER_CONFIG_GEMM_BF16`, search-space pruning, config caching |
| [optimization/roofline_and_bottlenecks.md](roofline_and_bottlenecks.md) | compute vs bandwidth bound, ~45% sustained-of-peak reality, arithmetic intensity, classifying a kernel |
| [optimization/kernel_fusion_strategy.md](kernel_fusion_strategy.md) | when to fuse (epilogue/prologue, norm+quant, rope+cache, comm+norm), fusion donors, when NOT to fuse |
| [optimization/numerical_stability.md](numerical_stability.md) | fp32 accumulate, online softmax, Welford, fp8 FNUZ vs OCP 2× trap, scaling |

## Applying a method to an operator

Instruction shape, tile size, occupancy and grid size are coupled choices. The useful values depend
on the actual GPU, compiler, dtype and shape; the historical examples in these guides are starting
points for measurement. Refer to an operator's source and case records for concrete constraints and
outcomes, such as [dense GEMM tuning](../../operator/dense_gemm/tuning.md) and the
[Qwen3.5-27B DB-tuning case](../../operator/dense_gemm/case_studies/gemm_aiter_db_tuning.md).

## Sources
- MI300X workload optimization guide (occupancy/VGPR/LDS/XCD levers): ROCm docs.
- CDNA3 ISA reference (register pools, MFMA, GLOBAL_LOAD_LDS): AMD Instinct MI300 ISA PDF.
- Per-file sources listed in each document.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/optimization/README.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
