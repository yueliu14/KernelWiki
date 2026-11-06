---
title: dense_gemm on asm — SOTA card
kind: sota_card
operator: dense_gemm
backend: asm
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp6, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - ROCm/aiter@a6bb4993:aiter/tuned_gemm.py
---
# dense_gemm × asm

## TL;DR
> Hand-written MFMA assembly (or Gluon, the asm-adjacent tile DSL) is the **peak** dense-GEMM path — use it
> when you need the last 10–30% over a tuned library kernel, or to exploit a brand-new ISA feature (CDNA4
> block-scaled fp4/fp6 MFMA) before libraries expose it. In aiter it is a **raced candidate**: the tuned DB
> picks `libtype=asm` when it wins the per-shape race, and there's an asm fast-path for bf16 + bpreshuffle.
> Most expensive to author, no standalone env seam — reserve it for the few shapes that dominate Amdahl.
> In recorded bf16 prefill races, aiter's shipped non-preshuffle bf16 asm kernels tied or lost to the library
> path (see *Recorded races*): race them, don't assume a win.

## SOTA implementation
asm is engaged through aiter's dispatcher. From `/sgl-workspace/aiter/aiter/tuned_gemm.py`
(`ROCm/aiter@a6bb4993`) — the asm branch and the bpreshuffle default that auto-selects asm on gfx950:

```python
if config["libtype"] == "asm" and not _no_asm:
    out = asm_gemm(inp_view, B, bias, otype, config["splitK"],
                   config["kernelName"], bpreshuffle)
# ... default when no tuned row, bpreshuffle=True, bf16, N%64==0, K%64==0, out in {bf16,fp32}:
default_config["libtype"] = "asm"; default_config["solidx"] = 0
default_config["splitK"] = None;   default_config["kernelName"] = None
```

`asm_gemm` itself is bf16-in / fp32-or-bf16-out (`# just support bf16gemm_outFp32`) and calls
`gemm_a16w16_asm(inp, weights, out, bias, splitK, KernelName, bpreshuffle)`.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `asm` GEMM (dispatched by tuned_gemm) | `gemm_a16w16_asm` (asm kernels) | gfx942/950; bf16/fp8 | raced per-shape inside aiter; wins specific skinny/prefill + bpreshuffle bf16 shapes @ MI300X, 2026-06-08 | shapes where asm beats hipBLASLt in the race |
| Gluon near-peak GEMM | ROCm Gluon GEMM tutorial | gfx950; bf8/mxfp4 | **BF8 3257 TFLOPS (99.72%)**, **MXFP4 5255 TFLOPS (92.41%)** of peak @ MI350/355 CDNA4 | fp4/fp8 near-peak on CDNA4 |
| CDNA4 block-scaled MFMA fp4/fp6 | `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4` | gfx950; fp4/fp6/fp8 | MI350: ~20 PFLOPs FP4/FP6 peak (≈4× gen-on-gen) | block-scaled low-precision GEMM |

## Recorded races (bf16, non-preshuffle, prefill)
- bf16 GEMM, N=6144, K=2048, M=5916–32768 (cold, cache flushed): the 256x256 asm kernel measured 0.96–1.00x of
  the existing path; small-tile pf3 kernels 0.16–0.45x; splitK > 1 is rejected by aiter's launcher above 1024 tiles.
- bf16 GEMM, N=2624, M=16955–32768 (zero-padded weight, direct launch): the 256x256 kernel took 1.013–1.078x
  the existing path's time.
- Same N=6144 GEMM at M=1936: the 256x256 kernel was faster back to back (41.7 vs 43.2 µs) but lost cold
  (56.8 vs 53.5–55.6 µs); the engineer attributes this to fewer tiles than CUs and a ~12.4 µs host path.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| MFMA op | `v_mfma_f32_16x16x16_bf16` / `..._32x32x8` | matrix-core shape — **16×16 usually wins** | per-kernel |
| scaled MFMA (CDNA4) | `mfma_scale_f32_32x32x64_f8f6f4` / `16x16x128` | block-scaled fp4/fp6/fp8 with E8M0 per-32 scales | — |
| `Atype/Btype` codes | 0=E4M3,1=E5M2,2=E2M3(fp6),3=E3M2(bf6),4=E2M1(fp4) | operand format select | — |
| operand width | 256-bit | scaled-MFMA needs 256-bit A/B; 32 fp4 = 128b → **pad to `fp4x64_t`** (upper 128b zero) | — |
| scale pipeline | GR→LW→LR | global-read scales → LDS re-layout → LDS-read into MFMA layout (no reg→scale path) | — |
| `splitK` | None / int | K split across CUs (set by tuned row) | None |
| `bpreshuffle` | bool | consume pre-shuffled weights (asm fast path on gfx950 bf16) | False |
| tiling/occupancy | ≥1024 workgroups, 8-multiple tiles | fill all CUs, same-XCD placement, deep K-pipelines | — |

## Numerics / parity
fp32 accumulate. fp4/fp6 are block-scaled (OCP microscaling, E8M0 per-32) → **task-accuracy gated**, see
[../numerics.md](../numerics.md). bf16 asm is parity-safe vs library bf16.

## Integration (rebind seam)
Two seams: (1) inside **aiter** as a raced candidate (`libtype=asm`) — engages the live path automatically
when it wins, or via the bpreshuffle bf16 default on gfx950; (2) direct call from a custom op / extension.
No env-overlay for a standalone asm blob — wire via aiter or your own dispatcher.

## Pitfalls & anti-patterns
- Authoring asm for shapes that aren't Amdahl-dominant — huge cost, no e2e move.
- Getting the fp4 **scale layout** wrong (the GR→LW→LR round-trip) → silently wrong results or stalls; this
  3-step round-trip is the main fp4 perf challenge.
- Forgetting the **256-bit operand padding** for scaled MFMA (32 fp4 elems = 128 bits).
- `asm_gemm` only supports bf16-in / bf16-or-fp32-out — don't route fp16/scaled shapes here.
- gfx942 has **no** native block-scaled MFMA; the scaled-MFMA intrinsics are gfx950-only.
- **Edge tiles.** The bf16 256x256 kernel needs N%256==0: a direct launch at N=2624 gave wrong output, and in one
  run a partial M tile (M=5916) caused a GPU memory access fault. Check edge tiles for shapes not a multiple of 256.
- **Launch quirks.** With an explicit kernel name use splitK=1, not 0 (0 failed to launch). On one image the
  `gemm_a16w16_asm` wrapper rejected every bf16 asm kernel with "invalid argument" while a direct HIP launch worked.
- `bpreshuffle` needs a pre-shuffled W cached outside the timed call; it is unavailable if the deployed weight
  layout can't provide one.

## How to verify (worked example)
```bash
# microbench the single asm kernel: TFLOP/s = 2*M*N*K / t_seconds
rocprofv3 --stats -- python bench_asm_gemm.py        # confirm asm kernel name in trace
hipblaslt-bench -m M -n N -k K --a_type bf16_r ...    # same shape baseline
# also time cold per call incl. host path; kernel-only timing can flip the verdict
# for fp4: gate output against a dequant reference AND a downstream eval (LAMBADA / lm-eval)
```

## Alternatives / cross-links
[operators/dense_gemm/backends/aiter](aiter.md) (dispatches asm) · [operators/dense_gemm/backends/hipblaslt](hipblaslt.md) ·
[operators/dense_gemm/backends/ck](ck.md) · [operators/scaled_quant_gemm/backends/asm](../../scaled_quant_gemm/backend/asm.md) (low-bit asm) ·
[operators/dense_gemm/overview](../README.md) · [optimization/mfma_scheduling](../../../general/optimization/mfma_scheduling.md) ·
language refs `languages/asm_mfma/`, `languages/hipkittens/`.

## Sources
- CDNA3/4 matrix-core programming + scaled-MFMA intrinsic/layout: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- fp8 GEMM on CDNA4: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- Near-peak BF8 99.72% / MXFP4 92.41% TFLOPS: https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- aiter asm dispatch + bpreshuffle default: `/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/dense_gemm/backends/asm.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
