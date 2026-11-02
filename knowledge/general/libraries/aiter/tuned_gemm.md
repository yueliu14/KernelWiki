---
title: aiter tuned_gemm — the live dense-GEMM dispatch + per-shape tuning recipe
kind: backend
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: historical_reference
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:gradlib/gradlib/gemm_tuner.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:gradlib/gradlib/GemmTuner.py
---

# aiter tuned_gemm — dispatch + tuning recipe

## Scope

This reference describes `aiter.tuned_gemm.gemm_a16w16` at the source revision listed above.
It applies when inspection confirms that the measured caller uses this dispatcher. The wrapper
selects an implementation from its database or fallback rules; a shipped selection is not a
performance ranking for a different environment.

Use the maintained [GEMM tuning workflow](../../workflows/gemm_tuning_workflow.md) for capture,
deployment and validation. PyTorch BLAS settings affect only paths that consult them; an AITER
call choosing a solution directly must be tuned or changed at that active layer.

## The dispatch path (`aiter/tuned_gemm.py`)

### Entry + key construction
`gemm_a16w16(A, B, bias, otype, scale_a, scale_b, scale_c)` (decorated with `@torch_compile_guard`):
- flattens `A` to 2-D `[m, k]`, `n = B.shape[0]`;
- detects `bpreshuffle` from `B.is_shuffled`;
- builds the lookup via `get_GEMM_A16W16_config(M=m, N=n, K=k, bias=use_bias, dtype=str(A.dtype),
  otype=str(otype), scaleAB=(scale_a or scale_b is not None), bpreshuffle=...)`.

The **DB index is the 9-tuple**:
`(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`
(`get_GEMM_A16W16_config_` sets the index on the CSV; `cu_num` from `get_cu_num()`). Every component must
match the live call — **`bias` and `cu_num` are the usual mismatch traps.**

### padded_M bucketing
The lookup is tried for three `padded_M` variants in order `[None, 0, 1]` where each non-None value calls
`get_padded_m(M, N, K, gl)` (`aiter/ops/gemm_op_common.py`). This lets the tuner bucket large prefill M
into a bounded set of rows so the DB stays small and decode/prefill M both hit an entry. First non-None
hit wins.

### libtype dispatch (`solMap`)
```python
solMap = {"torch": torch_gemm, "hipblaslt": hipb_gemm, "skinny": skinny_gemm,
          "asm": asm_gemm, "triton": triton_gemm}
```
- `hipblaslt` → `hipb_mm(inp, weights.t(), solidx, bias, otype, scale_a, scale_b, scale_c, bpreshuffle)`
  — calls a specific Tensile **solution index** directly (creates the hipBLASLt ext on first use).
- `asm` → `gemm_a16w16_asm(inp, weights, out, bias, splitK, kernelName, bpreshuffle)` (bf16 in, fp32/bf16
  out only).
- `skinny` → `wvSpltK` (solidx 0), `LLMM1` (1), `wv_splitk_small_fp16_bf16` (2); gated by
  `is_skinny_default_shape` (M≤16 with N/K bounds, K%8==0).
- `flydsl` → handled before `solMap`: re-resolves kernel params and calls `flydsl_hgemm` (see
  [flydsl_path.md](flydsl_path.md)). If FlyDSL isn't importable the config is dropped and lookup falls
  through.
- `triton` → `aiter.ops.triton.gemm.basic.gemm_a16w16` (no scaling, no bpreshuffle).

### No-hit defaults (`get_GEMM_A16W16_config`)
When nothing matches: `gfx12*` → `torch`; `bpreshuffle` on gfx942 → `hipblaslt solidx=-1` (let heuristic
pick), on gfx950 bf16 with N%64==K%64==0 → `asm`; else if `is_skinny_default_shape` → `skinny solidx=2`;
otherwise `torch`. A `logger.info(... not found tuned config ... will use default ...)` line is emitted.

## Tuning procedure

Follow the [GEMM tuning workflow](../../workflows/gemm_tuning_workflow.md). Check the installed
version's supported tuner options instead of assuming that archived CLI flags remain available.
The captured key must match the caller, and the selected configuration must reach every intended
worker after cache invalidation and graph recapture where required.

## Config space / knobs
- **DB key**: `(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)` — match all 9.
- **Tuner**: `--libtype`, `--errRatio` (0.05), `--mp`, `--warmup`/`--iters`, `--shape_grouped`,
  `--all_bias` (tune both bias/non-bias regardless of capture), `--all` (retune existing rows).
- **Per-libtype**: hipBLASLt `solidx` + `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` (workspace cap); asm
  `kernelName` + `splitK`; flydsl tiling (see [flydsl_path.md](flydsl_path.md)).

## Correctness and measurement limits

The tuner's error check establishes agreement only on its reference inputs and threshold. Validate
the target workload's numerical contract, boundary cases and serving accuracy separately. A partial
configuration keeps fallbacks for untouched shapes; it makes no guarantee about the shapes replaced
or the resulting end-to-end performance.

Verify both selected paths and expected fallbacks. Use interleaved measurements with uncertainty
appropriate to the actual environment. See the [integration method](../../corpus/decisions.md#method-live-binary-and-integration)
and [measurement method](../../corpus/decisions.md#method-paired-measurement-controls).

## Alternatives / cross-links
Operator reference: [`operators/dense_gemm/backends/aiter.md`](../../../operator/dense_gemm/backend/aiter.md).
Executed kernels: [`operators/dense_gemm/backends/hipblaslt.md`](../../../operator/dense_gemm/backend/hipblaslt.md)
· [flydsl_path.md](flydsl_path.md) · [configs_db.md](configs_db.md) · [integration.md](integration.md).

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`:
  `aiter/tuned_gemm.py` (dispatch, key, save_shapes, solMap),
  `gradlib/gradlib/gemm_tuner.py` (driver, retry, model_dir),
  `gradlib/gradlib/GemmTuner.py` (ladder, libtype_list, err_ratio gate),
  `aiter/utility/base_tuner.py` (`--mp`, `--errRatio`, `--shape_grouped` args),
  `aiter/jit/core.py` (`AITER_CONFIG_GEMM_BF16`, merge).
- Historical outcomes and applicability: [GEMM corpus](../../../operator/dense_gemm/corpus/decisions.md#gemm-shape-regime-routing).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/backends/aiter/tuned_gemm.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
