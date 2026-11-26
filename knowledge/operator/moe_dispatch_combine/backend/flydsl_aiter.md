---
title: "MoE dispatch and combine: AITER flydsl"
kind: implementation_knowledge
operator: moe_dispatch_combine
backend: flydsl
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# MoE dispatch and combine — AITER flydsl

Dispatch moves tokens into expert-local order; combine reverses that mapping and applies routing weights. Gather/scatter locality, atomics and reduction ordering determine the useful optimization space.

FlyDSL wrappers compile and cache specialized builders. The builder, layout helpers and wrapper arguments jointly define the implementation. Separate first-use compilation from steady-state timing.

## Scope and evidence

The source links below are pinned to AITER `e9995854abbd`. **Recent** means the file changed between 2026-07-21 and this snapshot. The symbol lists and parameter names are extracted from source; they include helpers as well as entry points. They do not certify runtime dispatch, hardware support or a measured speedup.

## Implementation files and entry points

### `aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py) · **Recent**

Source documentation: FlyDSL intranode dispatch/combine kernels for EP MoE.

Symbols: `_wave_uniform_i64`, `_pack_f32x4_to_fp8`, `make_dispatch_kernel`, `make_combine_kernel`, `make_dispatch_jit`, `make_combine_jit`.

### `aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py) · **Recent**

Source documentation: Resolved tuned CSV path (kernel_type / ep_size kept for API compat).

Symbols: `_device_cu_count`, `_check_block_num_resident`, `_resolve_launch_geometry`, `dtype_to_tuning_name`, `_detect_gpu_model`, `resolve_tuning_config_path`, `build_geometry_tuning_table_for_config`, `_dtype_elem_size`, `_is_fp4_dtype`, `_token_bytes_for`, `_token_view_dim_for`, `FlyDSLDispatchCombineConfig`, `build_p2p_table`, `FlyDSLDispatchGroupMajorOp`, `FlyDSLDispatchCombineIntraNodeOp`.

### `aiter/ops/flydsl/kernels/moe_g2l_lut.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_g2l_lut.py) · **Recent**

Source documentation: EP global->local expert LUT build (FlyDSL), single-block parallel scan. Collapses the host ne + cumsum + sub + where chain (6 elementwise/scan launches) into one kernel: read the (E_global,) 0/1 expert_mask, scan first within each wave and then across wave totals, and write g2l_lut[i] = mask[i] ? prefix_incl[i]-1 : E (sentinel E = dropped route).

Symbols: `build_moe_g2l_lut_module`.

### `aiter/ops/flydsl/kernels/moe_gather_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gather_reduce.py) · **Recent**

Source documentation: One-pass MoE gather-reduce (weighted) epilogue kernel (FlyDSL). Background ---------- After the per-expert stage2 GEMM, MoE output lives in a grouped layout grouped_out (E, max_m, model_dim): expert e holds its routed tokens in rows [0, counts[e]).

Symbols: `_GatherRouteLds`, `_lds_li32`, `_lds_si32`, `_unpack_pair_to_f32`, `_pack_pair_from_f32`, `build_moe_gather_reduce_module`.

Parameters referenced in this file: `BLOCK_THREADS`, `split_k`.

### `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/moe_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/moe_reduce.py) · **Recent**

Source documentation: Routing reductions with cached JIT launchers.

Symbols: `sorted_sum`, `invert_sorted_ids`.

### `aiter/ops/flydsl/kernels/moe_reduce.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_reduce.py) · **Recent**

Source documentation: MoE topk-reduction kernel (FlyDSL, layout API). Y[t, d] = sum_k X[t, k, d], optionally gated by the EP validity mask (valid[t,k] = expert_mask[topk_ids[t,k]] != 0). Epilogue of stage2 mode="reduce", shared by every dtype's reduce path. Build a per-shape launcher with compile_moe_reduction (cached); the kernel's compile-time params are Constexpr so flyc specializes per shape/dtype.

Symbols: `_moe_reduction_body`, `_pick_reduce_block`, `compile_moe_reduction`.

### `aiter/ops/flydsl/kernels/moe_scatter_copy_preshuffle_scale.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_scatter_copy_preshuffle_scale.py) · **Recent**

Source documentation: Fused MoE route-gather + e8m0-scale preshuffle kernel (FlyDSL). Background ---------- The grouped a8w4 stage1 path needs the per-token MXFP8 e8m0 scale both *route-gathered* into the grouped per-expert layout and *preshuffled* into the WMMA layout the masked grouped GEMM consumes. Previously this was two passes: 1. scatter-copy a1_scale_token_u8[tok] -> a1_scale_raw[e, m] (row-major) 2.

Symbols: `_emit_preshuffle_dword`, `build_moe_scatter_copy_preshuffle_scale_module`.

Parameters referenced in this file: `BLOCK_THREADS`, `tile_k`.

### `aiter/ops/flydsl/kernels/moe_scatter_copy_token.py`

[Source](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/kernels/moe_scatter_copy_token.py) · **Recent**

Source documentation: One-pass MoE route-gather (scatter-copy) kernel (FlyDSL). Background ---------- Before the stage1 GEMM, each token's quantized payload (and per-token scale) must be copied from the flat per-token layout into the grouped per-expert layout the masked grouped kernel consumes:: for e in range(E): toks = tokens routed to expert e # n = counts[e] of them grouped[e, :n] = a_payload[toks] # copy each token's row This kernel does that copy in one pass.

Symbols: `build_moe_scatter_copy_token_module`.

Parameters referenced in this file: `BLOCK_THREADS`.

## Tests and benchmarks

Related tests for this operator family are listed below. Select the entry point and backend that match the implementation; the presence of a test does not mean KernelWiki ran it or that every backend is covered.

- [op_tests/multigpu_tests/test_dispatch_combine.py](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_dispatch_combine.py)

## Performance and validation

Validate duplicate routes, empty experts and output initialization. Include communication where the selected implementation uses it.

No new GPU timings were collected for this page. Record GPU model, ROCm/compiler/library versions, full shape and strides, input/output dtypes, warmup and timing method, baseline implementation, latency and correctness tolerance before claiming a win. Compare speedups only within the same benchmark protocol and baseline.

## Sources

- [AITER source snapshot](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca/)
- [AITER contribution and benchmark conventions](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/CONTRIBUTE.md)
- File-level source, configuration and test links appear beside the corresponding material above.
