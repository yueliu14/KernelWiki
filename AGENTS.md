# KernelWiki — start here (kernel optimization agents)

This branch keeps only **kernel-level** knowledge: how GEMM, MoE, attention and indexer kernels and their fusions are
implemented, tuned, measured and validated on AMD Instinct GPUs. Model histories, serving/workload chains, library
integration and agent-process notes live on `main`, not here.

Use this page to pick at most a few files, read those, and go back to measuring. Do not read the tree top to bottom.

## 1. Find the operator your kernel belongs to

| Your kernel computes … | Read `knowledge/operator/…` |
| --- | --- |
| Dense projection / FFN GEMM (prefill-sized M) | [dense_gemm](knowledge/operator/dense_gemm/README.md) |
| Decode-sized GEMM / GEMV (M ≲ 16), weight-bandwidth bound | [skinny_gemv_decode](knowledge/operator/skinny_gemv_decode/README.md), then [splitk_streamk_gemm](knowledge/operator/splitk_streamk_gemm/README.md) |
| Split-K partials + reduction, poorly occupied grid | [splitk_streamk_gemm](knowledge/operator/splitk_streamk_gemm/README.md) |
| FP8 / INT8 / FP4 / MX block-scaled GEMM, packing, scales | [scaled_quant_gemm](knowledge/operator/scaled_quant_gemm/README.md) |
| GEMM with fused activation / residual / norm / quant epilogue | [gemm_epilogue_fused](knowledge/operator/gemm_epilogue_fused/README.md) |
| Batched / head-wise GEMM | [batched_gemm](knowledge/operator/batched_gemm/README.md) |
| MoE expert GEMMs (gate/up, down), low-bit experts, token imbalance | [grouped_gemm_moe](knowledge/operator/grouped_gemm_moe/README.md), [fused_moe_grouped_gemm](knowledge/operator/fused_moe_grouped_gemm/README.md) |
| MoE routing, top-k, token sorting / alignment | [moe_routing_topk](knowledge/operator/moe_routing_topk/README.md) |
| MoE dispatch / combine, shared experts | [moe_dispatch_combine](knowledge/operator/moe_dispatch_combine/README.md), [shared_expert_fusion](knowledge/operator/shared_expert_fusion/README.md) |
| Decode / verify attention over a (paged) KV cache, split-KV + reduce | [attention_decode_paged](knowledge/operator/attention_decode_paged/README.md), [gqa_mqa_attention](knowledge/operator/gqa_mqa_attention/README.md) |
| Sliding-window attention | [sliding_window_attention](knowledge/operator/sliding_window_attention/README.md) |
| Prefill / extend attention, cached prefixes | [attention_prefill_fmha](knowledge/operator/attention_prefill_fmha/README.md), [chunked_prefill](knowledge/operator/chunked_prefill/README.md) |
| MLA, sparse/NSA, compressed, linear (gated-delta) attention | [mla_attention](knowledge/operator/mla_attention/README.md), [sparse_attention_nsa](knowledge/operator/sparse_attention_nsa/README.md), [compressed_attention](knowledge/operator/compressed_attention/README.md), [linear_attention_gated_delta](knowledge/operator/linear_attention_gated_delta/README.md) |
| Indexer scores / block selection / top-k for sparse attention | [attention_indexer](knowledge/operator/attention_indexer/README.md) |
| Q/K norm, RoPE, quantization, KV-cache write fusions | [qk_norm_rope_cache](knowledge/operator/qk_norm_rope_cache/README.md) |
| RMSNorm / residual-add / quant boundary kernels (no dedicated operator) | [qk_norm_rope_cache](knowledge/operator/qk_norm_rope_cache/README.md) and [gemm_epilogue_fused](knowledge/operator/gemm_epilogue_fused/README.md) for the fusion contracts; then the method pages in §4 (it is a bandwidth/latency problem) |

Inside an operator directory, read in this order and stop when you have a direction:

1. `README.md` — what the operator is, its regimes, the TL;DR.
2. `cards.md` — decision cards. Read the table first and open only the cards whose **languages** match your
   kernel's. A card in another language is a porting idea, not a recipe.
3. `tuning.md`, `numerics.md`, `fusion.md` — short operator-specific notes.
4. `backend/<lang>.md` — older GEAK implementation notes for one language (dated; check the date).
5. `backend/<lang>_aiter.md` — walkthrough of the pinned AITER source for that language: file paths, entry points,
   parameters, tests. Long reference pages: search them for the symbol or file you need instead of reading them.

`corpus/decisions.yaml` is the machine-readable source of `cards.md`; read the card instead. `corpus/tuned_configs.yaml`
(dense_gemm, scaled_quant_gemm) lists the tile configurations AITER ships per shape bucket: seeds for a config search.

## 2. Your kernel language

| Language | Authoring and tuning pages |
| --- | --- |
| Triton | [triton_amd/](knowledge/general/backend/triton_amd/) (overview, patterns, knobs, pitfalls, deep_codegen, isa_verify), [geak_triton_optimization](knowledge/general/backend/triton/geak_triton_optimization.md), [tuning-triton](knowledge/general/expert_recipes/tuning/tuning-triton/SKILL.md) |
| Gluon | [gluon/](knowledge/general/backend/gluon/) (overview, programming_model, gemm_cookbook), [gluon_authoring skill](knowledge/general/expert_recipes/skills/gluon_authoring/skill.md) |
| FlyDSL | [flydsl/](knowledge/general/backend/flydsl/) (overview, authoring_*, knobs, patterns, debugging), [tuning-flydsl](knowledge/general/expert_recipes/tuning/tuning-flydsl/SKILL.md), `expert_recipes/skills/flydsl_*` |
| HIP / C++ | [hip_cpp/](knowledge/general/backend/hip_cpp/) (intrinsics, lds_async, patterns, pitfalls), [geak_hip_optimization](knowledge/general/backend/hip/geak_hip_optimization.md), [tuning-hip](knowledge/general/expert_recipes/tuning/tuning-hip/SKILL.md) |
| Composable Kernel | [composable_kernel/](knowledge/general/backend/composable_kernel/), [tuning-ck](knowledge/general/expert_recipes/tuning/tuning-ck/SKILL.md) |
| Assembly / MFMA intrinsics | [asm_mfma/](knowledge/general/backend/asm_mfma/) |
| TileLang | [tilelang/](knowledge/general/backend/tilelang/) |

[choosing_a_backend](knowledge/general/workflows/choosing_a_backend.md) compares them when the language is yours to choose.

## 3. The GPU

MI355X / MI350 = **gfx950, CDNA4**: [arch](knowledge/general/hardware/cdna4_mi350/arch.md),
[memory](knowledge/general/hardware/cdna4_mi350/memory.md), [ISA notes](knowledge/general/hardware/cdna4_mi350/isa_notes.md),
[peak tables](knowledge/general/hardware/cdna4_mi350/peak_tables.md),
[block-scaled matrix cores](knowledge/general/hardware/cdna4_mi350/matrix_core_blockscale.md),
[FP4/FP6 microscaling](knowledge/general/hardware/cdna4_mi350/fp4_fp6_microscaling.md),
[clocks and power](knowledge/general/hardware/cdna4_mi350/clocks_power.md).
Shared CDNA facts: [wavefront/VGPR/AGPR](knowledge/general/hardware/shared/wavefront_simd_vgpr_agpr.md),
[LDS banks and memory model](knowledge/general/hardware/shared/memory_model_lds_bank.md),
[MFMA/SMFMAC](knowledge/general/hardware/shared/matrix_core_mfma_smfmac.md),
[L2 and XCD swizzle](knowledge/general/hardware/shared/l2_xcd_swizzle.md),
[HBM and Infinity Fabric](knowledge/general/hardware/shared/hbm_infinity_fabric.md),
[dtype numerics](knowledge/general/hardware/shared/dtype_numerics.md). MI300X (gfx942, CDNA3): `hardware/cdna3_mi300/`.

## 4. Methods, by what limits the kernel

| Bottleneck you measured | Read `knowledge/general/optimization/…` |
| --- | --- |
| Not sure yet | [roofline_and_bottlenecks](knowledge/general/optimization/roofline_and_bottlenecks.md), then [reading a kernel bottleneck](knowledge/general/profile_bottleneck/reading_a_kernel_bottleneck.md) |
| HBM bandwidth (decode GEMV, norms, KV reads) | [vectorization_and_coalescing](knowledge/general/optimization/vectorization_and_coalescing.md), [memory_pipelining](knowledge/general/optimization/memory_pipelining.md), [xcd_l2_locality](knowledge/general/optimization/xcd_l2_locality.md) |
| Grid does not fill 256 CUs / tail effects | [wave_and_grid_sizing](knowledge/general/optimization/wave_and_grid_sizing.md), split-K cards in [splitk_streamk_gemm](knowledge/operator/splitk_streamk_gemm/cards.md) |
| MFMA throughput, instruction schedule | [mfma_scheduling](knowledge/general/optimization/mfma_scheduling.md), [MFMA tile selection case](knowledge/general/case_studies/by_kernel/mfma_tile_selection.md) |
| LDS traffic / bank conflicts | [lds_and_bank_conflicts](knowledge/general/optimization/lds_and_bank_conflicts.md) |
| Registers, spills, occupancy | [occupancy_and_registers](knowledge/general/optimization/occupancy_and_registers.md) |
| Tiny kernel, launch / host bound | [geak_wrapper_optimization](knowledge/general/optimization/geak_wrapper_optimization.md) (host savings vanish under CUDA-graph replay), [kernel_fusion_strategy](knowledge/general/optimization/kernel_fusion_strategy.md) |
| Precision / tolerance trade-offs | [numerical_stability](knowledge/general/optimization/numerical_stability.md), [quantization](knowledge/general/quantization/README.md) (formats, FNUZ vs OCP, MX block scaling) |
| Search over configs | [autotuning_methodology](knowledge/general/optimization/autotuning_methodology.md), [tuning-core search strategy](knowledge/general/expert_recipes/tuning/tuning-core/search_strategy.md) |
| Weighted multi-shape score | [geak_geomean_levers](knowledge/general/optimization/geak_geomean_levers.md), [strategy catalog](knowledge/general/optimization/geak_optimization_strategies.md) |

Cross-operator method cards: [general/cards.md](knowledge/general/cards.md).

## 5. Measure and validate

- Timing protocol and noise: [benchmarking_methodology](knowledge/general/profile_bottleneck/benchmarking_methodology.md),
  [tuning-core measurement](knowledge/general/expert_recipes/tuning/tuning-core/measurement.md),
  [CUDA-graph-captured benchmarking](knowledge/general/expert_recipes/tuning/tuning-core/graph_captured_benchmarking.md),
  [clocks and power](knowledge/general/expert_recipes/tuning/tuning-core/clocks_and_power.md).
- Correctness and "is my kernel actually running": [correctness gates](knowledge/general/expert_recipes/tuning/tuning-core/correctness_gates.md),
  [engagement verification](knowledge/general/profile_bottleneck/engagement_verification.md),
  [common pitfalls](knowledge/general/profile_bottleneck/common_pitfalls.md).
- Profilers: [rocprofv3 counters](knowledge/general/profile_bottleneck/rocprofv3_counters.md),
  [rocprof-compute workflow](knowledge/general/profile_bottleneck/rocprof_compute_workflow.md),
  [roofline on MI](knowledge/general/profile_bottleneck/roofline_on_mi.md), [trace analysis](knowledge/general/profile_bottleneck/trace_analysis.md).
- End-to-end procedure for one kernel: [optimize_single_kernel](knowledge/general/workflows/optimize_single_kernel.md).

## 6. How to use what you read

- Nothing here is a measured winner for your kernel. `reported` cards carry an upstream report that was not
  reproduced; `source-only` cards describe what shipped code does. Rank directions by them, then let your own
  benchmark decide; a lever that loses on your benchmark is refuted for your task.
- When a page drives a candidate, name the page or card id in your notes, so the outcome can be traced and fed back.
- Pages pin their source revision and date; prefer the newest when two disagree.
