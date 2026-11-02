---
title: "GEMM fusion boundaries: AITER dispatch"
kind: implementation_knowledge
operator: gemm_epilogue_fused
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# GEMM fusion boundaries in AITER

Treat normalization, quantization, gated activations and tensor-parallel communication as parts of the projection workload when they lie on its live path. This page keeps those contracts beside GEMM instead of maintaining separate small-operator collections.

## RMSNorm, FP8 quantization and GEMM

The [public wrapper](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/fused_norm_quant_gemm.py) exposes `fused_rmsnorm_quant_gemm`. Its inputs include BF16 activations `[M,K]`, FP8 weights `[N,K]`, normalization weights `[K]`, scalar FP32 activation/weight scales and a preallocated FP8 workspace `[M,K]`. It returns BF16 `[M,N]`.

The [native implementation](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/csrc/kernels/fused_norm_quant_gemm.cu) executes CK normalization/quantization followed by hipBLASLt GEMM. This is host dispatch fusion: it still launches separate device kernels and materializes the FP8 workspace. The Python wrapper falls back to separate calls when the native module is unavailable or fails. Verify which path executes before attributing a timing change to device fusion.

Measure the whole normalization-to-GEMM chain with identical scales, epsilon, workspace allocation policy and hipBLASLt solution. A faster inner GEMM can lose after repeated conversion or packing.

## Feed-forward activation fusion

The [Triton implementation references](triton_aiter.md) include full feed-forward paths and gated/ungated fused variants. Compare the projection pair plus activation at the actual token count and intermediate width. Preserve the gate split, activation approximation and intermediate rounding; a standalone activation benchmark does not describe the fused workload.

## Tensor-parallel projection boundary

For row-parallel projections followed by residual addition and RMSNorm, inspect [AITER's distributed API](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/dist/communication_op.py), especially `tensor_model_parallel_fused_allreduce_rmsnorm` and its quantized variant. The [communicator](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/dist/device_communicators/communicator_cuda.py) chooses paths using input layout, hidden width, message size, rank count and communicator availability. A fused API name alone does not establish that the custom fused path ran.

The quantized variant distinguishes per-token FP8, per-group FP8 and MXFP4. Scale layout, optional BF16 output and Gemma-style normalization are observable parts of the next GEMM's input contract. Include residual output and cross-rank correctness in validation.

## Evidence and validation

- For the norm/quant/GEMM chain, validate against the decomposed wrapper with identical scales and workspace contents. A dedicated test for this wrapper was not found in the pinned `op_tests` tree.
- [AllReduce + RMSNorm tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_rms.py).
- [Per-group quantization tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py).
- [Historical norm+GEMM campaign lessons](../corpus/decisions.md), moved from GEAK's `fused_norm_quant` category without changing their original measurements.

These are source-derived contracts and validation routes. KernelWiki has not run these GPU or multi-GPU tests and makes no new performance claim.
