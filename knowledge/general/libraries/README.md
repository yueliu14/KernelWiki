# Libraries

Reusable library APIs, configuration formats, tuning tools and integration guidance. GEMM-specific
libraries and tuners belong here when the procedure applies across models and GEMM variants.
Operator implementation contracts live in the [operator collection](../../operator/README.md), including
[AITER MLA](../../operator/mla_attention/backend/aiter.md) and
[AITER grouped MoE](../../operator/grouped_gemm_moe/backend/aiter.md).
Source revisions and evidence remain attached to individual documents.

- [aiter configs DB](aiter/configs_db.md)
- [aiter FlyDSL path](aiter/flydsl_path.md)
- [aiter integration](aiter/integration.md)
- [aiter — backend overview](aiter/overview.md)
- [aiter tuned_gemm — dispatch + tuning recipe](aiter/tuned_gemm.md)
- [Composable Kernel (consumed as a library)](composable_kernel_lib/api.md)
- [ckProfiler](composable_kernel_lib/ckprofiler.md)
- [CK instances — traits, coverage, selection](composable_kernel_lib/instances.md)
- [FlashAttention-ROCm — CK backend](flash_attention_rocm/ck_backend.md)
- [FlashAttention-ROCm (`fa_rocm`)](flash_attention_rocm/overview.md)
- [FlashAttention-ROCm — Triton backend](flash_attention_rocm/triton_backend.md)
- [hipBLASLt API](hipblaslt/api.md)
- [hipBLASLt environment variables](hipblaslt/env.md)
- [hipBLASLt offline tuning](hipblaslt/offline_tuning.md)
- [hipBLASLt TensileLite](hipblaslt/tensilelite.md)
- [hipBLASLt vs rocBLAS — when each wins](hipblaslt/when_wins.md)
- [MIOpen convolution (conv2d / depthwise / grouped)](miopen/conv.md)
- [MIOpen — the cuDNN analog on ROCm](miopen/overview.md)
- [DeepEP on AMD (ROCm port) + UCCL-EP](mori_rccl/deepep.md)
- [MoRI-EP — expert-parallel all-to-all (dispatch / combine)](mori_rccl/mori_ep.md)
- [MoRI + RCCL — the AMD communication backend](mori_rccl/overview.md)
- [RCCL tuning — the TP/collective backend on Instinct](mori_rccl/rccl_tuning.md)
- [Inductor max-autotune on ROCm](pytorch_inductor/max_autotune.md)
- [PyTorch Inductor on ROCm](pytorch_inductor/overview.md)
- [rocBLAS (Tensile GEMM API)](rocblas_tunableop/api.md)
- [PyTorch TunableOp](rocblas_tunableop/tunableop.md)
- [rocBLAS / TunableOp — when it wins](rocblas_tunableop/when_wins.md)
- [SGLang attention backends (ROCm)](sglang_kernels/attention_backends.md)
- [SGLang kernels on ROCm / MI300X](sglang_kernels/overview.md)
- [Where SGLang's kernels live (ROCm source map)](sglang_kernels/where_kernels_live.md)
- [vLLM × AITER integration](vllm_kernels/aiter_integration.md)
- [vLLM kernels on ROCm / MI300X (V1)](vllm_kernels/overview.md)
- [vLLM's own ROCm HIP kernels (`csrc/rocm/`)](vllm_kernels/rocm_kernels.md)
