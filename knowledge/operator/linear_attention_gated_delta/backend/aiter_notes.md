---
title: "Gated-delta attention: causal state mixing and QKV fusion"
kind: implementation_knowledge
operator: linear_attention_gated_delta
backend: aiter
provider: aiter
source_commit: e9995854abbd935f0190b19abebe9938b6b90cca
updated: 2026-10-21
evidence: source_reference
performance_status: not_measured_in_kernelwiki
---

# Causal state mixing inside gated-delta attention

Short causal convolution belongs here as the front end of the Gated DeltaNet/Mamba inference workload. Its useful optimization target is the state-mixing-to-attention chain, including fused Q/K/V splitting and state traffic.

## Prefill

The [Triton prefill wrapper](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py) consumes `[dim, total_tokens]`, with `dim = 2*k_dim + v_dim`, and produces split Q/K/V. Sequence boundaries, cache indices and initial-state flags participate in the contract.

`causal_conv1d_split_qkv_triton_fn` uses a per-token path; `causal_conv1d_split_qkv_triton_tile_fn` tiles features and tokens and requires width 2, 3 or 4. The [FlyDSL implementation](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/flydsl/causal_conv1d_flydsl.py) also fuses split output, with a fixed 64-by-64 tile and 256 threads in the pinned builder. The [HIP entry](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/causal_conv1d_fwd_split_qkv.py) is another route. Backend availability does not imply identical supported cases or a performance ranking.

## Decode and verification-sized token groups

The [decode wrapper](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/aiter/ops/triton/gated_delta_net/causal_conv1d_decode.py) accepts `[batch, dim, seqlen]` or a single-token matrix and updates convolution state while emitting Q/K/V. Its `use_gluon` switch defaults to true; a Triton namespace therefore does not identify the device backend by itself.

Validate consecutive token updates against a full-sequence reference. Check state mutation, padded slots, continuous-batching indices, optional bias/SiLU and multiple-token updates. For a fused comparison, time state mixing, splitting and recurrent attention together, then use component timings to locate the bottleneck.

## Source validation routes

- [Prefill split-QKV tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_causal_conv1d_prefill_split_qkv.py).
- [HIP update tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/test_causal_conv1d_update.py).
- [Triton single-token tests](https://github.com/ROCm/aiter/blob/e9995854abbd935f0190b19abebe9938b6b90cca/op_tests/triton_tests/conv/test_causal_conv1d_update_single_token.py).
- [Attention implementations](triton_aiter.md) and [historical run lessons](../corpus/decisions.md).

No GPU measurements were collected for this page.
