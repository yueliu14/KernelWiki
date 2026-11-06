---
title: sparse_attention_nsa on Triton — SOTA card
kind: sota_card
operator: sparse_attention_nsa
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://arxiv.org/html/2508.18224v1
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---
# sparse_attention_nsa × Triton

## TL;DR
Triton is the **SOTA and effectively only portable** NSA path on AMD. Two flavors: (a) the **research
reference** `fla-org/native-sparse-attention` (`parallel_nsa`, online top-k, fused selected+sliding) and
Flash Sparse Attention — CUDA-targeted but Triton runs on MI300X via `triton_amd` with the usual AMD
tuning; (b) the **production** DeepSeek sparse-MLA path shipped in aiter's Triton kernels
(`unified_attention_sparse_mla`, `pa_mqa_logits`, `fp8_mqa_logits`) and wired in sglang's `nsa_backend.py`
/ `nsa/nsa_indexer.py`. There is **no hand-tuned CK/asm NSA kernel** publicly available on AMD as of
2026-06; the production path is Triton + some HIP glue, with gfx942 falling back to generic Triton where
AITER's tuned path is missing or broken.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter sparse-MLA Triton | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/unified_attention_sparse_mla.py`, `pa_mqa_logits.py`, `fp8_mqa_logits.py` | gfx942/950; bf16/fp16/fp8 | DeepSeek-V4 MI300X: sparse MLA among most-expensive layers; bookkeeping fix +8.6% (2485→2699 tok/s/GPU) | DeepSeek-V3.2/V4 sparse MLA serving |
| sglang NSA backend | `sgl-project/sglang:python/sglang/srt/layers/attention/nsa_backend.py`, `nsa/nsa_indexer.py` | gfx942/950 | DeepSeek sparse attn; ROCm coverage maturing | sglang serving (`--attention-backend nsa`/`dsa`) |
| FLA `parallel_nsa` (reference) | `fla-org/native-sparse-attention` | CUDA-written; runs on MI300X via Triton ROCm | research-grade; not AMD-tuned | training/research, custom NSA |
| Flash Sparse Attention | `arXiv 2508.18224` | CUDA-targeted; portable | alternative selected-branch kernel design | when reference selected-branch stalls |

## Config space / knobs
- `block_size=64` (selected/compressed), `window_size`, `top_k` — match the model config.
- AMD Triton: `BLOCK_N == block_size`, `matrix_instr_nonkdim=16`, `num_warps=4`, `num_stages=1`,
  `waves_per_eu=2–3`, `schedule_hint=attention`, `knobs.amd.use_buffer_ops=ON` (indexed gather).
- gfx950 Gluon fp8 indexer prefill (recorded runs): the split arm ran `num_warps=2`, `BLOCK_M=4`,
  `BLOCK_KV=128`, `NUM_BUFFERS=2`. Treat the generic set above as a starting point, not as what this arm used.
- fp8 indexer: **fnuz** on gfx942. See [tuning.md](../tuning.md).

## Numerics / parity
Discrete top-k selection (selection-overlap gate), fp8 indexer (fnuz accuracy gate), three-branch fp32
gated sum. See [numerics.md](../numerics.md).

## Integration (rebind seam)
- sglang: `--attention-backend nsa` (or `dsa`/`dsv4`); `SGLANG_OPT_USE_AITER_INDEXER` toggles the aiter
  indexer. `nsa_backend.py` import block is the "which kernel ran" surface.
- The `@triton.jit` selected/indexer kernels are clean Python rebind seams — overlay a tuned config and
  verify via `TRITON_PRINT_AUTOTUNING=1`.

## Recorded optimization results (gfx950)
Scope: the fp8 (e4m3) MQA-logits indexer prefill in Gluon (32 heads × 128, M 626–10129, KV 40k–857k), the
paged MQA-logits decode, and sparse-MLA split-K decode (512 + rope tail, bs1–bs7). Most recorded steps that
read this page did not beat their parent.
- **Gate on ISA/counters before rewriting.** On the indexer prefill, engineers found that a targeted
  `gl.convert_layout` compiled to zero instructions. They also measured an LDS bank-conflict ratio of 49.5%
  that cost only 0.64–0.73% of CU-busy cycles; the live layout was already `PaddedSharedLayout`. Both axes
  were closed without an edit.
- **Prefill split arm: plan-time register knobs held, loop restructuring did not.** Two changes improved and
  were built on: lowering the fold `NUM_CHAINS` 4→2, and gating the head-bit assignment per arm
  (`block_kv>=128`). The engineer attributes the first to lower accumulator liveness (NC=1 spilled). The
  engineer also reports that gfx950 has packed f32 FMA but no packed f32 max, so the ReLU stays scalar.
  Sub-tiling the BLOCK_KV=128 loop into N_TILE=64 chunks to confine the fp32 accumulator regressed against
  its parent.
- **Sparse-MLA decode: fewer, wider dots per trip.** One 512-wide QK dot and one 512-wide PV dot per trip,
  instead of four 128-wide dots each, was associated with a gain. By the engineer's ISA count it cut loop
  barriers 18→6. Raising BLOCK_K 64→128, coupled to the split count, lost against its seed in these cases.
- Paged-decode instruction trimming (`v_mul_u32_u24` address math, dropping an always-true mask) did not
  beat its parent.

## Pitfalls & anti-patterns
- **gfx942 fallback cliff:** AITER prefill/sparse MQA-logits broken on CDNA3 → dispatch refused → generic
  Triton (several × slower). Paged MQA logits / sparse MLA prefill+decode were *missing entirely* on
  gfx942 → ROCm helper → Triton.
- **HIP-graph capture** of ragged index tensors → corruption; use static capture-safe metadata.
- Reference FLA/FSA kernels carry CUDA assumptions (`device='cuda'`, NVIDIA tile defaults) — re-tune for
  CDNA3 (wave64, 64 KB LDS, mfma_16x16).
- Selection done densely (materializing the score matrix) defeats the point — use online top-k.

## How to verify
Prefill time scales with `top_k·block_size`, not `seq`; selection-overlap vs reference indexer; greedy
temp=0 + gsm8k-style eval (sparse paths have shown eval regressions). `AITER_LOG_MORE=1` to confirm the
tuned vs fallback path.

## Alternatives / cross-links
[overview.md](../README.md) · [hip.md](hip.md) · [tilelang.md](tilelang.md) · languages: `triton_amd` ·
core: [mla_attention](../../mla_attention/README.md) · [sliding_window_attention](../../sliding_window_attention/README.md) · backend: `aiter` · `sglang_kernels`.

## Sources
- aiter sparse-MLA / mqa-logits kernels: `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/` (on-box).
- sglang NSA backend + indexer: https://github.com/sgl-project/sglang (`layers/attention/nsa_backend.py`, `nsa/nsa_indexer.py`).
- FLA NSA reference: https://github.com/fla-org/native-sparse-attention
- FSA: https://arxiv.org/html/2508.18224v1
- DeepSeek-V4 MI300X bring-up (+8.6%, gfx942 fallback): https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/sparse_attention_nsa/backends/triton.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
