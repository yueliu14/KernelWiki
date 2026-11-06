---
title: sparse_attention_nsa — numerics
kind: technique
operator: sparse_attention_nsa
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://arxiv.org/abs/2502.11089
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
---
# sparse_attention_nsa — numerics

## Three sources of numerical divergence
1. **Discrete block selection (top-k).** The selected branch is a hard argmax-style choice of blocks. A
   tie or a near-tie in the indexer score can flip *which* blocks are attended across backends/dtypes —
   a discontinuous change, unlike softmax's smooth rounding. This is the NSA-specific risk: small score
   noise → different selected set → visibly different output. Gate on the **selected-set overlap**, not
   only on logits.
2. **fp8 indexer.** The DeepSeek lightning indexer / MQA logits run in **fp8** (fnuz on gfx942). The
   off-by-one exponent bias vs OCP means a wrong-dialect read is off ~2×, which directly corrupts the
   top-k ranking. fp8 indexer **must** be accuracy-gated; sliding-window K-cache in the V4 bring-up was
   routed through a "fnuz-aware fused quantise-and-insert helper" so cache bytes matched.
3. **Three-branch gated sum.** `o = g_cmp·o_cmp + g_slc·o_slc + g_swa·o_swa`. The gates and the per-branch
   online-softmax must all accumulate in fp32; mixing branch dtypes or normalizing the combined output
   wrong changes scale. One sparse MLA prefill port fed bf16 `p` to the PV MFMA and kept `l` as the fp32
   row sum of that same bf16 `p`; the port passed its correctness gate.

## Parity gate
- Oracle: dense attention **masked to the same selected blocks + window + compressed** the kernel chose
  (extract the kernel's `block_indices` and replay them densely in fp32). This isolates kernel error from
  selection error.
- Separately gate: **selection agreement** — fraction of queries whose top-k set matches the reference
  indexer. A drop here is a selection bug (often fp8 dialect), not a softmax bug.
- Include **ragged and causal** cases in the fp32-oracle edge set.
- Decide what the logits hold **past each row's context**. In one fp8 indexer decode run, the candidate
  wrote -inf there while the reference left the buffer untouched. The valid-region test still passed.
  Confirm that the downstream top-k never reads those columns.
- Greedy temp=0, ≥10 prompts, several **longer than `top_k·block_size + window`** so sparsity actually
  drops tokens. End-to-end eval (gsm8k-style) is the real gate — AITER sparse/MLA paths have shown eval
  regressions; benign bf16 argmax flips must be distinguished from a real selection regression.

## Reduction order and precision (recorded fp8 indexer runs, gfx950)
- **An unchanged error is a codegen signal.** Reassociating the prefill head reduction (more FMA chains)
  gave max_rel_err=0.0 on every timed bucket. The engineer read this as backend normalisation, and the
  timing was null. Check the ISA before benchmarking such an edit.
- **Bit-identical edits keep parity free.** A decode store-epilogue change that used a select (no
  reassociation) kept the f32 values bit-identical and was associated with a gain.
- **A lower-precision matrix-core head reduce needs a proven range.** relu(S)ᵀ·w was moved onto a second
  fp16 MFMA in decode, with a 2^-9 E8M0 Q prescale and a per-CTA power-of-two w scale undone in the
  kv-scale multiply. It passed parity (max_rel_err 6.9e-4 vs tol 1e-2). The fp16 bound (50176 < 65504)
  was analytic only, so probe large-magnitude fp8 inputs before relying on it. In decode the change tied
  the seed. A prefill variant using bf16 hi/lo weight rows was a dead end.

## Correctness traps specific to AMD
- HIP-graph capture corrupting ragged metadata → wrong indices silently → wrong selected set. Use static
  capture-safe index tensors.
- gfx942 falling back to generic Triton can change reduction order vs the tuned path → benign bf16 flips;
  confirm with logits/selection overlap.

## Sources
- NSA branches + trainable selection: https://arxiv.org/abs/2502.11089
- FLA online top-k (avoids materializing scores): https://github.com/fla-org/native-sparse-attention
- fnuz quantise-and-insert, capture-safe metadata: https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/sparse_attention_nsa/numerics.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
