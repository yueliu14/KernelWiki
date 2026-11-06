---
title: attention_decode_paged — tuning
kind: operator_overview
operator: attention_decode_paged
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---
# attention_decode_paged — tuning

Paged decode is **memory-bandwidth + launch-latency bound**, not FLOP-bound (the query is one token).
So tuning is about (1) filling 304/256 CUs when batch is small, (2) coalescing the KV-cache reads, and
(3) cutting launch overhead. There is no big GEMM to tile.

## Pick the backend first (vLLM ROCm, Feb 2026)
Ranking: **ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN > TRITON_ATTN > ROCM_ATTN**. ROCM_AITER_FA vs legacy
ROCM_ATTN TPS (64/128 req): **MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×**; TPOT
**2.8–4.6×** faster. The single biggest decode "tune" is not landing on the ROCM_ATTN/Triton fallback.

## Lever 1 — splitKV / flash-decoding (the main lever)
With `sq=1` and small batch, a naive one-CU-per-(batch,head) grid starves the GPU. Split the KV history
into chunks across CUs, compute partial `(O, m, ℓ)` per chunk, then a **reduce kernel** combines them.
- aiter custom paged-attn: `_PARTITION_SIZE_ROCM = 256` (the per-split KV chunk);
  `num_partitions = ceil((max_seq_len + 256 - 1) / 256)`. `paged_attention_v2` is the split path,
  `paged_attention_v1` the single-pass path.
- aiter MLA decode: `num_kv_splits` is auto-picked by `get_meta_param` from batch/total-KV/heads — leave
  it `None` (auto) unless you measured a better value.
- Triton flash-decoding: `NUM_KV_SPLITS` constexpr; combine in a stage-2 reduce kernel.
- **Rule**: more splits when batch×heads is small (fill CUs); fewer when batch is already large (avoid
  reduce overhead). The auto heuristics encode this — trust them first.
- **Recorded (sliding-window paged decode, split-K, gfx950):** the best split count fell as batch grew:
  64 at bs1-2, 32 at bs3-4, 16 at bs5-8. Keeping 64 at bs5+ regressed by up to 10%; 16 → 64 at bs1
  gained about 10%. Keying the table on batch only kept it graph-safe. The reduce needed a power-of-2
  SPLITS where it sizes `tl.arange` by it.
- **Fusing the reduce into the split kernel did not pay in recorded runs.** In a sparse-MLA decode run
  the engineer measured a last-CTA arrival handshake alone (gpu-scope release atomic) at 5458–7725
  ns/call vs a 1790 ns dispatch floor. In a split-K decode chain, acq_rel atomics added 14.6 us at the
  smallest case and a polling sentinel protocol cost about as much as the launch it removed; the
  engineer reported normal and `.cs` stores were not coherent across XCDs. Only these forms were measured.

## Lever 2 — KV-cache layout & coalescing
- **NHD vs HND**: pick the layout the kernel reads coalesced. vLLM's custom path reshapes to
  `[2, num_blocks, block_size*num_kv_heads*head_size]` with `x = 16/elt_size` inner split on K so each
  lane reads a 128-bit chunk. `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` reshuffles for the AITER FA path at
  concurrency ≥32.
- **block_size / page_size**: 16 common; larger pages reduce block-table indirection but coarsen memory.
  Tune `--page-size N` per model.
- Verify coalescing in ISA: want `global_load_dwordx4` / `buffer_load_dwordx4` on the KV reads.
- Two kept split-K rewrites replaced masked KV loads (compiled to exec-mask branches) with unmasked
  clamped-index loads, issuing block-table and K/V loads before the first wait.

## Lever 3 — MFMA path vs small-head path
vLLM custom paged-attn has two kernels:
- `paged_attention_ll4mi_QKV_mfma16_kernel` — MFMA-16 main path (use matrix core for the q·Kᵀ).
- `paged_attention_ll4mi_QKV_mfma4_kernel` — MFMA-4 small-head path.
Plus `paged_attention_ll4mi_reduce_kernel` for the cross-split softmax reduce. Templated on `BLOCK_SIZE`,
KV dtype, fp8 KV. Use `matrix_instr_nonkdim=16`, `waves_per_eu` to trim VGPRs (decode is memory-bound →
favor occupancy 3-4).
- Recorded sweep (Triton split-K decode, gfx950): `num_warps=1` for partials and reduce gained 2-3% /
  2-4%; `num_stages=1` lost 13% at bs1; `waves_per_eu=1` lost 6-7%, 2/4 were neutral; `kpack` is a
  no-op on gfx950.

## Lever 4 — launch overhead (small batch)
Decode at small batch is launch-bound. Levers: HIP-graph capture (CUDA-graph equiv), persistent kernels
(`SGLANG_AITER_MLA_PERSIST=1`), and **unified attention** (one kernel for chunked-prefill+decode, avoids
two launches — see [../chunked_prefill/overview.md](../chunked_prefill/README.md)). `HSA_NO_SCRATCH_RECLAIM=1` is near-mandatory on
MI300X (stops scratch-reclaim idle gaps).
One graph-timed harness charged about 3 us per empty launched kernel, but removing the reduce
launch via in-kernel sync did not pay (Lever 1).

## Head-dim / GQA specifics
- Supported head sizes (aiter custom paged-attn): **64, 80, 96, 112, 120, 128, 192, 256** — 192/256 are
  CDNA3/4-relevant. An unsupported head size falls back to Triton decode (`ROCM_ATTN` cliff: 2.7–4.4×
  slower).
- `gqa_ratio` (q-heads per kv-head) must be **1..32** for the custom path; the kernel broadcasts each KV
  head across its query-head group in-register (no KV replication in HBM) — see
  [../gqa_mqa_attention/overview.md](../gqa_mqa_attention/README.md).
- `max_seq_len ≤ 65536` for the custom paged-attn fast path.
- Packing query tokens × heads of one KV head into M (each KV page gathered once per request) was kept
  in two recorded Triton split-K rewrites, with `matrix_instr_nonkdim=16` (the default 32x32 MFMA was
  slower). Raising BM through launch meta alone, body unchanged, regressed (BM=64: -16% at bs1).

## fp8 KV-cache
Store K/V as fp8 (half the KV bandwidth = directly faster decode), dequant on read. FNUZ on gfx942, OCP
on gfx950. `--kv-cache-dtype fp8_e4m3` (fnuz on MI300X). This is the **highest-leverage** decode tune for
long context — but it is an accuracy gate (see [numerics.md](numerics.md)). Note: aiter MLA decode does
**not** support fp8 KV-cache in vLLM upstream.

## How to verify a tune helped
Isolated decode bench at the served `(batch, heads, kv_heads, head_dim, context, block_size, dtype)`,
median ≥3 warm reps. rocprofv3 → confirm `paged_attention_ll4mi_*` (not a Triton fallback) ran. e2e:
TPOT at served concurrency + greedy temp=0 parity.

## Sources
- aiter paged-attn split size 256 / v1-v2 / supported head sizes / gqa 1-32 / max_seq_len 65536: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py`.
- vLLM custom paged-attn kernel names / KV layout / shuffle env: https://github.com/vllm-project/vllm/tree/main/csrc/rocm ; `backends/vllm_kernels/rocm_kernels.md`.
- HSA_NO_SCRATCH_RECLAIM / ≥1024 grid / mfma_16x16: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; `backends/sglang_kernels/overview.md`.
- Backend ranking + ROCM_AITER_FA vs ROCM_ATTN TPS (MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×; TPOT 2.8–4.6×) (vendor, Feb 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/attention_decode_paged/tuning.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
