---
title: mla_attention on Triton — SOTA card
kind: sota_card
operator: mla_attention
backend: triton
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/mla_decode.py
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---
# mla_attention × Triton

## TL;DR (one-line decision)
> Triton MLA (`mla_decode.py`, `mla_decode_rope.py`) is the **editable** and **reference** MLA path — it
> implements the absorbed latent decode with the `BLOCK_DMODEL=512` (latent) + `BLOCK_DPE=64` (rope)
> dim-split. It is **slower than aiter asm MLA** (aiter gives 1.2–1.6× TPOT over it), but it is portable,
> the correctness oracle, and on **gfx942 `ROCM_AITER_TRITON_MLA` shows +2–3% TPS** over the pure-asm path
> (vendor). Use it as the reference / gfx942 contender / Tier-C rewrite seam.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| Triton MLA decode (`mla_decode.py`) | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/mla_decode.py` | gfx90a/942/950; bf16/fp16/fp8 | reference; ~0.6–0.85× of asm MLA; **+2–3% TPS as `ROCM_AITER_TRITON_MLA` on gfx942** (vendor) | correctness oracle; gfx942 contender; editable |
| Triton MLA + RoPE (`mla_decode_rope.py`) | same | as above | fused-rope variant | when rope is fused into decode |

**Real dim-split + HIP block config** (`mla_decode.py`) — the latent (`Lk=576 = 512+64`) is split to stay
under the ROCm Triton compiler limit:
```python
# Split key dim into BLOCK_DMODEL + BLOCK_DPE to keep each block
# within ROCm Triton compiler limits (avoids PassManager::run failure).
if Lk == 576:                       # DeepSeek MLA: 512 latent + 64 rope
    BLOCK_DMODEL = 512
    BLOCK_DPE = 64
    if is_hip_:
        BLOCK = 4                   # KV chunk (HIP); BLOCK=64 on CUDA
        num_warps = 1
elif Lk == 288:
    BLOCK_DMODEL, BLOCK_DPE = 256, 32
else:
    BLOCK_DMODEL, BLOCK_DPE = triton.next_power_of_2(Lk), 0
# stage-1 flash-decoding over NUM_KV_SPLITS, then stage-2 reduce
kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
```

## Config space / knobs
| param | range / values | effect | default (DeepSeek MLA on HIP) |
|---|---|---|---|
| `BLOCK_DMODEL` | 512 (latent) / 256 / pow2 | latent score-dim tile | 512 |
| `BLOCK_DPE` | 64 / 32 / 0 | decoupled-RoPE score-dim tile | 64 |
| `BLOCK` (KV chunk) | **4 on HIP** (64 CUDA) | KV streaming tile | 4 |
| `NUM_KV_SPLITS` | flash-decoding split | parallelism over seq | shape-derived |
| `num_warps` | **1 on HIP** when grouped | wave64; avoid spill | 1 |
| `num_stages` | 1 | fused decode pipeline | 1 |
| `matrix_instr_nonkdim` | 16 | MFMA 16×16 | 16 |
| `waves_per_eu` | 2–4 | memory-bound → occupancy (helps only if the grid fills the extra slots; see recorded results) | 2 |

See `languages/triton_amd/knobs.md` and [../tuning.md](../tuning.md).

## Numerics / parity
fp32 two-part score accumulate (latent `BLOCK_DMODEL` + rope `BLOCK_DPE`), online softmax, split-KV
reduce (stage-1 → stage-2). Matrix absorption exact in bf16. fp8 scales — **FNUZ gfx942 / OCP gfx950**.
This is the **oracle** for the asm path — greedy temp=0 parity vs `mla_decode_fwd`. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
- **sglang:** `--attention-backend triton` (MLA path). **vLLM:** `ROCM_AITER_TRITON_MLA` / `TRITON_MLA`.
- The `@triton.jit` MLA kernel is the Tier-C edit seam.
- **Verify it engaged:** rocprofv3 → `_fwd_kernel_stage1`/`_fwd_kernel_stage2` Triton names (vs asm
  `mla_decode_stage1_asm_fwd`).

## Pitfalls & anti-patterns
- `num_stages>1` hurts the fused decode kernel; keep at 1.
- On HIP, `Lk==576` **must** split into 512+64 — a single block trips `PassManager::run failure` (the
  comment in-source). Don't "simplify" it back.
- Don't carry NVIDIA `num_warps=8` / `BLOCK=64` — HIP wants `num_warps=1`, `BLOCK=4` for grouped MLA
  (wave64 spill otherwise).
- Slower than asm MLA on most shapes (except the gfx942 +2–3% case) — don't ship as serving SOTA without
  measuring.

## Recorded optimization results (fp8 top-k sparse MLA, split-K decode)
From GEAK runs on a Triton split-K sparse MLA decode (512 latent + 64 rope, `BLOCK_H=16`, 64-row KV tile,
decode batch 1–7, OCP fp8) — geometry unlike the dense defaults above; associations, not rules. Fewer than a
third of the directions citing this page beat their parent, and on average they matched siblings that did not.
- **Occupancy pays only if the grid fills it.** A 32-row KV tile cut VGPR from 254+16 AGPR to 204 (occupancy
  1→2) but alone was slower (grid ≈ one workgroup per CU). With 2× splits (2 workgroups per CU) it gave 1.074×
  call-weighted over the seed; bs1, already one tile per split, kept the 64-row launch (32 rows: 1.30×→1.01×).
- **Full-width dots cut LDS barriers.** One 512-wide QK and PV dot per trip instead of four 128-wide ones took
  loop `s_barrier` 18→6 and `lgkmcnt(0)` 19→7 at equal MFMA count; bs2–bs7 gained 2.9–5.9%. The engineer traced
  the barriers to trips through one shared LDS buffer, not to softmax reductions.
- **Gluon single-tile path:** staging the gathered KV tile in LDS once as the B operand of both QK and PV gave
  +3.9% call-weighted (bs1 +8.7%, bs2 +10.8%); a direct global→LDS variant failed the ISA gate.
- **Did not pay there:** a 128-row tile where it cut serial depth (0.9916×); `.cs` partial stores (slower);
  `.cv` KV loads (gather de-vectorized to `global_load_ubyte`) — `.cg` stayed best; pipeline depth 2 at bs2;
  epilogue LDS-hop removal (none existed).
- **Split-K + reduce fusion:** one arrival-counter form (`scope="gpu"` release) cost 5458–7725 ns/call vs a
  1790 ns dispatch floor; other forms unmeasured. A faster bf16 reduce on gfx950 (4.2→3.4 µs microbench) did
  not move the ~33–38 µs graph-timed decode beyond noise.

## Worked example
gfx942 DeepSeek decode where you want to try the Triton MLA edge:
1. vLLM `--attention-backend ROCM_AITER_TRITON_MLA` (gfx942 favors this per vendor).
2. Confirm `_fwd_kernel_stage1` (Triton) fired; `Lk=576 → BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK=4`.
3. Bench TPS vs `ROCM_AITER_MLA` (asm); on gfx942 Triton may be +2–3%, on gfx950 asm wins.
4. Use this kernel as the parity oracle for any asm-path change.

## How to verify (bench + oracle)
Greedy temp=0 parity vs `mla_decode_fwd` (≥10 prompts) — this is the asm path's oracle. Isolated decode
bench vs asm at the served shape; `AMDGCN_ENABLE_DUMP=1` ISA check (no `scratch_`).

## Alternatives / cross-links
[./aiter.md](aiter.md) (serving SOTA) · [./ck.md](ck.md) · [../../attention_decode_paged/backends/triton.md](../../attention_decode_paged/backend/triton.md) ·
[../../attention_prefill_fmha/backends/triton.md](../../attention_prefill_fmha/backend/triton.md) · `languages/triton_amd/` · [../overview.md](../README.md).

## Sources
- Triton MLA kernels (on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/mla_decode.py`: `Lk==576 → BLOCK_DMODEL=512 + BLOCK_DPE=64`, HIP `BLOCK=4`/`num_warps=1`, `NUM_KV_SPLITS` stage1/stage2; `mla_decode_rope.py`).
- ROCM_AITER_TRITON_MLA +2–3% TPS on gfx942 (vendor): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- Triton AMD knobs (num_stages=1, wave64): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/mla_attention/backends/triton.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.
