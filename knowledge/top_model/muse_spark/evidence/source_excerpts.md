# Container source excerpts

These excerpts preserve original line numbers. The SHA256 is of the entire source file, not only the excerpt. Complete selected changes are in [source diffs](source_changes.json); exact full-source archives are identified by [image inventories](image_source_inventory.json). Source comments describe implementation intent; any measurements in upstream comments are not treated as locally verified results.

## E1

Image: `lmsysorg/sglang-rocm@sha256:7343002d0dc2c1e525c0563f0c5c0377b471f966b3d604343b93a6366c2f7935`

Path: `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/qk_norm_rope_cache.py`

File SHA256: `df7233ecd09eea43501bc03d36a247bd54cb9aef67b8865410449d877e761df3`

Lines 1–18:

```text
   1: """Per-head QK RMSNorm + interleaved RoPE + paged FP8 KV-cache store (Muse Spark, head_dim 64, page 16).
   2:
   3: * ``qk_norm_rope_cache``: from projected Q/K/V. One program per (token, Q head) normalizes, scales and rotates
   4:   Q; the first Q head of each GQA group also normalizes and rotates its K and stores K/V into the FP8 cache
   5:   (optionally through the full -> SWA slot mapping).
   6: * ``qkv_proj_qk_norm_rope_cache``: the same epilogue fed straight from the BF16 split-K partials of the fused
   7:   QKV + output-gate projection (``bf16_splitk_gemm``), which also emits V and the gate.
   8:
   9: The BF16 round trips of the unfused sequence (norm output, query scaling, RoPE products, projection store)
  10: are explicit, so the results match the separate norm / RoPE / cache-write kernels.
  11: """
  12:
  13: import torch
  14: import triton
  15: import triton.language as tl
  16:
  17: from sglang.kernels.ops.gemm.bf16_splitk_gemm import bf16_splitk_gemm_partials
  18:
```

Lines 86–109:

```text
  86:     q = tl.load(Q + token * QS + head * D + d)
  87:     q = _norm_rope(q, COS, POS, EPS, SCALE, D, ROPE, True, COS_STRIDE)
  88:     tl.store(QO + token * HQ * D + head * D + d, q)
  89:     if head % (HQ // HK) == 0:
  90:         kh = head // (HQ // HK)
  91:         k = tl.load(K + token * KSTR + kh * D + d)
  92:         k = _norm_rope(k, COS, POS, EPS, 1.0, D, ROPE, False, COS_STRIDE)
  93:         tl.store(KO + token * HK * D + kh * D + d, k)
  94:         slot = tl.load(LOC + token).to(tl.int64)
  95:         if slot >= 0:
  96:             if HAS_MAP:
  97:                 slot = tl.load(MAP + slot).to(tl.int64)
  98:             if slot >= 0:
  99:                 v = tl.load(V + token * VSTR + kh * D + d)
 100:                 if HAS_SCALE:
 101:                     k_value = k.to(tl.float32) / tl.load(KS)
 102:                     v_value = v.to(tl.float32) / tl.load(VS)
 103:                 else:
 104:                     k_value = k.to(tl.float32)
 105:                     v_value = v.to(tl.float32)
 106:                 offset = slot * HK * D + kh * D + d
 107:                 tl.store(KC + offset, k_value)
 108:                 tl.store(VC + offset, v_value)
 109:
```

Lines 233–278:

```text
 233: def qkv_proj_qk_norm_rope_cache(
 234:     x,
 235:     weight,
 236:     positions,
 237:     cos_sin,
 238:     cache_k,
 239:     cache_v,
 240:     locations,
 241:     mapping,
 242:     eps,
 243:     q_scale,
 244:     k_scale,
 245:     v_scale,
 246: ):
 247:     """Fused QKV + gate projection of ``x`` [M <= 32, 8192] with the QK-norm / RoPE / FP8-cache epilogue.
 248:
 249:     Returns (Q [M, 1024], K [M, 128], V [M, 128], gate [M, 1024]), BF16.
 250:     """
 251:     m, k = x.shape
 252:     assert 0 < m <= 32 and k == 8192 and weight.shape == (2304, 8192)
 253:     assert x.dtype == weight.dtype == torch.bfloat16
 254:     assert x.is_contiguous() and weight.is_contiguous()
 255:     assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
 256:     assert cache_k.is_contiguous() and cache_v.is_contiguous()
 257:     partials = torch.empty((8, m, 2304), device=x.device, dtype=torch.float32)
 258:     bf16_splitk_gemm_partials(x, weight, partials, block_n=32, block_k=128, split_k=8)
 259:     qo = torch.empty((m, 1024), device=x.device, dtype=torch.bfloat16)
 260:     ko = torch.empty((m, 128), device=x.device, dtype=torch.bfloat16)
 261:     vo = torch.empty_like(ko)
 262:     go = torch.empty_like(qo)
 263:     _qkv_epilogue[(m, 16)](
 264:         partials,
 265:         positions,
 266:         cos_sin,
 267:         locations,
 268:         mapping,
 269:         cache_k,
 270:         cache_v,
 271:         qo,
 272:         ko,
 273:         vo,
 274:         go,
 275:         k_scale,
 276:         v_scale,
 277:         m,
 278:         2304,
```

## E2

Image: `lmsysorg/sglang-rocm@sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`

Path: `/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/runner.py`

File SHA256: `267c5b33db7be6ce439a33e417166d6d4b91a2c45e33457d7a84a15539f8ffeb`

Lines 1–14:

```text
   1: # SPDX-License-Identifier: Apache-2.0
   2: """One rank's Muse Spark launches on gfx950.
   3:
   4: A decoder layer with MXFP4 experts runs as K1 (``stages/pre_attn``: the fused
   5: projection, the q / k norms and query scale, RoPE, the fp8 cache write), K-attn
   6: (``stages/attention``: the verify attention over the cache), K2
   7: (``stages/post_attn``: o-norm-gate, o_proj with the attention all-reduce, the
   8: attention boundary, router / pooler, top-8, the pre-expert norm) and K3
   9: (``stages/ffn``: the experts, finalize with the EP all-reduce, the sharded
  10: unpooler, the FFN boundary). The bf16-expert layers (0 and 61) run SGLang's own
  11: decoder layer, as the stage-0 plugin did. A launch takes ``(stream, attn_input)``,
  12: the fp32 residual stream (updated in place) and the layer's bf16 attention input,
  13: and hands on the next pair.
  14:
```

Lines 97–101:

```text
  97: # SGLANG_AMD_MONO_MUSE_KERNELS=0: every layer on SGLang's modules (the stage-0 plugin)
  98: USE_KERNELS = os.environ.get("SGLANG_AMD_MONO_MUSE_KERNELS", "1") == "1"
  99: # an MXFP4 layer's attention, K2, K3 and the next layer's K1 as one launch (the
 100: # default); SGLANG_AMD_MONO_MUSE_LAYER=0: the four launches
 101: USE_LAYER = os.environ.get("SGLANG_AMD_MONO_MUSE_LAYER", "1") == "1"
```

Lines 335–342:

```text
 335:     def merged(self, i: int, rows: int, q_len: int) -> bool:
 336:         """Layer ``i`` runs as one launch (the attention, K2, K3 and, when the next
 337:         layer takes the kernels too, its K1)."""
 338:         return self.use_layer and self.views[i].kind == "mxfp4" and rows % q_len == 0
 339:
 340:     def _next_k1(self, i: int) -> bool:
 341:         return i + 1 < len(self.views) and self.views[i + 1].kind == "mxfp4"
 342:
```

Lines 532–566:

```text
 532:         if self._k1_done != i:  # else the previous merged launch ran this layer's K1
 533:             self.k1_kernel(rows)(
 534:                 _ptr(attn_input), _ptr(L.w_qkvg), _ptr(cos_sin), _ptr(positions), _ptr(slots), _ptr(m),
 535:                 k_cache, v_cache, _ptr(self.q), _ptr(self.k), _ptr(self.v), _ptr(self.g),
 536:                 _ptr(self.scratch_k1), _ptr(self.epoch), i, int(L.use_rope), int(is_swa),
 537:                 int(slots.dtype == torch.int64), int(m.dtype == torch.int64), stream=cur,
 538:             )  # fmt: skip
 539:         self._k1_done = -1
 540:         last = i == len(self.modules) - 1
 541:         if self.merged(i, rows, q_len):
 542:             next_k1 = self._next_k1(i)
 543:             L1 = self.views[i + 1] if next_k1 else L
 544:             k_cache1, v_cache1, is_swa1 = self.caches[i + 1] if next_k1 else self.caches[i]
 545:             cos_sin1 = L1.cos_sin if L1.cos_sin is not None else self.any_cos_sin
 546:             self.layer_kernel(rows, q_len, next_k1)(
 547:                 _ptr(self.q), _ptr(self.o), *self._attn_args(i, rows, forward_batch, q_len),
 548:                 _ptr(self.g), _ptr(stream), _ptr(L.w_o), _ptr(L.alpha), _ptr(L.beta), _ptr(L.pre_w),
 549:                 _ptr(L.gate_hi), _ptr(L.gate_lo), _ptr(L.bias), _ptr(self._pool_w(i, rows)), _ptr(L.xn_w),
 550:                 _ptr(self.ids), _ptr(self.scores), _ptr(self.xin), _ptr(self.scratch_k2),
 551:                 int(self.modules[i].mlp.local_expert_start), *self._experts(i), _ptr(L.pen_w), _ptr(L.pfw), _ptr(L.alpha2), _ptr(L.beta2), _ptr(L.npw),
 552:                 _ptr(L.w_unpool_fp8 if self.fp8_dense else L.w_unpool), _ptr(self.attn_in), _ptr(self.scratch_k3),
 553:                 _ptr(L1.w_qkvg), _ptr(cos_sin1), _ptr(positions), _ptr(slots), _ptr(m), k_cache1,
 554:                 v_cache1, _ptr(self.k), _ptr(self.v), _ptr(self.scratch_k1), int(L1.use_rope),
 555:                 int(is_swa1), int(slots.dtype == torch.int64), int(m.dtype == torch.int64),
 556:                 self.peers.local, _ptr(self.peers.addresses), self.rank, _ptr(self.epoch), i,
 557:                 int(last), stream=cur,
 558:             )  # fmt: skip
 559:             if next_k1:
 560:                 self._k1_done = i + 1
 561:             return stream, self.attn_in[:rows]
 562:         self.kattn_kernel(rows, q_len)(
 563:             _ptr(self.q), _ptr(self.o), *self._attn_args(i, rows, forward_batch, q_len),
 564:             _ptr(self.epoch), i, stream=cur,
 565:         )  # fmt: skip
 566:         self.k2_kernel(rows)(
```

## E3

Image: `lmsysorg/sglang-rocm@sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`

Path: `/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/layer.py`

File SHA256: `507604a48233c776fad241df7b62ce5639bc862ecb53957f85d85675e82a2bed`

Lines 294–307:

```text
 294: def build_layer(key: LayerBuild):
 295:     """``@flyc.jit`` launcher of a whole MXFP4 layer: the attention, K2, K3 and,
 296:     with ``key.next_k1``, the next layer's K1, in one launch for ``key.tokens``
 297:     rows, ``key.q_len`` a request, on a rank of a ``key.npes``-way TP / EP group.
 298:
 299:     Every stage boundary is a mailbox flag: the attention's merge tasks flag each
 300:     (request, KV head, row tile)'s o rows for K2's og tasks; K2's route / xn tasks
 301:     flag each token for K3; K3's boundary task flags each token's x for the next
 302:     layer's K1 (under that layer's tag). The LDS is the union of the stages' (a
 303:     CTA is done with a stage's rows before it starts the next); the stages'
 304:     scratch regions stay separate. A CTA runs the next layer's K1 only after its
 305:     K3 stages, which every CTA's attention and K2 stages precede (K3's chain of
 306:     flags runs through every token's route and expert input), so K1's q / g
 307:     outputs are written after every read of this layer's.
```

Lines 393–419:

```text
 393:             }  # fmt: skip
 394:             c2 = k2_context(k2_key, lds.k2.peek(), k2_args, scratch2, sym, peers, rank, tag, k2_layout, layer=layer)
 395:             k3_args["xinq"] = c2["xinq"]  # (read by K3 only with xq_pre)
 396:             oflag, rflag, xinflag = c2["oflag"], c2["rflag"], c2["xinflag"]
 397:             ca = kattn_context(ka_key, lds, a_args, scratch_a, tag, ka_layout, layer=layer)
 398:             attention.run_kattn(ca, o_flag=oflag)
 399:             gpu.barrier()  # this CTA is done with the attention's LDS
 400:             post_attn.run_k2(c2, route_flag=rflag, xin_flag=xinflag, o_flag=oflag, QL=QL)
 401:             gpu.barrier()
 402:             c3 = k3_context(k3_key, lds.k3.peek(), k3_args, scratch3, sym, peers, rank, tag, k3_layout, sy, layer=layer)
 403:             if const_expr(next_k1):
 404:                 k1_args = {
 405:                     "x": c2["c1"] if defer else attn_out, "w": w1, "pfw": pfw, "beta2": beta2, "npw": npw, "cos_sin": cos_sin, "positions": positions, "slots": slots,
 406:                     "swa_map": swa_map, "k_cache": k_cache1, "v_cache": v_cache1, "q_out": q,
 407:                     "k_out": k_out, "v_out": v_out, "g_out": g, "use_rope": use_rope1,
 408:                     "use_swa": use_swa1, "slot64": slot64, "swa64": swa64,
 409:                 }  # fmt: skip
 410:                 c1 = k1_context(k1_key, lds.k1.peek(), k1_args, scratch1, tag1, k1_layout, merged=True, layer=layer + 1)
 411:                 if const_expr(defer):
 412:                     # the unpooled rows and the scalars are this layer's (K3's tag)
 413:                     c1.update(d_poll=c3["poll"], d_unp=c3["own_unp"], d_nscal=c3["nscal"])
 414:                 ffn.run_k3(c3, route_flag=rflag, xin_flag=xinflag, x_flag=c1["xflag"], x_put=c1["put"], res_flag=(c2["resflag"], S * post_attn.a_parts(S)))
 415:                 gpu.barrier()
 416:                 pre_attn.run_k1(c1, bid, x_flag=c1["xflag"])
 417:             else:
 418:                 ffn.run_k3(c3, route_flag=rflag, xin_flag=xinflag, res_flag=(c2["resflag"], S * post_attn.a_parts(S)))
 419:
```

Lines 433–445:

```text
 433:         sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32, last_in: Int32,
 434:         stream: fx.Stream = _STREAM,
 435:     ):  # fmt: skip
 436:         _ = keyed
 437:         layer_(
 438:             q, o, k_cache, v_cache, page_table, table_stride, table_bytes, seq_lens, lens64, extra,
 439:             window, part_in, scale_bits, scratch_a, g, res, w_o, alpha, beta, pre_w, w_hi, w_lo, bias,
 440:             w_pool, xn_w, ids, scores, xin, scratch2, local_start, w13, s13, w2, s2, pen_w, pfw, alpha2,
 441:             beta2, npw, w_unpool, attn_out, scratch3, w1, cos_sin, positions, slots, swa_map, k_cache1,
 442:             v_cache1, k_out, v_out, scratch1, use_rope1, use_swa1, slot64, swa64, sym, peers, rank,
 443:             epoch, layer, last_in,
 444:         ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip
 445:
```

## E4

Image: `lmsysorg/sglang-rocm@sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`

Path: `/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/finalize_allreduce.hip`

File SHA256: `8ab37d684c03d8063428854fd416da182280863f6b9cf83611d7f4fc4ba03e77`

Lines 1–19:

```text
   1: // Muse Spark MoE finalize fused with the TP4 all-reduce of its output (gfx950, xGMI peers).
   2: //
   3: // One 256-thread block per token row r of an M-row batch (M a multiple of 4, hidden 4096, top-8):
   4: //   1. finalize: the per-expert input-scale RMSNorm and route-weighted sum of the row's local-expert slots, with the
   5: //      thread mapping, reduction order and rounding points of aiter's FlyDSL muse_moe_finalize_vector_flydsl;
   6: //   2. the row's partial goes to its owner rank r / (M / 4), which sums the four partials in the order of aiter's
   7: //      cross_device_reduce_2stage (owner first, then owner+1, ...; FP32, one BF16 rounding) and pushes the sum to
   8: //      the other ranks; every rank writes the row to `out`.
   9: // So the result is bitwise the FlyDSL finalize followed by the aiter two-stage all-reduce, except that a -0.0 can
  10: // come out as +0.0.
  11: //
  12: // The data is its own flag (Lamport): inbox records start as the BF16 -0.0 pattern (0x8000), senders map -0.0 to
  13: // +0.0, readers spin with cache-bypassing volatile loads until no lane holds the sentinel. Each row keeps a call
  14: // counter and the owner of its previous call (advanced identically on every rank because every rank runs the same
  15: // shapes); row-call c uses inbox slot c % 3 and resets the records the previous row-call left in this rank's inbox
  16: // (slot (c + 2) % 3, placed by that call's owner), which no peer can write again before this rank's next call.
  17: // Inbox accesses are raw buffer ops with the sc0|sc1 cache policy on coarse-grained IPC memory.
  18: #include <hip/hip_runtime.h>
  19: #include <cstdint>
```

Lines 106–120:

```text
 106: }
 107:
 108: extern "C" __global__ void __launch_bounds__(256) muse_finalize_allreduce_tp4(Args a) {
 109:     const int r = blockIdx.x;
 110:     const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
 111:     const uint32_t call = (uint32_t)a.row_state[2 * r];
 112:     const int prev_owner = a.row_state[2 * r + 1];
 113:     const uint32_t slot = call % 3u, prev = (call + 2u) % 3u;
 114:     const int owner = r / (a.rows / NR);
 115:     __shared__ float sums[TOPK * 4];
 116:     __shared__ uint16_t rowbuf[HID];
 117:
 118:     // ---- finalize (FlyDSL muse_moe_finalize_vector_flydsl numerics: H 4096, 256 threads, vec 4) ----
 119:     float scl[16], st[16];
 120: #pragma unroll
```

Lines 142–185:

```text
 142:                 for (int j = 0; j < 4; ++j) {
 143:                     const float val = bf2f(f2bf(bf2f(yw[j]) * scl[c * 4 + j]));
 144:                     v[c * 4 + j] = val;
 145:                     ss = ss + val * val;
 146:                 }
 147:             }
 148: #pragma unroll
 149:             for (int sh = 32; sh >= 1; sh >>= 1) ss = ss + __shfl_xor(ss, sh, 64);
 150:             if (lane == 0) sums[s * 4 + wave] = ss;
 151:             __syncthreads();
 152:             float total = 0.f;
 153: #pragma unroll
 154:             for (int w = 0; w < 4; ++w) total = total + sums[s * 4 + w];
 155:             const float mean = total * (1.0f / HID) + a.eps;
 156:             sc = sc * (1.0f / MUSE_SQRT(mean));
 157: #pragma unroll
 158:             for (int k = 0; k < 16; ++k) st[k] = st[k] + sc * v[k];
 159:         }
 160:     }
 161: #pragma unroll
 162:     for (int c = 0; c < 4; ++c) {
 163:         uint2 o;
 164:         o.x = no_neg_zero(f2bf(st[c * 4])) | (no_neg_zero(f2bf(st[c * 4 + 1])) << 16);
 165:         o.y = no_neg_zero(f2bf(st[c * 4 + 2])) | (no_neg_zero(f2bf(st[c * 4 + 3])) << 16);
 166:         *(uint2*)(rowbuf + tid * 4 + c * 1024) = o;
 167:     }
 168:     __syncthreads();
 169:     uint4 mine[2];
 170: #pragma unroll
 171:     for (int i = 0; i < 2; ++i) mine[i] = ((const uint4*)rowbuf)[tid + i * 256];
 172:
 173:     const rsrc_t my = make_rsrc(a.peers.inbox[a.rank]);
 174:     uint4* outrow = a.out + (size_t)r * PACKS_PER_ROW;
 175:     if (owner != a.rank) {
 176:         // ---- push the partial to the owner, then wait for the owner's sum ----
 177:         const rsrc_t dst = make_rsrc(a.peers.inbox[owner]);
 178:         const uint32_t base = s1_off(a, slot, a.rank, r);
 179: #pragma unroll
 180:         for (int i = 0; i < 2; ++i) bst_x4(dst, base + (tid + i * 256) * 16, mine[i]);
 181:         const uint32_t rbase = s2_off(a, slot, r);
 182: #pragma unroll
 183:         for (int i = 0; i < 2; ++i) {
 184:             uint4 t;
 185:             do { t = bld_x4(my, rbase + (tid + i * 256) * 16); } while (has_sentinel(t));
```

## E5

Image: `lmsysorg/sglang-rocm@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2`

Path: `/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/attention_kv_write.py`

File SHA256: `a21de58492ca4d4d59bbbb2d57d168aa5f77cb2a696e236277e1ce25d32548c0`

Lines 32–39:

```text
  32: logger = logging.getLogger(__name__)
  33:
  34: QK_ROPE_CACHE = envs.SGLANG_MUSE_QK_ROPE_CACHE.get()
  35: FUSED_QKV_CACHE = envs.SGLANG_MUSE_FUSED_QKV_CACHE.get()
  36: QK_ROPE_CACHE_MAX_ROWS = envs.SGLANG_MUSE_QK_ROPE_CACHE_MAX_ROWS.get()
  37: _logged: set = set()
  38:
  39:
```

Lines 68–83:

```text
  68:     backend = get_attn_backend()
  69:     layer = attention.attn
  70:     if not (
  71:         isinstance(backend, AiterAttnBackend)
  72:         and not backend.use_mla
  73:         and backend.page_size in (16, 32, 64, 128)
  74:         and not backend.kv_cache_is_vectorized_5d
  75:         and backend.use_triton_unified_attention
  76:         and not layer.is_cross_attention
  77:         and backend.kv_cache_dtype == torch.float8_e4m3fn
  78:     ):
  79:         return None
  80:     rope = attention.rotary_emb
  81:     if rope is not None and not (
  82:         type(rope).__name__ == "RotaryEmbedding"
  83:         and not rope.is_neox_style
```

Lines 173–189:

```text
 173:     q: torch.Tensor,
 174:     k: torch.Tensor,
 175:     v: torch.Tensor,
 176:     forward_batch: ForwardBatch,
 177: ):
 178:     """(q, k) normalized and rotated with K/V already in the cache, or None when the kernel does not apply."""
 179:     if not (
 180:         0 < q.shape[0] <= QK_ROPE_CACHE_MAX_ROWS
 181:         and q.dtype == k.dtype == v.dtype == torch.bfloat16
 182:         and q.stride(1) == k.stride(1) == v.stride(1) == 1
 183:         and _attention_shape_ok(attention, forward_batch)
 184:     ):
 185:         return None
 186:     target = _kv_write_target(attention, forward_batch)
 187:     if target is None:
 188:         return None
 189:     _note("QK norm/RoPE/FP8 cache", q.shape[0])
```

## E6

Image: `lmsysorg/sglang-rocm@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2`

Path: `/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/finalize_allreduce.py`

File SHA256: `b27664445f8928ac5b358ba15b82a0bfd3e3d113874200f40ffa922bf978bd92`

Lines 146–163:

```text
 146:         max_rows: int = 512,
 147:         sqrt: str = "__builtin_amdgcn_sqrtf",
 148:     ):
 149:         self.rank = dist.get_rank(group=cpu_group)
 150:         self.nr = dist.get_world_size(group=cpu_group)
 151:         if self.nr not in (4, 8):
 152:             raise ValueError("finalize_allreduce supports TP4 and TP8 only")
 153:         # TP8 rows where this beats FlyDSL finalize + aiter all-reduce (MI355X graph replay): one-shot at 8 rows
 154:         # (22.1 -> 19.0 us; aiter's one-stage order there), two-stage from 128 (24.4 -> 22.1 us, 39.7 -> 35.5 at
 155:         # 512). Between, both forms are slower (one-shot 25.9 vs 19.3 us at 32 rows).
 156:         self.oneshot_below = 16 if self.nr == 8 else 0
 157:         self.twostage_from = 128 if self.nr == 8 else 0
 158:         self.device = device
 159:         self.max_rows = int(max_rows)
 160:         h = _hip_lib()
 161:         with torch.cuda.device(device):
 162:             arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
 163:             co = _build(arch, sqrt, self.nr)
```

Lines 207–219:

```text
 207:             torch.cuda.synchronize(device)
 208:         dist.barrier(group=cpu_group)
 209:
 210:     def applies(self, rows: int) -> bool:
 211:         return (
 212:             0 < rows <= self.max_rows
 213:             and rows % self.nr == 0
 214:             and (rows < self.oneshot_below or rows >= self.twostage_from)
 215:         )
 216:
 217:     def __call__(
 218:         self,
 219:         y,
```

## E7

Image: `lmsysorg/sglang-rocm@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2`

Path: `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/muse_extend_flydsl.py`

File SHA256: `cdfa8cadad2f252028875ef1b4e9506281161f7b974f7cfe47326bc49fcf882e`

Lines 1–43:

```text
   1: """Muse Spark full-attention extend (prefill) on gfx950 in FlyDSL.
   2:
   3: FP8 e4m3 Q [tokens, 8 * KV heads, 64] (per-tensor descale) over a paged FP8 e4m3 KV cache [pages, 16, KV heads, 64],
   4: causal, BF16 output. Same API as ``muse_extend_gluon.muse_extend_attention`` (``muse_extend_attention_fly``).
   5:
   6: One workgroup = (16-token query block, KV head) = 128 query columns (16 tokens x GQA 8); 4 waves of 32 columns
   7: (4 tokens x 8 heads), 3 waves / SIMD (waves_per_eu=3, ~156 VGPRs). Transposed form on 32x32x64 scaled FP8 MFMAs:
   8:   * S^T = K Q^T per 128-key tile: 4 MFMAs (key tiles of 32), K A-fragments from LDS (2 x ds_read_b128), Q^T as the
   9:     B fragment in 8 VGPRs for the whole kernel (lane half hl holds dims 32 hl .. 32 hl + 31 for both operands).
  10:   * Softmax along keys = accumulator rows, in-lane over the 64 rows a lane holds (v_max3); the rescale runs only
  11:     when a score exceeds the running max (ballot, wave-uniform, llvm.expect -> out-of-line block): P = 448 *
  12:     exp2(s c - m) as in the Gluon kernel. All 32 pk_fma exponent arguments are formed before the exps.
  13:   * P^T accumulators, packed to FP8, are the PV B operand as-is (byte b of half hl = key
  14:     64 ks + 32 (b // 16) + 8 ((b % 16) // 4) + 4 hl + b % 4); V^T A-fragments come from ds_read_b64_tr_b8 in that key
  15:     order. Row sums on the matrix core (ones x P^T, 2 extra MFMAs / tile: cheaper than 32 VALU adds).
  16:   * K / V tiles (128 keys x 64 B each) are LDS-DMA'd (buffer_load ... lds) by the 4 waves, one 16-key page per
  17:     instruction, through a per-page buffer descriptor (64-bit page base, num_records = valid rows): rows past the
  18:     request read as zero (no stale NaN bytes reach P.V) and there is no 2^31 offset limit. 2 LDS stages, one bare
  19:     s_barrier per tile, the loop unrolled per stage with one LLVM alias scope per stage so the compiler waits only
  20:     for the DMA of the stage being read. 16-byte chunks XOR-swizzled by row ((r>>1) + 2 (r>>3)) & 3: the b128 K
  21:     reads and the transposed V reads are bank-conflict free.
  22:   * Page ids are vector loads (SMEM loads placed after LDS DMAs get a conservative vmcnt(0)).
  23: See DEVLOG.md for the version history, measurements and dead ends.
  24: """
  25:
  26: import functools
  27: import math
  28:
  29: import flydsl.compiler as flyc
  30: import flydsl.expr as fx
  31: from flydsl._mlir import ir
  32: from flydsl._mlir.dialects import llvm
  33: from flydsl._mlir.dialects import rocdl as _rocdl
  34: from flydsl.expr import arith, gpu, range_constexpr
  35: from flydsl.expr import math as fmath
  36: from flydsl.expr.typing import T
  37:
  38: from aiter.ops.flydsl.kernels import buffer_ops
  39: from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, _to_raw
  40:
  41: GQA = 8
  42: HD = 64
  43: BQ = 16  # query tokens per workgroup
```

## E8

Image: `lmsysorg/sglang-rocm@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2`

Path: `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/muse_verify_balanced.py`

File SHA256: `7ab7b5084c44fd90cd128ed4304d4e33939bb56b0f5590a9b5dc83563484b19a`

Lines 1–28:

```text
   1: """Muse Spark full-layer target verify with a length-balanced split-KV work list.
   2:
   3: The FlyDSL verify kernel (``muse_verify_flydsl``) used to give every request the same number of splits, so in a batch
   4: of very uneven contexts the longest request set the time. Here one tiny planning kernel splits the batch's total KV
   5: chunks into equal segments of whole slot groups (each request gets ``ceil(chunks / segment)`` of them, at least one),
   6: the verify kernel runs one segment per program and writes its partials to the segment's slot, and the merge reads
   7: each request's own segments. Everything stays on the device, so the step is graph-capturable.
   8:
   9: Partials: ``[segments + 1, 8, heads, 64]`` FP32 with max / denom ``[segments + 1, 8, heads]`` (log2 domain); the last
  10: slot takes the idle programs' writes.
  11: """
  12:
  13: import torch
  14: import triton
  15: import triton.language as tl
  16: from sglang.kernels.ops.attention.muse_verify_flydsl import (
  17:     CHUNK,
  18:     QL,
  19:     _launcher,
  20:     _run_compiled,
  21:     fx,
  22: )
  23:
  24: # slot groups per chunk group of the kernel's main loop (``slots`` of verify_fly_partials)
  25: NS = 2
  26: NUM_CUS = 256
  27: # smallest segment, in chunks (a segment pays ~2 us of fixed cost: Q load, prefetch fill, epilogue)
  28: MIN_SEGMENT_CHUNKS = 4
```

Lines 40–86:

```text
  40:     SEGMENTS: tl.constexpr,
  41:     BLOCK_R: tl.constexpr,
  42:     CHUNK_: tl.constexpr,
  43:     NS_: tl.constexpr,
  44:     MIN_CHUNKS: tl.constexpr,
  45:     BLOCK_P: tl.constexpr,
  46: ):
  47:     # every program recomputes the per-request counts (BLOCK_R values) and fills BLOCK_P plan rows
  48:     r = tl.arange(0, BLOCK_R)
  49:     rmask = r < num_seqs
  50:     lens = tl.load(SEQ_LENS + r, mask=rmask, other=0).to(tl.int32)
  51:     nchunks = (lens + CHUNK_ - 1) // CHUNK_
  52:     if WINDOW > 0:
  53:         # sliding layers: from the chunk of the first key any of the 8 queries sees
  54:         first = tl.maximum(lens - 8 + 1 - WINDOW, 0) // CHUNK_
  55:     else:
  56:         first = tl.zeros_like(nchunks)
  57:     chunks = nchunks - first
  58:     total = tl.sum(chunks, axis=0)
  59:     # segment length: the total over the segments left after one per request, in whole slot groups
  60:     budget = tl.maximum(SEGMENTS - num_seqs, 1)
  61:     seg = (total + budget - 1) // budget
  62:     seg = tl.maximum(seg, MIN_CHUNKS)
  63:     seg = (seg + NS_ - 1) // NS_ * NS_
  64:     count = tl.where(rmask, tl.maximum((chunks + seg - 1) // seg, 1), 0)
  65:     start = tl.cumsum(count, axis=0) - count
  66:     if tl.program_id(0) == 0:
  67:         tl.store(SEG_START + r, start, mask=rmask)
  68:         tl.store(SEG_COUNT + r, count, mask=rmask)
  69:     used = tl.sum(count, axis=0)
  70:     p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
  71:     # owner request of program p: the last request whose first segment is <= p
  72:     owner = tl.sum((start[None, :] <= p[:, None]) & rmask[None, :], axis=1) - 1
  73:     owner = tl.maximum(owner, 0)
  74:     own_start = tl.sum(
  75:         tl.where(r[None, :] == owner[:, None], start[None, :], 0), axis=1
  76:     )
  77:     own_chunks = tl.sum(
  78:         tl.where(r[None, :] == owner[:, None], chunks[None, :], 0), axis=1
  79:     )
  80:     own_first = tl.sum(
  81:         tl.where(r[None, :] == owner[:, None], first[None, :], 0), axis=1
  82:     )
  83:     i = p - own_start
  84:     begin = own_first + i * seg
  85:     end = own_first + tl.minimum(i * seg + seg, own_chunks)
  86:     busy = p < used
```
