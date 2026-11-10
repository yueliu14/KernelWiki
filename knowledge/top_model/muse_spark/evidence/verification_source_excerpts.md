# Supplemental container-source excerpts

These excerpts come from the CPU-exported immutable images, not a guessed upstream commit. Full file and symbol hashes are in [verified_source_audit.json](verified_source_audit.json). Line numbers refer to the container files.

## V03: attention_kv_write.py, lines 118–124

Image `lmsysorg/sglang-rocm:muse-spark-open-1027@sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`. Path `/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/attention_kv_write.py`. File SHA256 `34933e2895490a3719f918779df60c9575a7b05abccf3d51a03a396b31752adf`.

```python
 118:
 119:
 120: def _note(name: str, rows: int) -> None:
 121:     if name not in _logged:
 122:         _logged.add(name)
 123:         logger.info("Muse fast path: %s, rows=%d", name, rows)
 124:
```

## V04: router_pooler.py, lines 37–63

Image `lmsysorg/sglang-rocm:muse-spark-open-mi355-1106@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2`. Path `/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/router_pooler.py`. File SHA256 `d1e1447d7980b69734bc17e7f49fc106dafc2042adb2f5f6ff4f474d949362d5`.

```python
  37: # Rows above 64 go to the joint form wherever it has a row-major tile (BM 128 in bf16_tiled_small_m_gemm.ROW_MAJOR).
  38: JOINT_MAX_ROWS = envs.SGLANG_MUSE_JOINT_ROUTER_MAX_ROWS.get()
  39: _logged: set = set()
  40:
  41:
  42: def can_share_router_pooler(moe, x: torch.Tensor) -> bool:
  43:     """The Muse Spark TP8 routing front: 8192 hidden, 256 experts top-8 sigmoid, 4096-wide pooler + norm."""
  44:     m = x.shape[0]
  45:     if not (0 < m <= max(64, JOINT_MAX_ROWS) and _routing_front_matches(moe, x)):
  46:         return False
  47:     if m <= 64:
  48:         return True
  49:     # Above 64 rows only the joint form serves: the shared form's completed-pooler epilogue stops at 64.
  50:     packed = getattr(moe, "router_pooler_packed_weight", None)
  51:     return (
  52:         JOINT_ROUTER_POOLER
  53:         and envs.SGLANG_MUSE_DENSE_TILED.get()
  54:         and packed is not None
  55:         and joint_router_pooler_applies(x, packed)
  56:     )
  57:
  58:
  59: def can_route_fp8_joint(moe, x: torch.Tensor, x_fp8) -> bool:
  60:     """The FP8 joint form's wider admission: the same routing front for up to ``FP8_JOINT_MAX_M`` rows when the
  61:     boundary kernel handed over the FP8 activation and the FP8 pooler copy exists."""
  62:     return (
  63:         JOINT_POOLER_FP8
```

## V04: router_pooler.py, lines 116–169

Image `lmsysorg/sglang-rocm:muse-spark-open-mi355-1106@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2`. Path `/sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/router_pooler.py`. File SHA256 `d1e1447d7980b69734bc17e7f49fc106dafc2042adb2f5f6ff4f474d949362d5`.

```python
 116:                 m,
 117:             )
 118:         return *joint_router_pooler_topk_fp8(
 119:             x_fp8[0],
 120:             x_fp8[1],
 121:             x,
 122:             moe.router_pooler_packed_weight,
 123:             (moe.router_pooler_fp8_weight, moe.router_pooler_fp8_scale),
 124:             moe.expert_bias,
 125:             norm_weight,
 126:             float(moe.router_multiplier),
 127:             float(moe.route_scale),
 128:             float(moe.pre_expert_norm.eps),
 129:             quant_out=quant_out,
 130:         ), quant_out
 131:     if JOINT_ROUTER_POOLER and (
 132:         m in _JOINT_ROWS
 133:         or (
 134:             envs.SGLANG_MUSE_DENSE_TILED.get()
 135:             and joint_router_pooler_applies(x, moe.router_pooler_packed_weight)
 136:         )
 137:     ):
 138:         packed = moe.router_pooler_packed_weight
 139:         assert moe.pre_expert_proj.weight.data_ptr() == packed.data_ptr()
 140:         key = ("joint", moe.layer_idx, m)
 141:         if key not in _logged:
 142:             _logged.add(key)
 143:             logger.info(
 144:                 "Muse joint router/pooler active: layer=%d rows=%d", moe.layer_idx, m
 145:             )
 146:         return *joint_router_pooler_topk(
 147:             x,
 148:             packed,
 149:             moe.expert_bias,
 150:             norm_weight,
 151:             float(moe.router_multiplier),
 152:             float(moe.route_scale),
 153:             float(moe.pre_expert_norm.eps),
 154:             quant_out=quant_out,
 155:         ), quant_out
 156:     key = "shared" if m <= 16 else "shared_completed"
 157:     if key not in _logged:
 158:         _logged.add(key)
 159:         logger.info(
 160:             "Muse fast path: %s, rows=%d",
 161:             (
 162:                 "shared router/preproject epilogue"
 163:                 if m <= 16
 164:                 else "shared router/norm with hipBLASLt pooler"
 165:             ),
 166:             m,
 167:         )
 168:     return *router_pooler_topk(
 169:         x,
```

## V04: router_pooler_topk.py, lines 410–456

Image `lmsysorg/sglang-rocm:muse-spark-open-mi355-1106@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2`. Path `/sgl-workspace/sglang/python/sglang/kernels/ops/moe/router_pooler_topk.py`. File SHA256 `8492b6be85065fe9dab1863e2624bf94583f328f513c65f7b018fe931b758131`.

```python
 410: # FP8 joint form, BM -> (BN, BK, SK, warps, stages): one launch whose first 4096 / BN programs per K split run the
 411: # pooler tile from the FP8 activation and weight and whose last 512 / BN run the router rows from BF16, all
 412: # writing the same ``[SK, M, 4608]`` partials. Tuned on MI355X (bench_joint_fp8.py).
 413: FP8_JOINT = {
 414:     32: (64, 256, 4, 8, 3),
 415:     64: (32, 128, 4, 4, 2),
 416:     128: (64, 128, 4, 8, 3),
 417:     256: (64, 128, 4, 8, 3),
 418:     512: (128, 128, 4, 8, 2),
 419: }
 420: # Measured 10-23 (bench_joint_fp8.py): 21-22 us at 24-32 rows vs 26.9 for the aiter BF16 launch. MI355X, whole
 421: # front (GEMM + epilogue) in CUDA graphs, vs the BF16 row-major joint: 28.3 -> 25.4 us at 40 rows, 30.0 -> 27.3 at
 422: # 64, 35.7 -> 30.1 at 72, 41.6 -> 34.9 at 128; vs the unfused router GEMM + routing + BF16 pooler + norm above 128
 423: # rows: 79.1 -> 38.2 us at 136, 89.2 -> 46.7 at 256, 106.0 -> 54.1 at 384, 131.5 -> 71.5 at 512.
 424: FP8_JOINT_MIN_M = 17
 425: FP8_JOINT_MAX_M = 512
 426: # Largest row tile of the FP8 joint partials; more rows take more tiles along the grid's third axis.
 427: FP8_JOINT_MAX_TILE_M = 128
 428:
 429:
 430: @triton.jit
 431: def _joint_fp8_partial(
 432:     XQ,
 433:     X,
 434:     WQ,
 435:     WR,
 436:     P,
 437:     M,
 438:     BM: tl.constexpr,
 439:     BN: tl.constexpr,
 440:     BK: tl.constexpr,
 441:     SK: tl.constexpr,
 442:     POOLER_TILES: tl.constexpr,
 443: ):
 444:     N: tl.constexpr = 4608
 445:     K: tl.constexpr = 8192
 446:     KT_PER: tl.constexpr = K // BK // SK
 447:     pn, pk, pm = tl.program_id(0), tl.program_id(1), tl.program_id(2)
 448:     m = pm * BM + tl.arange(0, BM)
 449:     kk = pk * KT_PER * BK + tl.arange(0, BK)
 450:     nn = tl.arange(0, BN)
 451:     acc = tl.zeros((BM, BN), tl.float32)
 452:     # Distinct names per branch: Triton merges same-named values across the if / else and the FP8 and BF16
 453:     # pointers do not share a type.
 454:     if pn < POOLER_TILES:
 455:         n = pn * BN + nn
 456:         xq_ptr = XQ + m[:, None] * K + kk[None, :]
```

## V03: draft.py, lines 1–22

Image `lmsysorg/sglang-rocm:muse-spark-open-1027@sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`. Path `/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/draft.py`. File SHA256 `f1778a6cbe79a9501ba2a1728eccf85cfa1d256ffde9218179086d3bfc5c17ca`.

```python
   1: # SPDX-License-Identifier: Apache-2.0
   2: """Muse Spark's DFlash2 draft on gfx950: its spec and forward (RFC #43949).
   3:
   4: SGLang attaches the draft model as it does the target: a routed draft step (a
   5: block of DRAFT_TOKENS rows a request, run as a TARGET_VERIFY batch) runs
   6: ``_Draft.forward``, which takes and returns what ``DFlash2DraftModel.forward``
   7: does: the final-normed hidden states of the block, which the worker's selector
   8: turns into the draft tokens. The draft is a dense Qwen3-style stack (5 sliding
   9: layers, hidden 8192, 64 / 8 heads of 128, intermediate 28672, a grouped two-tap
  10: convolution around each sublayer) with no head of its own.
  11:
  12: Stage 2 (the default): each layer runs as four launches, D-pre (the input norm's
  13: fused add, the attention convolution's coefficients and conv0, the qkv projection,
  14: the per-head q / k norm + table RoPE, q out and k / v into the fp8 cache), D-attn
  15: (the block attention over the compact windowed cache), D-post (the o projection,
  16: the TP sum, conv1) and D-MLP (the post norm's fused add, the convolutions, gate /
  17: up / down, the TP sum) — ``gfx950/draft_attn_half.py``, ``draft_attn.py``,
  18: ``draft_mlp.py``. ``SGLANG_AMD_MONO_DRAFT_STAGE=1`` keeps the attention half on
  19: SGLang's modules; ``SGLANG_AMD_MONO_DRAFT_KERNELS=0`` (stage 0) runs the model's
  20: own forward, so the wiring (attach, widths, block rows, graph capture) can be
  21: exercised with an output identical to the normal path. Check mode
  22: (``SGLANG_AMD_MONO_CHECK=1``, graphs off) runs SGLang's halves on copies and logs
```

## V03: draft.py, lines 264–324

Image `lmsysorg/sglang-rocm:muse-spark-open-1027@sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`. Path `/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/draft.py`. File SHA256 `f1778a6cbe79a9501ba2a1728eccf85cfa1d256ffde9218179086d3bfc5c17ca`.

```python
 264:     def launch_attn(self, i: int, rows: int, hidden, residual, first: bool, positions, forward_batch, stream, trace: bool = False):
 265:         """Layer ``i``'s attention half on ``rows`` rows (D-pre, D-attn, D-post): (attn_out,
 266:         the residual after the input norm's fused add), bf16. ``trace`` (check mode):
 267:         every launch synchronized and logged, so a hang names its launch."""
 268:         from sgl_amd_mono.muse_spark.gfx950.runner import _f32_bits, _ptr
 269:
 270:         def mark(what):
 271:             if trace:
 272:                 torch.cuda.synchronize()
 273:                 logger.info("monokernel check: draft layer %d %s done (seq_lens %s, slots %s, table %s)", i, what, lens.tolist()[:4], slots.tolist()[:4], tuple(table.shape))  # fmt: skip
 274:
 275:         backend = self._backend()
 276:         if not self._cache_checked:
 277:             self._check_cache(backend)
 278:         v = self.aviews[i]
 279:         attn = self.layers[i].self_attn.attn
 280:         k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(attn.layer_id)
 281:         slots = forward_batch.out_cache_loc
 282:         table = backend.req_to_token_pool.req_to_token
 283:         rpi = forward_batch.req_pool_indices
 284:         lens = forward_batch.seq_lens
 285:         if slots.numel() < rows or positions.numel() < rows or lens.numel() * DRAFT_TOKENS < rows:
 286:             raise RuntimeError(f"Muse Spark draft: slots {tuple(slots.shape)} positions {tuple(positions.shape)} seq_lens {tuple(lens.shape)} for {rows} rows")  # fmt: skip
 287:         pre, dattn, post = self.attn_kernels(rows)
 288:         res_pre, q, o, coef, attn_out = self.res_pre[:rows], self.q[:rows], self.o[:rows], self.coef[:rows], self.attn_out[:rows]
 289:         pre(
 290:             _ptr(hidden), _ptr(residual), int(first), _ptr(v["w_in"]), _ptr(v["w_kpa"]), _ptr(v["base_a"]), _ptr(v["w_qkv"]),
 291:             _ptr(v["q_norm"]), _ptr(v["k_norm"]), _ptr(v["cos_sin"]), _ptr(positions), int(positions.dtype == torch.int64),
 292:             _ptr(slots), int(slots.dtype == torch.int64), _ptr(k_cache), _ptr(v_cache), _ptr(res_pre), _ptr(q), _ptr(coef),
 293:             _ptr(self.scratch_pre), _ptr(self.epoch), i, stream=stream,
 294:         )  # fmt: skip
 295:         mark("D-pre")
 296:         # the draft batch is a TARGET_VERIFY batch: seq_lens is the (compact) prefix, the
 297:         # block's DRAFT_TOKENS keys follow it; the compact cache holds the window, so no
 298:         # window mask (the window is the cache)
 299:         dattn(
 300:             _ptr(q), _ptr(o), _ptr(k_cache), _ptr(v_cache), _ptr(table), table.shape[1], table.numel() * 4, _ptr(rpi),
 301:             int(rpi.dtype == torch.int64), _ptr(lens), int(lens.dtype == torch.int64), DRAFT_TOKENS, 0, 0,
 302:             _f32_bits(float(attn.scaling)), _ptr(self.scratch_attn), _ptr(self.epoch), i, stream=stream,
 303:         )  # fmt: skip
 304:         mark("D-attn")
 305:         post(
 306:             _ptr(o), _ptr(v["w_o"]), _ptr(coef), _ptr(v["base_a"]), _ptr(attn_out), _ptr(self.scratch_post),
 307:             self.peers_post.local, _ptr(self.peers_post.addresses), self.rank, _ptr(self.epoch), i, stream=stream,
 308:         )  # fmt: skip
 309:         mark("D-post")
 310:         return attn_out, res_pre
 311:
 312:     def launch(self, i: int, rows: int, attn_out: torch.Tensor, residual: torch.Tensor, stream):
 313:         """Layer ``i``'s MLP half on ``rows`` rows: (out, the updated residual), bf16."""
 314:         from sgl_amd_mono.muse_spark.gfx950.runner import _ptr
 315:
 316:         v = self.views[i]
 317:         res_out, out = self.res_out[:rows], self.out[:rows]
 318:         self.kernel(rows)(
 319:             _ptr(residual), _ptr(attn_out), _ptr(v["w_post"]), _ptr(v["w_kp"]), _ptr(v["base"]), _ptr(v["w_gu"]), _ptr(v["w_down"]),
 320:             _ptr(res_out), _ptr(out), _ptr(self.scratch), self.peers.local, _ptr(self.peers.addresses), self.rank,
 321:             _ptr(self.epoch), i, stream=stream,
 322:         )  # fmt: skip
 323:         return out, res_out
 324:
```

## V03: draft.py, lines 475–519

Image `lmsysorg/sglang-rocm:muse-spark-open-1027@sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`. Path `/opt/sgl-amd-mono/sgl_amd_mono/muse_spark/draft.py`. File SHA256 `f1778a6cbe79a9501ba2a1728eccf85cfa1d256ffde9218179086d3bfc5c17ca`.

```python
 475: def _shard_fc(model) -> bool:
 476:     """Replace ``model.project_target_hidden`` with the TP-sharded projection: the rank's
 477:     rows of ``fc.weight`` (a view: the loader's storage) against the replicated input,
 478:     the ranks' parts all-gathered in rank order, then ``hidden_norm`` as the model does.
 479:     The draft's context rows only (the target's distribution is the verify's); a call
 480:     that does not match (another TP size or shape) runs the model's own."""
 481:     fc = getattr(model, "fc", None)
 482:     w = getattr(fc, "weight", None)
 483:     if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[0] % TP or getattr(fc, "bias", None) is not None:
 484:         return False
 485:     original = model.project_target_hidden
 486:
 487:     def project_target_hidden(target_hidden: torch.Tensor) -> torch.Tensor:
 488:         from sglang.srt.runtime_context import get_parallel
 489:
 490:         p = get_parallel()
 491:         if p.tp_size != TP or target_hidden.ndim != 2 or target_hidden.shape[-1] != w.shape[1]:
 492:             return original(target_hidden)
 493:         n = w.shape[0] // TP
 494:         part = torch.nn.functional.linear(target_hidden.to(w.dtype), w.narrow(0, p.tp_rank * n, n))
 495:         return model.hidden_norm(p.tp_group.all_gather(part, dim=-1))
 496:
 497:     model.project_target_hidden = project_target_hidden
 498:     return True
 499:
 500:
 501: def _build(model, hf_config, ctx) -> _Draft:
 502:     original = type(model).forward.__get__(model, type(model))
 503:     if not USE_KERNELS:
 504:         logger.info("monokernel: muse_spark DFlash2 draft attached at stage 0 (the model's own forward)")
 505:         return _Draft(model, original)
 506:     from sglang.srt.runtime_context import get_parallel
 507:
 508:     p = get_parallel()
 509:     runner = DraftRunner(list(model.layers), p.tp_rank, p.tp_size, p.tp_group.cpu_group, model.norm.weight.device)
 510:     runner.set_deadline(ctx.deadline_s)
 511:     widths = list(ctx.widths)
 512:     start = len(widths) * p.tp_rank // max(p.tp_size, 1)
 513:     for rows in widths[start:] + widths[:start]:
 514:         runner.precompile(rows)
 515:     if SHARD_FC and p.tp_size == TP and _shard_fc(model):
 516:         logger.info("monokernel: muse_spark DFlash2 draft: the context projection TP-sharded (%s rows a rank)", model.fc.weight.shape[0] // TP)  # fmt: skip
 517:     logger.info("monokernel: muse_spark DFlash2 draft attached at stage %d (%s)", STAGE, "the MLP half as D-MLP launches" if STAGE < 2 else "D-pre, D-attn, D-post and D-MLP launches a layer")  # fmt: skip
 518:     return _Draft(model, original, runner)
 519:
```
