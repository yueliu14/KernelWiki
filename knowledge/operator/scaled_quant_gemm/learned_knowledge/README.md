# scaled_quant_gemm: learned knowledge

One transferable lesson per card. Preserve the run, environment, workload and outcome that support it. A kernel result and an end-to-end result have different scopes; imported cards carry `evidence_scope`.

- [Arg-plan replay beats graph replay at low dispatch counts, and pays a second time](kernel-arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed.md)
- [Axes that closed on a parity-gated fp8 GEMM](kernel-axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [Bit-exact integer re-encode of the fp8-FNUZ upcast](kernel-bit-exact-integer-re-encode-of-the-fp8-fnuz-upcast-quantized-gemm-gfx950-compute-bound.md)
- [Collapse the dequant chain in a block-scaled fp8 GEMM](kernel-collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [Collapse the fp8 dequant chain into one scaled convert](kernel-collapse-the-fp8-dequant-chain-into-one-scaled-convert-quantized-gemm-gfx950-compute-bound.md)
- [De-scale the fp8 GEMM K-loop, then feed the native non-scaled MFMA](kernel-de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound.md)
- [Five closed axes above an ILP-bound block-scaled fp8 GEMM](kernel-five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [Reinterpret legacy fp8 bits to the arch-native fp8 type to reach the matrix core](kernel-reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound.md)
- [Split K by 2 to fill the grid on the tiny-M case](kernel-split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch.md)
- [Sub-K coarsening: regroup the same reduction order into fewer wider dots](kernel-sub-k-coarsening-regroup-the-same-reduction-order-into-fewer-quantized-gemm-gfx950-compute-bound.md)
- [The operand-feed residual of a scale-free fp8 GEMM is a closed axis](kernel-the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound.md)
- [The residual axes on a decoded fp8 GEMM are already closed](kernel-the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound.md)
- [Where the headroom is not, and the two floors that tell you so](kernel-where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound.md)

- [gfx950 sglang fp8 a8w8 blockscale — the live kernel is CK **bpreshuffle**, so tune THAT DB](e2e-fp8-a8w8-blockscale-bpreshuffle-ck-tune-gfx950-sglang.md)

- [gfx950 vLLM fp8 a8w8 blockscale — per-shape CK tune DB (no overlay needed)](e2e-fp8-a8w8-blockscale-ck-tune-gfx950-vllm.md)

- [fp8 a8w8 blockscale GEMM → per-(N,K) M-bucketed Triton config-JSON overlay  (DEPRECATED — see banner)](e2e-fp8-a8w8-blockscale-overlay-gfx942.md)

- [MXFP8 (1x32 E8M0 microscale) DENSE LINEAR GEMM — route via Tier-C author, not env](e2e-mxfp8-e8m0-dense-linear-gfx950.md)

- [MXFP8 dense-linear decode-tile rewrite is the main e2e lever (decode-bound serving)](e2e-mxfp8-linear-decode-rewrite-gfx950.md)

## New lesson fields

Record the problem, applicable operator/backend/GPU/dtype/shapes, attempted change, positive or negative outcome, explanation, limitations, run reference and optional patch reference. Keep raw run logs in the run/patch store.
