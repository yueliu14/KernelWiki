# Learned knowledge

Methods that transfer across operators: measurement controls, engagement, graph-safe integration,
occupancy analysis and optimization experiments. A worked example may name kernels without becoming
an operator-specific recommendation. Source revisions and evidence stay attached to each document.

Specific FP8/MXFP8 GEMM and attention-backend outcomes live with the
[operator learned knowledge](../../operator/README.md).

- [Stacking small kernel wins under an end-to-end Amdahl budget](e2e-editable-triton-cluster-amdahl.md)
- [Make an optimized kernel survive CUDA/HIP-graph capture (or the win vanishes e2e)](e2e-method-cudagraph-safe-integration.md)
- [Honest e2e A/B: tight interleave + non-overlap gate (not just a positive median)](e2e-method-e2e-ab-harness.md)
- [Prove the optimized kernel ran on the LIVE serving path (don't infer it from an e2e wiggle)](e2e-method-verify-engagement.md)
- [A/B in the graded case mix, and price a direction against the noise floor first](kernel-a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a.md)
- [A/B protocol and oracle confounds on a power-capped GPU](kernel-a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a.md)
- [A closure is conditional on the body that measured it](kernel-a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a.md)
- [A fixed-order A/A control measures order bias and then hides it](kernel-a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)
- [A stuck tiny case may be floored by the timing bracket, not the GPU](kernel-a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound.md)
- [CDNA4 sums ArchVGPR and AGPR for occupancy](kernel-cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
- [Fill the CUs with a hidden-dim block axis, then hoist the k-loop guard](kernel-fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
- [Force the rebuild, pair the blocks, dump the registers before stacking](kernel-force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a.md)
- [Gate a tiny-kernel win on a median or a paired A/B](kernel-gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound.md)
- [Hand-count the bytes and build a read-only twin before staffing a memory round](kernel-hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode.md)
- [Measure a tiny op with the harness's own protocol, rotated, against an A/A null](kernel-measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both.md)
- [Per-region ISA census before hot-loop tuning; locate on CPU, price on GPU](kernel-per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a.md)
- [Per-rep geomean plus a two-sided negative control](kernel-per-rep-geomean-plus-a-two-sided-negative-control-method-gfx950-n-a.md)
- [Prove the edit built and prove the win separately from the harness flag](kernel-prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a.md)
- [Re-measure shelved partials after the bound class moves](kernel-re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a.md)
- [Re-price the dead list when the operating point moves](kernel-re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound.md)
- [Scale percent-of-peak to the CUs the box actually exposes](kernel-scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound.md)
- [Scope a closure, a gate and a stack by case regime](kernel-scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a.md)
- [Test the allocator before designing a kernel fix for a periodic per-case split](kernel-test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
- [The host lane pays once; the exhaustion test is submit cost versus the smallest case's kernel time](kernel-the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both.md)
