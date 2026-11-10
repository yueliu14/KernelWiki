# Additional image-wide observations

These source observations from the initial review are not separate measured pair attributions. The shared [change records](../kernel_evolution.md) own the reviewed per-configuration claims; source diffs retain the detailed implementation context.

- Baseline→1020 also introduced unpooler/boundary and sequence-parallel prefill wiring in `model.py` / `sequence_parallel.py`. Sequence parallelism and Ulysses communication are distinct from a fused device launch. See [the captured source diff](source_diffs/01_to_02.diff); no independent kernel speedup is assigned.
- In 1027, `SGLANG_AMD_MONO_MUSE_LAYER=1` is the default for the eligible layer fusion. The first necessary K1 remains separate, and BF16-expert layers 0 and 61 retain ordinary execution. See [source excerpts](source_excerpts.md) and the [implementation browser](../kernel_implementations.md).
- In 1106, standalone finalize/all-reduce support expands to TP8 with a one-shot protocol below 16 rows and a two-stage protocol from 128 rows; intermediate sizes fall back. This TP8 expansion cannot explain the measured TEP4 c28/c36/c40 edges. See [source diff](source_diffs/03_to_04.diff). The latest TEP8 c10 point has no measured same-configuration predecessor.
