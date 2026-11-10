# Muse Spark Open — MI355X / SGLang

Start with the **[configuration × image matrix](configuration_history.md)**, then the pair for the requested TEP/concurrency. This is `Muse-Spark-Open` (`musespark`), collected on **2026-11-09** from [InferenceX](https://inferencex.semianalysis.com/infx/inference/muse-spark-open).

The retained history covers **2026-10-16 → 2026-11-08: 54 observations, four images, 27 configuration rows and 11 pairs**. [All observations](performance_points.md) retain repeats; [pair index](pair_changes.md) includes the TEP4 c16 gap. TEP8 c2 and latest c10 have no same-configuration predecessor. Shared [metric and evidence rules](../README.md) apply.

| Need | Authoritative reading entry |
| --- | --- |
| Performance at a specific configuration | [Matrix](configuration_history.md) → pair page |
| Shared body/fusion changes and guards | [Reviewed image changes](kernel_evolution.md) → one shared change record |
| Exact container paths and function bodies | [Implementation browser](kernel_implementations.md) |
| Dispatch observations and unchanged-body audit | [Runtime recheck](runtime_verification.md) |
| Audit metadata or reproduction | [Evidence guide](evidence/README.md) · [Collection reference](../../../tools/inferencex/skills/inferencex-kernel-evolution/references/muse-spark-pilot.md) |

**Evidence limit:** all four images were pulled and their source exported. All 19 distinct pair endpoint server artifacts were inspected and phase-aligned. The [bounded GPU attempt](gpu_validation/README.md) obtained no allocation; GPU-time top-kernel ranks, actual launch counts and causal contributions remain unavailable.

**Workload:** FP4, aggregate AgentX replay, TP=EP, DFLASH with eight draft tokens and simulated expected acceptance length **4.05**. TEP8 uses eight chips. `HiCache` means DRAM KV offload; `GPU` means offload is off. Same chart labels can change server admission, graph capture, cache layout and other recipe settings.

**Source identity:** all four images report SGLang HEAD `c722c0a160981f21c81cffaab7bbcc117089a2a6`, despite differing file bodies. Use the recorded image digest and source hashes. The 1027→1106 audit found all 97 captured monokernel package files unchanged; broader ordinary-path guards must not be described as a new monokernel body. Exact findings belong to the linked shared change and runtime records.

## Image timeline

Tags have prefix `lmsysorg/sglang-rocm:`. These are dashboard dates, not job execution dates.

| Version | First → last observed | Image tag | Digest prefix |
| --- | --- | --- | --- |
| V01 | 2026-10-16 → 2026-10-19 | `muse-spark-open` | `72065dfc0b87` |
| V02 | 2026-10-22 | `muse-spark-open-1020` | `7343002d0dc2` |
| V03 | 2026-10-30 | `muse-spark-open-1027` | `7184d3ae2bfd` |
| V04 | 2026-11-08 | `muse-spark-open-mi355-1106` | `d231181df079` |
