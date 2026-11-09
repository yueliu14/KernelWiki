# Point-to-point kernel changes

Each row links to the exact configuration's dates, performance, CI commands and kernel/fusion changes. Pair the next observed image in each configuration row; skipped image columns remain explicit. Singleton rows remain in the [configuration matrix](configuration_history.md).

| Configuration | Dates | Δ interactivity | Δ output/chip | Skipped images | Kernel record |
| --- | --- | ---: | ---: | ---: | --- |
| TEP4 conc16 / HiCache | 2026-10-16 → 2026-10-30 | +295.46% | +11.76% | 1 | [Paths, bodies, fusion and CI](pairs/tep4_conc16_on_2026-10-16_to_2026-10-30.md) |
| TEP4 conc28 / HiCache | 2026-10-30 → 2026-11-08 | +19.17% | +24.49% | 0 | [Paths, bodies, fusion and CI](pairs/tep4_conc28_on_2026-10-30_to_2026-11-08.md) |
| TEP4 conc36 / HiCache | 2026-10-30 → 2026-11-08 | +78.52% | +29.62% | 0 | [Paths, bodies, fusion and CI](pairs/tep4_conc36_on_2026-10-30_to_2026-11-08.md) |
| TEP4 conc40 / HiCache | 2026-10-30 → 2026-11-08 | -24.42% | +29.02% | 0 | [Paths, bodies, fusion and CI](pairs/tep4_conc40_on_2026-10-30_to_2026-11-08.md) |
| TEP8 conc1 / GPU | 2026-10-16 → 2026-10-22 | +71.84% | +5.40% | 0 | [Paths, bodies, fusion and CI](pairs/tep8_conc1_off_2026-10-16_to_2026-10-22.md) |
| TEP8 conc1 / GPU | 2026-10-22 → 2026-10-30 | +48.36% | +11.29% | 0 | [Paths, bodies, fusion and CI](pairs/tep8_conc1_off_2026-10-22_to_2026-10-30.md) |
| TEP8 conc4 / GPU | 2026-10-16 → 2026-10-22 | +69.17% | +9.62% | 0 | [Paths, bodies, fusion and CI](pairs/tep8_conc4_off_2026-10-16_to_2026-10-22.md) |
| TEP8 conc4 / GPU | 2026-10-22 → 2026-10-30 | +34.33% | +12.37% | 0 | [Paths, bodies, fusion and CI](pairs/tep8_conc4_off_2026-10-22_to_2026-10-30.md) |
| TEP8 conc8 / GPU | 2026-10-16 → 2026-10-22 | +91.07% | +21.67% | 0 | [Paths, bodies, fusion and CI](pairs/tep8_conc8_off_2026-10-16_to_2026-10-22.md) |
| TEP8 conc8 / GPU | 2026-10-22 → 2026-10-30 | +62.91% | +2.86% | 0 | [Paths, bodies, fusion and CI](pairs/tep8_conc8_off_2026-10-22_to_2026-10-30.md) |
| TEP8 conc16 / HiCache | 2026-10-16 → 2026-10-22 | +130.69% | +6.93% | 0 | [Paths, bodies, fusion and CI](pairs/tep8_conc16_on_2026-10-16_to_2026-10-22.md) |

The older [adjacent-image CSV](data/point_comparisons.csv) remains a narrower view. [Structured pair records](data/pair_records.json) include all rows above. Kernel annotations are pinned to reviewed endpoint digests and configuration/CLI guards in [kernel_pair_rules.json](evidence/kernel_pair_rules.json); a new image requires a new source review.
