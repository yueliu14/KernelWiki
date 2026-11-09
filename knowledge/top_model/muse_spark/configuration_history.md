# Performance history by configuration

Each cell is **(p90 interactivity in tok/s/user, output throughput in tok/s/chip)** and links to its CI job. Columns are image versions; dates are first observed dashboard dates. The earliest valid observation per image/configuration is shown; [all measured points](performance_points.md) preserve repeats. `—` means no retained observation.

The last column links each available point-to-point pair to its kernel record. Pairs connect successive observed versions within the row, including across blank columns. Singleton rows have no pair. [Pair index](pair_changes.md) · [Structured pair records](data/pair_records.json).

| Configuration | 2026-10-16 / `muse-spark-open` | 2026-10-22 / `muse-spark-open-1020` | 2026-10-30 / `muse-spark-open-1027` | 2026-11-08 / `muse-spark-open-mi355-1106` | Kernel changes for pairs |
| --- | ---: | ---: | ---: | ---: | --- |
| TEP4 conc1 / GPU | [(138.889, 30.857)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688746) | — | — | — | — |
| TEP4 conc4 / GPU | [(108.578, 47.874)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688626) | — | — | — | — |
| TEP4 conc8 / GPU | [(73.475, 80.897)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688898) | — | — | — | — |
| TEP4 conc12 / GPU | [(49.261, 104.620)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688964) | — | — | — | — |
| TEP4 conc16 / GPU | [(29.291, 109.165)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688887) | — | — | — | — |
| TEP4 conc10 / HiCache | — | — | [(189.394, 116.002)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910587886) | — | — |
| TEP4 conc12 / HiCache | [(51.894, 107.376)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38947852942/job/117132179064) | — | — | — | — |
| TEP4 conc16 / HiCache | [(38.256, 127.527)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38947852942/job/117132178838) | — | [(151.286, 142.520)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910588096) | — | [10-16 → 10-30](pairs/tep4_conc16_on_2026-10-16_to_2026-10-30.md) |
| TEP4 conc24 / HiCache | — | — | [(111.359, 206.260)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910588410) | — | — |
| TEP4 conc28 / HiCache | — | — | [(105.152, 212.740)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910587901) | [(125.313, 264.845)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42394597374/job/128469427930) | [10-30 → 11-08](pairs/tep4_conc28_on_2026-10-30_to_2026-11-08.md) |
| TEP4 conc36 / HiCache | — | — | [(54.437, 237.046)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910588046) | [(97.182, 307.267)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42394597374/job/128469428100) | [10-30 → 11-08](pairs/tep4_conc36_on_2026-10-30_to_2026-11-08.md) |
| TEP4 conc40 / HiCache | — | — | [(95.602, 258.197)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910587816) | [(72.254, 333.119)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42394597374/job/128469427944) | [10-30 → 11-08](pairs/tep4_conc40_on_2026-10-30_to_2026-11-08.md) |
| TEP8 conc1 / GPU | [(184.162, 17.808)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688634) | [(316.456, 18.770)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428436852) | [(469.484, 20.888)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910588033) | — | [10-16 → 10-22](pairs/tep8_conc1_off_2026-10-16_to_2026-10-22.md); [10-22 → 10-30](pairs/tep8_conc1_off_2026-10-22_to_2026-10-30.md) |
| TEP8 conc2 / GPU | — | [(295.858, 22.076)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428436648) | — | — | — |
| TEP8 conc4 / GPU | [(164.204, 25.608)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688547) | [(277.778, 28.073)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428436698) | [(373.134, 31.544)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910587858) | — | [10-16 → 10-22](pairs/tep8_conc4_off_2026-10-16_to_2026-10-22.md); [10-22 → 10-30](pairs/tep8_conc4_off_2026-10-22_to_2026-10-30.md) |
| TEP8 conc6 / GPU | — | — | — | [(515.464, 38.953)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42394597374/job/128469427998) | — |
| TEP8 conc8 / GPU | [(116.822, 43.152)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688890) | [(223.214, 52.504)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428437135) | [(363.636, 54.006)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910588087) | — | [10-16 → 10-22](pairs/tep8_conc8_off_2026-10-16_to_2026-10-22.md); [10-22 → 10-30](pairs/tep8_conc8_off_2026-10-22_to_2026-10-30.md) |
| TEP8 conc10 / GPU | — | — | — | [(373.134, 64.960)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42394597374/job/128469428179) | — |
| TEP8 conc12 / GPU | [(88.652, 64.724)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688909) | — | — | — | — |
| TEP8 conc13 / GPU | — | — | [(362.319, 66.098)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40714803265/job/122910587874) | — | — |
| TEP8 conc16 / GPU | [(65.920, 80.844)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38877885539/job/116897688917) | — | — | — | — |
| TEP8 conc28 / GPU | — | — | — | [(221.729, 137.471)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42394597374/job/128469427954) | — |
| TEP8 conc12 / HiCache | [(87.260, 64.389)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38947852942/job/117132178702) | — | — | — | — |
| TEP8 conc16 / HiCache | [(66.181, 80.089)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38947852942/job/117132178956) | [(152.672, 85.642)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428436736) | — | — | [10-16 → 10-22](pairs/tep8_conc16_on_2026-10-16_to_2026-10-22.md) |
| TEP8 conc32 / HiCache | — | [(96.618, 151.010)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428436857) | — | — | — |
| TEP8 conc48 / HiCache | — | [(77.640, 206.105)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428436854) | — | — | — |
| TEP8 conc64 / HiCache | — | [(64.767, 252.913)](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39659301076/job/119428436843) | — | — | — |
