# Recovered-container point-to-point kernel comparisons

[Full 11-image matrix](configuration_history.md) · [Original 110 pairs](pair_changes.md) · [Model overview](README.md)

This supplement compares the four recovered container snapshots V01, V09, V10 and V11 using **31 exact endpoint server artifacts**, recovered vLLM/AITER bodies, and log timestamps aligned to the AgentX measurement phase. No new GPU job was run. There are **16 same-configuration pairs**: ten existing pairs now have deeper log/source evidence, and six supplemental V01→V09 TP4 pairs span the unavailable intermediate images. The original 110-pair chronology remains intact. V01 is a current tag snapshot without a historical digest; the other three match historical API digests.

Logs can establish startup configuration, entry into guarded dispatch code, exact HIT shapes/tiles, and named JIT compilations. Source inspection then identifies bodies and fusion boundaries. Logs cannot establish a complete kernel execution list, GPU-time ranking, call counts, or causal speedups. `warning_once` / `info_once`, JIT caches, and monitor activation after server warmup make silence inconclusive. AgentX/AIPerf files named `profile_export` and its `Profiling` phase describe E2E requests/metrics, not a GPU kernel trace.

## Findings strengthened by logs and recovered device bodies

- **Tiny-M quant+GEMM:** V11 has explicit HIT shapes; TP2 c64 logs `M=4, N=5120, K=15360, tile 8×16×256, k_pack=2` at **2026-11-07 06:14:55**, inside its E2E measurement window. This is direct evidence that a client concurrency of 64 can contain a four-row kernel. Source folds activation quantization into the packed GEMM K loop, saving the standalone quantization launch/intermediate; split-K reduction may still be separate.
- **TP8 c1 V10→V11:** `rocm_block32_gemm.py` is byte-identical and both endpoints log the same 41 unique HIT shape/tile signatures before measurement. That fusion was already present in V10; these log counts are not invocation counts. V11 adds the gfx950 Markov eligibility described below.
- **Mono and Markov:** every V09/V10/V11 endpoint logs target and draft mono active messages. V11 additionally logs fused Markov sampling (5 steps, vocabulary 129280, rank 256, 505 blocks). All first active/Markov messages precede E2E measurement. Their source call sites prove the guarded path was entered, but do not give its measurement-phase coverage.
- **Ordinary mHC:** recovered AITER wrapper/device code proves two principal Triton launches (main + reduce). Those specific files and gfx950 tuning config are identical across V09/V10/V11, despite changing AITER package versions. The separate V11 mHC overlap path is source-verified, with no dedicated overlap log found.
- **Sparse indexer:** inspected MXFP4 Q/K producer files and the score kernel are unchanged across V09/V10/V11. The score is implemented in Gluon even though the generic JIT monitor labels it Triton. Ordinary sparse MLA bodies do change; the reports separate numerical/addressing/layout changes from new cross-operator fusion.
- **Measurement-phase JIT:** TP8 c1 V10 logs `_block32_packed_kernel` and a FlyDSL BF16 GEMM compilation inside measurement; V11 logs a FlyDSL radix-top-k specialization. This proves those compilation events happened, not that the other version never executed those cached kernels or that one caused the performance delta.

## Same-configuration recovered-image pairs

Cells use `(p90 interactivity tok/s/user, output tok/s/chip)`. Version IDs refer to the [image index](README.md#image-versions). Source-only guards and observed log events are separate in every report.

| Configuration | Versions / dashboard dates | Before (x,y) | After (x,y) | Δ x | Δ y | Kernel + fusion + logs |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TP2/EP1 conc1 | V09→V11; 2026-10-20→2026-11-09 | (465.116, 86.818) | (452.489, 84.948) | -2.71% | -2.15% | [Detailed comparison](recovered_pairs/tp2_ep1_conc1_v09_to_v11.md) |
| TP2/EP1 conc2 | V09→V11; 2026-10-20→2026-11-09 | (423.729, 102.187) | (574.713, 104.894) | +35.63% | +2.65% | [Detailed comparison](recovered_pairs/tp2_ep1_conc2_v09_to_v11.md) |
| TP2/EP1 conc4 | V09→V11; 2026-10-20→2026-11-09 | (404.858, 127.315) | (578.035, 130.209) | +42.77% | +2.27% | [Detailed comparison](recovered_pairs/tp2_ep1_conc4_v09_to_v11.md) |
| TP2/EP1 conc8 | V09→V11; 2026-10-20→2026-11-09 | (288.184, 224.835) | (452.489, 227.961) | +57.01% | +1.39% | [Detailed comparison](recovered_pairs/tp2_ep1_conc8_v09_to_v11.md) |
| TP2/EP1 conc16 | V09→V11; 2026-10-20→2026-11-09 | (198.020, 365.450) | (302.115, 370.949) | +52.57% | +1.50% | [Detailed comparison](recovered_pairs/tp2_ep1_conc16_v09_to_v11.md) |
| TP2/EP1 conc32 | V09→V11; 2026-10-20→2026-11-09 | (120.192, 660.261) | (186.916, 645.735) | +55.51% | -2.20% | [Detailed comparison](recovered_pairs/tp2_ep1_conc32_v09_to_v11.md) |
| TP2/EP1 conc64 | V09→V11; 2026-10-20→2026-11-09 | (64.392, 1341.846) | (62.696, 1349.502) | -2.63% | +0.57% | [Detailed comparison](recovered_pairs/tp2_ep1_conc64_v09_to_v11.md) |
| TP4/EP1 conc1 | V01→V09; 2026-09-12→2026-10-20 | (35.727, 8.052) | (588.235, 45.710) | +1546.47% | +467.66% | [Detailed comparison](recovered_pairs/tp4_ep1_conc1_v01_to_v09.md) |
| TP4/EP1 conc1 | V09→V11; 2026-10-20→2026-11-09 | (588.235, 45.710) | (675.676, 47.423) | +14.86% | +3.75% | [Detailed comparison](recovered_pairs/tp4_ep1_conc1_v09_to_v11.md) |
| TP4/EP1 conc2 | V01→V09; 2026-09-12→2026-10-20 | (30.769, 12.403) | (526.316, 51.941) | +1610.53% | +318.79% | [Detailed comparison](recovered_pairs/tp4_ep1_conc2_v01_to_v09.md) |
| TP4/EP1 conc2 | V09→V11; 2026-10-20→2026-11-09 | (526.316, 51.941) | (666.667, 53.374) | +26.67% | +2.76% | [Detailed comparison](recovered_pairs/tp4_ep1_conc2_v09_to_v11.md) |
| TP4/EP1 conc4 | V01→V09; 2026-09-12→2026-10-20 | (31.201, 21.253) | (485.437, 66.917) | +1455.83% | +214.85% | [Detailed comparison](recovered_pairs/tp4_ep1_conc4_v01_to_v09.md) |
| TP4/EP1 conc8 | V01→V09; 2026-09-12→2026-10-20 | (26.441, 42.076) | (400.000, 116.254) | +1412.80% | +176.29% | [Detailed comparison](recovered_pairs/tp4_ep1_conc8_v01_to_v09.md) |
| TP4/EP1 conc16 | V01→V09; 2026-09-12→2026-10-20 | (19.662, 70.708) | (284.900, 189.076) | +1349.00% | +167.41% | [Detailed comparison](recovered_pairs/tp4_ep1_conc16_v01_to_v09.md) |
| TP4/EP1 conc32 | V01→V09; 2026-09-12→2026-10-20 | (13.179, 120.971) | (177.305, 362.620) | +1245.39% | +199.76% | [Detailed comparison](recovered_pairs/tp4_ep1_conc32_v01_to_v09.md) |
| TP8/EP1 conc1 | V10→V11; 2026-11-03→2026-11-09 | (684.932, 23.654) | (704.225, 24.005) | +2.82% | +1.48% | [Detailed comparison](recovered_pairs/tp8_ep1_conc1_v10_to_v11.md) |

## Evidence and reproduction

[Endpoint artifact manifests](evidence/recovered_endpoint_logs.json) · [Phase-aligned runtime facts](evidence/recovered_runtime_facts.json) · [Recovered pair records](data/recovered_pair_records.json) · [Source inventories](evidence/recovered_source_inventory.json) · [Reviewed kernel/fusion rules](evidence/recovered_kernel_reviews.json).

The 31 archives total 353,654,031 compressed bytes and are retained outside Git with full logs, AIPerf request exports and source snapshots. There are 298 JIT warning lines, 50 mono active lines, 10 Markov dispatch lines and 481 tiny-M HIT lines in the selected server streams. These totals describe diagnostic evidence only. Some messages occur during request warmup or graph preparation; each event retains its phase, exact file/line and file hash. All 31 jobs have recovered measurement windows.

Run `collect_recovered_logs.py`, `prepare_recovered_sources.py`, `analyze_recovered_logs.py` with `--cache ... --knowledge knowledge/top_model/deepseek_v41_flash`, followed by `render_recovered.py --knowledge ...`, then the common `build_pair_history.py`. See [collection tools](../../../tools/inferencex/deepseek_flash/README.md). Curated fusion rules are reviewed separately; discovering a new symbol does not automatically assign a speedup.
