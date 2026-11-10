# Verified Muse Spark pilot (2026-11-09 UTC)

The public page is <https://inferencex.semianalysis.com/infx/inference/muse-spark-open>.
The API model parameter is `Muse-Spark-Open`; returned `model` is `musespark`.

## Collect and replay

From this skill directory:

```bash
python3 scripts/collect_history.py \
  --cache /path/to/external/raw-cache \
  --output /path/to/KernelWiki/knowledge/top_model/muse_spark/data --ci
```

Use `--offline` with the same cache to verify reproducibility. Use a fresh cache directory for a refresh; cached URLs are intentionally not silently overwritten. GitHub requests use the authenticated `gh api` client without copying its credentials into artifacts. The collector handles published **aggregate AgentX** rows; extend its normalization explicitly before using it for disaggregated pools or fixed-length benchmarks.

After collecting each paired endpoint's CI launch log and reviewing source applicability, run:

```bash
python3 scripts/build_pair_history.py --knowledge /path/to/KernelWiki/knowledge/top_model/muse_spark
```

This renders the matrix, pair index, compact pair pages, shared `changes/` records, `kernel_evolution.md` change index and expanded `data/pair_records.json`. Pair pages retain endpoint metrics and applicability; shared records own the full body/fusion/source explanation. It pairs successive observed images within each configuration row, including explicitly marked gaps. The pilot has 27 configuration rows, 38 representatives, 11 pairs and 19 unique pair endpoint jobs; ten pairs connect adjacent images, while TEP4 c16 HiCache spans baseline → 1027. TEP8 c2 remains a singleton. `evidence/kernel_pair_rules.json` contains reviewed findings scoped to exact endpoint digests and configuration/CLI guards; source hashes are filled from verified inventories. The renderer rejects missing endpoint logs, missing reviewed rules and incompatible CLI guards instead of silently reusing an image-level finding.

The matrix and per-pair records are the user-approved primary presentation. Each pair must describe actual language, source path/body and fusion changes relevant to that configuration, with source-only eligibility separated from runtime evidence and measured GPU time.

- `/api/v1/benchmarks?model=Muse-Spark-Open`: current published snapshot.
- `/api/v1/benchmarks/history?model=Muse-Spark-Open&benchmarkType=agentic_traces`: currently retained published historical snapshots. This is not a census of every CI run or discarded benchmark.
- Historical rows carry `date`/`workflow_run_id` and `curve_date`/`curve_workflow_run_id`. Append-only snapshots can inherit earlier points, so keep snapshot occurrences and deduplicate measured IDs separately.
- `workflow_run_id` is an internal database ID. Parse the GitHub run ID from `run_url`, which can end with `/attempts/1`.
- `image` already contains a fully qualified tag and digest. Primary version key = the part before `@`; retain the complete reference for pulls.
- Actual model revision and dataset revision are not guaranteed by these rows. Report descriptive matches on available fields and extract missing controls from logs.

The upstream inspected implementation is InferenceX-app commit `d475545122661430a1f6de280e81418b0ba2f198`; this is a source inspection revision, not a verified deployed-build SHA. Relevant files are `packages/db/src/queries/benchmarks.ts` (history seed and append-only recursion) and `packages/app/src/lib/benchmarks/benchmark-transform.ts` (AgentX aliases).

For AgentX, x is `1 / p90_full_response_itl` in **seconds**, falling back to `p90_itl` only when no full-response ITL fields exist. A row with other full-response percentiles but a missing p90 remains missing. Do not use `p90_full_response_intvty`: producer percentile-rate definitions can differ. y is `output_tput_per_gpu`, already in tok/s/chip. On an aggregate TEP8 row, divide deployment `output_tput_tps` by 8, not by 64 or by the sum of the prefill/decode GPU fields. ISL/OSL are null because AgentX replays variable-length traces.

## CPU container evidence

After reading each corresponding run/job, pull its complete `tag@sha256:...` reference on a Docker-capable CPU allocation. Record `docker image inspect` fields and the scheduler's CPU/GPU request. Run `scripts/capture_image_sources.py /output` inside the image with the script mounted read-only and an output directory mounted writable. Set an ordinary UID/GID, use `--network none`, and pass no GPU device mounts. The script uses package metadata and filesystem reads, without importing GPU frameworks or executing a model.

The source exporter selects text/source extensions and skips build/cache directories, symlinks, files above 10 MB, tuning CSVs and compiled binaries. Its manifest is an inventory of captured source, not a claim that every runtime file was copied. Expand the selection when a conclusion depends on an omitted table, binary, symlink target, compiler, or external library.

Run `scripts/summarize_image_sources.py --images /path/to/images --image-timeline /path/to/data/images.json --output /path/to/evidence` to verify exported bytes against manifests, compare kernel paths/hashes, and write selected unified diffs. Image directories use the first 12 hex characters of the immutable digest. Keep raw archives outside the Git tree.

CI `server_logs_*` artifacts are ZIPs that contain another archive, `srt-single-node-logs.tar.gz`. Inspect that nested archive for the server stdout/stderr; searching only outer `benchmark.log` misses initialization and dispatch evidence. `srt-slurm-sha.txt` identifies an additional launcher revision. Preserve both container source and mounted launcher provenance. A “built” or “enabled” log confirms initialization, not GPU time share or an exact benchmark-phase launch count.

CPU probe job `275453` and source capture job `275462` completed on `crsuse2-m2m-v3-237` with 1 CPU and no GPU requested. The initial evidence cache is `/shared_nfs/yueliu14/inferencex_muse_spark_research/`; this machine-specific location is a pilot receipt, not a requirement for future runs.

## Pilot pitfalls

- The dashboard dates are 2026-10-16, 10-19, 10-22, 10-30, and 11-08, whereas some associated CI jobs ran the preceding day or earlier. The first tag appears in three runs and on two dashboard dates.
- All four image trees report SGLang HEAD `c722c0a160981f21c81cffaab7bbcc117089a2a6`, and newer images contain many modified files. Git HEAD alone is insufficient source identity.
- All sampled performance logs set `SGLANG_SIMULATE_ACC_LEN=4.05`, `match-expected`, and `real-draft-token`. This is part of the workload protocol and must be preserved when explaining changes.
- Client concurrency is not server admission: TEP8 c8 changed from `max-running-requests=128` to 2 with monokernel enabled. Same chart label does not isolate a kernel's causal effect.
- Newest TEP8 c10 has no same-configuration predecessor in retained history. TEP4 c40 improves throughput and regresses interactivity. Preserve both facts.
- The four images permit source-level fusion analysis. Artifact names and AgentX request profiles do not establish a GPU-time top-kernel ranking.
