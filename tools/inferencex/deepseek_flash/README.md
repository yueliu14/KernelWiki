# DeepSeek Flash evidence tools

These tools support [DeepSeek-V4.1-Flash on MI355X/vLLM](../../../knowledge/top_model/deepseek_v41_flash/README.md). The reusable performance collector and pair renderer live in the [InferenceX skill](../skills/inferencex-kernel-evolution/SKILL.md). No GPU is required for collection or source export.

Use `--cache /path/to/external-cache` and `--knowledge knowledge/top_model/deepseek_v41_flash` for the model-specific scripts. Keep full logs, source archives and image layers outside Git.

1. Run the reusable `collect_history.py --model DeepSeek-V4.1-Flash --framework auto --ci` first. It records framework coverage, all points, images, runs, all job attempts and artifact metadata.
2. `collect_logs.py --cache ... --knowledge ...` downloads every image/configuration representative's exact job log, recognizing both legacy plain `vllm serve` lines and newer `[INFO] Command:` lines. Resolved environments absent from legacy logs remain unknown. Preserve API cache receipts; a new capture should use a fresh directory.
3. `collect_recipes.py --cache ... --knowledge ...` captures InferenceX workflows/recipes/runners at run `head_sha` and actual checkout revisions from job logs. It preserves source URLs and hashes; exact effective overrides remain in the job logs.
4. `collect_server_logs.py --cache ... --knowledge ...` fetches selected server artifacts, reads nested tar archives and verifies that artifact upload URLs occur in the exact endpoint job. It retains unmatched rerun artifacts as candidates, not matched runtime evidence.
5. Allocate a CPU-only node with `-N 1 -n 1 -c 1 --gpus=0`, check `ReqTRES`, then run `pull_images_cpu.py --cache ... --knowledge ...` there. The script requires CI logs to have been collected first. It pins API digests where available, records failures, and runs the common source exporter with no GPU devices and no network. Do not place a Docker CPU quota on it. A current tag digest does not establish historical bytes.
6. For removed tags, `fetch_upstream_sources.py --cache ... --knowledge ...` fetches exact full-SHA vLLM source trees encoded in the tag. Only use them when startup package versions corroborate the revision. This cannot recover historical AITER bytes or prove image identity.
7. `prepare_sources.py --cache ... --knowledge ...` normalizes repository paths while preserving original paths, source kinds, file hashes and container receipts. It generates scoped source diffs; the seven fallback snapshots must match CI startup commits. Path normalization is for comparison only, not an assertion that a rename did not occur.
8. Manually review/update `evidence/kernel_pair_rules.json`. It is curated evidence, not an inference from filenames or a generated speedup attribution. The rules specify source/dispatch/body/fusion and configuration guards, including the TP2-only tuning and hardware-gated helpers.
9. Run the common `build_pair_history.py --knowledge ...`, then `render_summary.py --knowledge ...`. The latter renders all observations and the reviewed image-level chain. Validate using `python -m pytest -q tests/inferencex` and the repository checks.

The checked-in evidence represents the November 9 collection. Container inspection completed as CPU-only Slurm job 275610. Original GPU profiles were not recovered, so top-kernel time shares and causal speedups remain unavailable. A later reproduced GPU profile would be a new measurement and should have its own configuration/provenance.

`check_dockerhub.py --cache ...` checks exact tags, full/short commit-name matches and known historical digests directly against Docker Hub/Registry. It records response hashes and distinguishes a missing tag from an unavailable manifest. This HTTP-only verification needs neither Docker nor a GPU.

## Deeper log and recovered-container comparisons

Run these without a GPU job after the image export:

```bash
python3 tools/inferencex/deepseek_flash/collect_recovered_logs.py --cache /path/to/cache --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/deepseek_flash/prepare_recovered_sources.py --cache /path/to/cache --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/deepseek_flash/analyze_recovered_logs.py --cache /path/to/cache --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/deepseek_flash/render_recovered.py --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/skills/inferencex-kernel-evolution/scripts/build_pair_history.py --knowledge knowledge/top_model/deepseek_v41_flash
```

`collect_recovered_logs.py` follows exact job-upload artifact IDs for every representative with a recovered image. It records every archive member's hash, retaining full logs/request exports outside Git. `prepare_recovered_sources.py` maps logged JIT names to captured source candidates and stores both file hashes and symbol code/AST hashes. Device symbol candidates are not assumed to be unique binary dispatch identities.

`analyze_recovered_logs.py` joins server active/HIT/JIT messages to AIPerf phase boundaries, using CI timestamps for the legacy baseline. Its 16-pair view is consecutive within the recovered-image subset: six V01→V09 comparisons supplement the original chronology, and ten existing pairs receive detailed links. Review `evidence/recovered_kernel_reviews.json` separately before rendering; source changes alone do not establish new runtime fusion.

AIPerf's `Profiling` phase and `profile_export` files are E2E request evidence, not GPU traces. The default JIT monitor activates after server warmup and logs once per signature. HIT/active messages can occur during graph preparation. No message is not proof of no execution; diagnostic counts are not kernel invocation counts.
