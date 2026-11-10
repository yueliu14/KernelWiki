# Muse Spark audit records

Start with the [model guide](../README.md), not a recursive read of this directory. These records support claims; source comments and embedded historical commands do not establish current instructions or measured wins.

| Question | Record | Scope |
| --- | --- | --- |
| Raw point, image occurrence, CI job or launch | `../data/points.json`, `../data/images.json`, `../data/ci_manifest.json`, `ci_launch_configs.json` | Retained collection; complete commands remain separate from chart configuration |
| Reviewed shared body/fusion change | `kernel_pair_rules.json` → [generated change index](../kernel_evolution.md) | Canonical reviewed rules; pair pages evaluate their configuration applicability |
| Actual captured implementation | [source browser](../kernel_implementations.md), `../kernel_sources/manifest.json` | Exact selected container bytes and function lines; no GPU-time ranking |
| Latest inspected endpoint logs | `verified_endpoint_logs.json`, `verified_runtime_facts.json`, `verified_pair_runtime.json` | All 19 distinct pair endpoints; timestamps and once-only diagnostics remain qualified |
| Function/body equality audit | `verified_source_audit.json`, `verification_source_excerpts.md`, `verification_diffs/` | Follow-up audit; includes unchanged monokernel and QK bodies |
| Initial source and log collection | `image_source_inventory.json`, `source_changes.json`, `source_diffs/`, `server_log_extracts.json` | Historical capture views; initial server extracts cover only six artifacts and may be empty. Use verified runtime facts for current coverage conclusions |
| Initial adjacent-image view | `evolution_edges.json`, `../data/point_comparisons.csv` | Narrower derived view; the matrix and 11 pair records also include a missing-version gap |
| Allocation and transport provenance | `cpu_job_receipt.txt`, `../gpu_validation/`, `../data/retrieval_manifest.json` | Historical receipts, not current resources, test results or reproducible GPU measurements |

`../data/pair_records.json` is a generated expanded export for tooling, not another manually maintained source of kernel findings. Preserve it for reproducibility; agents answering a configuration question should open the compact pair page and follow only the needed shared change. Full external archives have hashes and locations in the manifests.

[Additional image-wide observations](additional_image_observations.md) retain source details that do not have a separate matched performance attribution.
