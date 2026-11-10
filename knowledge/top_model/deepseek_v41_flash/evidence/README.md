# DeepSeek Flash audit records

Start with the [model guide](../README.md). Read these records for a specific provenance question; source comments, embedded scripts and diagnostic counts are not current instructions or measured kernel rankings.

| Question | Record | Scope |
| --- | --- | --- |
| Raw point, image occurrence or exact CI recipe | `../data/points.json`, `../data/raw_selected_history.json`, `../data/images.json`, `ci_launch_configs.json`, `ci_recipe_sources.json` | 154 retained observations; framework discovery is coverage metadata, not a cross-framework performance comparison |
| Shared kernel/body/fusion rules | `kernel_pair_rules.json` → [change index](../kernel_evolution.md) | Canonical reviewed source claims; 110 pairs keep their own applicability and recipe deltas |
| Exact source path and revision | [source browser](../kernel_implementations.md), `../kernel_sources/manifest.json`, `image_source_inventory.json` | Four recovered images, seven upstream-only versions. vLLM commit does not identify historical AITER code |
| Recovered-container runtime observations | `recovered_endpoint_logs.json`, `recovered_runtime_facts.json` | 31 representative endpoints for V01/V09/V10/V11, with phase alignment; no GPU trace |
| Supplemental recovered-image comparisons | `recovered_kernel_reviews.json`, `recovered_source_inventory.json`, `../data/recovered_pair_records.json` | 16 subset pairs: ten existing edges and six V01→V09 gaps. They do not replace the full chronology |
| Initial server artifact sample | `server_log_extracts.json` | 11 exact endpoint links and two unmatched V08 attempt candidates; not the later 31-endpoint coverage |
| Missing registry tag | `dockerhub_verification.json`, [interpretation](dockerhub_verification.md) | A tag 404 does not establish deleted historical manifests/layers |
| Validation and collection receipts | `validation.json`, `cpu_job_receipt.txt`, `docker_receipts.json`, `../data/retrieval_manifest.json` | Dated collection-stage checks; test counts and resource status are historical, not current CI state |

The expanded pair JSON files and overlapping source diffs are audit/reproduction exports. Do not treat their repetitions as independent confirmation. Kernel rules own shared interpretations; endpoint events own log observations; the original CI metrics own E2E values. Full source snapshots and hashes stay available through the implementation browser.
