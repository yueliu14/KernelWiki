# DeepSeek V4.1 Flash — MI355X / vLLM

Start with the **[configuration × image matrix](configuration_history.md)**, then the linked pair for the requested TP/concurrency. This is the catalog model `DeepSeek-V4.1-Flash` (`dsv41flash`), collected on **2026-11-09** from [InferenceX](https://inferencex.semianalysis.com/infx/inference/deepseek-v41-flash).

The retained history covers **2026-09-12 → 2026-11-09: 154 observations, 11 images, 17 configuration rows and 110 pairs**. [All observations](performance_points.md) retain repeats; [pair index](pair_changes.md) retains missing-version gaps. Read the shared [metric and evidence rules](../README.md) before interpreting a gain.

| Need | Authoritative reading entry |
| --- | --- |
| Performance at a specific configuration | [Matrix](configuration_history.md) → pair page |
| Kernel/body/fusion change shared across concurrencies | [Reviewed image changes](kernel_evolution.md) → one shared change record |
| Exact implementation and framework revision | [Commit/source browser](kernel_implementations.md) |
| Observed dispatch and JIT events | [Recovered-container supplement](recovered_image_history.md) → endpoint pair |
| Audit metadata or reproduction | [Evidence guide](evidence/README.md) · [Collection tools](../../../tools/inferencex/deepseek_flash/README.md) |

**Evidence limit:** no GPU-time top-kernel ranking or causal kernel speedup is established. The 31 recovered endpoint server artifacts support phase-aligned diagnostics, not launch counts. The 16-pair recovered-image supplement does not replace the full 110-pair chronology.

**Workload:** FP4, aggregate AgentX replay, EP1 (TP2/TP4/TP8), DSpark with five speculative tokens and synthetic acceptance length **3.51**. `GPU` in the matrix means KV offload is off; Engram CPU placement is a separate launch setting. Same chart configuration can still change admission, graphs, batching, quick-reduce quantization and model/data revisions.

**Source scope:** V01/V09/V10/V11 have recovered container bytes. V01 is a current tag snapshot without a historical digest; V09–V11 match historical API digests. V02–V08 use CI-corroborated upstream vLLM commits; their historical container and AITER internals remain unavailable. Framework discovery lists other published frameworks for coverage only; this trajectory compares vLLM. [Registry evidence](evidence/dockerhub_verification.md).

## Image versions

Tags have prefix `vllm/vllm-openai-rocm:`. These are dashboard dates. A vLLM commit does not identify AITER or prove historical container bytes.

| Version | First → last dashboard date | Image tag | Representatives / observations | Source recovery |
| --- | --- | --- | ---: | --- |
| V01 | 2026-09-12 → 2026-09-12 | `deepseekv41-flash-0909` | 6 / 6 | Current tag; historical digest unknown |
| V02 | 2026-09-13 → 2026-09-13 | `nightly-eed1f3d0c6043bd494424a22443ee198dd56f657` | 6 / 6 | CI-corroborated upstream commit; tag unavailable |
| V03 | 2026-09-22 → 2026-09-22 | `nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984` | 16 / 16 | CI-corroborated upstream commit; tag unavailable |
| V04 | 2026-09-24 → 2026-09-24 | `nightly-rocm100-7f1a5398e9610d96c473931a26c0e12bbe0d0423` | 16 / 16 | CI-corroborated upstream commit; tag unavailable |
| V05 | 2026-09-25 → 2026-09-25 | `nightly-rocm100-29468dde8b515031dc6d4d9d06bf0a2fa0442098` | 16 / 16 | CI-corroborated upstream commit; tag unavailable |
| V06 | 2026-10-02 → 2026-10-02 | `nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516` | 14 / 14 | CI-corroborated upstream commit; tag unavailable |
| V07 | 2026-10-12 → 2026-10-12 | `nightly-rocm100-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe` | 14 / 14 | CI-corroborated upstream commit; tag unavailable |
| V08 | 2026-10-13 → 2026-10-13 | `nightly-rocm100-5555a2d8f7116b2b4877f66516f4128176cf722a` | 14 / 14 | CI-corroborated upstream commit; tag unavailable |
| V09 | 2026-10-20 → 2026-11-03 | `nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f` | 14 / 41 | Container; API digest verified |
| V10 | 2026-11-03 → 2026-11-03 | `nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1` | 1 / 1 | Container; API digest verified |
| V11 | 2026-11-09 → 2026-11-09 | `nightly-4fab5a07475822c6eaa971dceb2e2220d0446808` | 10 / 10 | Container; API digest verified |
