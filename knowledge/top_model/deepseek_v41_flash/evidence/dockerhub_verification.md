# Docker Hub and Registry availability verification

Checked directly on **2026-11-09 UTC** against Docker Hub's tag API and the authenticated Docker Registry manifest API. The Registry token was held in memory and is not included in the evidence. Full response bodies are in the external research cache; [receipts](dockerhub_verification.json) retain URLs, timestamps, HTTP status, response hashes, selected headers and error bodies.

**V02–V08 tags are absent from the queried Docker Hub repository.** Each exact-tag Hub request returns HTTP 404 and each Registry tag lookup returns HTTP 404 / `MANIFEST_UNKNOWN`. Searching tag names for either the full commit SHA or its first eight characters returns zero results. These searches do not establish the absence of aliases with unrelated names.

**Tag absence does not establish that an image manifest or its layers were deleted.** V09 is a concrete example: its tag also returns 404, but its API-recorded digest returns HTTP 200. The earlier CPU job successfully pulled/exported V09 by that digest. V02–V08 currently lack a recovered historical digest, so their underlying manifest/blob availability remains unknown. The deletion time and reason were not established.

| Version | Commit/tag identifier | Docker Hub tag | Registry by tag | Registry by historical digest | Full/short SHA tag search |
| --- | --- | ---: | --- | --- | --- |
| V01 | `deepseekv41-flash-0909` | [200](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/deepseekv41-flash-0909/) | 200 | Historical digest unavailable | Not queried |
| V02 | `eed1f3d0c604` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-eed1f3d0c6043bd494424a22443ee198dd56f657/) | 404 / MANIFEST_UNKNOWN | Historical digest unavailable | 0 / 0 |
| V03 | `3df4ae153eb3` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984/) | 404 / MANIFEST_UNKNOWN | Historical digest unavailable | 0 / 0 |
| V04 | `7f1a5398e961` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-rocm100-7f1a5398e9610d96c473931a26c0e12bbe0d0423/) | 404 / MANIFEST_UNKNOWN | Historical digest unavailable | 0 / 0 |
| V05 | `29468dde8b51` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-rocm100-29468dde8b515031dc6d4d9d06bf0a2fa0442098/) | 404 / MANIFEST_UNKNOWN | Historical digest unavailable | 0 / 0 |
| V06 | `ac9126e58aa7` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516/) | 404 / MANIFEST_UNKNOWN | Historical digest unavailable | 0 / 0 |
| V07 | `7d0b4e57aac4` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-rocm100-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe/) | 404 / MANIFEST_UNKNOWN | Historical digest unavailable | 0 / 0 |
| V08 | `5555a2d8f711` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-rocm100-5555a2d8f7116b2b4877f66516f4128176cf722a/) | 404 / MANIFEST_UNKNOWN | Historical digest unavailable | 0 / 0 |
| V09 | `e68efc685ab1` | [404](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f/) | 404 / MANIFEST_UNKNOWN | 200 | Not queried |
| V10 | `335dd77b0916` | [200](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1/) | 200 | 200 | Not queried |
| V11 | `4fab5a074758` | [200](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-4fab5a07475822c6eaa971dceb2e2220d0446808/) | 200 | 200 | Not queried |

V09's historical digest remains addressable:

```text
vllm/vllm-openai-rocm@sha256:520e098f54ec96837228aaddea0a833cbf6374dbdffd480c1d65f26ffab7701b
```

V01, V10 and V11 exact tags return HTTP 200, providing successful controls against the same repository and APIs. The seven failures are explicit missing-tag responses, not transport or authentication errors.

This check uses HTTP metadata only and does not repeat Docker pulls. The previous [CPU pull receipts](docker_receipts.json) remain the evidence for successful image exports. Recovery of V02–V08 requires a historical digest, an independent archive, or a CI image/squashfs cache. Upstream vLLM commits alone do not recover historical AITER binaries/sources.
