# DeepSeek Flash: MI355X / vLLM pilot

The requested name resolves to catalog `DeepSeek-V4.1-Flash`, DB key `dsv41flash`, [page](https://inferencex.semianalysis.com/infx/inference/deepseek-v41-flash). Query `/api/v1/benchmarks?model=DeepSeek-V4.1-Flash` and `/api/v1/benchmarks/history?model=DeepSeek-V4.1-Flash&benchmarkType=agentic_traces`. Never silently substitute another model version.

As of 2026-11-09, MI355X history contains vLLM 154 rows, SGLang 56, ATOM 65 and TileRT 1. The user selected vLLM; the auto policy also chooses vLLM because it has the newest published curve. A future model's framework must be discovered from its own coverage. Explicit selection overrides auto; preserve other-framework coverage without mixing performance pairs.

The pilot has 11 Docker names, 17 configuration rows, 127 image/configuration representatives and 110 consecutive-observed-image pairs; 9 skip V10, which has only TP8 c1. All selected points are EP1, so use TP2/EP1, TP4/EP1 and TP8/EP1, not TEP. KV offload off does not mean Engram offload off. Preserve server admission and active token-row limits separately from client concurrency.

CI handling learned here:

- Fetch GitHub jobs with `filter=all`. A dashboard result can come from an earlier run attempt or a canary. Select successful matching TP/EP, concurrency, framework and offload jobs from the latest eligible attempt, respecting `/attempts/N` in point URLs. Preserve ambiguity rather than choosing arbitrarily.
- Some vLLM job labels omit `/EP1`. SGLang can appear as `sgl`. Match the actual framework labels and model keys, not hardcoded Muse names.
- Legacy logs contain timestamped plain `vllm serve` with escaped JSON; newer logs use `[INFO] Command:`. Parse shell quoting with `shlex`. An absent resolved environment line means unknown environment, not no settings.
- Follow actual checkout revisions as well as run `head_sha`. Nested server artifact names may repeat across attempts; verify the artifact ID's upload link in the exact endpoint log before claiming a match.

Container recovery learned here:

- V01, V09, V10 and V11 pulled on a CPU-only node. V02–V08 tags were removed. Full-SHA tags plus matching CI startup package versions identify upstream vLLM commits, but not historical image bytes or AITER dependencies.
- V01's present tag resolves and its package commit matches CI; no historical digest was recovered. Only V09–V11 match API-recorded immutable digests. Direct Hub/Registry checks subsequently confirmed V02–V08 tag 404s and zero full/short-SHA tag matches. V09 also returns 404 by tag but HTTP 200 by historical digest, demonstrating that absent tags do not establish deleted image bytes. See the model's `evidence/dockerhub_verification.md`.
- Use inventories keyed by image name/source snapshot, not `None` digests. Preserve source_kind (`container` or `upstream_commit`), historical digests, current resolved digests and original/repository paths. Normalize `deepseek_v4_1` → `deepseek_v41` for diffs only.
- vLLM can be installed under `/usr/local/lib/python3.12/dist-packages/vllm`, with separate `/app/vllm` build copies. Prefer the installed runtime package and retain original paths/hashes.

Fusion review learned here:

- The two persistent FlyDSL mono launches have explicit row, TP, cache, graph and routing guards. At index sources, compressor/indexer execute between them. Engram injection remains outside them.
- V07 uses `VLLM_ROCM_MONO_DECODE=1`; V08 migrates to `--monokernel` and broader backbone coverage; V09 adds draft routing; V10 adds TP8; V11 enables existing gfx942 metadata/Markov helpers on gfx950 and adds TP2 ratio-2-only tuning.
- A helper existing in an image does not establish MI355X eligibility. A TP2-only body change must not be attributed to a TP8 pair. Client concurrency cannot determine how much of a run fits the 48-token-row mono guard.
- API-level AITER fusion contracts are weaker evidence than recovered device bodies. Mark dependency gaps and do not assert exact historical launch counts from a wrapper name.
- These benchmarks use five DSpark drafts with synthetic acceptance length 3.51. Performance deltas do not establish natural acceptance or causal kernel effects.

[Knowledge](../../../../../knowledge/top_model/deepseek_v41_flash/README.md) · [Evidence refresh tools](../../../deepseek_flash/README.md)

The [recovered-container log supplement](../../../../../knowledge/top_model/deepseek_v41_flash/recovered_image_history.md) inspects 31 exact endpoint artifacts and 16 pairs. It aligns server diagnostics to request warmup and E2E measurement, maps named JIT symbols to file/body hashes, and separates unchanged device bodies from new fusion. V11 TP2 c64 records a four-row quant+GEMM HIT during measurement, while both TP8 c1 endpoints retain the same block32 fusion body and 41 pre-measurement HIT shape signatures. Every first mono/Markov active message is before measurement. These examples show why client concurrency, active logs and JIT warnings cannot substitute for a complete GPU trace.
