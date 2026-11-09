#!/usr/bin/env python3
"""Render the DeepSeek Flash image index, complete observation table and reviewed source chain."""
import argparse
import json
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--knowledge',type=Path,required=True)
a=p.parse_args();root=a.knowledge
read=lambda path:json.loads((root/path).read_text())
points=read('data/points.json');images=read('data/images.json');pairs=read('data/pair_records.json');rules=read('evidence/kernel_pair_rules.json');inventory=read('evidence/image_source_inventory.json')
versions={x['image_name']:f'V{i:02d}' for i,x in enumerate(images,1)}
intro='''# DeepSeek-V4.1-Flash: MI355X / vLLM evolution

Collected **2026-11-09 UTC** from the [InferenceX dashboard](https://inferencex.semianalysis.com/infx/inference/deepseek-v41-flash). The requested “DeepSeek v4 flash” resolves to the catalog's **DeepSeek-V4.1-Flash** (`dsv41flash`); no different model version is merged into this history. [Reusable workflow skill](../../../tools/inferencex/skills/inferencex-kernel-evolution/SKILL.md).

The primary [configuration × image matrix](configuration_history.md) covers **2026-09-12 → 2026-11-09**, **11 Docker image names**, **154 observations**, **17 configuration rows**, and **127 image/configuration representatives**. All observations have unique CI job matches. Every one of the **110 consecutive-observed-image pairs** has its own kernel record; **9** span a missing V10 column. The narrower adjacent-image-only CSV contains 101 pairs.

- [Full configuration matrix](configuration_history.md): all observed TP2/TP4/TP8 + concurrency points, with blanks and per-pair links.
- [Pair index](pair_changes.md): both dates, performance changes, source paths/hashes, implementation languages, bodies, fusion and exact launch changes.
- [All 154 observations](performance_points.md): repeats are preserved; deduplication chooses one representative per image/configuration.
- [Kernel/fusion evolution](kernel_evolution.md): image-to-image source chain, including transitions with no measured matching point.
- [Machine-readable pairs](data/pair_records.json), [raw selected history](data/raw_selected_history.json), [coverage](data/coverage.json), [source inventory](evidence/image_source_inventory.json), [source changes](evidence/source_changes.json), [reviewed applicability rules](evidence/kernel_pair_rules.json).

**Container evidence is incomplete for historical tags.** CPU-only job 275610 successfully pulled/exported V01, V09, V10 and V11; V02–V08 tags are unavailable. Those seven snapshots are pinned upstream vLLM commits corroborated by the actual CI startup version, not recovered container bytes. V01's current tag matches the logged vLLM commit but lacks a historical digest. V09–V11 match the API's immutable digests. Historical AITER dependency internals for V02–V08 remain unavailable. These distinctions are carried into every pair record. [Direct Docker Hub/Registry verification](evidence/dockerhub_verification.md) confirms V02–V08 tag 404s; V09 also has a missing tag but its historical digest remains addressable. Missing tags do not prove deleted image bytes.

## Scope and comparison semantics

Hardware is fixed to **MI355X**, and the framework is **vLLM**, as explicitly selected by the user. [Framework discovery](data/framework_discovery.json) records all MI355X coverage: vLLM 154 rows, SGLang 56, ATOM 65 and TileRT 1. The reusable workflow now inspects actual coverage to select vLLM or SGLang; its automatic policy also chooses vLLM here. It never combines different frameworks into one same-stack pair.

These are aggregate AgentX replay points with FP4 model precision and **EP1**, so label them **TP2/EP1, TP4/EP1, TP8/EP1**, not TEP. The matrix's `GPU` means **KV offload is off**. It does not mean all model state is on GPU: the launch's Engram CPU-offload setting is separately retained (often on for TP2). All inspected representative commands use DSpark, **five speculative tokens**, probabilistic draft sampling and **synthetic acceptance length 3.51**. That protocol is not a measurement of natural model acceptance. ISL/OSL vary over the replay dataset.

**x = p90 interactivity = 1 / p90 full-response ITL in seconds (tok/s/user)**. **y = output_tput_per_gpu (output tok/s/chip)**. Both axes are better when larger. Throughput is already per chip: TP4/EP1 uses four chips, not TP×EP or eight chips. The collector uses legacy p90 ITL only when the full-response schema is absent. The inspected chart implementation is [benchmark-transform.ts](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d475545122661430a1f6de280e81418b0ba2f198/packages/app/src/lib/benchmarks/benchmark-transform.ts); this inspected revision does not prove the live deployment SHA.

Version nodes use the full Docker name/tag. Representatives are the earliest valid point for each image and full available dashboard configuration. Repeated same-image observations, their recipe fingerprints and spreads remain in [images.json](data/images.json). Pairing connects consecutive **observed** image versions within a row; it never interpolates a blank or compares different TP/concurrency values. Dates are dashboard dates, distinct from CI creation/execution and image creation times.

Same dashboard configuration does not guarantee identical execution: model/dataset revisions are not fully pinned at all endpoints; Engram placement, server admission, graph capture, batched-token limits, quick-reduce quantization, ROCm/AITER versions and recipes can change. Each pair includes its own two launch commands and environment delta. These are deployment-evolution measurements, not isolated causal kernel speedups.

## Image versions

All tags below have prefix `vllm/vllm-openai-rocm:`. A source commit corroborated by CI is evidence of vLLM revision only, not of all historical image bytes.

| Version | First → last dashboard date | Image tag | Representatives / observations | Source recovery |
| --- | --- | --- | ---: | --- |
'''
for i,(im,inv) in enumerate(zip(images,inventory),1):
    count=sum(p['image_name']==im['image_name'] for p in points)
    recovery='Current tag; historical digest unknown' if i==1 else 'CI-corroborated upstream commit; tag unavailable' if i<9 else 'Container; API digest verified'
    intro+=f"| V{i:02d} | {im['first_date']} → {im['last_date']} | `{im['image_name'].split(':',1)[1]}` | {len(im['representative_ids'])} / {count} | {recovery} |\n"
intro+='''
## Selected point-to-point examples

Cells are `(x, y)` in the units above; the full matrix preserves every configuration, including points without a later version.

| Configuration | Dates / versions | Before (x, y) | After (x, y) | Δ x | Δ y | Kernel record |
| --- | --- | ---: | ---: | ---: | ---: | --- |
'''
for tp,conc,from_v,to_v in [(4,1,'V01','V02'),(4,1,'V06','V07'),(4,1,'V07','V08'),(4,1,'V08','V09'),(4,2,'V09','V11'),(2,1,'V09','V11'),(2,32,'V09','V11'),(2,64,'V09','V11'),(8,1,'V10','V11')]:
    x=next(p for p in pairs if p['config']['decode_tp']==tp and p['config']['conc']==conc and versions[p['before']['image_name']]==from_v and versions[p['after']['image_name']]==to_v)
    b,c=x['before'],x['after']
    intro+=f"| TP{tp}/EP1 c{conc} | {b['date']} → {c['date']} ({from_v}→{to_v}) | ({b['p90_interactivity_tok_s_user']:.3f}, {b['output_tok_s_chip']:.3f}) | ({c['p90_interactivity_tok_s_user']:.3f}, {c['output_tok_s_chip']:.3f}) | {x['interactivity_change_pct']:+.2f}% | {x['throughput_change_pct']:+.2f}% | [Changes]({x['report']}) |\n"
intro+='''
## Evidence and reproduction

Performance values are **measured** E2E CI results. Kernel facts are **source_verified** candidate paths, with historical-container limitations above. **GPU-time top-kernel ranks, benchmark-phase launch counts, and per-kernel causal contributions are unavailable**: no compatible original trace was recovered. Server startup/mono messages establish initialization eligibility and may include warmup; they do not rank hot kernels.

The [127 endpoint CI extracts](evidence/ci_log_extracts.json), [launch configurations](evidence/ci_launch_configs.json), [13-run metadata](data/ci_manifest.json), [pinned recipes](evidence/ci_recipe_sources.json) and [server artifacts](evidence/server_log_extracts.json) preserve provenance. Eleven selected server artifacts have an exact upload link in the endpoint's own job log. Two older V08 attempt artifacts are retained as inspected candidates but are explicitly not matched to the representative endpoint. Canary and rerun jobs are retained where the dashboard used them. Early log environments without a resolved environment line are marked unknown.

[CPU receipt](evidence/cpu_job_receipt.txt) records a completed **1 CPU / 0 GPU** allocation. [Docker receipts](evidence/docker_receipts.json) retain successes and failed historical pulls. Full images, raw logs, artifact ZIPs and source bundles remain outside Git in `/shared_nfs/yueliu14/inferencex_deepseek_flash_research/`; archive/file hashes and compact diffs are committed. [Retrieval manifest](data/retrieval_manifest.json) records API URLs and response identities.

From the repository root:

```bash
python3 tools/inferencex/skills/inferencex-kernel-evolution/scripts/collect_history.py \\
  --model DeepSeek-V4.1-Flash --framework auto \\
  --cache /shared_nfs/yueliu14/inferencex_deepseek_flash_research/raw \\
  --output knowledge/top_model/deepseek_v41_flash/data --ci --offline
python3 tools/inferencex/deepseek_flash/prepare_sources.py \\
  --cache /shared_nfs/yueliu14/inferencex_deepseek_flash_research \\
  --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/skills/inferencex-kernel-evolution/scripts/build_pair_history.py \\
  --knowledge knowledge/top_model/deepseek_v41_flash
python3 tools/inferencex/deepseek_flash/render_summary.py \\
  --knowledge knowledge/top_model/deepseek_v41_flash
```

The cached run is deterministic for performance/pair selection. For fresh data, use a fresh raw cache and omit `--offline`; review new source rules before rendering new transitions. [Collection tools](../../../tools/inferencex/deepseek_flash/README.md) describe evidence refresh. Coverage is the currently retained published history, not every CI run ever executed or every historical deployed frontier.
'''
(root/'README.md').write_text(intro)
text='''# Every retained DeepSeek Flash observation

[Configuration matrix](configuration_history.md) · [Image index](README.md#image-versions)

All 154 distinct selected API records are retained. `Representative` marks the earliest valid record per full configuration and Docker image name. The two axes are p90 interactivity (tok/s/user) and output token throughput/chip (tok/s/chip). `KV offload=off` does not describe Engram placement. Dates are dashboard dates.

| Point | Date | Image | TP / EP | Concurrency | KV offload | x | y | Representative | CI |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
'''
representatives={pid for im in images for pid in im['representative_ids']}
for x in sorted(points,key=lambda x:(x['date'],versions[x['image_name']],x['config']['decode_tp'],x['config']['conc'],x['id'])):
    c=x['config']; text+=f"| {x['id']} | {x['date']} | {versions[x['image_name']]} | {c['decode_tp']} / {c['decode_ep']} | {c['conc']} | {c['offload_mode']} | {x['p90_interactivity_tok_s_user']:.6f} | {x['output_tok_s_chip']:.5f} | {'yes' if x['id'] in representatives else 'repeat'} | [Job]({x['job_url']}) |\n"
(root/'performance_points.md').write_text(text)
text='''# DeepSeek Flash kernel and fusion evolution

[Full performance matrix](configuration_history.md) · [110 measured pair records](pair_changes.md) · [Source inventory](evidence/image_source_inventory.json)

This is a source-verified candidate-path history, not a measured GPU-time ranking. Source kinds and hashes are preserved per file. V02–V08 use exact upstream commits corroborated by CI; their historical container/AITER bytes are unavailable. V01 is a current tag snapshot, while V09–V11 are API-digest-matched containers. Each source edge can contain many unrelated changes; the reviewed pipelines below identify relevant changes and their applicability, without claiming exhaustive kernel coverage or causal E2E speedups.

The main fusion progression is ordinary staged operators → fused reduction/inverse RoPE → reduction/inverse RoPE/MXFP8 output → fused mHC API → two persistent FlyDSL launches → broader backbone coverage → draft-layer coverage → TP8 eligibility → MI355X metadata/Markov helper eligibility and TP2-specific internal tuning. Compressor/indexer and Engram are not silently included in the two persistent launches.

'''
for n,r in enumerate(rules):
    av,bv=versions[r['from_image_name']],versions[r['to_image_name']]
    text+=f"## {av} → {bv}"+(f" (TP{r['config_selector']['decode_tp']})" if r['config_selector'] else '')+'\n\n'
    count=sum(p['before']['image_name']==r['from_image_name'] and p['after']['image_name']==r['to_image_name'] and all(p['config'].get(k)==v for k,v in r['config_selector'].items()) for p in pairs)
    text+=f"{count} measured same-configuration pairs. "+('This is an image-source transition only; V10 has only TP8 c1, absent from V09.\n\n' if not count else 'See the pair index for exact dates, both axes and endpoint-specific commands.\n\n')
    for k in r['kernel_changes']:
        text+=f"### {k['pipeline']}\n\n**Before:** {k['before']}\n\n**After:** {k['after']}\n\n**Language:** {k['language']}\n\n**Body:** {k['body_change']}\n\n**Fusion:** {k['fusion']}\n\n**Guards:** {k['applicability']}\n\n"
        text+='Source paths: '+', '.join('`'+(s['after_path'] or s['before_path'])+'`' for s in k['sources'])+'. Full original paths, before/after SHA256 and source kind are in the inventory and per-pair records.\n\n'
        text+='Evidence: '+', '.join(f'[{ref}]({ref})' for ref in k['evidence'])+'.\n\n'
    if r['excluded_attributions']:text+='Excluded attributions:\n\n'+''.join('- '+x+'\n' for x in r['excluded_attributions'])+'\n'
(root/'kernel_evolution.md').write_text(text.rstrip()+'\n')
print(json.dumps({'observations':len(points),'images':len(images),'pairs':len(pairs),'summary_documents':3}))
