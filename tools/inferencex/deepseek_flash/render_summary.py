#!/usr/bin/env python3
"""Render the DeepSeek Flash image index, complete observation table."""
import argparse
import json
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--knowledge',type=Path,required=True)
a=p.parse_args();root=a.knowledge
read=lambda path:json.loads((root/path).read_text())
points=read('data/points.json');images=read('data/images.json');pairs=read('data/pair_records.json');inventory=read('evidence/image_source_inventory.json')
versions={x['image_name']:f'V{i:02d}' for i,x in enumerate(images,1)}
intro='''# DeepSeek V4.1 Flash — MI355X / vLLM

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
'''
for i,(im,inv) in enumerate(zip(images,inventory),1):
    count=sum(p['image_name']==im['image_name'] for p in points)
    recovery='Current tag; historical digest unknown' if i==1 else 'CI-corroborated upstream commit; tag unavailable' if i<9 else 'Container; API digest verified'
    intro+=f"| V{i:02d} | {im['first_date']} → {im['last_date']} | `{im['image_name'].split(':',1)[1]}` | {len(im['representative_ids'])} / {count} | {recovery} |\n"
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
print(json.dumps({'observations':len(points),'images':len(images),'pairs':len(pairs),'summary_documents':2}))
