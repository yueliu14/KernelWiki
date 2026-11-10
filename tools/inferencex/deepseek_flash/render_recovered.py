#!/usr/bin/env python3
"""Render per-configuration comparisons grounded in recovered sources and phase-aligned logs."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read(p):return json.loads(p.read_text())


def source_cell(f):
    if not f:
        return 'Absent from captured source selection'
    text = f"`{f['original_path']}`<br>`{f['sha256']}`"
    implementation = f.get('implementation')
    if implementation:
        url = '../' + implementation['snapshot_path']
        text += f'<br>[Complete implementation]({url})'
        text += ''.join(f"<br>[{e['name']}]({url}#L{e['line']})" for e in implementation['entrypoints'])
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--knowledge',type=Path,required=True)
    root=p.parse_args().knowledge
    pairs=read(root/'data/recovered_pair_records.json')
    facts={f['point_id']:f for f in read(root/'evidence/recovered_runtime_facts.json')}
    inventory={x['version']:x for x in read(root/'evidence/recovered_source_inventory.json')}
    files={v:{f['path']:f for f in x['files']} for v,x in inventory.items()}
    source_index_path = root / 'kernel_sources/manifest.json'
    source_index = {s['version']: s for s in read(source_index_path)['snapshots']} if source_index_path.exists() else {}
    for version, snapshot in source_index.items():
        for f in snapshot['files']:
            if version in files and f['path'] in files[version]:
                assert files[version][f['path']]['sha256'] == f['sha256']
                files[version][f['path']]['implementation'] = f
    reviews={(r['from_version'],r['to_version']):r for r in read(root/'evidence/recovered_kernel_reviews.json')}
    (root/'recovered_pairs').mkdir(exist_ok=True)
    index='''# Recovered-container point-to-point kernel comparisons

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
'''
    links=[]
    for pair in pairs:
        av,bv=pair['from_version'],pair['to_version'];b,c=pair['before'],pair['after'];cfg=pair['config'];bf,cf=facts[b['id']],facts[c['id']]
        title=f"TP{cfg['decode_tp']}/EP{cfg['decode_ep']} conc{cfg['conc']}"
        index+=f"| {title} | V{av:02d}→V{bv:02d}; {b['date']}→{c['date']} | ({b['p90_interactivity_tok_s_user']:.3f}, {b['output_tok_s_chip']:.3f}) | ({c['p90_interactivity_tok_s_user']:.3f}, {c['output_tok_s_chip']:.3f}) | {pair['interactivity_change_pct']:+.2f}% | {pair['throughput_change_pct']:+.2f}% | [Detailed comparison]({pair['report']}) |\n"
        text=f"# {title}: V{av:02d} → V{bv:02d}\n\n[Recovered-image comparisons](../recovered_image_history.md) · [Full matrix](../configuration_history.md)\n\n"
        if source_index:
            text += '**vLLM commits:** ' + ' → '.join(f"[{source_index[v]['vllm_commit'][:12]}]({source_index[v]['vllm_commit_url']})" for v in [av,bv]) + '. [Kernel implementation browser](../kernel_implementations.md). Container file identity is retained separately from the reported framework commit.\n\n'
        if pair['original_pair_report']:text+=f"[Original pair and exact launch changes](../{pair['original_pair_report']}).\n\n"
        else:text+='Supplemental comparison across V02–V08; intermediate measured points remain in the original matrix. This is not an adjacent-image kernel attribution.\n\n'
        text+='| Endpoint | Docker image | Point / exact CI | (p90 interactivity, output/chip) | Server artifact |\n| --- | --- | --- | ---: | --- |\n'
        for side,point,fact,version in [('Before',b,bf,av),('After',c,cf,bv)]:
            text+=f"| {side}: {point['date']} | `{point['image_name']}` | [{point['id']}]({point['job_url']}) | ({point['p90_interactivity_tok_s_user']:.6f}, {point['output_tok_s_chip']:.5f}) | `{fact['artifact_id']}` |\n"
        text+=f"\nInteractivity **{pair['interactivity_change_pct']:+.2f}%**; throughput/chip **{pair['throughput_change_pct']:+.2f}%**. E2E measurements do not isolate kernel causality.\n\n"
        text+='## Exact log evidence and measurement windows\n\nBoth artifacts are joined through their upload IDs in their own endpoint job logs. Server messages are aligned against that job\'s AIPerf/CI phase boundaries; timestamps are treated as UTC, and events within one second of a boundary are marked uncertain. AIPerf `profiling` means E2E measurement.\n\n| Endpoint | E2E measurement start | E2E measurement end |\n| --- | --- | --- |\n'
        for side,fact in [('Before',bf),('After',cf)]:text+=f"| {side} | {fact['phase_windows'].get('profiling_start',{}).get('timestamp','unknown')} | {fact['phase_windows'].get('profiling_end',{}).get('timestamp','unknown')} |\n"
        text+='\n| Diagnostic | Before observed phases | After observed phases | Interpretation |\n| --- | --- | --- | --- |\n'
        for kind,description in [('mono_dispatch','Reached guarded mono path; no sustained coverage fraction'),('markov_dispatch','Reached guarded sampling path; first message only'),('quant_gemm_dispatch','Explicit HIT includes row/weight/tile shape; logs once per shape'),('jit_compile','Named compilation events; not launches or total GPU time')]:
            def phases(fact):
                counts=Counter(e['phase'] for e in fact['events'] if e['kind']==kind)
                return '; '.join(f'{phase}: {n} diagnostic messages' for phase,n in sorted(counts.items())) or 'No matching message (not proof of non-use)'
            text+=f"| {kind} | {phases(bf)} | {phases(cf)} | {description} |\n"
        text+='\nSelected exact log lines (all retained events and source mappings are in [runtime facts](../evidence/recovered_runtime_facts.json)):\n\n'
        for side,fact in [('Before',bf),('After',cf)]:
            chosen=[]
            for kind in ['mono_dispatch','markov_dispatch','quant_gemm_dispatch']:
                matches=[e for e in fact['events'] if e['kind']==kind]
                measured=[e for e in matches if e['phase']=='e2e_measurement']
                chosen.extend(measured or matches[:1])
            chosen.extend(e for e in fact['events'] if e['kind']=='jit_compile' and e['phase']=='e2e_measurement')
            for e in chosen:
                text+=f"- **{side}, {e['phase']}**, `{e['file']}:{e['line']}`: `{e['text']}`\n"
        text+='\n## Reviewed kernel bodies and fusion boundaries\n\nThese findings use recovered container file bytes. V01\'s tag was not historically digest-pinned; V09–V11 were. Source facts establish possible launch boundaries; only the log observations above establish which diagnostic path was reached.\n'
        excluded=[]
        for review in reviews[av,bv]['pipelines']:
            if review['tp_selector'] and cfg['decode_tp'] not in review['tp_selector']:
                excluded.append(review['title']+': excluded by TP guard.');continue
            text+=f"\n### {review['title']}\n\n"
            for label,key in [('Before','before'),('After','after'),('Device language','language'),('Code/body','body_change'),('Fusion boundaries','fusion'),('Applicability / log limit','guards')]:text+=f"**{label}:** {review[key]}\n\n"
            text+='| Comparison path | Before installation / implementation | After installation / implementation | File relation |\n| --- | --- | --- | --- |\n'
            for path in review['paths']:
                x,y=files[av].get(path),files[bv].get(path)
                relation='identical bytes' if x and y and x['sha256']==y['sha256'] else 'changed bytes' if x and y else 'added/absent in snapshot'
                text+=f"| `{path}` | {source_cell(x)} | {source_cell(y)} | {relation} |\n"
        if excluded:text+='\nExcluded from this configuration:\n\n'+''.join('- '+x+'\n' for x in excluded)
        text+=f"\nPinned source evidence: [inventory](../evidence/recovered_source_inventory.json), [reviewed rules](../evidence/recovered_kernel_reviews.json), [V{av:02d}→V{bv:02d} source diff](../evidence/recovered_source_diffs/v{av:02d}_to_v{bv:02d}.diff). Installation paths are preserved in the inventory; canonical comparison paths above normalize the earlier deepseek_v4_1 directory name.\n"
        text+='\n## Named JIT kernels: observed sets and body comparisons\n\nThis table lists every symbol mentioned by the selected endpoints\' JIT diagnostics. Missing messages do not imply missing execution: the monitor activates after server warmup and uses warning_once. A changed file can contain an unchanged symbol; AST comparisons below separate those cases. A name match is a source candidate, not proof of a unique binary implementation.\n\n| Symbol | Before logged phases | After logged phases | Candidate source / symbol-body relation |\n| --- | --- | --- | --- |\n'
        for comparison in pair['jit_kernel_comparisons']:
            refs=[]
            for candidate in comparison['source_comparisons']:
                ref=f"`{candidate['path']}`: {candidate['symbol_body_relation']}"
                for side,version in [('before',av),('after',bv)]:
                    f=files[version].get(candidate['path'],{}).get('implementation')
                    if f:
                        symbol=next((s for s in f['symbols'] if s['name'].rsplit('.',1)[-1]==comparison['symbol']),None)
                        line=symbol['line'] if symbol else 1
                        ref+=f" [{side} source](../{f['snapshot_path']}#L{line})"
                refs.append(ref)
            refs='; '.join(refs) or 'No mapped source'
            text+=f"| `{comparison['symbol']}` | {', '.join(comparison['before_log_phases']) or 'not logged'} | {', '.join(comparison['after_log_phases']) or 'not logged'} | {refs} |\n"
        text+='\nThe [structured pair record](../data/recovered_pair_records.json) retains complete generated kernel names (including FlyDSL variant parameters), candidate source paths, file hashes, symbol code/AST hashes and line ranges. Triton JIT log labeling can include Gluon implementations; source decorators distinguish them.\n\nNo GPU trace, kernel time ranking, per-step launch count, or causal kernel speedup is claimed. Admission, graph capture, batching, quick-reduce quantization and synthetic DSpark acceptance are retained in the original CI evidence and can affect E2E deltas.\n'
        (root/pair['report']).write_text(text)
        if pair['original_pair_id']:links.append({'pair_id':pair['original_pair_id'],'report':pair['report'],'before_point_id':b['id'],'after_point_id':c['id'],'before_artifact_id':bf['artifact_id'],'after_artifact_id':cf['artifact_id'],'evidence_status':'phase_aligned_log_observations_and_recovered_container_source','gpu_profile_status':'unavailable'})
    index+='''
## Evidence and reproduction

[Endpoint artifact manifests](evidence/recovered_endpoint_logs.json) · [Phase-aligned runtime facts](evidence/recovered_runtime_facts.json) · [Recovered pair records](data/recovered_pair_records.json) · [Source inventories](evidence/recovered_source_inventory.json) · [Reviewed kernel/fusion rules](evidence/recovered_kernel_reviews.json).

The 31 archives total 353,654,031 compressed bytes and are retained outside Git with full logs, AIPerf request exports and source snapshots. There are 298 JIT warning lines, 50 mono active lines, 10 Markov dispatch lines and 481 tiny-M HIT lines in the selected server streams. These totals describe diagnostic evidence only. Some messages occur during request warmup or graph preparation; each event retains its phase, exact file/line and file hash. All 31 jobs have recovered measurement windows.

Run `collect_recovered_logs.py`, `prepare_recovered_sources.py`, `analyze_recovered_logs.py` with `--cache ... --knowledge knowledge/top_model/deepseek_v41_flash`, followed by `render_recovered.py --knowledge ...`, then the common `build_pair_history.py`. See [collection tools](../../../tools/inferencex/deepseek_flash/README.md). Curated fusion rules are reviewed separately; discovering a new symbol does not automatically assign a speedup.
'''
    (root/'recovered_image_history.md').write_text(index)
    (root/'evidence/recovered_pair_links.json').write_text(json.dumps(links,indent=2)+'\n')
    print(json.dumps({'pair_reports':len(pairs),'original_pair_links':len(links)}))


if __name__=='__main__':main()
