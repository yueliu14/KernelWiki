#!/usr/bin/env python3
"""Align kernel diagnostics to AgentX phase windows and pair recovered-image endpoints."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def read(p):return json.loads(p.read_text())


def timestamp(text,year):
    match=re.search(r'\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)',text)
    if match:return datetime.fromisoformat(match[1]+'T'+match[2][:15])
    match=re.search(r'\b(\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})',text)
    return datetime.fromisoformat(str(year)+'-'+match[1]+'T'+match[2]) if match else None


def phase_at(t,windows):
    if t is None or 'profiling_start' not in windows:return 'unknown'
    start=datetime.fromisoformat(windows['profiling_start']['timestamp'])
    end=datetime.fromisoformat(windows['profiling_end']['timestamp']) if 'profiling_end' in windows else None
    if abs((t-start).total_seconds())<1:return 'phase_boundary_uncertain'
    if t<start:
        warm=windows.get('warmup_start')
        return 'benchmark_warmup' if warm and t>=datetime.fromisoformat(warm['timestamp']) else 'server_setup_or_graph_warmup'
    if end and t>end:return 'after_measurement'
    return 'e2e_measurement'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--knowledge',type=Path,required=True)
    a=p.parse_args();root=a.knowledge
    points={x['id']:x for x in read(root/'data/points.json')}
    images=read(root/'data/images.json');snapshots=read(root/'evidence/recovered_source_inventory.json')
    inv={x['version']:{f['path']:f for f in x['files']} for x in snapshots}
    logs=read(root/'evidence/recovered_endpoint_logs.json');facts=[]
    for row in logs:
        point=points[row['point_id']];year=int(point['date'][:4]);windows={}
        phase_sources=[f for f in row['files'] if f['name'].endswith('/aiperf.log')]
        phase_sources.append({'name':'CI job log','external_file':str(a.cache/'raw/ci'/f"{row['job_id']}.log")})
        for f in phase_sources:
            for n,line in enumerate(Path(f['external_file']).read_text(errors='replace').splitlines(),1):
                key=None
                if 'Phase profiling (profiling) started' in line:key='profiling_start'
                elif 'Phase profiling (profiling) complete |' in line:key='profiling_end'
                elif 'Phase warmup (warmup) started' in line:key='warmup_start'
                elif 'Phase warmup (warmup) complete |' in line:key='warmup_end'
                if key and key not in windows:
                    t=timestamp(line,year)
                    if t:windows[key]={'timestamp':t.isoformat(),'file':f['name'],'line':n,'text':line}
        events=[]
        servers=[f for f in row['files'] if f['name']=='results/server.log' or re.search(r'_agg_w\d+\.out$',f['name'])]
        for f in servers:
            text=Path(f['external_file']).read_text(errors='replace')
            assert hashlib.sha256(Path(f['external_file']).read_bytes()).hexdigest()==f['sha256']
            for n,line in enumerate(text.splitlines(),1):
                kind=None
                if 'JIT compilation during inference:' in line:kind='jit_compile'
                elif 'tiny-M fused MXFP8 quant+GEMM HIT' in line:kind='quant_gemm_dispatch'
                elif 'DSpark fused Markov sampling:' in line:kind='markov_dispatch'
                elif 'DSv4.1 mono decode' in line:kind='mono_dispatch' if 'active' in line else 'mono_initialization'
                if not kind:continue
                t=timestamp(line,year)
                e={'kind':kind,'file':f['name'],'file_sha256':f['sha256'],'line':n,'text':line,'timestamp':t.isoformat() if t else None,'phase':phase_at(t,windows)}
                if kind=='jit_compile':
                    m=re.search(r'(Triton|FlyDSL|TileLang|CuTeDSL) kernel JIT compilation during inference: (.*)\. This causes',line)
                    if not m:m=re.search(r'(Triton|FlyDSL|TileLang|CuTeDSL) JIT compilation during inference: (.*)\. This causes',line)
                    if m:
                        e['logger_backend']=m[1];e['logged_name']=m[2];name=m[2].split(' (',1)[0];e['symbol']=name.split('.')[0]
                        candidates=[]
                        for path,source in inv[row['version']].items():
                            if any('/'+arch+'/' in path for arch in ['gfx942','gfx1250','gfx1100']):continue
                            if e['symbol'] in source['matched_log_symbols']:
                                candidates.append({'path':path,'original_path':source['original_path'],'file_sha256':source['sha256'],**source['matched_log_symbols'][e['symbol']]})
                        e['source_candidates']=candidates
                        e['mapping_status']='unique_name_match' if len(candidates)==1 else 'ambiguous_name_match' if candidates else 'unresolved'
                if kind=='quant_gemm_dispatch':
                    match=re.search(r'M=(\d+), N=(\d+), K=(\d+), packed tile (\d+)x(\d+)x(\d+) k_pack=(\d+)(?:, fused_splits=(\d+))?',line)
                    if match:e['shape_and_tile']=dict(zip(['M','N','K','BLOCK_M','BLOCK_N','BLOCK_K','K_PACK','FUSED_SPLITS'],[int(x) if x else None for x in match.groups()]))
                events.append(e)
        facts.append({'version':row['version'],'point_id':row['point_id'],'job_id':row['job_id'],'job_url':row['job_url'],'artifact_id':row['artifact_id'],'config':row['config'],'phase_windows':windows,'phase_semantics':'AgentX/AIPerf profiling is E2E measurement, not a GPU kernel profiler. UTC/log-clock alignment; second-resolution server events within one second of the measurement boundary are marked uncertain.','events':events,'event_counts_by_kind_and_phase':dict(Counter(e['kind']+'/'+e['phase'] for e in events)),'absence_semantics':'No diagnostic does not prove no execution; info_once/warning_once suppress repetitions, cached compilations are silent, and JIT monitoring activates after server warmup.'})
    (root/'evidence/recovered_runtime_facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    bypoint={x['point_id']:x for x in facts};groups=defaultdict(list)
    for version,image in enumerate(images,1):
        if version not in inv:continue
        for pid in image['representative_ids']:
            point=points[pid];groups[json.dumps(point['config'],sort_keys=True)].append((version,point))
    original={(p['before']['id'],p['after']['id']):p for p in read(root/'data/pair_records.json')}
    pairs=[]
    for group in groups.values():
        for (av,b),(bv,c) in zip(group,group[1:]):
            bf,cf=bypoint[b['id']],bypoint[c['id']]
            old=original.get((b['id'],c['id']))
            slug=f"tp{b['config']['decode_tp']}_ep{b['config']['decode_ep']}_conc{b['config']['conc']}_v{av:02d}_to_v{bv:02d}"
            jit_a={e['symbol'] for e in bf['events'] if e['kind']=='jit_compile' and 'symbol' in e}
            jit_b={e['symbol'] for e in cf['events'] if e['kind']=='jit_compile' and 'symbol' in e}
            comparisons=[]
            for symbol in sorted(jit_a|jit_b):
                aa=[e for e in bf['events'] if e.get('symbol')==symbol];bb=[e for e in cf['events'] if e.get('symbol')==symbol]
                candidates=[]
                for path in sorted(inv[av].keys()|inv[bv].keys()):
                    if any('/'+arch+'/' in path for arch in ['gfx942','gfx1250','gfx1100']):continue
                    sf,tf=inv[av].get(path),inv[bv].get(path)
                    sm=sf and sf['matched_log_symbols'].get(symbol);tm=tf and tf['matched_log_symbols'].get(symbol)
                    if not sm and not tm:continue
                    candidates.append({'path':path,'before_source':sf if sm else None,'after_source':tf if tm else None,'symbol_body_relation':'identical_ast' if sm and tm and sm['ast_sha256']==tm['ast_sha256'] else 'changed_ast' if sm and tm else 'added_or_removed_or_not_captured'})
                comparisons.append({'symbol':symbol,'before_log_phases':sorted({e['phase'] for e in aa}),'after_log_phases':sorted({e['phase'] for e in bb}),'before_logged_names':sorted({e['logged_name'] for e in aa}),'after_logged_names':sorted({e['logged_name'] for e in bb}),'source_comparisons':candidates,'interpretation':'Logged compilation occurrence, not kernel launch count. Missing log evidence is not proof a kernel was unused.'})
            pairs.append({'id':slug,'original_pair_id':old['id'] if old else None,'original_pair_report':old['report'] if old else None,'from_version':av,'to_version':bv,'before':b,'after':c,'config':b['config'],'pairing_policy':'next observed recovered-container version within the full configuration; separate from the complete image history','skipped_versions':list(range(av+1,bv)),'interactivity_change_pct':100*(c['p90_interactivity_tok_s_user']/b['p90_interactivity_tok_s_user']-1),'throughput_change_pct':100*(c['output_tok_s_chip']/b['output_tok_s_chip']-1),'before_artifact_id':bf['artifact_id'],'after_artifact_id':cf['artifact_id'],'jit_kernel_comparisons':comparisons,'report':'recovered_pairs/'+slug+'.md','gpu_trace_available':False})
    pairs.sort(key=lambda p:(p['config']['decode_tp'],p['config']['conc'],p['from_version']))
    (root/'data/recovered_pair_records.json').write_text(json.dumps(pairs,indent=2)+'\n')
    print(json.dumps({'endpoints':len(facts),'phase_windows':sum('profiling_start' in f['phase_windows'] for f in facts),'pairs':len(pairs),'existing_pairs_enriched':sum(p['original_pair_id'] is not None for p in pairs),'supplemental_gap_pairs':sum(p['original_pair_id'] is None for p in pairs),'events':dict(Counter(e['kind'] for f in facts for e in f['events']))}))


if __name__=='__main__':main()
