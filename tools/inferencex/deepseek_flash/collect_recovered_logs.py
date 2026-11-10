#!/usr/bin/env python3
"""Collect exact endpoint server logs for recovered images; retain diagnostic evidence, not GPU profiles."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def classify(line):
    if re.search(r'mono decode|monokernel|mono.*(?:built|enable)', line, re.I): return 'mono'
    if re.search(r'fused Markov sampling', line, re.I): return 'markov'
    if re.search(r'markov|swa.*meta|mhc|engram', line, re.I): return 'fusion_helpers'
    if re.search(r'using.*backend|selected.*backend|attention backend|moe.*(?:backend|kernel)|fused_moe|sparse.*(?:mla|attention)', line, re.I): return 'backend'
    if re.search(r'\[aiter\].*(?:shape is|solution|tuned|compile|build|kernel)', line, re.I): return 'aiter_diagnostics'
    if re.search(r'graph captur|warm.?up|profil|application startup|Starting vLLM|vLLM API server|version \d', line, re.I): return 'phase_context'
    return None


def normalized(line):
    line=re.sub(r'\x1b\[[0-9;]*m', '', line)
    line=re.sub(r'\bpid[=:]\d+', 'pid=*', line)
    line=re.sub(r'\b(?:\d{4}-)?\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?', 'TIME', line)
    return re.sub(r'^.*?(?:\] |\[aiter\] )', '', line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--knowledge',type=Path,required=True)
    a=p.parse_args();out=a.cache/'raw/server_logs';out.mkdir(parents=True,exist_ok=True)
    points={p['id']:p for p in read(a.knowledge/'data/points.json')}
    images=read(a.knowledge/'data/images.json')
    snapshots={x['image_name']:x for x in read(a.knowledge/'evidence/image_source_inventory.json')}
    runs=read(a.knowledge/'data/ci_manifest.json')
    artifacts={x['id']:x for r in runs for x in r['artifacts']}
    selected=[]
    for version,image in enumerate(images,1):
        if snapshots[image['image_name']]['source_kind']!='container':continue
        for id in image['representative_ids']:
            point=points[id]
            text=(a.cache/'raw/ci'/f"{point['github_job_id']}.log").read_text()
            links=[{'line':n+1,'text':line} for n,line in enumerate(text.splitlines()) if re.search(r'/artifacts/\d+',line)]
            own={int(x) for x in re.findall(r'/artifacts/(\d+)',text)}
            server=[artifacts[x] for x in own if x in artifacts and artifacts[x]['name'].startswith('server_logs_')]
            if len(server)!=1:raise ValueError(f"Need one exact server artifact for job {point['github_job_id']}: {server}")
            artifact=server[0]
            selected.append((version,point,artifact,[l for l in links if f"/artifacts/{artifact['id']}" in l['text']]))

    def get(item):
        version,point,artifact,links=item
        f=out/f"{artifact['id']}.zip"
        if not f.exists():
            temporary=f.with_suffix('.partial')
            with temporary.open('wb') as stream:
                subprocess.run(['gh','api',artifact['archive_download_url']],stdout=stream,check=True)
            temporary.replace(f)
        directory=a.cache/'recovered_server_logs'/str(point['github_job_id']);directory.mkdir(parents=True,exist_ok=True)
        events={};counts=Counter();entries=[]
        def scan(name,data):
            safe=hashlib.sha256(name.encode()).hexdigest()[:12]+'_'+re.sub(r'[^A-Za-z0-9_.-]', '_', name)[-170:]
            target=directory/safe
            target.write_bytes(data)
            entries.append({'name':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'external_file':str(target)})
            if not name.endswith(('.log','.txt','.out','.err')):return
            for n,line in enumerate(data.decode(errors='replace').splitlines(),1):
                kind=classify(line)
                if not kind or len(line)>6000:continue
                key=(kind,normalized(line))
                counts[kind]+=1
                if key not in events:events[key]={'category':kind,'file':name,'line':n,'text':line,'normalized_message':key[1],'diagnostic_occurrences':0}
                events[key]['diagnostic_occurrences']+=1
        with zipfile.ZipFile(f) as z:
            archive_entries=z.namelist()
            for name in archive_entries:
                if name.endswith('.tar.gz'):
                    with tarfile.open(fileobj=io.BytesIO(z.read(name))) as tar:
                        for entry in tar:
                            if entry.isfile():scan(name+':'+entry.name,tar.extractfile(entry).read())
                elif not name.endswith('/'):scan(name,z.read(name))
        event_file=directory/'diagnostic_events.json';event_file.write_text(json.dumps(list(events.values()),indent=2)+'\n')
        # Keep all mono/helper/backend/context messages; bound potentially huge GEMM shape diagnostics.
        extracts=[]
        for kind in ['mono','markov','fusion_helpers','backend','phase_context','aiter_diagnostics']:
            subset=[e for e in events.values() if e['category']==kind and len(e['text'])<1500]
            extracts.extend(subset[:8])
        record={'version':version,'point_id':point['id'],'job_id':point['github_job_id'],'job_url':point['job_url'],'config':point['config'],'artifact_id':artifact['id'],'artifact_name':artifact['name'],'artifact_url':artifact['archive_download_url'],'archive_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'archive_bytes':f.stat().st_size,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'exact_job_link_verified':True,'job_artifact_link_evidence':links,'archive_entries':archive_entries,'files':entries,'diagnostic_line_counts':dict(counts),'unique_diagnostic_messages':len(events),'diagnostic_events_sha256':hashlib.sha256(event_file.read_bytes()).hexdigest(),'extracts':extracts,'evidence_limit':'diagnostic messages can include warmup/graph capture; occurrence counts are log lines, not GPU launch counts or timings'}
        (directory/'endpoint_record.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f"V{version:02d} TP{point['config']['decode_tp']} c{point['config']['conc']}: {len(events)} unique diagnostics",flush=True)
        return record
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        rows=list(pool.map(get,selected))
    (a.knowledge/'evidence/recovered_endpoint_logs.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'endpoint_artifacts':len(rows),'archive_bytes':sum(r['archive_bytes'] for r in rows),'exact_job_links':sum(r['exact_job_link_verified'] for r in rows)}))


if __name__=='__main__':main()
