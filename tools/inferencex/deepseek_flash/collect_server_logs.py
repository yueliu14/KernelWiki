#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,zipfile,tarfile,io,re,hashlib,concurrent.futures
import argparse
p=argparse.ArgumentParser(description='Collect selected server artifacts and verify exact endpoint upload links.')
p.add_argument('--cache',type=Path,required=True);p.add_argument('--knowledge',type=Path,required=True)
a=p.parse_args();base=a.cache;root=a.knowledge;out=base/'raw/server_logs';out.mkdir(parents=True,exist_ok=True)
points={x['id']:x for x in json.loads((root/'data/points.json').read_text())};runs={x['run_id']:x for x in json.loads((root/'data/ci_manifest.json').read_text())};selected=[]
for version,image in enumerate(json.loads((root/'data/images.json').read_text()),1):
 p=min((points[x] for x in image['representative_ids']),key=lambda p:(p['config']['conc'],abs(p['config']['decode_tp']-4)))
 for a in runs[p['github_run_id']]['artifacts']:
  if a['name'].startswith('server_logs_') and f"_tp{p['config']['decode_tp']}_conc{p['config']['conc']}_" in a['name']:selected.append((version,p,a))
def get(item):
 version,point,a=item;f=out/f"{a['id']}.zip"
 if not f.exists():f.write_bytes(subprocess.check_output(['gh','api',a['archive_download_url']]))
 extracts=[];names=[]
 def scan(name,data):
  for i,line in enumerate(data.decode(errors='replace').splitlines()):
   if any(s in line.lower() for s in ['version','mono','aiter','deepseek_v4','rocm','sparse_mla']) and not any(s in line.lower() for s in ['cache hit','prompt throughput','generation throughput','running:','waiting:']):
    if len(line)<2000:extracts.append({'file':name,'line':i+1,'text':line})
 with zipfile.ZipFile(f) as z:
  for name in z.namelist():
   names.append(name)
   if name.endswith('.tar.gz'):
    with tarfile.open(fileobj=io.BytesIO(z.read(name))) as tar:
     for entry in tar:
      names.append(name+':'+entry.name)
      if entry.isfile() and entry.name.endswith(('.log','.out','.err','.txt')):scan(entry.name,tar.extractfile(entry).read())
   elif name.endswith(('.log','.txt','.out')):scan(name,z.read(name))
 unique=[];seen=set()
 for e in extracts:
  if not any(w in e['text'].lower() for w in ['version','mono decode','monokernel','attention backend']):continue
  key=re.sub(r'pid=\d+', 'pid=*', re.sub(r'^.*?\] ', '', e['text']))
  if key not in seen:unique.append(e);seen.add(key)
 lines=(base/'raw/ci'/f"{point['github_job_id']}.log").read_text().splitlines()
 hits=[{'line':i+1,'text':l} for i,l in enumerate(lines) if f"/artifacts/{a['id']}" in l]
 return {'exact_job_link_verified':bool(hits),'job_artifact_link_evidence':hits,'version':version,'point_id':point['id'],'job_id':point['github_job_id'],'artifact_id':a['id'],'artifact_name':a['name'],'artifact_url':a['archive_download_url'],'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'entries':names,'extracts':unique[:100],'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(get,selected))
(root/'evidence/server_log_extracts.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 versions=sorted(set(re.findall(r'\bversion\s+([0-9][A-Za-z0-9_.+]+)','\n'.join(e['text'] for e in r['extracts']))));print(r['version'],r['artifact_id'],versions,len(r['extracts']))
