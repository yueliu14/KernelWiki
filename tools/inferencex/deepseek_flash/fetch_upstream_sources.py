#!/usr/bin/env python3
from pathlib import Path
import json,urllib.request,concurrent.futures,hashlib,time
from datetime import datetime,timezone
import argparse
p=argparse.ArgumentParser(description='Fetch scoped sources at exact commits for unavailable image tags; requires cached tree and selection manifests.')
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--knowledge',type=Path,required=True)
a=p.parse_args();base=a.cache/'upstream'
import re,subprocess
images=json.loads((a.knowledge/'data/images.json').read_text())
for version,image in enumerate(images,1):
 receipt=a.cache/'images'/f'v{version:02d}'/'receipt.json'
 if not receipt.exists() or json.loads(receipt.read_text())['status']=='captured':continue
 match=re.search(r'([0-9a-f]{40})$',image['image_name'])
 if not match:raise ValueError('Unavailable tag does not identify a full upstream commit')
 sha=match[1];directory=base/f'v{version:02d}';directory.mkdir(parents=True,exist_ok=True)
 if not (directory/'tree.json').exists():
  (directory/'tree.json').write_bytes(subprocess.check_output(['gh','api',f'repos/vllm-project/vllm/git/trees/{sha}?recursive=1']))
 if not (directory/'selection.json').exists():
  (directory/'selection.json').write_text(json.dumps({'commit':sha,'image_name':image['image_name'],'source_kind':'upstream_commit'},indent=2)+'\n')
def selected(path):
 model=('/deepseek_v4_1/' in path or '/deepseek_v41/' in path) and not any(x in path for x in ['/nvidia/','/cpu/','/xpu/']) and path.startswith('vllm/models/')
 shared=path.startswith(('vllm/models/deepseek_v4/amd/','vllm/models/deepseek_v4/common/ops/'))
 support=any(x in path for x in ['rocm_aiter_fa.py','rocm_aiter_mla.py','rocm_aiter_fused_moe.py','rocm_aiter_ops.py','rocm_aiter_all_reduce.py'])
 support = support or (path.startswith('vllm/') and ('aiter' in path.lower() or path in ['vllm/envs.py', 'vllm/config/kernel.py', 'vllm/model_executor/layers/fused_moe/oracle/mxfp4.py']))
 return model or shared or support or path.startswith(('vllm/model_executor/kernels/mhc/', 'vllm/model_executor/layers/mhc.py'))
jobs=[]
for d in sorted(base.glob('v*')):
 meta=json.loads((d/'selection.json').read_text())
 for path in [x['path'] for x in json.loads((d/'tree.json').read_text())['tree'] if x['type']=='blob' and x['path'].endswith('.py')]:
  if selected(path):jobs.append((d,meta['commit'],path))
def get(item):
 d,sha,path=item;f=d/'sources'/path;url=f'https://raw.githubusercontent.com/vllm-project/vllm/{sha}/{path}'
 try:
  if not f.exists():
   data=urllib.request.urlopen(url,timeout=60).read();f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
  data=f.read_bytes();return {'version':d.name,'commit':sha,'path':path,'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'source_kind':'upstream_commit; historical container unavailable'}
 except Exception as e:return {'version':d.name,'path':path,'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(8) as pool:results=list(pool.map(get,jobs))
for d in sorted(base.glob('v*')):
 rows=[r for r in results if r['version']==d.name];(d/'source_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');print(d.name,len(rows),'errors',sum('error' in r for r in rows))
