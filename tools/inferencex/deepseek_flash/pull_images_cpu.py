#!/usr/bin/env python3
import concurrent.futures,json,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
import argparse
p=argparse.ArgumentParser(description='Pull and export sources on a previously allocated CPU-only Docker node.')
p.add_argument('--cache',type=Path,required=True);p.add_argument('--knowledge',type=Path,required=True)
a=p.parse_args();base=a.cache.resolve();knowledge=a.knowledge.resolve();wiki=Path(__file__).resolve().parents[3]
images=json.loads((knowledge/'data/images.json').read_text());reviewed=json.loads((knowledge/'evidence/ci_launch_configs.json').read_text());assert all(any(r['image_name']==im['image_name'] and r.get('launch_command') for r in reviewed) for im in images)
script=wiki/'tools/inferencex/skills/inferencex-kernel-evolution/scripts/capture_image_sources.py'
def capture(item):
 i,entry=item;ref=entry['image_name'];historic=[d for d in entry['digests'] if d]
 if len(historic)==1:ref+='@'+historic[0]
 out=base/'images'/f'v{i+1:02d}';out.mkdir(parents=True,exist_ok=True)
 result={'version':i+1,'image_name':entry['image_name'],'requested_reference':ref,'historical_api_digests':historic,'historical_digest_status':'api_pinned' if historic else 'not_pinned_in_api','retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
 print('PULL',result['version'],ref,flush=True)
 try:
  with (out/'docker_pull.log').open('w') as log:subprocess.run(['docker','pull',ref],stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
  data=subprocess.check_output(['docker','image','inspect',ref]);(out/'image_inspect.json').write_bytes(data);inspection=json.loads(data)[0];result.update({'repo_digests':inspection['RepoDigests'],'image_id':inspection['Id'],'created':inspection['Created']})
  run=subprocess.run(['docker','run','--rm','--network','none','--env','HIP_VISIBLE_DEVICES=','--env','ROCR_VISIBLE_DEVICES=','--mount',f'type=bind,src={script},dst=/capture.py,readonly','--mount',f'type=bind,src={out},dst=/capture-output','--entrypoint','python3',ref,'/capture.py','/capture-output'],capture_output=True,text=True,timeout=900)
  (out/'capture.log').write_text(run.stdout+run.stderr);run.check_returncode();result['status']='captured'
  print('CAPTURED',result['version'],flush=True)
 except Exception as e:
  result['status']='failed';result['error']=str(e);print('FAILED',result['version'],str(e),flush=True)
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result
with concurrent.futures.ThreadPoolExecutor(2) as pool:results=list(pool.map(capture,enumerate(images)))
(base/'images/receipts.json').write_text(json.dumps(results,indent=2)+'\n')
