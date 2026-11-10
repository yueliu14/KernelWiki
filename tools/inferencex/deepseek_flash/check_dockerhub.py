#!/usr/bin/env python3
"""Verify tag and known historical digest availability directly with Docker Hub/Registry."""
import concurrent.futures
import hashlib
import json
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cache',type=Path,required=True)
base=parser.parse_args().cache
out=base/'dockerhub_verification';out.mkdir(exist_ok=True)
rows=json.loads((base/'images/receipts.json').read_text())
repo='vllm/vllm-openai-rocm'
headers={'User-Agent':'KernelWiki-evidence-check/1.0'}
auth_url='https://auth.docker.io/token?'+urllib.parse.urlencode({'service':'registry.docker.io','scope':f'repository:{repo}:pull'})
with urllib.request.urlopen(urllib.request.Request(auth_url,headers=headers),timeout=30) as r:
 token=json.load(r)['token']

def fetch(url,filename,registry=False):
 h=dict(headers)
 if registry:
  h['Authorization']='Bearer '+token
  h['Accept']=','.join(['application/vnd.oci.image.index.v1+json','application/vnd.docker.distribution.manifest.list.v2+json','application/vnd.oci.image.manifest.v1+json','application/vnd.docker.distribution.manifest.v2+json'])
 started=datetime.now(timezone.utc).isoformat()
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=40) as r:
   status=r.status;data=r.read();rh=dict(r.headers)
 except urllib.error.HTTPError as e:
  status=e.code;data=e.read();rh=dict(e.headers)
 except Exception as e:
  return {'url':url,'retrieved_at_utc':started,'transport_error':str(e)}
 path=out/filename;path.write_bytes(data)
 record={'url':url,'retrieved_at_utc':started,'http_status':status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'file':filename,'response_headers':{k:v for k,v in rh.items() if k.lower() in ['content-type','docker-content-digest','docker-distribution-api-version','date','ratelimit-remaining']}}
 try:
  parsed=json.loads(data)
 except Exception:parsed={}
 record['summary']={k:parsed[k] for k in ['message','detail','errors','name','count','next','last_updated','tag_status','digest','mediaType'] if k in parsed}
 if 'results' in parsed:record['summary']['matching_tags']=[{'name':x['name'],'last_updated':x.get('last_updated'),'digest':x.get('digest')} for x in parsed['results']]
 path.with_suffix(path.suffix+'.meta.json').write_text(json.dumps(record,indent=2)+'\n')
 return record

def check(row):
 version=row['version'];tag=row['image_name'].split(':',1)[1]
 checks={
  'hub_tag':fetch(f'https://hub.docker.com/v2/repositories/{repo}/tags/{tag}/',f'v{version:02d}.hub_tag.json'),
  'registry_manifest':fetch(f'https://registry-1.docker.io/v2/{repo}/manifests/{tag}',f'v{version:02d}.registry_manifest.json',True)}
 for digest in row.get('historical_api_digests',[]):
  checks['registry_historical_digest']=fetch(f'https://registry-1.docker.io/v2/{repo}/manifests/{digest}',f'v{version:02d}.registry_historical_digest.json',True)
 if row['status']=='failed':
  commit=tag.rsplit('-',1)[-1]
  checks['hub_commit_search']=fetch(f'https://hub.docker.com/v2/repositories/{repo}/tags/?page_size=100&name={commit}',f'v{version:02d}.hub_commit_search.json')
  checks['hub_short_commit_search']=fetch(f'https://hub.docker.com/v2/repositories/{repo}/tags/?page_size=100&name={commit[:8]}',f'v{version:02d}.hub_short_commit_search.json')
 result={'version':version,'image_name':row['image_name'],'previous_pull_status':row['status'],'checks':checks}
 print(json.dumps({'version':version,**{k:{'status':v.get('http_status'),'summary':v.get('summary'),'error':v.get('transport_error')} for k,v in checks.items()}}),flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 results=list(pool.map(check,rows))
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
