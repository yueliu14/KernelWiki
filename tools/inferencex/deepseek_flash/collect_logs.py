#!/usr/bin/env python3
import concurrent.futures,hashlib,json,re,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
import argparse
p=argparse.ArgumentParser(description='Collect every representative endpoint CI log, including legacy vLLM commands.')
p.add_argument('--cache',type=Path,required=True);p.add_argument('--knowledge',type=Path,required=True)
a=p.parse_args();base=a.cache;root=a.knowledge
raw=base/'raw/ci';raw.mkdir(parents=True,exist_ok=True)
points=json.loads((root/'data/points.json').read_text());images=json.loads((root/'data/images.json').read_text());byid={p['id']:p for p in points}
ids=[pid for im in images for pid in im['representative_ids']]
def capture(point):
 job=point['github_job_id'];path=raw/f'{job}.log';url=f'https://api.github.com/repos/SemiAnalysisAI/InferenceX/actions/jobs/{job}/logs'
 if not path.exists():
  result=subprocess.run(['gh','api',url],capture_output=True)
  if result.returncode:return {'job_id':job,'error':result.stderr.decode().strip()}
  path.write_bytes(result.stdout);path.with_suffix('.log.meta.json').write_text(json.dumps({'url':url,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(result.stdout).hexdigest(),'bytes':len(result.stdout)},indent=2)+'\n')
 lines=path.read_text().splitlines()
 patterns=['[INFO] Command:','[INFO] Env:','git checkout --progress --force','--public-dataset ','Digest: sha256:','Resolved image','MONO_DECODE','--monokernel',' vllm serve ','Using vLLM version','vLLM API server version']
 extracts=[{'line':i+1,'text':line} for i,line in enumerate(lines) if any(k in line for k in patterns)]
 command=[x['text'].split('[INFO] Command: ',1)[1] for x in extracts if '[INFO] Command:' in x['text'] and any(k in x['text'] for k in ['vllm serve','vllm.entrypoints','vllm launch'])]
 if not command:
  command=[re.sub(r'^\S+Z\s+', '', line).strip() for line in lines if re.match(r'^\S+Z\s+vllm serve ',line)]
 env=[x['text'].split('[INFO] Env: ',1)[1] for x in extracts if '[INFO] Env:' in x['text']]
 return {'run_id':point['github_run_id'],'job_id':job,'job_url':point['job_url'],'point_id':point['id'],'image_name':point['image_name'],'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'log_bytes':path.stat().st_size,'launch_command':command[0] if command else None,'server_env':dict(v.split('=',1) for v in shlex.split(env[0]) if '=' in v) if env else {},'effective_checkouts':sorted(set(re.findall(r'git checkout --progress --force ([0-9a-f]{40})','\n'.join(lines)))),'datasets':sorted(set(x.strip("'\"") for x in re.findall(r'--public-dataset ([^\s]+)','\n'.join(lines)))),'launch_format':'info_command' if any('[INFO] Command:' in x['text'] and 'vllm serve' in x['text'] for x in extracts) else 'legacy_plain_command','server_env_status':'resolved_env_line' if env else 'not_reported_as_resolved_env_in_legacy_log','vllm_versions':sorted(set(re.findall(r'\bversion\s+([0-9][\w.+]*\+g[0-9a-f]+)', '\n'.join(lines)))),'extracts':extracts}
(root/'evidence').mkdir(exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(5) as pool:all_logs=list(pool.map(capture,[byid[id] for id in ids]))
(root/'evidence/ci_log_extracts.json').write_text(json.dumps(all_logs,indent=2)+'\n')
(root/'evidence/ci_launch_configs.json').write_text(json.dumps([{k:v for k,v in x.items() if k!='extracts'} for x in all_logs],indent=2)+'\n')
print('ALL_ENDPOINTS',len(all_logs),'missing launches',sum(not x.get('launch_command') for x in all_logs),'errors',sum('error' in x for x in all_logs),flush=True)
