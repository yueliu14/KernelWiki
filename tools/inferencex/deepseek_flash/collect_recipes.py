#!/usr/bin/env python3
"""Archive pinned InferenceX recipes/workflows used by the selected CI jobs."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--knowledge', type=Path, required=True)
a = p.parse_args()
launches = json.loads((a.knowledge / 'evidence/ci_launch_configs.json').read_text())
runs = json.loads((a.knowledge / 'data/ci_manifest.json').read_text())
refs = {}
for run in runs:
    refs.setdefault(run['head_sha'], set()).add(run['run_id'])
for launch in launches:
    for sha in launch['effective_checkouts']:
        refs.setdefault(sha, set()).add(launch['run_id'])

def fetch(sha):
    directory = a.cache / 'recipes' / sha
    directory.mkdir(parents=True, exist_ok=True)
    tree = directory / 'tree.json'
    if not tree.exists():
        tree.write_bytes(subprocess.check_output(['gh', 'api', f'repos/SemiAnalysisAI/InferenceX/git/trees/{sha}?recursive=1']))
    paths = [x['path'] for x in json.loads(tree.read_text())['tree'] if x['type'] == 'blob']
    selected = [s for s in paths if (('dsv41flash' in s.lower() or 'deepseek-v4' in s.lower()) and ('mi355' in s and 'vllm' in s))
                or s.startswith('.github/workflows/') and s.endswith(('.yml', '.yaml'))
                or (s.startswith('runners/') or s.startswith('inferencex-e2e/runners/')) and any(x in s for x in ['mi355x', 'runtime_settings.sh', 'slurm_utils.sh', 'hooks/common.sh'])]
    records = []
    for path in selected:
        url = f'https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/{sha}/{path}'
        dest = directory / path
        try:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(urllib.request.urlopen(url, timeout=60).read())
            body = dest.read_bytes()
            excerpt = [{'line': i+1, 'text': line} for i, line in enumerate(body.decode().splitlines())
                       if any(word in line for word in ['VLLM_', 'AITER_', 'pip install', 'git checkout', 'git apply', 'docker ', 'container-', 'uses:', 'benchmarks/', 'setup', 'launch_'])]
            records.append({'revision': sha, 'path': path, 'url': url, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body), 'extracts': excerpt[:80] if not path.startswith('.github/') else [e for e in excerpt if 'uses:' in e['text']][:20]})
        except Exception as error:
            records.append({'revision': sha, 'path': path, 'url': url, 'error': str(error)})
    return {'revision': sha, 'run_ids': sorted(refs[sha]), 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'files': records}

with concurrent.futures.ThreadPoolExecutor(6) as pool:
    rows = list(pool.map(fetch, sorted(refs)))
(a.knowledge / 'evidence/ci_recipe_sources.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps({'revisions': len(rows), 'files': sum(len(x['files']) for x in rows), 'errors': sum('error' in f for x in rows for f in x['files'])}))
