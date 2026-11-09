#!/usr/bin/env python3
"""Export reviewed and log-named kernel sources from recovered container archives."""
import argparse
import ast
import difflib
import hashlib
import json
import re
import tarfile
from pathlib import Path


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--knowledge',type=Path,required=True)
    a=p.parse_args();out=a.cache/'recovered_sources';out.mkdir(exist_ok=True)
    logs=read(a.knowledge/'evidence/recovered_endpoint_logs.json')
    wanted=set()
    for endpoint in logs:
        events=read(a.cache/'recovered_server_logs'/str(endpoint['job_id'])/'diagnostic_events.json')
        for e in events:
            match=re.search(r'JIT compilation during inference: ([^ .]+(?:\.<locals>\.[^ .]+)?)',e['text'])
            if match:wanted.add(match[1].split('.')[0])
    snapshots=read(a.knowledge/'evidence/image_source_inventory.json');inventories=[];contents={};lookup={}
    aiters={
        'aiter/ops/mhc.py','aiter/ops/triton/fusions/mhc.py','aiter/ops/triton/_triton_kernels/fusions/mhc.py',
        'aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py','aiter/ops/triton/_triton_kernels/fusions/mhc_fused_post_pre_delayed_rmsnorm.py',
        'aiter/ops/triton/fusions/k_norm_rope_mxfp4_cache.py','aiter/ops/triton/_triton_kernels/fusions/k_norm_rope_mxfp4_cache.py',
        'aiter/ops/triton/rope/q_rope_mxfp4_quant.py','aiter/ops/triton/_triton_kernels/rope/q_rope_mxfp4_quant.py',
        'aiter/ops/triton/attention/sparse_mla.py','aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla.py',
        'aiter/ops/triton/attention/pa_decode_sparse.py','aiter/ops/triton/utils/mhc_config_utils.py',
        'aiter/fused_moe.py','aiter/fused_moe_registry.py'}
    for snap in snapshots:
        if snap['source_kind']!='container':continue
        v=snap['version'];version=f'v{v:02d}';data={};files=[];symbols={}
        with tarfile.open(a.cache/'images'/version/'sources.tar.gz') as archive:
            for entry in archive:
                match=re.search(r'/(?:dist|site)-packages/((?:vllm|aiter)/.*)$','/'+entry.name)
                if not entry.isfile() or not match:continue
                path=match[1]
                if not path.endswith(('.py','.json')):continue
                if '/tests/' in path:continue
                body=archive.extractfile(entry).read();text=body.decode(errors='replace')
                defs={m[1]:(text[:m.start()].count('\n')+1) for m in re.finditer(r'^[ \t]*(?:async )?def ([A-Za-z0-9_]+)\(',text,re.M)}
                found=defs.keys() & wanted
                for name in found:symbols.setdefault(name,[]).append(path)
                model=path.startswith(('vllm/models/deepseek_v4_1/','vllm/models/deepseek_v41/')) and not any(s in path for s in ['/nvidia/','/cpu/','/xpu/'])
                select=model or path in aiters or bool(found) or path.endswith('/jit_monitor.py') or '/kernels/mhc/' in path or path in ['vllm/model_executor/layers/mhc.py','vllm/_aiter_ops.py'] or ('/configs/gfx950/' in path and 'mhc' in path)
                if not select:continue
                canonical=path.replace('deepseek_v4_1','deepseek_v41')
                target=out/version/canonical;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
                data[canonical]=text
                matches={}
                if found:
                    tree=ast.parse(text)
                    for node in ast.walk(tree):
                        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in found:
                            start=min([node.lineno]+[d.lineno for d in node.decorator_list])
                            snippet="\n".join(text.splitlines()[start-1:node.end_lineno])+"\n"
                            language='Gluon' if '@gluon.jit' in snippet else 'Triton' if '@triton.jit' in snippet else 'FlyDSL' if any(t in snippet for t in ['@flyc.', '@fx.', '@flydsl.']) else 'Python wrapper/factory'
                            matches[node.name]={'line':start,'end_line':node.end_lineno,'code_sha256':hashlib.sha256(snippet.encode()).hexdigest(),'ast_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'language':language}
                files.append({'path':canonical,'repository_path':path,'original_path':'/'+entry.name,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'source_kind':'container','matched_log_symbols':matches})
        inventories.append({'version':v,'image_name':snap['image_name'],'digest':snap['digest'],'historical_api_digests':snap['historical_api_digests'],'archive_sha256':snap['archive_sha256'],'packages':snap['packages'],'files':sorted(files,key=lambda f:f['path'])})
        contents[v]=data;lookup[v]=symbols
        print(version,len(files),'files',len(symbols),'log-named symbols',flush=True)
    evidence=a.knowledge/'evidence';(evidence/'recovered_source_inventory.json').write_text(json.dumps(inventories,indent=2)+'\n')
    edges=[];diffdir=evidence/'recovered_source_diffs';diffdir.mkdir(exist_ok=True)
    for old,new in [(1,9),(9,11),(10,11)]:
        x,y=contents[old],contents[new];changes=[];focus=[]
        with (diffdir/f'v{old:02d}_to_v{new:02d}.diff').open('w') as output:
            for path in sorted(x.keys()|y.keys()):
                before,after=x.get(path,''),y.get(path,'')
                if before==after:continue
                changes.append(path)
                # Device and dispatch files central to the reviewed fusion comparisons.
                if path in aiters or '/deepseek_v41/amd/mono/' in path or path.endswith(('amd/model.py','model_executor/layers/mhc.py','rocm_block32_gemm.py','radix_topk_one_block.py')) or path.endswith('/jit_monitor.py') or path.endswith(('amd/mono_decode.py','amd/mhc.py','amd/engram.py','amd/ops/fused_markov.py','amd/ops/swa_decode_meta.py','amd/ops/__init__.py','amd/mono/layer.py')):
                    output.writelines(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=f'V{old:02d}:{path}',tofile=f'V{new:02d}:{path}',n=3))
                    focus.append(path)
        edges.append({'from_version':old,'to_version':new,'changed_paths':changes,'diff_paths':focus,'unchanged_paths':sorted(p for p in x.keys()&y.keys() if x[p]==y[p]),'diff':f'evidence/recovered_source_diffs/v{old:02d}_to_v{new:02d}.diff'})
    (evidence/'recovered_source_changes.json').write_text(json.dumps(edges,indent=2)+'\n')
    (diffdir/'.gitattributes').write_text('*.diff -whitespace\n')


if __name__=='__main__':main()
