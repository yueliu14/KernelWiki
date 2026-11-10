#!/usr/bin/env python3
"""Verify exported image bytes and compare dispatch/device symbols across Muse versions."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knowledge', type=Path, required=True)
    args = parser.parse_args()
    inventories = json.loads((args.knowledge / 'evidence/image_source_inventory.json').read_text())
    relative = [
        'sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/attention_kv_write.py',
        'sgl-workspace/sglang/python/sglang/srt/rocm_muse_spark/router_pooler.py',
        'sgl-workspace/sglang/python/sglang/kernels/ops/attention/qk_norm_rope_cache.py',
        'sgl-workspace/sglang/python/sglang/kernels/ops/moe/router_pooler_topk.py',
        'opt/sgl-amd-mono/sgl_amd_mono/muse_spark/draft.py',
        'opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/draft_attn_half.py',
        'opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/draft_attn.py',
        'opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/draft_mlp.py',
    ]
    snapshots = []
    contents = []
    for version, inventory in enumerate(inventories, 1):
        archive = Path(inventory['external_archive'])
        assert sha(archive.read_bytes()) == inventory['archive_sha256']
        with tarfile.open(archive) as tar:
            members = {m.name.lstrip('/'): m for m in tar if m.isfile()}
            for f in inventory['relevant_files']:
                assert sha(tar.extractfile(members[f['path'].lstrip('/')]).read()) == f['sha256'], f['path']
            files = []
            texts = {}
            for path in relative:
                if path not in members:
                    continue
                data = tar.extractfile(members[path]).read()
                text = data.decode()
                texts[path] = text
                tree = ast.parse(text)
                symbols = []
                def visit(node, parent=''):
                    for child in ast.iter_child_nodes(node):
                        name = parent
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            name = parent + child.name
                            if not isinstance(child, ast.ClassDef):
                                source = ast.get_source_segment(text, child)
                                symbols.append({'name': name, 'line': child.lineno, 'end_line': child.end_lineno,
                                                'source_sha256': sha(source.encode()),
                                                'ast_sha256': sha(ast.dump(child, include_attributes=False).encode())})
                            name += '.'
                        visit(child, name)
                visit(tree)
                files.append({'path': '/' + path, 'sha256': sha(data), 'bytes': len(data), 'symbols': symbols})
            mono = {n: sha(tar.extractfile(m).read()) for n, m in members.items() if n.startswith('opt/sgl-amd-mono/sgl_amd_mono/')}
        snapshots.append({'version': version, 'image_name': inventory['image_name'], 'digest': inventory['digest'],
                          'verified_existing_source_files': len(inventory['relevant_files']), 'archive_sha256_verified': True,
                          'files': files, 'mono_package_files': mono})
        contents.append(texts)
    transitions = []
    diffs = args.knowledge / 'evidence/verification_diffs'
    diffs.mkdir(exist_ok=True)
    for old, new, before, after in zip(snapshots, snapshots[1:], contents, contents[1:]):
        changes = []
        for path in relative:
            a = next((f for f in old['files'] if f['path'] == '/' + path), None)
            b = next((f for f in new['files'] if f['path'] == '/' + path), None)
            aa = {s['name']: s for s in a['symbols']} if a else {}
            bb = {s['name']: s for s in b['symbols']} if b else {}
            if not a and not b:
                continue
            changes.append({'path': '/' + path, 'before_sha256': a['sha256'] if a else None, 'after_sha256': b['sha256'] if b else None,
                            'symbols': [{'name': n, 'relation': 'identical_ast' if n in aa and n in bb and aa[n]['ast_sha256'] == bb[n]['ast_sha256'] else 'changed_ast' if n in aa and n in bb else 'added' if n in bb else 'removed'} for n in sorted(aa.keys() | bb.keys())]})
        diffname = f"{old['version']:02d}_to_{new['version']:02d}.diff"
        delta = ''.join(''.join(difflib.unified_diff(before.get(path, '').splitlines(keepends=True), after.get(path, '').splitlines(keepends=True), fromfile=f"v{old['version']}/{path}", tofile=f"v{new['version']}/{path}", n=0)) for path in relative if not path.startswith('opt/'))
        (diffs / diffname).write_text(delta)
        transitions.append({'from_version': old['version'], 'to_version': new['version'], 'files': changes,
                            'mono_package_identical': bool(old['mono_package_files']) and old['mono_package_files'] == new['mono_package_files'],
                            'dispatch_device_diff': 'evidence/verification_diffs/' + diffname})
    result = {'snapshots': snapshots, 'transitions': transitions,
              'limits': 'AST identity omits comments/locations but includes the function syntax. Identical function bodies can behave differently because globals, guards, callees, launch grids, flags or shapes change. File bytes and package file sets are recorded separately. Source inspection cannot rank GPU time.'}
    (args.knowledge / 'evidence/verified_source_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verified_files': sum(s['verified_existing_source_files'] for s in snapshots), 'mono_1027_to_1106_identical': transitions[-1]['mono_package_identical'], 'mono_files': len(snapshots[-1]['mono_package_files'])}))


if __name__ == '__main__':
    main()
