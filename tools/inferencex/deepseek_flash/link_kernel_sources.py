#!/usr/bin/env python3
"""Link DeepSeek image revisions and cited kernel implementations without conflating source kinds."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile


def read(path):
    return json.loads(path.read_text())


def definitions(text, path, logged):
    if not path.endswith('.py'):
        return [], []
    found = []
    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            name = prefix
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name += child.name
                decorators = [ast.unparse(d) for d in getattr(child, 'decorator_list', [])]
                found.append({'name': name, 'line': child.lineno, 'end_line': child.end_lineno,
                              'device_decorator': any(any(x in d for x in ['triton.jit', 'gluon.jit', 'flyc.kernel', 'flyc.jit', 'flydsl.kernel', 'fx.kernel']) for d in decorators)})
                name += '.'
            visit(child, name)
    visit(ast.parse(text))
    preferred = [s for s in found if s['name'].rsplit('.', 1)[-1] in logged]
    preferred += [s for s in found if s['name'].rsplit('.', 1)[-1] in ['_main_kernel', '_reduce_kernel', '_block32_packed_kernel', 'build_k1', 'build_k2', 'build_layer', 'run_k1', 'run_k2']]
    preferred += [s for s in found if s['device_decorator']]
    preferred += [s for s in found if '.' not in s['name']]
    unique = list({s['name']: s for s in preferred}.values())
    return found, unique[:4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--knowledge', type=Path, required=True)
    args = parser.parse_args()
    root = args.knowledge
    inventories = read(root / 'evidence/image_source_inventory.json')
    recovered = {s['version']: s for s in read(root / 'evidence/recovered_source_inventory.json')}
    wanted = defaultdict(set)
    for pair in read(root / 'data/pair_records.json'):
        for kernel in pair['kernel_changes']:
            for source in kernel['sources']:
                for side in ['before', 'after']:
                    if source.get(side + '_path'):
                        wanted[pair[side]['image_name']].add(source[side + '_path'])
    for edge in read(root / 'evidence/recovered_kernel_reviews.json'):
        for version in [edge['from_version'], edge['to_version']]:
            for pipeline in edge['pipelines']:
                wanted[inventories[version - 1]['image_name']].update(pipeline['paths'])
    for pair in read(root / 'data/recovered_pair_records.json'):
        for comparison in pair['jit_kernel_comparisons']:
            for candidate in comparison['source_comparisons']:
                for side, version in [('before', pair['from_version']), ('after', pair['to_version'])]:
                    if candidate.get(side + '_source'):
                        wanted[inventories[version - 1]['image_name']].add(candidate['path'])
    out = root / 'kernel_sources'
    out.mkdir(exist_ok=True)
    snapshots = []
    for inventory in inventories:
        version = inventory['version']
        files = {f['path']: f for f in inventory['relevant_files']}
        files.update({f['path']: f for f in recovered.get(version, {}).get('files', [])})
        # Retain the shared DeepSeek-V4 implementation as well as model-specific kernels.
        # Newer versions can move device bodies into the shared directory.
        wanted[inventory['image_name']].update(path for path in files
            if '/mono/' in path and path.startswith('vllm/models/deepseek_v4')
            and path.endswith(('/layer.py', '/runner.py', '/stages/seam.py', '/stages/moe.py', '/stages/attn.py')))
        commit = inventory.get('source_revision')
        if not commit:
            match = re.search(r'([0-9a-f]{40})$', inventory['image_name'])
            # Full revision resolved from the GitHub commits API using CI's g79a7108d9.
            if not match:
                assert version == 1 and inventory['image_name'].endswith(':deepseekv41-flash-0909')
            commit = match[1] if match else '79a7108d9aea27ddab99ce1779290d300b17fc23'
        assert any(commit.startswith(v.split('+g')[1]) for v in inventory['ci_reported_versions'])
        container = inventory['source_kind'] == 'container'
        tar = None
        if container:
            archive = Path(inventory['external_archive'])
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == inventory['archive_sha256']
            tar = tarfile.open(archive)
            members = {m.name.lstrip('/'): m for m in tar if m.isfile()}
        entries = []
        try:
            for path in sorted(wanted[inventory['image_name']] & files.keys()):
                source = files[path]
                if container:
                    data = tar.extractfile(members[source['original_path'].lstrip('/')]).read()
                else:
                    data = (args.cache / 'upstream' / f'v{version:02d}' / 'sources' / source['repository_path']).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                assert digest == source['sha256'], (version, path)
                defs, entrypoints = definitions(data.decode(), path, source.get('matched_log_symbols', {}))
                entry = {k: source[k] for k in ['path', 'original_path', 'repository_path', 'source_kind']}
                entry.update(sha256=digest, bytes=len(data), symbols=defs, entrypoints=entrypoints)
                if container:
                    target = out / 'blobs' / digest[:16] / Path(path).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        assert target.read_bytes() == data
                    else:
                        target.write_bytes(data)
                    entry['snapshot_path'] = str(target.relative_to(root))
                    entry['url_kind'] = 'container_snapshot'
                else:
                    entry['source_url'] = f"https://github.com/vllm-project/vllm/blob/{commit}/{source['repository_path']}"
                    entry['url_kind'] = 'upstream_commit_not_recovered_container'
                entries.append(entry)
        finally:
            if tar:
                tar.close()
        snapshots.append({'version': version, 'image_name': inventory['image_name'], 'source_kind': inventory['source_kind'],
                          'vllm_commit': commit, 'vllm_commit_url': f'https://github.com/vllm-project/vllm/commit/{commit}',
                          'commit_basis': 'CI package revision; full SHA from tag/source selection or GitHub resolution of V01 short SHA',
                          'ci_reported_versions': inventory['ci_reported_versions'], 'digest': inventory.get('digest'),
                          'historical_api_digests': inventory['historical_api_digests'],
                          'aiter_package_version': inventory.get('packages', {}).get('amd-aiter'),
                          'aiter_commit': None, 'files': entries})
    (out / 'manifest.json').write_text(json.dumps({'snapshots': snapshots,
        'limits': 'vLLM commit identifies the CI-reported framework revision; it does not identify AITER or prove unmodified container files. Container files are keyed by digest and byte hash. V02-V08 upstream links are not historical container recovery; their AITER implementation is unavailable. JIT name mappings are source candidates, not measured top kernels.'}, indent=2) + '\n')
    text = '# DeepSeek V4.1 Flash commits and kernel locations\n\n'
    text += '[110 pair reports](pair_changes.md) · [16 recovered-container comparisons](recovered_image_history.md) · [Machine-readable source index](kernel_sources/manifest.json)\n\n'
    text += 'This index is for **MI355X / vLLM**. Each version has a CI-corroborated vLLM commit and direct kernel source links. For V01/V09/V10/V11, links open complete selected container files with SHA256-verified bytes. For V02–V08, links open the pinned upstream commit because the historical image could not be recovered. A vLLM commit is not an AITER commit; historical AITER code for V02–V08 remains unavailable. V01 is a current tag snapshot without a historical digest.\n\n'
    text += '| Version | Docker tag | vLLM commit | Kernel source identity |\n| --- | --- | --- | --- |\n'
    for s in snapshots:
        identity = s['digest'] if s['source_kind'] == 'container' else 'Upstream commit only; container unavailable'
        text += f"| [V{s['version']:02d}](#v{s['version']:02d}) | `{s['image_name'].split(':')[-1]}` | [{s['vllm_commit'][:12]}]({s['vllm_commit_url']}) | {identity} |\n"
    for s in snapshots:
        text += f"\n## V{s['version']:02d}\n\nCommit [{s['vllm_commit']}]({s['vllm_commit_url']}); source kind `{s['source_kind']}`. "
        text += f"AITER package `{s['aiter_package_version']}`; AITER Git revision not established.\n\n" if s['aiter_package_version'] else 'Historical AITER package/body identity is unavailable.\n\n'
        text += '| Exact installation or upstream repository path | Complete implementation | Function entry points |\n| --- | --- | --- |\n'
        for f in s['files']:
            url = f.get('snapshot_path') or f['source_url']
            entries = ', '.join(f"[{e['name']}]({url}#L{e['line']})" for e in f['entrypoints'])
            text += f"| `{f['original_path']}` | [source]({url}) | {entries or 'configuration / metadata'} |\n"
    text += '\nOriginal paths are preserved; canonical comparison paths may normalize `deepseek_v4_1` to `deepseek_v41`. Identical container bytes share a source URL. The named functions are navigation entries or JIT source candidates, not GPU-time rankings. Existing source notices remain in the files; see [third-party notices](../../../THIRD_PARTY_NOTICES.md).\n\n'
    text += 'Regenerate with `python3 tools/inferencex/deepseek_flash/link_kernel_sources.py --cache /path/to/cache --knowledge knowledge/top_model/deepseek_v41_flash`, then run `render_recovered.py` and the common `build_pair_history.py`.\n'
    (root / 'kernel_implementations.md').write_text(text)
    unique = {f['snapshot_path']: f['bytes'] for s in snapshots for f in s['files'] if f.get('snapshot_path')}
    print(json.dumps({'versions': len(snapshots), 'file_entries': sum(len(s['files']) for s in snapshots),
                      'container_files': len(unique), 'container_bytes': sum(unique.values())}))


if __name__ == '__main__':
    main()
