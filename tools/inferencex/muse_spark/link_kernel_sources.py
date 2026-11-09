#!/usr/bin/env python3
"""Publish byte-exact selected container source files and build a browsable implementation index."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile


PREFERRED = {
    'layer.py': ['build_layer', 'build_k1', 'build_kattn'],
    'draft.py': ['DraftRunner.launch_attn', 'DraftRunner.launch', '_Draft.forward', '_shard_fc'],
    'draft_attn_half.py': ['run_dpre', 'run_dpost', 'build_dpre', 'build_dpost'],
    'draft_attn.py': ['run_dattn', 'build_dattn'],
    'draft_mlp.py': ['run_dmlp', 'build_dmlp'],
    'qk_norm_rope_cache.py': ['_muse_qk_rope_cache', '_qkv_epilogue', 'qkv_proj_qk_norm_rope_cache'],
    'muse_qk_norm.py': ['_qk_norm_kernel', 'fused_qk_norm'],
    'attention_kv_write.py': ['_attention_shape_ok', '_kv_write_target', 'qkv_proj_norm_rope_kv_write', 'qk_norm_rope_kv_write'],
    'router_pooler_topk.py': ['_joint_epilogue', '_joint_fp8_partial', 'joint_router_pooler_topk', 'joint_router_pooler_topk_fp8'],
    'router_pooler.py': ['can_share_router_pooler', 'route_with_shared_pooler'],
    'mxfp4_compact_a4_moe.py': ['_prepare_a4', '_scatter_a4', '_expert_a4', 'compact_a4_experts'],
    'mxfp4_moe.py': ['run_compact_a4', '_compact_a4_operands', 'run_a4w4'],
    'muse_moe_gemm.py': ['_launcher', 'launch_muse_moe_gemm'],
    'muse_moe_finalize.py': ['build_finalize', 'build_splitk_silu'],
    'finalize_allreduce.py': ['MuseFinalizeAllreduce', 'run_compact_a4_allreduce', 'ensure_finalize_allreduce'],
    'fused_norm_gate.py': ['_rms_norm_gate_fwd_kernel', 'FusedRMSNormGate.forward_verify_partials'],
    'fused_norm_residual.py': ['_fused_norm_residual_triton_kernel', 'FusedPostNormResidualPreNorm.forward_fp8'],
    'verify_reduce_norm_gate.py': ['_reduce_norm_gate', 'verify_reduce_norm_gate'],
    'muse_extend_gluon.py': ['_muse_extend_kernel', 'muse_extend_attention'],
    'muse_extend_flydsl.py': ['_launcher', 'muse_extend_attention_fly'],
    'muse_verify_flydsl.py': ['_launcher', 'verify_fly_partials', 'gqa_verify_flydsl'],
    'muse_verify_balanced.py': ['_plan_kernel', '_seg_reduce', 'segmented_reduce_norm_gate', 'gqa_verify_flydsl_balanced'],
    'aiter_backend.py': ['_muse_verify_attention', 'AiterAttnBackend.forward_extend', 'AiterAttnBackend.forward_decode'],
    'model.py': ['MuseSpark1xAttention.forward', 'MuseSpark1xMoE.forward'],
    'runner.py': ['MuseMonoRunner.launch', 'MuseMonoRunner.layer_kernel', 'MuseMonoRunner.merged'],
    'runtime.py': ['install_monokernel_forward', '_install_forward'],
}


def symbols(path, text):
    result = []
    if path.endswith('.py'):
        def visit(node, prefix=''):
            for child in ast.iter_child_nodes(node):
                name = prefix
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name += child.name
                    result.append({'name': name, 'line': child.lineno, 'end_line': child.end_lineno})
                    name += '.'
                visit(child, name)
        visit(ast.parse(text))
    else:
        for n, line in enumerate(text.splitlines(), 1):
            if '__global__' in line:
                m = re.search(r'\b(muse_\w+)\s*\(', line)
                if m:
                    result.append({'name': m[1], 'line': n})
    preferred = PREFERRED.get(Path(path).name)
    selected = [s for name in preferred or [] for s in result if s['name'] == name]
    if not selected:
        selected = [s for s in result if '.' not in s['name'] and s['name'].startswith(('run_', 'build_'))][:4]
    if not selected:
        selected = [s for s in result if '.' not in s['name']][:4]
    return result, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knowledge', type=Path, required=True)
    args = parser.parse_args()
    root = args.knowledge
    pairs = json.loads((root / 'data/pair_records.json').read_text())
    inventories = json.loads((root / 'evidence/image_source_inventory.json').read_text())
    needed = defaultdict(set)
    for pair in pairs:
        for kernel in pair['kernel_changes']:
            for source in kernel['sources']:
                for side in ['before', 'after']:
                    if source.get(side + '_path'):
                        needed[pair[side]['image_digest']].add(source[side + '_path'])
    out = root / 'kernel_sources'
    out.mkdir(exist_ok=True)
    snapshots = []
    for version, inventory in enumerate(inventories, 1):
        archive = Path(inventory['external_archive'])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == inventory['archive_sha256']
        with tarfile.open(archive) as tar:
            members = {m.name.lstrip('/'): m for m in tar if m.isfile()}
            selected = set(needed[inventory['digest']])
            # Include the target layer's device stages behind build_layer's function calls.
            base = 'opt/sgl-amd-mono/sgl_amd_mono/muse_spark/gfx950/'
            selected.update('/' + p for p in members if p.startswith(base + 'stages/') and p.endswith('.py'))
            selected.update('/' + base + p for p in ['layer.py', 'build_k2.py', 'build_k3.py', 'device.py', 'config.py', 'layout.py'] if base + p in members)
            files = []
            expected = {f['path']: f['sha256'] for f in inventory['relevant_files']}
            for path in sorted(selected):
                data = tar.extractfile(members[path.lstrip('/')]).read()
                digest = hashlib.sha256(data).hexdigest()
                if path in expected:
                    assert digest == expected[path]
                target = out / 'blobs' / digest[:16] / Path(path).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert target.read_bytes() == data, 'Content-address collision'
                else:
                    target.write_bytes(data)
                definitions, entries = symbols(path, data.decode())
                files.append({'container_path': path, 'sha256': digest, 'bytes': len(data),
                              'snapshot_path': str(target.relative_to(root)), 'symbols': definitions,
                              'entrypoints': entries, 'referenced_by_pair': path in needed[inventory['digest']]})
        snapshots.append({'version': version, 'image_name': inventory['image_name'], 'image_digest': inventory['digest'],
                          'archive_sha256': inventory['archive_sha256'], 'files': files})
    manifest = {'source_kind': 'byte_exact_container_snapshot', 'snapshots': snapshots,
                'policy': 'Only cited files and target monokernel stage implementations are published. Identical file bytes share a content-addressed path. Symbols are navigation anchors, not measured hot-kernel rankings. These links do not assert byte identity with an upstream Git commit.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    by_version = {s['version']: {f['container_path']: f for f in s['files']} for s in snapshots}
    paths = sorted({p for files in by_version.values() for p in files})
    text = '# Muse Spark kernel implementations\n\n'
    text += '[Pair reports](pair_changes.md) · [Runtime recheck](runtime_verification.md) · [Source manifest](kernel_sources/manifest.json)\n\n'
    text += 'Click a version below to open the **complete, byte-exact file exported from that Docker image**, at an implementation entry point. Every pair report also links its Before/After source cells and named functions. The container paths are retained verbatim. Identical files share a snapshot URL; file SHA256 and all function line numbers are in the manifest. These are selected reading snapshots, not an executable package or a GPU-time ranking.\n\n'
    for s in snapshots:
        text += f"- **V{s['version']:02d}**: `{s['image_name']}` — `{s['image_digest']}`.\n"
    text += '\nThe images report the same SGLang HEAD despite changed file bytes. For example, the captured 1027 `layer.py` does not match the known public sgl-amd-mono commit `7a4a552db119d1c3689165a182f151388a636272`. Therefore these implementation links point to verified container snapshots, not an assumed upstream revision. Existing upstream notices remain in the files; see [third-party notices](../../../THIRD_PARTY_NOTICES.md).\n\n'
    text += '| Exact container path | V01 baseline | V02 1020 | V03 1027 | V04 1106 | Implementation entry points |\n| --- | --- | --- | --- | --- | --- |\n'
    for path in paths:
        cells = []
        names = []
        for version in range(1, 5):
            file = by_version[version].get(path)
            if not file:
                cells.append('—')
                continue
            line = file['entrypoints'][0]['line'] if file['entrypoints'] else 1
            cells.append(f"[source]({file['snapshot_path']}#L{line})")
            for entry in file['entrypoints']:
                if entry['name'] not in names:
                    names.append(entry['name'])
        text += '| `' + path + '` | ' + ' | '.join(cells) + ' | ' + ', '.join('`' + n + '`' for n in names) + ' |\n'
    text += '\nA dash means the file is not published for that version in this selected index; it does not assert absence from the image. Target monokernel `layer.py` calls `build_k2.py`, `build_k3.py` and the included `stages/` files. Draft `draft.py` calls the three `draft_*` device files. The full dependency trees remain in the external source archives.\n'
    text += '\nThe example [upstream byte comparison](kernel_sources/upstream_comparison.json) records why the captured layer kernel needs its own snapshot. To regenerate this index, run `python3 tools/inferencex/muse_spark/link_kernel_sources.py --knowledge knowledge/top_model/muse_spark`, then regenerate the pair reports with `build_pair_history.py`.\n'
    (root / 'kernel_implementations.md').write_text(text)
    unique = {f['snapshot_path']: f['bytes'] for s in snapshots for f in s['files']}
    print(json.dumps({'image_path_entries': sum(len(s['files']) for s in snapshots), 'unique_files': len(unique), 'bytes': sum(unique.values())}))


if __name__ == '__main__':
    main()
