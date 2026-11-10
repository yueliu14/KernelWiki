"""Regression checks for framework discovery, CI joins and dated source evidence."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'tools/inferencex/skills/inferencex-kernel-evolution/scripts'
K = ROOT / 'knowledge/top_model/deepseek_v41_flash'


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collector = module('collect_history')
builder = module('build_pair_history')
read = lambda p: json.loads((K / p).read_text())


def row(id, framework, date, hardware='mi355x'):
    return dict(id=id, framework=framework, date=date, hardware=hardware)


def test_framework_discovery_uses_hardware_date_coverage_and_explicit_choice():
    data = [row(1, 'sglang', '2026-10-22'), row(2, 'vllm', '2026-11-09'),
            row(3, 'atom', '2026-11-10'), row(4, 'sglang', '2026-11-11', 'gb300')]
    chosen, detail = collector.discover_framework(data, data, 'mi355x', 'auto')
    assert chosen == 'vllm'
    assert len(detail['available_frameworks']) == 3
    assert collector.discover_framework(data, data, 'mi355x', 'sglang')[0] == 'sglang'
    data[0]['date'] = '2026-11-09'
    data.append(row(5, 'sglang', '2026-10-20'))
    assert collector.discover_framework(data, data, 'mi355x', 'auto')[0] == 'sglang'


@pytest.mark.parametrize('framework,tag,tp,ep', [('vllm', 'vllm', 4, 1), ('sglang', 'sgl', 8, 8)])
def test_job_aliases_ep1_canary_and_attempt_selection(framework, tag, tp, ep):
    point = {'config': dict(framework=framework, hardware='mi355x', decode_tp=tp, decode_ep=ep, conc=2, offload_mode='off'), 'github_run_attempt': 2}
    name = f'canary / model mi355x {tag} TP{tp}' + (f'/EP{ep}' if ep != 1 else '') + ' mtp c2'
    jobs = [dict(id=i, name=name, run_attempt=i, conclusion='success') for i in [1, 2, 3]]
    jobs += [dict(id=4, name=name.replace('c2', 'c4'), run_attempt=2, conclusion='success'), dict(id=5, name=name, run_attempt=2, conclusion='failure')]
    assert [j['id'] for j in collector.matching_jobs(point, jobs)] == [2]


def test_all_raw_metric_values_and_chip_counts():
    raw = {p['id']: p for p in read('data/raw_selected_history.json')}
    points = read('data/points.json')
    assert len(points) == len(raw) == 154
    assert len({p['github_job_id'] for p in points}) == 154
    for p in points:
        r = raw[p['id']]
        assert p['config']['framework'] == 'vllm'
        assert p['config']['decode_ep'] == 1
        assert p['physical_chips'] == p['config']['decode_tp']
        assert p['p90_interactivity_tok_s_user'] == 1 / r['metrics'][p['interactivity_source_field']]
        assert p['output_tok_s_chip'] == r['metrics']['output_tput_per_gpu']
        if r['metrics'].get('output_tput_tps') is not None:
            assert math.isclose(p['output_tok_s_chip'], r['metrics']['output_tput_tps'] / p['physical_chips'], rel_tol=1e-5)


def test_pair_coverage_identity_gap_and_source_provenance():
    points = {p['id']: p for p in read('data/points.json')}
    pairs = read('data/pair_records.json')
    inventory = {x['image_name']: {f['path']: f for f in x['relevant_files']} for x in read('evidence/image_source_inventory.json')}
    assert len(pairs) == 110
    assert len({p['id'] for p in pairs}) == 110
    assert sum(bool(p['skipped_images']) for p in pairs) == 9
    assert len({json.dumps(p['config'], sort_keys=True) for p in points.values()}) == 17
    for p in pairs:
        b, a = p['before'], p['after']
        assert points[b['id']]['config'] == points[a['id']]['config'] == p['config']
        assert p['throughput_change_pct'] == 100 * (a['output_tok_s_chip']/b['output_tok_s_chip']-1)
        assert p['interactivity_change_pct'] == 100 * (a['p90_interactivity_tok_s_user']/b['p90_interactivity_tok_s_user']-1)
        assert p['top_kernels'] is None and p['causal_attribution'] == 'not_established'
        for kernel in p['kernel_changes']:
            for ref in kernel['sources']:
                for side, endpoint in [('before', b), ('after', a)]:
                    if ref[side+'_path']:
                        f = inventory[endpoint['image_name']][ref[side+'_path']]
                        assert f['sha256'] == ref[side+'_sha256']
                        assert f['source_kind'] == ref[side+'_source_kind']
            for evidence in kernel['evidence']:
                assert (K / evidence).is_file()
        for side in ['before', 'after']:
            src = p['source_provenance'][side]
            if src['source_kind'] == 'upstream_commit':
                assert src['digest'] is None
                assert len(src['source_revision']) == 40


def test_latest_tp8_excludes_tp2_tuned_schedule():
    pairs = read('data/pair_records.json')
    tp8 = [p for p in pairs if p['config']['decode_tp'] == 8]
    assert len(tp8) == 1 and tp8[0]['config']['conc'] == 1
    assert all('TP2 ratio-2' not in k['pipeline'] for k in tp8[0]['kernel_changes'])
    assert any('TP2/ratio2-only' in text for text in tp8[0]['excluded_attributions'])
    assert all('synthetic_acceptance_length' in p[side]['launch']['launch_command'] for p in pairs for side in ['before', 'after'])


def test_server_artifact_exact_endpoint_evidence():
    artifacts = read('evidence/server_log_extracts.json')
    assert sum(a['exact_job_link_verified'] for a in artifacts) == 11
    assert sum(not a['exact_job_link_verified'] for a in artifacts) == 2
    for a in artifacts:
        if a['exact_job_link_verified']:
            assert any('/artifacts/'+str(a['artifact_id']) in e['text'] for e in a['job_artifact_link_evidence'])


def test_legacy_escaped_json_parses_without_shell_execution():
    launches = read('evidence/ci_launch_configs.json')
    for launch in launches:
        parsed = builder.flags(launch['launch_command'])
        spec = json.loads(parsed['speculative-config'])
        assert spec['num_speculative_tokens'] == 5
        assert spec['synthetic_acceptance_length'] == 3.51
