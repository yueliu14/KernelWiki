"""Keep runtime diagnostics distinct from GPU traces and verify recovered pair provenance."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
K=ROOT/'knowledge/top_model/deepseek_v41_flash'
read=lambda p:json.loads((K/p).read_text())
spec=importlib.util.spec_from_file_location('recovered',ROOT/'tools/inferencex/deepseek_flash/analyze_recovered_logs.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_phase_parser_handles_legacy_midnight_and_boundary_uncertainty():
    t=analysis.timestamp('(Worker) INFO 09-12 00:01:00 [file.py] warning',2026)
    windows={'profiling_start':{'timestamp':'2026-09-11T23:44:43.164'},'profiling_end':{'timestamp':'2026-09-12T00:45:23.165'},'warmup_start':{'timestamp':'2026-09-11T23:43:22'}}
    assert analysis.phase_at(t,windows)=='e2e_measurement'
    near=analysis.timestamp('2026-09-11T23:44:43.900000Z message',2026)
    assert analysis.phase_at(near,windows)=='phase_boundary_uncertain'
    early=analysis.timestamp('09-11 23:43:40 message',2026)
    assert analysis.phase_at(early,windows)=='benchmark_warmup'
    assert analysis.phase_at(None,windows)=='unknown'


def test_all_recovered_endpoints_have_exact_artifacts_and_phase_windows():
    logs=read('evidence/recovered_endpoint_logs.json');facts=read('evidence/recovered_runtime_facts.json')
    assert len(logs)==len(facts)==31
    assert len({r['artifact_id'] for r in logs})==31
    for row in logs:
        assert row['exact_job_link_verified']
        assert any(f"/artifacts/{row['artifact_id']}" in e['text'] for e in row['job_artifact_link_evidence'])
    for fact in facts:
        assert set(fact['phase_windows'])=={'warmup_start','warmup_end','profiling_start','profiling_end'}
        assert fact['phase_windows']['profiling_start']['timestamp']<fact['phase_windows']['profiling_end']['timestamp']


def test_active_messages_do_not_establish_measurement_phase_coverage():
    facts=read('evidence/recovered_runtime_facts.json')
    active=[e for r in facts for e in r['events'] if e['kind'] in ['mono_dispatch','markov_dispatch']]
    assert len(active)==60
    assert all(e['phase']=='server_setup_or_graph_warmup' for e in active)


def test_large_client_concurrency_has_small_actual_kernel_hit_during_measurement():
    fact=next(r for r in read('evidence/recovered_runtime_facts.json') if r['version']==11 and r['config']['decode_tp']==2 and r['config']['conc']==64)
    hits=[e for e in fact['events'] if e['kind']=='quant_gemm_dispatch' and e['phase']=='e2e_measurement']
    assert len(hits)==1
    shape=hits[0]['shape_and_tile']
    assert (shape['M'],shape['N'],shape['K'])==(4,5120,15360)
    assert hits[0]['timestamp']=='2026-11-07T06:14:55'


def test_tp8_quant_fusion_is_retained_not_new():
    facts=read('evidence/recovered_runtime_facts.json')
    b=next(r for r in facts if r['version']==10)
    a=next(r for r in facts if r['version']==11 and r['config']['decode_tp']==8)
    signatures=lambda r:{json.dumps(e['shape_and_tile'],sort_keys=True) for e in r['events'] if e['kind']=='quant_gemm_dispatch'}
    assert len(signatures(b))==len(signatures(a))==41
    assert signatures(b)==signatures(a)
    inventory={r['version']:{f['path']:f for f in r['files']} for r in read('evidence/recovered_source_inventory.json')}
    path='vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py'
    assert inventory[10][path]['sha256']==inventory[11][path]['sha256']


def test_recovered_subset_keeps_original_chronology_and_enriches_ten_pairs():
    pairs=read('data/recovered_pair_records.json');original=read('data/pair_records.json')
    assert len(pairs)==16 and len(original)==110
    assert len({p['id'] for p in pairs})==16
    assert sum(p['original_pair_id'] is not None for p in pairs)==10
    assert sum(p.get('runtime_log_evidence') is not None for p in original)==10
    for pair in pairs:
        assert pair['before']['config']==pair['after']['config']==pair['config']
        assert not pair['gpu_trace_available']
        if pair['original_pair_id'] is None:
            assert (pair['from_version'],pair['to_version'])==(1,9)
            assert pair['skipped_versions']==list(range(2,9))


def test_unchanged_symbol_is_separate_from_changed_file():
    pair=next(p for p in read('data/recovered_pair_records.json') if p['from_version']==9 and p['to_version']==11)
    symbol=next(s for s in pair['jit_kernel_comparisons'] if s['symbol']=='_block32_tiled_kernel')
    change=next(s for s in symbol['source_comparisons'] if s['path'].endswith('/rocm_block32_gemm.py'))
    assert change['before_source']['sha256']!=change['after_source']['sha256']
    assert change['symbol_body_relation']=='identical_ast'


def test_log_backend_label_does_not_override_source_language():
    events=[e for r in read('evidence/recovered_runtime_facts.json') for e in r['events'] if e.get('symbol')=='_pa_mqa_logits_mxfp4_kernel']
    assert events and all(e['logger_backend']=='Triton' for e in events)
    assert all(any(s['language']=='Gluon' for s in e['source_candidates']) for e in events)
