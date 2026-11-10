#!/usr/bin/env python3
"""Align Muse dispatch diagnostics to historical AgentX phase windows."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def read(p):return json.loads(p.read_text())


def timestamp(text,year):
    match=re.search(r'\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)',text)
    if match:return datetime.fromisoformat(match[1]+'T'+match[2][:15])
    match=re.search(r'\b(\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})',text)
    return datetime.fromisoformat(str(year)+'-'+match[1]+'T'+match[2]) if match else None


def phase_at(t,windows):
    if t is None or 'profiling_start' not in windows:return 'unknown'
    start=datetime.fromisoformat(windows['profiling_start']['timestamp'])
    end=datetime.fromisoformat(windows['profiling_end']['timestamp']) if 'profiling_end' in windows else None
    if abs((t-start).total_seconds())<1:return 'phase_boundary_uncertain'
    if t<start:
        warm=windows.get('warmup_start')
        return 'benchmark_warmup' if warm and t>=datetime.fromisoformat(warm['timestamp']) else 'server_setup_or_graph_warmup'
    if end and t>end:return 'after_measurement'
    return 'e2e_measurement'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--knowledge', type=Path, required=True)
    args = parser.parse_args()
    facts = []
    for row in read(args.knowledge / 'evidence/verified_endpoint_logs.json'):
        windows = {}
        for f in row['files']:
            if not f['name'].endswith('/aiperf.log'):
                continue
            for n, line in enumerate(Path(f['external_file']).read_text(errors='replace').splitlines(), 1):
                for phase in ['warmup', 'profiling']:
                    for action, key in [('started', 'start'), ('complete |', 'end')]:
                        if f'Phase {phase} ({phase}) {action}' in line:
                            t = timestamp(line, 2026)
                            if t:
                                windows[f'{phase}_{key}'] = {'timestamp': t.isoformat(), 'file': f['name'], 'line': n, 'text': line}
        events = []
        for f in row['files']:
            if not re.search(r'_agg_w\d+\.out$', f['name']):
                continue
            raw = Path(f['external_file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == f['sha256']
            for n, line in enumerate(raw.decode(errors='replace').splitlines(), 1):
                if len(line) > 1600:
                    continue
                m = re.search(r'(Muse fast path: .*?), rows=(\d+)', line)
                kind = 'dispatch'
                if m:
                    family, shape = m[1], int(m[2])
                else:
                    m = re.search(r'(Muse joint router/pooler(?: \(FP8 pooler\))? active): layer=(\d+) rows=(\d+)', line)
                    if m:
                        family, shape = m[1], int(m[3])
                    elif re.search(r'monokernel.*(?:built in|attached at stage|context projection)|Muse fused finalize .*enabled|DFLASH fused KV materialization enabled', line):
                        family = re.sub(r'^.*?\] ', '', line)
                        family = re.sub(r'built in .*', 'built (timing/size omitted)', family)
                        family = re.sub(r'\(rank \d+\)', '(rank *)', family)
                        shape = None
                        kind = 'initialization'
                    elif 'JIT compilation during inference:' in line:
                        family = line.split('JIT compilation during inference:', 1)[1].split('. This causes', 1)[0].strip()
                        shape = None
                        kind = 'jit_compile'
                    else:
                        continue
                t = timestamp(line, 2026)
                events.append({'kind': kind, 'family': family, 'rows': shape, 'phase': phase_at(t, windows),
                               'timestamp': t.isoformat() if t else None, 'file': f['name'], 'file_sha256': f['sha256'],
                               'line': n, 'text': line})
        external = args.cache / 'verified_server_logs' / str(row['job_id']) / 'phase_events.json'
        external.write_text(json.dumps(events, indent=2) + '\n')
        groups = defaultdict(list)
        for e in events:
            groups[(e['kind'], e['family'])].append(e)
        summaries = []
        for (kind, family), group in sorted(groups.items()):
            phases = {}
            for phase in sorted({e['phase'] for e in group}):
                subset = [e for e in group if e['phase'] == phase]
                phases[phase] = {'diagnostic_lines': len(subset),
                                 'observed_rows': sorted({e['rows'] for e in subset if e['rows'] is not None}),
                                 'first': subset[0], 'last': subset[-1]}
            summaries.append({'kind': kind, 'family': family, 'phases': phases})
        facts.append({'point_id': row['point_id'], 'version': row['version'], 'job_id': row['job_id'], 'job_url': row['job_url'],
                      'artifact_id': row['artifact_id'], 'config': row['config'], 'phase_windows': windows,
                      'families': summaries, 'event_counts_by_kind_and_phase': dict(Counter(e['kind']+'/'+e['phase'] for e in events)),
                      'external_events': str(external), 'external_events_sha256': hashlib.sha256(external.read_bytes()).hexdigest(),
                      'limits': 'Diagnostic line counts are not invocations. Once-only logs suppress repetitions; absence does not prove non-use. AIPerf Profiling is an E2E request phase, not a GPU trace. Timestamps use the logged clock; events within one second of the phase start are uncertain.'})
    (args.knowledge / 'evidence/verified_runtime_facts.json').write_text(json.dumps(facts, indent=2) + '\n')
    bypoint = {f['point_id']: f for f in facts}
    pairs = []
    for pair in read(args.knowledge / 'data/pair_records.json'):
        endpoints = [bypoint[pair[side]['id']] for side in ['before', 'after']]
        pair_facts = {'id': pair['id'], 'endpoint_points': [e['point_id'] for e in endpoints], 'families': []}
        for family in sorted({f['family'] for e in endpoints for f in e['families']}):
            record = {'family': family}
            for side, endpoint in zip(['before', 'after'], endpoints):
                f = next((f for f in endpoint['families'] if f['family'] == family), None)
                record[side] = None if f is None else {'kind': f['kind'], 'phases': {phase: {'diagnostic_lines': value['diagnostic_lines'], 'observed_rows': value['observed_rows']} for phase, value in f['phases'].items()}}
            pair_facts['families'].append(record)
        pairs.append(pair_facts)
    (args.knowledge / 'evidence/verified_pair_runtime.json').write_text(json.dumps(pairs, indent=2) + '\n')
    print(json.dumps({'endpoints': len(facts), 'windows': sum(len(f['phase_windows']) == 4 for f in facts), 'pairs': len(pairs),
                      'events': dict(sum((Counter(f['event_counts_by_kind_and_phase']) for f in facts), Counter()))}))


if __name__ == '__main__':
    main()
