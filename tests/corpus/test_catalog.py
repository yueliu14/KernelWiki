"""Operator ownership, grounded references and reproducible source extraction."""
from copy import deepcopy
from pathlib import Path
import re
import subprocess

import pytest

import build
import extract
import _extract_impl_facts as core
from validate import Catalog, qualified


def test_every_operator_has_owned_knowledge(catalog):
    directories = {p.name for p in (catalog.root / 'knowledge/operator').iterdir() if p.is_dir()}
    assert directories == set(catalog.operators)
    for op, bundle in catalog.operators.items():
        assert bundle['path'] == catalog.root / 'knowledge/operator' / op / 'corpus'
        assert bundle['scope']['sources']
        assert bundle['evidence']['source_evidence']
        assert bundle['decisions']['cards']
    assert len(catalog.cards) == sum(len(b['decisions']['cards']) for b in catalog.operators.values())
    assert len(catalog.sources) == sum(len(b['evidence']['source_evidence']) for b in catalog.operators.values())


def test_snapshot_qualified_references_resolve_to_full_commit(catalog):
    for ref, row in catalog.sources.items():
        assert ref == qualified(row)
        assert catalog.resolve_source(ref) is row
        assert '/blob/' + catalog.snapshot(row['snapshot'])['commit'] + '/' in catalog.source_url(row)
    with pytest.raises(ValueError, match='unresolved source reference'):
        catalog.resolve_source(next(iter(catalog.sources.values()))['evidence_id'])


@pytest.mark.parametrize('op', list(Catalog().operators))
def test_generated_pages_and_relative_links(op, catalog):
    for path, text in build.pages(catalog, [op], include_index=False).items():
        assert path.read_text() == text, f'regenerate {path}'
        for target in re.findall(r'\]\(([^\s)]+)\)', text):
            if target.startswith(('https://', 'http://')):
                continue
            relative, _, fragment = target.partition('#')
            destination = path.parent / relative if relative else path
            assert destination.exists(), (path, target)
            if fragment:
                assert f'<a id="{fragment}"></a>' in destination.read_text(), (path, target)


def test_root_corpus_is_a_thin_catalog(catalog):
    directory = catalog.root / 'corpus'
    assert {p.name for p in directory.iterdir() if p.is_file()} == {'README.md', 'catalog.yaml'}
    path, text = build.render_index(catalog)
    assert path.read_text() == text


def test_shared_cards_are_links_to_one_canonical_body(catalog):
    for op, bundle in catalog.operators.items():
        page, text = build.render_decisions(catalog, op)
        for ref in bundle['decisions'].get('shared_cards', []):
            assert catalog.card_owners[ref] != op
            assert op in catalog.cards[ref]['match']['operator_families']
            assert build.card_link(catalog, ref, page) in text
            assert f'<a id="{ref}"></a>' not in text


@pytest.mark.parametrize('mutation,error', [
    ('quote', 'source ID'), ('speedup', 'unexpected source fields'),
    ('snapshot', 'unknown snapshot'), ('source_ref', 'source evidence ID not present'),
    ('shared_card', 'invalid shared card'), ('dependency', 'unknown card in dependency'),
    ('scope', 'source outside declared scope'), ('language', 'unknown source language'),
    ('axis', 'unknown performance axis'), ('trait', 'trait'),
    ('measurement', 'measured evidence'), ('path', 'invalid upstream path'),
    ('commit', 'full commit'), ('identity', 'mismatched operator identity'),
])
def test_invalid_knowledge_is_rejected(catalog, mutation, error):
    broken = deepcopy(catalog)
    bundle = broken.operators['attention_indexer']
    card = bundle['decisions']['cards'][0]
    row = bundle['evidence']['source_evidence'][0]
    if mutation == 'quote':
        row['evidence'][0] += ' corrupted'
    elif mutation == 'speedup':
        row['speedup'] = 1.5
    elif mutation == 'snapshot':
        row['snapshot'] = 'unknown_snapshot'
    elif mutation == 'source_ref':
        card['source_evidence'] = ['aiter_b3ad6c46:src_0000000000000000']
    elif mutation == 'shared_card':
        bundle['decisions']['shared_cards'] = ['nonexistent-card']
    elif mutation == 'dependency':
        bundle['axes']['dependencies'][0]['decision_refs'] = ['nonexistent-card']
    elif mutation == 'scope':
        bundle['scope']['sources'] = []
    elif mutation == 'language':
        row['language'] = 'unknown_language'
        row['evidence_id'] = core.source_evidence_id(row)
    elif mutation == 'axis':
        card['axes'].append('unknown_axis')
    elif mutation == 'trait':
        card['match'].setdefault('requires', []).append('unknown_trait')
    elif mutation == 'measurement':
        card['measurement_evidence'] = ['unverified_speedup']
    elif mutation == 'path':
        row['file'] = '../outside.py'
        row['evidence_id'] = core.source_evidence_id(row)
    elif mutation == 'commit':
        broken.snapshots[row['snapshot']]['commit'] = 'main'
    elif mutation == 'identity':
        bundle['scope']['operator'] = 'dense_gemm'
    with pytest.raises(ValueError, match=error):
        broken.validate()


def test_source_file_ownership_cannot_overlap(catalog):
    broken = deepcopy(catalog)
    first, second = list(broken.operators.values())[:2]
    second['scope']['sources'].append(deepcopy(first['scope']['sources'][0]))
    with pytest.raises(ValueError, match='duplicate source ownership'):
        broken.validate()


def test_shipped_groups_preserve_provenance_without_timings(catalog):
    for row in catalog.seeds.values():
        assert row['source_files']
        assert not set(row) & {'us', 'ms', 'tflops', 'speedup', 'bw', 'errRatio'}
    for op, bundle in catalog.operators.items():
        if bundle['tuning']['records']:
            _, text = build.render_seeds(catalog, op)
            assert 'not measured winners' in text


def test_mixed_file_keeps_nested_kernel_languages(tmp_path):
    source = tmp_path / 'mixed.py'
    source.write_text(
        'def host():\n'
        '    @triton.jit\n'
        '    def helper(x):\n'
        '        BLOCK_M = 16\n'
        '        return x\n'
        '    @flyc.kernel\n'
        '    def kernel(x):\n'
        '        BLOCK_N = 32\n'
        '        return x\n'
    )
    rows, _, _ = core.scan_file(str(source), source.name, 'flydsl', include_symbols=True)
    declarations = {r['captured'][0]: r for r in rows if r['category'] == 'symbol_declaration'}
    assert declarations['helper']['language'] == 'triton'
    assert declarations['kernel']['language'] == 'flydsl'
    assert declarations['host']['language'] == 'flydsl'
    assert {r['language'] for r in rows if r['category'] == 'tile_shape'} == {'triton', 'flydsl'}


def test_interface_anchors_ignore_comments_and_keep_native_line():
    rows = extract.anchor_records('// cp_rank\nint cp_rank;\n', 'kernel.cu', 'asm', ['cp_rank'])
    assert len(rows) == 1
    assert rows[0]['line'] == 2
    assert rows[0]['captured'] == ['cp_rank']
    with pytest.raises(ValueError, match='invalid interface anchor'):
        extract.anchor_records('int cp_rank;', 'kernel.cu', 'asm', ['cp_*'])


@pytest.fixture
def pinned_catalog(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    (tmp_path / 'kernel.py').write_text('def forward(x):\n    return x\n')
    git('add', 'kernel.py')
    git('-c', 'user.name=Corpus test', '-c', 'user.email=corpus@example.invalid', 'commit', '-qm', 'source')
    catalog = Catalog.__new__(Catalog)
    catalog.snapshots = {'test': {'commit': git('rev-parse', 'HEAD')}}
    catalog.operators = {'test': {
        'scope': {'sources': [{'snapshot': 'test', 'file': 'kernel.py', 'language': 'triton',
                               'include_symbols': True}]},
        'decisions': {'cards': []}, 'tuning': {'records': []},
    }}
    payload = extract.extract_operator(catalog, 'test', tmp_path)
    catalog.sources = {qualified(r): r for r in payload['source_evidence']}
    catalog.source_owners = {ref: 'test' for ref in catalog.sources}
    return catalog, payload


def test_extraction_and_audit_read_pinned_objects_not_working_files(tmp_path, pinned_catalog):
    catalog, payload = pinned_catalog
    (tmp_path / 'kernel.py').write_text('# different working tree\n')
    assert extract.equivalent(payload, extract.extract_operator(catalog, 'test', tmp_path))
    assert catalog.verify_source(tmp_path) == len(catalog.sources) > 0
    next(iter(catalog.sources.values()))['evidence'][0] = '1: def invented(x):'
    with pytest.raises(ValueError, match='quote does not resolve'):
        catalog.verify_source(tmp_path)


def test_missing_git_object_aborts_extraction(tmp_path, pinned_catalog):
    catalog, _ = pinned_catalog
    catalog.snapshots['test']['commit'] = '0' * 40
    with pytest.raises(subprocess.CalledProcessError):
        extract.extract_operator(catalog, 'test', tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        extract.extract_operator(catalog, 'test', tmp_path / 'not-a-repository')
