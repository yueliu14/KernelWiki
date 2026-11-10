"""Keep both model histories reproducible through the shared pair renderer."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / 'tools/inferencex/skills/inferencex-kernel-evolution/scripts/build_pair_history.py'
INPUTS = [
    'data/points.json',
    'data/images.json',
    'evidence/ci_launch_configs.json',
    'evidence/image_source_inventory.json',
    'evidence/kernel_pair_rules.json',
    'evidence/server_log_extracts.json',
    'evidence/recovered_endpoint_logs.json',
    'evidence/recovered_pair_links.json',
    'evidence/verified_runtime_facts.json',
    'kernel_sources/manifest.json',
]


@pytest.mark.parametrize('model,pair_count', [('muse_spark', 11), ('deepseek_v41_flash', 110)])
def test_shared_renderer_preserves_published_pairs_and_source_links(tmp_path, model, pair_count):
    source = ROOT / 'knowledge/top_model' / model
    target = tmp_path / model
    for relative in INPUTS:
        if (source / relative).exists():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)

    subprocess.run([sys.executable, str(BUILDER), '--knowledge', str(target)],
                   check=True, capture_output=True, text=True)
    actual = json.loads((target / 'data/pair_records.json').read_text())
    expected = json.loads((source / 'data/pair_records.json').read_text())
    assert len(actual) == pair_count
    assert actual == expected
    for pair in expected:
        assert (target / pair['report']).read_text() == (source / pair['report']).read_text()
        change = target / pair['shared_change_report']
        assert change.read_text() == (source / pair['shared_change_report']).read_text()
        for index, kernel in enumerate(pair['kernel_changes'], 1):
            assert f"../{pair['shared_change_report']}#kernel-{index}" in (target / pair['report']).read_text()
            assert kernel['applicability'] in (target / pair['report']).read_text()
            assert f"**Body change:** {kernel['body_change']}" in change.read_text()
            assert f"**Fusion:** {kernel['fusion']}" in change.read_text()
            assert '**Body change:**' not in (target / pair['report']).read_text()

    # Re-rendering must not drop runtime annotations or change source/line links.
    first = {p.relative_to(target): p.read_bytes() for p in (target / 'pairs').glob('*.md')}
    subprocess.run([sys.executable, str(BUILDER), '--knowledge', str(target)],
                   check=True, capture_output=True, text=True)
    assert first == {p.relative_to(target): p.read_bytes() for p in (target / 'pairs').glob('*.md')}


def test_default_discovery_omits_model_payloads_but_keeps_direct_source_access():
    rg = shutil.which('rg')
    if not rg:
        pytest.skip('ripgrep is not installed')
    visible = subprocess.check_output([rg, '--files', 'knowledge/top_model'], cwd=ROOT, text=True).splitlines()
    assert 'knowledge/top_model/README.md' in visible
    assert any('/changes/' in p for p in visible)
    assert all(not any('/' + part + '/' in p for part in ['data', 'evidence', 'kernel_sources']) for p in visible)
    all_files = subprocess.check_output([rg, '--files', '--no-ignore', 'knowledge/top_model'], cwd=ROOT, text=True).splitlines()
    assert any('/kernel_sources/blobs/' in p for p in all_files)
    assert any('/evidence/' in p for p in all_files)


def test_shared_records_cover_source_edges_without_performance_pairs():
    root = ROOT / 'knowledge/top_model/deepseek_v41_flash'
    # V10 only has TP8 c1, absent in V09: source evolution exists, but no E2E pair.
    change = (root / 'changes/v09_to_v10.md').read_text()
    assert 'no matching measured pair' in change
    pairs = json.loads((root / 'data/pair_records.json').read_text())
    assert all(p['shared_change_report'] != 'changes/v09_to_v10.md' for p in pairs)
