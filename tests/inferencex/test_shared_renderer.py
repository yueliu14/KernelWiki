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

    # Re-rendering must not drop runtime annotations or change source/line links.
    first = {p.relative_to(target): p.read_bytes() for p in (target / 'pairs').glob('*.md')}
    subprocess.run([sys.executable, str(BUILDER), '--knowledge', str(target)],
                   check=True, capture_output=True, text=True)
    assert first == {p.relative_to(target): p.read_bytes() for p in (target / 'pairs').glob('*.md')}
