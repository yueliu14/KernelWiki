"""Compact pages retain source identity and lossless configuration semantics."""
from copy import deepcopy
import json
import re
import subprocess

import build
from validate import qualified


def test_source_pages_retain_every_owned_anchor_construct_and_pinned_location(catalog):
    for op, bundle in catalog.operators.items():
        _, text = build.render_source(catalog, op)
        for row in bundle["evidence"]["source_evidence"]:
            assert text.count(f'<a id="{build.anchor(qualified(row))}"></a>') == 1
            assert catalog.source_url(row) in text
            assert build.code(row["match"]) in text
            assert f'### `{row["snapshot"]}` / `{row["file"]}`' in text


def test_factored_seeds_reconstruct_all_canonical_records(catalog):
    for op, bundle in catalog.operators.items():
        original = bundle["tuning"]["records"]
        restored = []
        for snapshot, files, common, members in build.seed_groups(original):
            for row in members:
                assert not common.keys() & row.keys()
                restored.append(dict(snapshot=snapshot, source_files=list(files), **common, **row))
        keyed = lambda rows: {qualified(r): json.dumps(r, sort_keys=True) for r in rows}
        assert keyed(restored) == keyed(original)
        if original:
            _, text = build.render_seeds(catalog, op)
            for row in original:
                assert text.count(f'<a id="{build.anchor(qualified(row))}"></a>') == 1
                for file in row["source_files"]:
                    assert catalog.source_url(dict(snapshot=row["snapshot"], file=file)) in text


def test_factoring_does_not_invent_defaults_coerce_types_or_cross_snapshots():
    rows = [dict(snapshot="pin_a", source_files=["config.json"], config_id=f"cfg_{i}",
                 kind="triton", m_bucket="any", shape_configs=1, knobs={"tile": 32}) for i in range(3)]
    rows[0].update(optional=None, typed=1)
    rows[1].update(optional="None", typed=True)
    rows[2]["typed"] = 1.0
    other = deepcopy(rows[0])
    other.update(snapshot="pin_b")
    groups = list(build.seed_groups(rows + [other]))
    assert len(groups) == 2
    _, _, common, members = groups[0]
    assert common == dict(kind="triton", shape_configs=1, knobs={"tile": 32})
    assert "optional" not in members[2]
    assert members[0]["optional"] is None and members[1]["optional"] == "None"
    assert [type(row["typed"]) for row in members] == [int, bool, float]


def test_historical_overviews_are_linked_but_not_default_discovery(catalog):
    root = catalog.root
    archives = list((root / "knowledge/operator").glob("*/geak_reference.md"))
    assert len(archives) == 20
    discovered = subprocess.check_output(["rg", "--files", "knowledge/operator"], cwd=root, text=True).splitlines()
    for archive in archives:
        overview = archive.with_name("README.md").read_text()
        assert "(geak_reference.md)" in overview
        assert "## Contract and applicability" in overview
        assert "## TL;DR" not in overview and "🟢 sota" not in overview
        assert str(archive.relative_to(root)) not in discovered
        text = archive.read_text()
        assert "(README.md)" in text and "## Migration provenance" in text
        assert re.search(r"https://github.com/AMD-AGI/GEAK/blob/[0-9a-f]{40}/", text)
        assert subprocess.check_output(["rg", "--no-ignore", "--files", str(archive)], cwd=root, text=True).strip() == str(archive)
    # Structured decisions and raw source observations are still discoverable.
    assert "knowledge/operator/dense_gemm/corpus/decisions.yaml" in discovered
    assert "knowledge/operator/dense_gemm/corpus/source_evidence.yaml" in discovered
