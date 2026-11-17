"""Evidence statistics must not turn links or missing evidence into successful trials."""
from copy import copy, deepcopy
import json
import re
from types import SimpleNamespace

import pytest

import build
import stats


@pytest.fixture(scope="module")
def data(catalog):
    return stats.collect(catalog)


def by_id(data, key):
    return next(row for row in data["results"] if row["id"] == key)


def test_reference_multiplicity_does_not_inflate_results(catalog, data):
    modified = copy(catalog)
    modified.cards = deepcopy(catalog.cards)
    card = modified.cards["indexer-score-work-partition"]
    card["experience"].append(deepcopy(next(e for e in card["experience"]
                                         if e["provenance"]["kind"] == "upstream_report")))
    changed = stats.collect(modified)
    for field in ("canonical_cards", "canonical_upstream_results", "numeric_upstream_results", "metric_count"):
        assert changed["summary"][field] == data["summary"][field]
    assert changed["summary"]["coverage"] == data["summary"]["coverage"]
    assert changed["summary"]["experience_entries"] == data["summary"]["experience_entries"] + 1
    for row in changed["results"]:
        assert len(row["linked_card_ids"]) == len(set(row["linked_card_ids"]))


def test_multiple_metrics_are_not_independent_results(data):
    serving = by_id(data, "aiter_pr_4963:serving")
    assert serving["context"]["scope"] == "end_to_end"
    assert len(serving["metrics"]) > 1
    assert data["summary"]["canonical_upstream_results"] == len(data["results"])
    assert data["summary"]["metric_count"] == sum(len(r["metrics"]) for r in data["results"])
    assert {m["kind"] for m in serving["metrics"]} == {"latency_reduction", "throughput_increase"}
    assert next(m["speedup"] for m in serving["metrics"] if m["kind"] == "throughput_increase") == pytest.approx(1.066)


def test_missing_measurement_is_not_zero_or_one(data):
    missing = [r for r in data["results"] if r["outcome"] == "not_reported"]
    assert missing and all(r["metrics"] == [] for r in missing)
    row = by_id(data, "aiter_pr_4963:long_prefill")
    assert row["context"]["candidate_commit"] is None
    assert "candidate_commit" in row["null_context_fields"]
    assert row["confidence"]["missing_evidence"]
    assert stats.coverage(0, 0)["fraction"] is None


def test_contradictions_lifecycle_and_attribution_survive(data):
    assert {r["outcome"] for r in data["results"] if r["pr_id"] == "aiter_pr_4538"} == {"beneficial", "regressed"}
    assert all(r["pr_state"] == "closed" for r in data["results"] if r["pr_id"] == "aiter_pr_1909")
    assert all(r["verification"] == "not_reproduced" for r in data["results"])
    assert any(a["assessment"]["attribution"]["scope"] == "bundle"
               for r in data["results"] for a in r["associations"])
    assert "max" in data["summary"]["metrics_by_statistic"]
    assert "geomean" in data["summary"]["metrics_by_statistic"]
    assert not {"success_rate", "average_speedup"} & data["summary"].keys()


def test_internal_validation_coverage_requires_explicit_provenance(catalog, data):
    modified = copy(catalog)
    modified.cards = deepcopy(catalog.cards)
    card = next(iter(modified.cards.values()))
    card.setdefault("experience", []).append({
        "outcome": "constraint", "conditions": ["specified test context"],
        "summary": "A validated interface constraint, without a numerical performance claim.",
        "provenance": {"kind": "private_summary", "verification": "internally_validated",
                       "artifacts": "not_published", "references": []},
    })
    changed = stats.collect(modified)
    assert changed["summary"]["coverage"]["cards_with_internal_validation"]["numerator"] == 1
    assert changed["summary"]["coverage"]["cards_with_numeric_upstream_results"] == data["summary"]["coverage"]["cards_with_numeric_upstream_results"]
    assert changed["summary"]["metric_count"] == data["summary"]["metric_count"]


def test_generated_statistics_are_integrated_and_links_resolve(catalog):
    outputs = stats.pages(catalog)
    built = build.pages(catalog, [])
    for path, text in outputs.items():
        assert built[path] == text == path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            json.loads(text)
            continue
        for target in re.findall(r"\]\(([^\s)]+)\)", text):
            if target.startswith("https://"):
                continue
            relative, _, fragment = target.partition("#")
            destination = path.parent / relative
            assert destination.exists(), (path, target)
            if fragment:
                assert f'<a id="{fragment}"></a>' in destination.read_text(), target


def test_stale_check_fails_without_writing(tmp_path, monkeypatch):
    path = tmp_path / "knowledge/statistics/README.md"
    path.parent.mkdir(parents=True)
    path.write_text("stale")
    catalog = SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(stats, "Catalog", lambda: SimpleNamespace(validate=lambda: catalog))
    monkeypatch.setattr(stats, "pages", lambda _: {path: "current\n"})
    monkeypatch.setattr("sys.argv", ["stats.py", "--check"])
    with pytest.raises(SystemExit) as error:
        stats.main()
    assert error.value.code == 1
    assert path.read_text() == "stale"
    monkeypatch.setattr("sys.argv", ["stats.py", "--emit"])
    assert stats.main() == 0
    assert path.read_text() == "current\n"
