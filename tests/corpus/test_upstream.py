"""PR reports retain context, uncertainty and attribution through validation and rendering."""
from copy import deepcopy
import subprocess

import pytest

import build
import collect_pr
import experience
import upstream
import _render_upstream as render


def report(catalog, number=4963):
    return deepcopy(catalog.upstream[f"aiter_pr_{number}"])


def entry(catalog):
    return deepcopy(catalog.cards["indexer-score-work-partition"]["experience"][0])


@pytest.mark.parametrize("metric,expected", [
    ({"kind": "latency", "unit": "us", "statistic": "single_case", "before": 100, "after": 80}, 1.25),
    ({"kind": "throughput", "unit": "tokens/s", "statistic": "mean", "before": 80, "after": 100}, 1.25),
    ({"kind": "latency_reduction", "unit": "percent", "statistic": "median", "value": 20}, 1.25),
    ({"kind": "throughput_increase", "unit": "percent", "statistic": "mean", "value": 20}, 1.2),
    ({"kind": "speedup", "unit": "x", "statistic": "max", "value": 1.8}, 1.8),
])
def test_metric_semantics_and_statistics(metric, expected):
    upstream.validate_metric(metric)
    assert upstream.speedup(metric) == pytest.approx(expected)
    assert metric["statistic"] in render.metric_text(metric)


@pytest.mark.parametrize("mutation", ["zero", "nan", "infinite", "boolean", "missing", "unit", "double_value", "impossible_percent"])
def test_invalid_metrics_cannot_become_benefits(mutation):
    metric = {"kind": "latency", "unit": "us", "statistic": "single_case", "before": 100, "after": 80}
    if mutation in {"zero", "nan", "infinite", "boolean", "missing"}:
        metric["after"] = {"zero": 0, "nan": float("nan"), "infinite": float("inf"), "boolean": True, "missing": None}[mutation]
    elif mutation == "unit":
        metric["unit"] = "TFLOP/s"
    elif mutation == "double_value":
        metric["value"] = 1.25
    else:
        metric = {"kind": "latency_reduction", "unit": "percent", "statistic": "mean", "value": 100}
    with pytest.raises(ValueError):
        upstream.validate_metric(metric)


@pytest.mark.parametrize("mutation,message", [
    ("quote", "fingerprint"), ("merge", "merge state"), ("revision", "full commit"),
    ("promote", "high confidence"), ("context", "comparison context"),
    ("reverse", "numeric value"), ("outcome", "contradicts metric"), ("missing_not_disclosed", "missing context"),
    ("duplicate", "duplicate/invalid result"), ("foreign_source", "this PR"),
])
def test_report_integrity(catalog, mutation, message):
    data = report(catalog)
    if mutation == "quote":
        data["sources"][0]["excerpt"] += " ungrounded"
    elif mutation == "merge":
        data["state"] = "closed"
    elif mutation == "revision":
        data["head_commit"] = "main"
    elif mutation == "promote":
        data["results"][0]["confidence"]["level"] = "high"
    elif mutation == "context":
        data["results"][0]["context"]["baseline"] = None
    elif mutation == "reverse":
        data["results"][0]["metrics"][0]["after"] = 99999
    elif mutation == "outcome":
        data["results"][0]["outcome"] = "regressed"
    elif mutation == "missing_not_disclosed":
        data["results"][0]["confidence"]["missing_evidence"] = []
    elif mutation == "duplicate":
        data["results"].append(deepcopy(data["results"][0]))
    elif mutation == "foreign_source":
        data["sources"][0]["url"] = "https://github.com/ROCm/aiter/pull/1"
    with pytest.raises(ValueError, match=message):
        upstream.validate_record(data)


def test_absent_report_is_not_zero_gain(catalog):
    data = report(catalog, 3231)
    result = data["results"][0]
    assert result["outcome"] == "not_reported" and result["confidence"]["level"] == "unknown"
    assert not result["metrics"]
    result["metrics"] = [{"kind": "speedup", "unit": "x", "statistic": "mean", "value": 1}]
    with pytest.raises(ValueError, match="not_reported cannot"):
        upstream.validate_record(data)


def test_bundle_gain_cannot_be_assigned_to_an_individual_axis(catalog):
    data = entry(catalog)
    data["upstream"]["assessment"]["attribution"]["confidence"]["level"] = "high"
    assert "independent attribution" in " ".join(experience.validate("test", [data]))
    data["upstream"]["assessment"]["attribution"]["scope"] = "isolated"
    assert "explicit evidence" in " ".join(experience.validate("test", [data]))


def test_pr_report_cannot_self_promote_to_reproduction(catalog):
    data = entry(catalog)
    data["provenance"]["verification"] = "internally_validated"
    assert "cannot claim reproduced" in " ".join(experience.validate("test", [data]))


@pytest.mark.parametrize("mutation,message", [
    ("pr", "unresolved PR"), ("result", "unresolved change/result"),
    ("axis", "unrelated axis/backend"), ("outcome", "contradicts referenced outcomes"),
    ("status", "requires a review state"), ("missing_status", "invalid fields"),
])
def test_card_links_are_resolved_and_scoped(catalog, mutation, message):
    card = deepcopy(catalog.cards["indexer-score-work-partition"])
    evidence = card["experience"][0]
    if mutation == "pr":
        evidence["upstream"]["pr"] = "aiter_pr_1"
    elif mutation == "result":
        evidence["upstream"]["result_refs"] = ["invented_result"]
    elif mutation == "axis":
        evidence["upstream"]["assessment"]["applies_to"]["axes"] = ["invented_axis"]
    elif mutation == "outcome":
        evidence["outcome"] = "regressed"
    elif mutation == "status":
        card["upstream_review"] = {"status": "unreviewed"}
    else:
        card.pop("upstream_review")
    with pytest.raises(ValueError, match=message):
        upstream.validate_cards({card["id"]: card}, catalog.upstream)


def test_serving_operator_and_conflicting_results_remain_separate(catalog):
    results = {r["id"]: r for r in catalog.upstream["aiter_pr_4963"]["results"]}
    assert results["long_prefill"]["context"]["scope"] == "operator"
    assert results["serving"]["context"]["scope"] == "end_to_end"
    assert results["long_prefill"]["context"]["candidate_commit"] is None
    contrary = catalog.upstream["aiter_pr_4538"]["results"]
    assert {r["outcome"] for r in contrary} == {"beneficial", "regressed"}
    _, text = build.render_decisions(catalog, "attention_indexer")
    assert "end_to_end" in text and "later_regression" in text


def test_closed_unmerged_proposal_and_unreviewed_cards_are_visible(catalog):
    pr = catalog.upstream["aiter_pr_1909"]
    assert pr["state"] == "closed" and pr["merge_commit"] is None
    _, text = build.render_decisions(catalog, "moe_routing_topk")
    assert "closed" in text and "not reproduced" in text and "Review: unreviewed" in text


def test_shared_decision_does_not_duplicate_pr_effects(catalog):
    card = catalog.cards["triton-dot-accumulator-form"]
    assert "attention_prefill_fmha" in card["match"]["operator_families"]
    _, text = build.render_decisions(catalog, "attention_prefill_fmha")
    assert 'id="triton-dot-accumulator-form"' not in text
    assert "triton-dot-accumulator-form" in text
    assert len(catalog.upstream["aiter_pr_3231"]["results"]) == 1


def test_private_capture_checks_metadata_and_quotation(catalog):
    record = report(catalog, 3231)
    fake = {"captured_at": record["captured_at"], "pr": {
        "number": record["number"], "title": record["title"], "merged_at": "2026-01-01T00:00:00Z",
        "state": "closed", "head": {"sha": record["head_commit"]}, "base": {"sha": record["base_commit"]},
        "merge_commit_sha": record["merge_commit"], "updated_at": record["updated_at"],
        "html_url": record["sources"][0]["url"], "body": record["sources"][0]["excerpt"]},
        "commits": [], "files": [{"filename": c["file"]} for c in record["changes"]]}
    assert collect_pr.verify_capture(record, fake) == 1
    newer = deepcopy(fake)
    newer["captured_at"] = "2026-11-05T12:00:00Z"
    newer["pr"]["updated_at"] = "2026-11-05T11:00:00Z"
    newer["pr"]["body"] = "An edited PR description without the old quotation."
    retained = deepcopy(record)
    retained["captured_at"] = newer["captured_at"]
    retained["updated_at"] = newer["pr"]["updated_at"]
    assert collect_pr.verify_capture(retained, [fake, newer]) == 1
    with pytest.raises(ValueError, match="quotation"):
        collect_pr.verify_capture(retained, newer)
    record["sources"][0]["excerpt"] += " invented performance"
    record["sources"][0]["sha256"] = upstream.digest(record["sources"][0]["excerpt"])
    with pytest.raises(ValueError, match="quotation"):
        collect_pr.verify_capture(record, fake)


def test_capture_paginates_and_rejects_truncated_api_results():
    calls = []
    def pages(endpoint):
        calls.append(endpoint)
        return list(range(100)) if endpoint.endswith("page=1") else [100]
    assert len(collect_pr.paginate("issues/1/comments", pages)) == 101
    assert len(calls) == 2
    def incomplete(endpoint):
        return {"changed_files": 1, "commits": 1} if endpoint == "pulls/1" else []
    with pytest.raises(ValueError, match="incomplete"):
        collect_pr.capture(1, incomplete)


def test_git_audit_checks_real_parent_and_changed_excerpts(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    file = tmp_path / "kernel.py"
    file.write_text("acc += dot(a, b)\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "before")
    parent = git("rev-parse", "HEAD")
    file.write_text("acc = dot(a, b, acc=acc)\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "after")
    change = {"commit": git("rev-parse", "HEAD"), "parent_commit": parent, "file": "kernel.py",
              "before": "acc += dot(a, b)", "after": "acc = dot(a, b, acc=acc)"}
    records = {"test": {"changes": [change]}}
    file.write_text("unrelated working tree\n")
    assert upstream.verify_sources(records, tmp_path) == 1
    change["after"] = "invented"
    with pytest.raises(ValueError, match="after excerpt"):
        upstream.verify_sources(records, tmp_path)
    change["parent_commit"] = change["commit"]
    with pytest.raises(ValueError, match="parent does not match"):
        upstream.verify_sources(records, tmp_path)


def test_report_pages_are_reproducible_and_escape_upstream_html(catalog):
    for pr in catalog.upstream.values():
        path, text = render.render_record(catalog.root, pr)
        assert path.read_text() == text
    pr = report(catalog)
    pr["sources"][0]["excerpt"] = '<script>alert("upstream")</script>'
    _, text = render.render_record(catalog.root, pr)
    assert "<script>" not in text and "&lt;script&gt;" in text
