"""Observable diff/citation behavior of the portable AITER refresh planner."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "tools/corpus/skills/aiter-corpus-refresh/scripts/plan_refresh.py"
spec = importlib.util.spec_from_file_location("refresh_plan", SCRIPT)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value))


@pytest.fixture
def repository(tmp_path):
    upstream = tmp_path / "aiter"
    upstream.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(upstream), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def put(file, text):
        path = upstream / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(message):
        git("add", "-A")
        git("-c", "user.name=Corpus test", "-c", "user.email=corpus@example.invalid",
            "commit", "-qm", message)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    put("aiter/ops/shared.py", "def shared():\n    return 1\n")
    put("aiter/ops/gemm.py", "def gemm():\n    return 'exact rename example'\n")
    put("aiter/ops/gone.py", "def gone():\n    return 0\n")
    put("aiter/ops/old_unscoped.py", "def older():\n    return 0\n")
    put("aiter/configs/gemm.csv", "tile\n32\n")
    put("op_tests/test_gemm.py", "assert True\n")
    base = commit("base")
    put("aiter/ops/shared.py", "def shared():\n    return 2\n")
    git("mv", "aiter/ops/gemm.py", "aiter/ops/renamed gemm.py")
    (upstream / "aiter/ops/gone.py").unlink()
    put("aiter/configs/gemm.csv", "tile\n64\n")
    put("op_tests/test_gemm.py", "assert 2 > 1\n")
    put("aiter/ops/fp8_mqa_logits.py", "raise RuntimeError('must never execute this')\n")
    put("aiter/ops/attention_bwd.py", "raise RuntimeError('must never execute this')\n")
    put("aiter/ops/unknown_helper.py", "raise RuntimeError('must never execute this')\n")
    put("hsa/attention.co", "opaque binary fixture")
    target = commit("new kernels and updates")

    wiki = tmp_path / "KernelWiki"
    snapshot = "aiter_test"
    catalog = {"schema_version": 2, "snapshots": {snapshot: {"repository": planner.AITER + ".git", "commit": base}},
               "operators": [{"id": op, "directory": f"knowledge/operator/{op}/corpus"}
                             for op in ["dense_gemm", "grouped_gemm_moe"]]}
    write_yaml(wiki / "knowledge/operator/catalog.yaml", catalog)
    for op in ["dense_gemm", "grouped_gemm_moe"]:
        directory = wiki / "knowledge/operator" / op / "corpus"
        primary = op == "dense_gemm"
        files = ["aiter/ops/shared.py", "aiter/ops/gemm.py", "aiter/ops/gone.py"] if primary else []
        bindings = [{"snapshot": snapshot, "file": file, "language": "triton"} for file in files]
        write_yaml(directory / "scope.yaml", {"sources": bindings, "validation_sources": [
            {"snapshot": snapshot, "file": "op_tests/test_gemm.py"}] if primary else []})
        records = [dict(row, evidence_id=f"src_{i:016x}") for i, row in enumerate(bindings)]
        write_yaml(directory / "source_evidence.yaml", {"source_evidence": records})
        cards = [
            {"id": "shared-card", "source_evidence": [snapshot + ":src_0000000000000000"]},
            {"id": "tune-card", "source_evidence": [], "shipped_evidence": [snapshot + ":cfg_0000000000000001"]},
        ] if primary else []
        write_yaml(directory / "decisions.yaml", {"cards": cards, "shared_cards": [] if primary else ["shared-card"]})
        write_yaml(directory / "performance_axes.yaml", {"dependencies": [] if primary else [{"decision_refs": ["tune-card"]}]})
        if primary:
            write_yaml(directory / "tuned_configs.yaml", {"records": [{"snapshot": snapshot,
                       "config_id": "cfg_0000000000000001", "source_files": ["aiter/configs/gemm.csv"]}]})
    return wiki, upstream, base, target, git, put, commit


def test_changed_shared_sources_reach_all_consumers(repository):
    wiki, upstream, base, target, *_ = repository
    report = planner.plan(wiki, upstream, target)
    rows = {r["path"]: r for r in report["changes"]}
    shared = rows["aiter/ops/shared.py"]
    assert shared["cited_cards"] == ["shared-card"]
    assert shared["affected_consumers"] == ["dense_gemm", "grouped_gemm_moe"]
    assert shared["scoped_bindings"][0]["commit"] == base
    assert shared["scoped_bindings"][0]["differs_from_pin"] is True
    assert rows["op_tests/test_gemm.py"]["scoped_bindings"][0]["kind"] == "validation"
    config = rows["aiter/configs/gemm.csv"]
    assert config["cited_cards"] == ["tune-card"]
    assert config["affected_consumers"] == ["dense_gemm", "grouped_gemm_moe"]


def test_discovery_keeps_new_unknown_training_and_binary_candidates(repository):
    wiki, upstream, _, target, *_ = repository
    rows = {r["path"]: r for r in planner.plan(wiki, upstream, target)["changes"]}
    assert rows["aiter/ops/fp8_mqa_logits.py"]["unscoped_candidate"]
    assert "indexer" in rows["aiter/ops/fp8_mqa_logits.py"]["family_hints"]
    assert rows["aiter/ops/unknown_helper.py"]["unscoped_candidate"]
    assert rows["aiter/ops/unknown_helper.py"]["family_hints"] == []
    assert rows["aiter/ops/attention_bwd.py"]["training_name_hint"]
    assert rows["aiter/ops/attention_bwd.py"]["unscoped_candidate"]
    assert rows["hsa/attention.co"]["kind"] == "binary"


def test_exact_rename_and_deleted_binding_remain_visible(repository):
    wiki, upstream, _, target, *_ = repository
    rows = {r["path"]: r for r in planner.plan(wiki, upstream, target)["changes"]}
    renamed = rows["aiter/ops/renamed gemm.py"]
    assert renamed["status"] == "R100"
    assert renamed["old_path"] == "aiter/ops/gemm.py"
    assert not renamed["scoped_bindings"][0]["differs_from_pin"]
    assert renamed["affected_consumers"] == ["dense_gemm"]
    assert rows["aiter/ops/gone.py"]["status"] == "D"
    assert rows["aiter/ops/gone.py"]["scoped_bindings"][0]["differs_from_pin"]
    assert "target_object" not in rows["aiter/ops/gone.py"]


def test_first_inventory_finds_unchanged_unscoped_files(repository):
    wiki, upstream, _, target, *_ = repository
    report = planner.plan(wiki, upstream, target, full_inventory=True)
    assert "aiter/ops/old_unscoped.py" in {r["path"] for r in report["unscoped_inventory"]}
    assert "aiter/ops/old_unscoped.py" not in {r["path"] for r in report["changes"]}
    assert "aiter/ops/shared.py" not in {r["path"] for r in report["unscoped_inventory"]}


def test_dirty_working_files_are_not_read_and_wiki_is_unchanged(repository):
    wiki, upstream, _, target, _, put, _ = repository
    expected = planner.plan(wiki, upstream, target)
    before = {p: p.read_bytes() for p in wiki.rglob("*") if p.is_file()}
    put("aiter/ops/shared.py", "raise RuntimeError('working file must not execute')\n")
    put("aiter/ops/new_untracked.py", "not a committed file")
    assert planner.plan(wiki, upstream, target) == expected
    assert before == {p: p.read_bytes() for p in wiki.rglob("*") if p.is_file()}


def test_completed_interval_is_a_noop_and_since_limits_new_work(repository):
    wiki, upstream, _, target, _, put, commit = repository
    assert planner.plan(wiki, upstream, target, since=target)["changes"] == []
    put("aiter/ops/shared.py", "def shared():\n    return 3\n")
    latest = commit("later")
    report = planner.plan(wiki, upstream, latest, since=target)
    assert [r["path"] for r in report["changes"]] == ["aiter/ops/shared.py"]
    assert report["discovery_base"] == target
    assert report["target_commit"] == latest


def test_newest_ancestral_snapshot_and_schema3_general_cards(repository):
    wiki, upstream, _, target, _, put, commit = repository
    catalog_path = wiki / "knowledge/operator/catalog.yaml"
    data = planner.read_yaml(catalog_path)
    data["schema_version"] = 3
    data["snapshots"]["aiter_later"] = {"repository": planner.AITER, "commit": target}
    data["general_corpus"] = "knowledge/general/corpus"
    write_yaml(catalog_path, data)
    write_yaml(wiki / data["general_corpus"] / "decisions.yaml", {"cards": [
        {"id": "general-method", "source_evidence": [], "experience": [{"summary": "unchanged"}]}]})
    put("aiter/ops/shared.py", "def shared():\n    return 4\n")
    latest = commit("newest")
    report = planner.plan(wiki, upstream, latest)
    assert report["catalog_schema"] == 3
    assert report["discovery_base"] == target
    assert [r["path"] for r in report["changes"]] == ["aiter/ops/shared.py"]


def test_invalid_interval_or_missing_pin_stops_before_a_report(repository):
    wiki, upstream, base, target, *_ = repository
    with pytest.raises(ValueError, match="ancestor"):
        planner.plan(wiki, upstream, base, since=target)
    data = planner.read_yaml(wiki / "knowledge/operator/catalog.yaml")
    data["snapshots"]["aiter_test"]["commit"] = "0" * 40
    write_yaml(wiki / "knowledge/operator/catalog.yaml", data)
    with pytest.raises(ValueError, match="git rev-parse failed"):
        planner.plan(wiki, upstream, target)


def test_cli_failure_does_not_overwrite_a_prior_report(repository, tmp_path):
    wiki, upstream, *_ = repository
    output = tmp_path / "report.json"
    output.write_text("previous complete report")
    result = subprocess.run([sys.executable, str(SCRIPT), "--kernelwiki", str(wiki), "--aiter", str(upstream),
                             "--target", "missing-revision", "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Cannot plan corpus refresh" in result.stderr
    assert output.read_text() == "previous complete report"


def test_disconnected_shallow_pins_require_history_before_comparing(repository, tmp_path):
    wiki, upstream, base, target, *_ = repository
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth=1", upstream.as_uri(), str(shallow)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(shallow), "fetch", "-q", "--depth=1", "origin", base], check=True,
                   capture_output=True)
    with pytest.raises(ValueError, match="deepen a shallow cache"):
        planner.plan(wiki, shallow, target, since=base)


def test_unknown_shared_reference_is_not_silently_ignored(repository):
    wiki, upstream, _, target, *_ = repository
    p = wiki / "knowledge/operator/grouped_gemm_moe/corpus/decisions.yaml"
    write_yaml(p, {"cards": [], "shared_cards": ["missing-card"]})
    with pytest.raises(ValueError, match="unknown shared/dependency"):
        planner.plan(wiki, upstream, target)


def test_escaping_scope_path_is_rejected(repository):
    wiki, upstream, _, target, *_ = repository
    p = wiki / "knowledge/operator/dense_gemm/corpus/scope.yaml"
    write_yaml(p, {"sources": [{"snapshot": "aiter_test", "file": "../outside.py"}]})
    with pytest.raises(ValueError, match="invalid upstream path"):
        planner.plan(wiki, upstream, target)


def test_initial_review_finds_older_pins_before_the_newest_snapshot(repository):
    wiki, upstream, base, target, *_ = repository
    path = wiki / "knowledge/operator/catalog.yaml"
    data = planner.read_yaml(path)
    data["snapshots"]["aiter_newest"] = {"repository": planner.AITER, "commit": target}
    write_yaml(path, data)
    report = planner.plan(wiki, upstream, target, full_inventory=True)
    assert report["changes"] == []
    rows = {r["path"]: r for r in report["pinned_differences"]}
    shared = rows["aiter/ops/shared.py"]
    assert shared["affected_consumers"] == ["dense_gemm", "grouped_gemm_moe"]
    assert shared["scoped_bindings"][0]["commit"] == base
    assert rows["aiter/configs/gemm.csv"]["cited_cards"] == ["tune-card"]
    assert rows["op_tests/test_gemm.py"]["scoped_bindings"][0]["kind"] == "validation"
    assert rows["aiter/ops/gone.py"]["status"] == "D"
    assert rows["aiter/ops/gemm.py"]["status"] == "D"
    assert "aiter/ops/old_unscoped.py" not in rows
    assert report["summary"]["pinned_different_files"] == len(rows)


def test_retained_historical_pins_do_not_repeat_after_completed_checkpoint(repository):
    wiki, upstream, _, target, *_ = repository
    report = planner.plan(wiki, upstream, target, since=target)
    assert report["changes"] == report["pinned_differences"] == []
    assert report["summary"]["affected_consumers"] == []
    # A deliberate full review can reopen differences to retained historic evidence.
    assert planner.plan(wiki, upstream, target, since=target, full_inventory=True)["pinned_differences"]


def test_full_inventory_keeps_new_unscoped_changes_and_unselected_configs(repository):
    wiki, upstream, base, target, *_ = repository
    path = wiki / "knowledge/operator/dense_gemm/corpus/tuned_configs.yaml"
    data = planner.read_yaml(path)
    data["records"] = []
    data["inventory"] = [{"snapshot": "aiter_test", "file": "aiter/configs/gemm.csv"}]
    write_yaml(path, data)
    cards_path = path.with_name("decisions.yaml")
    cards = planner.read_yaml(cards_path)
    cards["cards"][1]["shipped_evidence"] = []
    write_yaml(cards_path, cards)
    report = planner.plan(wiki, upstream, target, full_inventory=True)
    assert "aiter/ops/unknown_helper.py" in {r["path"] for r in report["unscoped_inventory"]}
    assert any(r["path"] == "aiter/configs/gemm.csv" and r["affected_consumers"] == ["dense_gemm"]
               for r in report["pinned_differences"])
