"""Workload ownership, source boundaries and confidence survive generation and audits."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("workload_build", ROOT / "tools/workload/build.py")
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


@pytest.fixture
def catalog():
    return workload.Catalog().validate()


def test_generated_pages_and_workload_navigation(catalog):
    generated = workload.pages(catalog)
    assert all(path.read_text() == text for path, text in generated.items())
    for path in (ROOT / "knowledge/workload").rglob("*.md"):
        if "_templates" in path.parts:
            continue
        for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text()):
            if target.startswith(("http://", "https://")):
                continue
            relative, _, fragment = target.partition("#")
            destination = (path.parent / relative).resolve() if relative else path
            assert destination.is_file(), (path, target)
            if fragment:
                content = destination.read_text()
                headings = re.findall(r"^#{1,6}\s+(.+)$", content, re.M)
                slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                assert fragment in slugs or f'<a id="{fragment}"></a>' in content, (path, target)


@pytest.mark.parametrize("mutation,message", [
    ("omit", "catalog is incomplete"),
    ("duplicate", "duplicate/invalid case"),
    ("owner", "path/owner mismatch"),
    ("phase", "phases disagree"),
    ("source", "unresolved case source"),
    ("backend", "invalid backends"),
    ("escape", "invalid repository path"),
])
def test_catalog_rejects_lost_or_misattributed_cases(catalog, mutation, message):
    case = catalog.data["cases"][-1]
    if mutation == "omit":
        catalog.data["cases"].pop()
    elif mutation == "duplicate":
        catalog.data["cases"].append(deepcopy(case))
    elif mutation == "owner":
        case["chain"] = "decoder_layer"
    elif mutation == "phase":
        case["phases"] = ["decode"]
    elif mutation == "source":
        case["source_refs"] = ["nonexistent:source"]
    elif mutation == "backend":
        case["backends"] = ["monokernel"]
    else:
        case["path"] = "../elsewhere.md"
    with pytest.raises(ValueError, match=message):
        catalog.validate()


@pytest.mark.parametrize("mutation,message", [
    ("effect", "source inspection cannot"),
    ("attribution", "source inspection cannot"),
    ("metric", "unexpected or missing"),
    ("validated", "cannot claim validated"),
    ("foreign_source", "must belong to cited cases"),
    ("foreign_case", "owning-chain case"),
])
def test_source_inspection_cannot_be_promoted_or_reassigned(catalog, mutation, message):
    key = "stack-preserve-verify-state"
    record = deepcopy(catalog.decisions[key])
    catalog.decisions.clear()
    if mutation in {"effect", "attribution"}:
        record["assessment"][mutation]["level"] = "high"
    elif mutation == "metric":
        record["effect"]["speedup"] = 1.2
    elif mutation == "validated":
        record["status"] = "internally_validated"
    elif mutation == "foreign_source":
        record["source_refs"] = ["muse_spark_c8:moe_source"]
    else:
        record["case_refs"] = ["minimax_m31_decode"]
    with pytest.raises(ValueError, match=message):
        catalog._decision("decoder_stack", record)


def test_forms_do_not_encode_launch_counts_or_duplicate_cards(catalog):
    mono = [c for c in catalog.cases.values() if "monokernel" in c["execution_forms"]]
    assert {c["chain"] for c in mono} == {"decoder_layer", "attention_output_moe", "decoder_stack"}
    rendered = workload.pages(catalog)
    for key in catalog.decisions:
        assert sum(f'<a id="{key}"></a>' in text for text in rendered.values()) == 1
    for d in catalog.decisions.values():
        assert d["effect"]["outcome"] == "not_assessed"
        assert d["assessment"]["effect"]["level"] == "unknown"
        assert d["assessment"]["attribution"]["level"] == "unknown"


def test_source_audit_reads_fixed_git_objects_not_working_files(catalog, tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    git("remote", "add", "origin", "https://github.com/example/source.git")
    (tmp_path / "kernel.py").write_text("# inspected source\ndef entry():\n    pass\n")
    git("add", "kernel.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "source")
    sha = git("rev-parse", "HEAD")
    # A working copy that cannot be imported must not affect the fixed-object audit.
    (tmp_path / "kernel.py").write_text("not valid Python and must never be executed")
    row = dict(availability="public", reference=f"https://github.com/example/source/blob/{sha}/kernel.py#L2-L3")
    catalog.sources = {"fixture:kernel": row}
    assert catalog.verify_source(tmp_path) == (1, 0)
    row["reference"] = row["reference"].replace("L2-L3", "L2-L99")
    with pytest.raises(ValueError, match="line range"):
        catalog.verify_source(tmp_path)
    row["reference"] = f"https://github.com/example/source/blob/{sha}/../kernel.py"
    with pytest.raises(ValueError, match="upstream path"):
        catalog.verify_source(tmp_path)
    row["reference"] = f"https://github.com/another/source/blob/{sha}/kernel.py"
    with pytest.raises(ValueError, match="matches no public"):
        catalog.verify_source(tmp_path)
