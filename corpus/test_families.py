"""Cross-family knowledge contracts; no consumer, GPU or upstream checkout required."""
from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _build_corpus as build
import _extract_impl_facts as extract


@pytest.fixture(scope="module")
def catalog():
    return build.families()


@pytest.fixture(scope="module")
def inputs(catalog):
    return {family["id"]: build.load_inputs(family) for family in catalog}


def test_all_operator_directories_have_one_discovery_owner(catalog):
    operators = {p.name for p in (build.HERE.parent / "knowledge/operator").iterdir() if p.is_dir()}
    mapped = [p for f in catalog for p in f["patterns"] if p in operators]
    assert set(mapped) == operators
    assert len(mapped) == len(set(mapped))
    assert next(f["id"] for f in catalog if "grouped_gemm_moe" in f["patterns"]) == "moe"


@pytest.mark.parametrize("name", ["gemm", "attention", "moe", "indexer"])
def test_family_schema_and_grounding(name, catalog, inputs):
    family = next(f for f in catalog if f["id"] == name)
    build.validate_family(family, inputs[name])


@pytest.mark.parametrize("name", ["attention", "moe", "indexer"])
def test_new_family_has_cards_for_each_operator(name, catalog, inputs):
    family = next(f for f in catalog if f["id"] == name)
    cards = inputs[name][0]["cards"]
    operators = {op for c in cards for op in c["match"]["operator_families"]}
    assert operators == set(family["patterns"])
    records = inputs[name][1]["source_evidence"]
    assert all(not any(word in r["file"] for word in ("_bwd", "_backward", "_wgrad", "_train")) for r in records)
    assert not inputs[name][1]["missing"], "the shipped scope must resolve at its snapshot"


@pytest.mark.parametrize("name", ["gemm", "attention", "moe", "indexer"])
def test_generated_pages_and_relative_links(name, catalog):
    family = next(f for f in catalog if f["id"] == name)
    for path, text in build.pages(family).items():
        assert path.read_text() == text, f"regenerate {path.name}"
        for target in re.findall(r"\]\(([^\s)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path.name, target)


def test_edited_source_invalidates_content_identity(catalog, inputs):
    family = next(f for f in catalog if f["id"] == "attention")
    broken = deepcopy(inputs["attention"])
    broken[1]["source_evidence"][0]["evidence"][0] += " corrupted"
    with pytest.raises(ValueError, match="source ID"):
        build.validate_family(family, broken)


def test_unknown_dependency_ref_is_rejected(catalog, inputs):
    family = next(f for f in catalog if f["id"] == "moe")
    broken = deepcopy(inputs["moe"])
    broken[4]["dependencies"][0]["decision_refs"] = ["nonexistent-card"]
    with pytest.raises(ValueError, match="unknown card"):
        build.validate_family(family, broken)


def test_measurement_cannot_be_inserted_into_source_evidence(catalog, inputs):
    family = next(f for f in catalog if f["id"] == "indexer")
    broken = deepcopy(inputs["indexer"])
    broken[1]["source_evidence"][0]["speedup"] = 1.5
    with pytest.raises(ValueError, match="unexpected source fields"):
        build.validate_family(family, broken)


def test_scope_without_tuning_does_not_scan_checkout_root(tmp_path):
    (tmp_path / "gfx950-not-a-family.json").write_text('{"any":{"BLOCK_M":99}}')
    scope = extract.Scope("test", {"family": "test", "impls": []})
    assert extract.tuned_configs(scope, str(tmp_path)) == ([], [])
    assert extract.collect(scope, str(tmp_path)) == ([], [], [], [], [])


def test_scope_deduplicates_files_and_excludes_backward(tmp_path):
    (tmp_path / "forward.py").write_text("def forward(x):\n    return x\n")
    (tmp_path / "other_bwd.py").write_text("def backward(x):\n    return x\n")
    scope = extract.Scope("test", {
        "family": "test", "include_symbols": True, "exclude": ["*bwd*"],
        "impls": [{"language": "triton", "subtree": ".", "globs": ["*.py", "forward.py"]}],
    })
    records, _, _, coverage, _ = extract.collect(scope, str(tmp_path))
    assert len(records) == 1
    assert [row["file"] for row in coverage] == ["forward.py"]
    assert records[0]["end_line"] == 2


def test_mixed_file_keeps_nested_kernel_languages(tmp_path):
    source = tmp_path / "mixed.py"
    source.write_text(
        "def host():\n"
        "    @triton.jit\n"
        "    def helper(x):\n"
        "        BLOCK_M = 16\n"
        "        return x\n"
        "    @flyc.kernel\n"
        "    def kernel(x):\n"
        "        BLOCK_N = 32\n"
        "        return x\n"
    )
    rows, _, _ = extract.scan_file(str(source), source.name, "flydsl", include_symbols=True)
    declarations = {r["captured"][0]: r for r in rows if r["category"] == "symbol_declaration"}
    assert declarations["helper"]["language"] == "triton"
    assert declarations["kernel"]["language"] == "flydsl"
    assert declarations["host"]["language"] == "flydsl"
    assert {r["language"] for r in rows if r["category"] == "tile_shape"} == {"triton", "flydsl"}


def test_citation_audit_uses_pinned_git_objects(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    file = tmp_path / "kernel.py"
    file.write_text("def forward(x):\n    return x\n")
    git("add", "kernel.py")
    git("-c", "user.name=Corpus test", "-c", "user.email=corpus@example.invalid", "commit", "-qm", "source")
    commit = git("rev-parse", "HEAD")
    rows, _, _ = extract.scan_file(str(file), file.name, "triton", include_symbols=True)
    source = {"provenance": {"aiter_commit": commit}, "source_evidence": rows}
    monkeypatch.setattr(build, "load_inputs", lambda family: ({}, source))
    file.write_text("# a different working tree\n")
    assert build.verify_source({}, tmp_path) == 1
    rows[0]["evidence"][0] = "1: def invented(x):"
    with pytest.raises(ValueError, match="quote does not resolve"):
        build.verify_source({}, tmp_path)
