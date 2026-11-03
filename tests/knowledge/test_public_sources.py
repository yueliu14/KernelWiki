"""Public workload evidence records availability without private artifact locators."""
from copy import deepcopy
from pathlib import Path
import re

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PINNED = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/blob/[0-9a-f]{40}/\S+")


def check_index(data):
    assert set(data) == {"schema_version", "reviewed_on", "evidence_kind", "access", "sources"}
    assert data["schema_version"] == 1
    assert data["access"].strip()
    assert data["sources"]
    ids = set()
    for row in data["sources"]:
        fields = {"id", "kind", "availability", "verification", "supports"}
        if row["availability"] == "public":
            fields.add("reference")
            assert PINNED.fullmatch(row["reference"])
            assert row["verification"] == "source_linked"
        else:
            assert row["availability"] == "private"
            assert row["verification"] == "curator_inspected"
        assert set(row) == fields
        assert re.fullmatch(r"[a-z][a-z0-9_]*", row["id"])
        assert row["id"] not in ids
        ids.add(row["id"])
        assert row["kind"] in {"source_code", "extraction_metadata", "environment_summary", "extraction_report"}
        assert row["supports"].strip()
    assert not re.search(r"/(?:shared_nfs|home|mnt|tmp|workspace)/", yaml.safe_dump(data))


@pytest.mark.parametrize("path", sorted((ROOT / "knowledge/workload/_sources").glob("*.yaml")))
def test_public_workload_source_contract(path):
    check_index(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("mutation", ["root", "path", "hash", "inline_path", "duplicate", "false_validation", "unversioned"])
def test_public_contract_rejects_private_locators_and_unsupported_validation(mutation):
    data = deepcopy(yaml.safe_load((ROOT / "knowledge/workload/_sources/muse_spark_c8.yaml").read_text()))
    row = data["sources"][0]
    if mutation == "root":
        data["roots"] = {"private": "/home/example/run"}
    elif mutation == "path":
        row["path"] = "run/REPORT.md"
    elif mutation == "hash":
        row["sha256"] = "a" * 64
    elif mutation == "inline_path":
        row["supports"] += " at /home/example/run"
    elif mutation == "duplicate":
        data["sources"].append(deepcopy(row))
    elif mutation == "false_validation":
        row["verification"] = "publicly_reproduced"
    else:
        row.update(availability="public", verification="source_linked",
                   reference="https://github.com/ROCm/aiter/blob/main/kernel.py")
    with pytest.raises(AssertionError):
        check_index(data)


def test_public_commit_reference_is_supported():
    data = yaml.safe_load((ROOT / "knowledge/workload/_sources/muse_spark_c8.yaml").read_text())
    data["sources"][0].update(availability="public", verification="source_linked",
        reference="https://github.com/ROCm/aiter/blob/" + "a" * 40 + "/aiter/ops/kernel.py")
    check_index(data)
