"""Configuration claims must reproduce from declared, pinned inputs."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

import _extract_impl_facts as core
import seeds


def test_missing_knobs_are_not_agreement_and_null_is_not_a_string():
    fixed, varies, partial = core.summarize_knobs([
        {"BLOCK_SIZE_M": 32, "kpack": 2, "cache_modifier": None},
        {"BLOCK_SIZE_M": 32, "cache_modifier": "None"},
        {"BLOCK_SIZE_M": 32},
    ])
    assert fixed == {"BLOCK_SIZE_M": 32}
    assert varies == {}
    assert partial["kpack"] == {"present": 1, "missing": 2, "values": [{"value": 2, "count": 1}]}
    assert partial["cache_modifier"]["values"] == [{"value": None, "count": 1}, {"value": "None", "count": 1}]


@pytest.fixture
def pinned(tmp_path):
    upstream = tmp_path / "aiter"
    upstream.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(upstream), *args], text=True).strip()
    git("init", "-q")
    configs = upstream / "configs"
    configs.mkdir()
    (configs / "gfx942-GEMM-N=32-K=32.json").write_text(json.dumps({"any": {"BLOCK_SIZE_M": 32, "kpack": 2}}))
    (configs / "gfx942-GEMM-N=64-K=32.json").write_text(json.dumps({"any": {"BLOCK_SIZE_M": 64}}))
    (configs / "bf16_tuned_gemm.csv").write_text(
        "gfx,cu_num,M,N,K,libtype,kernelName,us\ngfx942,304,8,32,32,asm,opaque,999\n")
    (configs / "bf16_tuned_batched_gemm.csv").write_text(
        "gfx,cu_num,M,N,K,us\ngfx942,304,8,32,32,123\n")
    git("add", ".")
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
    commit = git("rev-parse", "HEAD")
    class Catalog:
        snapshots = {"pin": {"commit": commit}}
        operators = {"gemm": {"scope": {"tuning_sources": [{"snapshot": "pin",
                     "tuned_json": {"dir": "configs"}, "tuned_db": {"dirs": ["configs"],
                     "name_pattern": r"(?P<kind>bf16_tuned_(?:batched_)?gemm)", "context_columns": []}}]}}}
        def snapshot(self, name):
            return self.snapshots[name]
    return Catalog(), upstream


def test_pinned_inventory_reproduces_all_groups_and_gaps_ignoring_dirty_files(pinned):
    catalog, upstream = pinned
    actual = seeds.extract_operator(catalog, "gemm", upstream)
    assert len(actual["records"]) == 2
    assert len(actual["inventory"]) == 2
    assert len(actual["missing"]) == 1
    config, selection = actual["records"]
    assert config["shape_configs"] == 2
    assert config["same_across_configs"] == {}
    assert config["partially_specified"]["kpack"]["missing"] == 1
    assert selection["selected_backend_rows"] == {"asm": 1}
    assert '"us"' not in json.dumps(actual)
    for row in actual["records"]:
        seeds.validate_record(row)
    (upstream / "configs/gfx942-GEMM-N=32-K=32.json").write_text("working file must not be read")
    (upstream / "configs/gfx942-GEMM-N=99-K=32.json").write_text("untracked file")
    catalog.operators["gemm"]["tuning"] = actual
    assert seeds.check_operator(catalog, "gemm", upstream) == 2


@pytest.mark.parametrize("field", ["value", "count", "selection", "inventory", "gap", "drop_group"])
def test_audit_rejects_changed_values_counts_inventory_and_gaps(pinned, field):
    catalog, upstream = pinned
    actual = seeds.extract_operator(catalog, "gemm", upstream)
    bad = deepcopy(actual)
    if field == "value":
        bad["records"][0]["partially_specified"]["kpack"]["values"][0]["value"] = 4
    elif field == "count":
        bad["records"][0]["shape_configs"] = 3
    elif field == "selection":
        bad["records"][1]["selected_backend_rows"] = {"triton": 1}
    elif field == "inventory":
        bad["inventory"][0]["rows"] += 1
    elif field == "gap":
        bad["missing"] = []
    else:
        bad["records"].pop()
    catalog.operators["gemm"]["tuning"] = bad
    with pytest.raises(ValueError, match="stale tuning inventory"):
        seeds.check_operator(catalog, "gemm", upstream)


@pytest.mark.parametrize("field", ["tile", "count", "partial", "backend", "id", "unknown"])
def test_offline_schema_rejects_impossible_seed_claims(catalog, field):
    row = deepcopy(next(r for r in catalog.seeds.values() if r['kind'] == 'triton' and 'knobs' in r))
    if field == "tile":
        row["knobs"]["BLOCK_SIZE_M"] = -256
    elif field == "count":
        row["shape_configs"] = -1
    elif field == "partial":
        row["partially_specified"] = {"kpack": {"present": 1, "missing": 1, "values": [{"value": 2, "count": 1}]}}
    elif field == "backend":
        row = deepcopy(next(r for r in catalog.seeds.values() if r['kind'] == 'selection'))
        row["rows"] += 1
    elif field == "id":
        row["config_id"] = "cfg_" + "0" * 16
    else:
        row["latency_us"] = 10
    with pytest.raises(ValueError):
        seeds.validate_record(row)
