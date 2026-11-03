#!/usr/bin/env python3
"""Rebuild shipped configuration summaries from scoped, pinned Git objects."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tempfile

import yaml

import _extract_impl_facts as core


def positive(value):
    return type(value) is int and value > 0


def relative(path):
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError(f"invalid tuning source path: {path!r}")


def validate_scopes(scopes, snapshots):
    seen = set()
    for scope in scopes:
        if set(scope) - {"snapshot", "tuned_json", "tuned_db"} or scope.get("snapshot") not in snapshots:
            raise ValueError("invalid tuning scope or snapshot")
        if scope["snapshot"] in seen:
            raise ValueError("combine tuning inputs for the same snapshot in one scope")
        seen.add(scope["snapshot"])
        if not (scope.get("tuned_json") or scope.get("tuned_db")):
            raise ValueError("empty tuning scope")
        if scope.get("tuned_json"):
            if set(scope["tuned_json"]) != {"dir"}:
                raise ValueError("tuned_json needs a directory")
            relative(scope["tuned_json"]["dir"])
        if scope.get("tuned_db"):
            db = scope["tuned_db"]
            if set(db) != {"dirs", "name_pattern", "context_columns"} or not db["dirs"]:
                raise ValueError("tuned_db needs directories, a kind pattern and context columns")
            for path in db["dirs"]:
                relative(path)
            if "kind" not in re.compile(db["name_pattern"]).groupindex:
                raise ValueError("tuning database pattern needs a named kind group")


def validate_record(row):
    """Offline structural checks; exact values are audited by regeneration."""
    common = {"snapshot", "kind", "gfx", "m_bucket", "source_files"}
    kind = row.get("kind")
    if kind == "selection":
        required = common | {"selection_id", "cu_num", "db_kind", "rows", "selected_backend_rows"}
        if set(row) != required or not positive(row.get("rows")):
            raise ValueError("invalid selection fields or row count")
        counts = row["selected_backend_rows"]
        if (not isinstance(counts, dict) or not counts or not all(positive(n) for n in counts.values())
                or sum(counts.values()) != row["rows"]):
            raise ValueError("backend counts must sum to the selection row count")
        expected_id = core._short_id("sel_", ["aiter_tuned_db", row["gfx"], row["cu_num"],
                                             row["db_kind"], row["m_bucket"]])
        identity = row["selection_id"]
    elif kind in {"triton", "flydsl"}:
        required = common | {"config_id", "shape_configs"}
        required |= {"tags"} if kind == "triton" else {"cu_num", "db_kind", "family", "dtype"}
        count = row.get("shape_configs")
        if not positive(count):
            raise ValueError("shape_configs must be a positive integer")
        required |= {"knobs"} if count == 1 else {"same_across_configs", "varies_by_shape"}
        if not required <= set(row) or set(row) - required - {"partially_specified"}:
            raise ValueError("invalid configuration summary fields")
        groups = [row.get(key, {}) for key in ("knobs", "same_across_configs", "varies_by_shape", "partially_specified")]
        if not all(isinstance(group, dict) for group in groups):
            raise ValueError("knob summaries must be mappings")
        if any(n != 1 for n in Counter(key for group in groups for key in group).values()):
            raise ValueError("a knob cannot be both complete and partially specified")
        values = dict(row.get("knobs", row.get("same_across_configs", {})))
        for key, data in row.get("partially_specified", {}).items():
            if (not isinstance(data, dict) or set(data) != {"present", "missing", "values"}
                    or not positive(data["present"]) or not positive(data["missing"])
                    or data["present"] + data["missing"] != count):
                raise ValueError("partial knob counts must cover the whole group")
            entries = data["values"]
            if (not isinstance(entries, list) or not entries
                    or any(not isinstance(e, dict) or set(e) != {"value", "count"}
                           or not positive(e["count"]) for e in entries)
                    or sum(e["count"] for e in entries) != data["present"]):
                raise ValueError("partial knob value counts must match present declarations")
            for entry in entries:
                validate_knob(key, entry["value"])
        for key, value in values.items():
            validate_knob(key, value)
        if kind == "triton":
            parts = [row["gfx"], row["tags"], row["m_bucket"]]
        else:
            parts = ["aiter_tuned_db", row["gfx"], row["cu_num"], row["db_kind"],
                     row["family"], row["dtype"], row["m_bucket"]]
        expected_id = core._short_id("cfg_", parts)
        identity = row["config_id"]
    else:
        raise ValueError(f"unsupported seed kind: {kind}")
    if identity != expected_id:
        raise ValueError("seed ID does not match its grouping key")
    if "cu_num" in row and not positive(row["cu_num"]):
        raise ValueError("cu_num must be positive")
    files = row["source_files"]
    if not isinstance(files, list) or not files or len(files) != len(set(files)):
        raise ValueError("seed needs unique source files")
    for file in files:
        relative(file)


def validate_knob(key, value):
    if key in {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "tile_m", "tile_n", "tile_k",
               "num_warps", "kpack", "split_k", "NUM_KSPLIT", "matrix_instr_nonkdim"}:
        if not positive(value):
            raise ValueError(f"{key} must be a positive integer")
    json.dumps(value, allow_nan=False)


def extract_operator(catalog, op, checkout):
    scopes = catalog.operators[op]["scope"].get("tuning_sources", [])
    validate_scopes(scopes, catalog.snapshots)
    records, inventory, missing = [], [], []
    for spec in scopes:
        snapshot = spec["snapshot"]
        commit = catalog.snapshot(snapshot)["commit"]
        scope = core.Scope("pinned tuning scope", dict(spec, family=op, impls=[]))
        tree = subprocess.check_output(["git", "-C", str(checkout), "ls-tree", "-r", "--name-only", "-z", commit])
        files = []
        for raw in tree.split(b"\0"):
            if not raw:
                continue
            file = raw.decode("utf-8")
            parent, name = str(Path(file).parent), Path(file).name
            if (parent == scope.tuned_config_dir and name.endswith(".json")
                    or parent in scope.tuned_db_dirs and name.endswith(".csv")
                    and "untuned" not in name and "fmoe" not in name and scope.tuned_db_kind.search(name)):
                relative(file)
                files.append(file)
        with tempfile.TemporaryDirectory(prefix="kernelwiki-seeds-") as temporary:
            for file in files:
                target = Path(temporary) / file
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.check_output(["git", "-C", str(checkout), "show", f"{commit}:{file}"]))
            tuned, json_gaps = core.tuned_configs(scope, temporary)
            rows, db_inventory, db_gaps = core.tuned_db_rows(scope, temporary)
            flydsl, flydsl_gaps = core.summarize_flydsl_tuned(rows)
            for kind, groups in (("triton", core.summarize_tuned(tuned)), ("flydsl", flydsl),
                                 ("selection", core.summarize_backend_selection(rows))):
                records.extend(dict(snapshot=snapshot, kind=kind, **row) for row in groups)
            inventory.extend(dict(snapshot=snapshot, **row) for row in db_inventory)
            missing.extend(dict(snapshot=snapshot, **row) for row in json_gaps + db_gaps + flydsl_gaps)
    return dict(schema_version=3, operator=op, records=records, inventory=inventory, missing=missing)


def check_operator(catalog, op, checkout):
    actual = extract_operator(catalog, op, checkout)
    stored = catalog.operators[op]["tuning"]
    if stored != actual:
        raise ValueError(f"{op}: stale tuning inventory; regenerate with tools/corpus/seeds.py --emit")
    return len(actual["records"])


def main():
    from validate import Catalog
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--operator")
    group.add_argument("--family")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--emit", action="store_true")
    args = parser.parse_args()
    catalog = Catalog()
    selected = [op for op in catalog.selected(args.operator, args.family)
                if catalog.operators[op]["scope"].get("tuning_sources") or catalog.operators[op]["tuning"]["records"]]
    # Complete every read before emitting any files.
    products = {op: extract_operator(catalog, op, args.aiter) for op in selected}
    if args.check:
        stale = [op for op, data in products.items() if data != catalog.operators[op]["tuning"]]
        if stale:
            parser.exit(1, "STALE tuning inventory: " + ", ".join(stale) + "\n")
    else:
        for op, data in products.items():
            path = catalog.operators[op]["path"] / "tuned_configs.yaml"
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=105), encoding="utf-8")
    print(f"OK: {sum(len(data['records']) for data in products.values())} shipped groups across {len(products)} operators")


if __name__ == "__main__":
    main()
