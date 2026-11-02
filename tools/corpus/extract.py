#!/usr/bin/env python3
"""Extract one or more operators from the git snapshots declared in their local scopes.

Only source_evidence.yaml is produced. Curated cards and shipped tuning inventories
remain separate inputs. Working-tree edits are never read as committed evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tempfile

import yaml

import _extract_impl_facts as core
from validate import Catalog, qualified


def anchor_records(raw, file, language, anchors):
    """Record literal interface-parameter occurrences without interpreting their behavior."""
    view = core.strip_comments(raw, python=file.endswith(".py"))
    lines = raw.splitlines()
    result = {}
    for word in anchors:
        if not re.fullmatch(r"[A-Za-z_]\w*", word):
            raise ValueError(f"invalid interface anchor: {word!r}")
        for match in re.finditer(r"\b" + re.escape(word) + r"\b", view):
            line = view.count("\n", 0, match.start()) + 1
            row = dict(category="interface_parameter", language=language, file=file,
                       line=line, match=word, captured=[word], arch_scope="",
                       evidence=core.excerpt(lines, line - 1))
            row["evidence_id"] = core.source_evidence_id(row)
            result[row["evidence_id"]] = row
    return list(result.values())


def extract_operator(catalog, op, checkout):
    bundle = catalog.operators[op]
    records, coverage = [], []
    for binding in bundle["scope"]["sources"]:
        snapshot = catalog.snapshot(binding["snapshot"])
        file, language = binding["file"], binding["language"]
        catalog._upstream_path(file)
        raw = subprocess.run(["git", "-C", str(checkout), "show", f"{snapshot['commit']}:{file}"],
                             check=True, capture_output=True, text=True).stdout
        with tempfile.TemporaryDirectory(prefix="kernelwiki-source-") as directory:
            temporary = Path(directory) / ("source" + Path(file).suffix)
            temporary.write_text(raw, encoding="utf-8")
            if binding.get("scan") == "routing":
                scope = core.Scope("operator scope", {"family": op, "impls": [],
                                   "routing": {"targets": binding["targets"]}})
                rows, _ = core.routing_records(scope, str(temporary), file)
            else:
                rows, _, _ = core.scan_file(str(temporary), file, language,
                                            include_symbols=binding.get("include_symbols", False))
        rows += anchor_records(raw, file, language, binding.get("anchors", []))
        rows = list({row["evidence_id"]: row for row in rows}.values())
        records.extend(dict(snapshot=binding["snapshot"], **row) for row in rows)
        coverage.append(dict(snapshot=binding["snapshot"], file=file, language=language,
                             lines=raw.count("\n") + 1, evidence_records=len(rows)))
    records.sort(key=lambda r: (r["snapshot"], r["file"], r["line"], r["category"], r["evidence_id"]))
    return dict(schema_version=2, operator=op, provenance={"extractor": "tools/corpus/extract.py"},
                coverage=coverage, missing=bundle["scope"].get("coverage_gaps", []),
                source_evidence=records)


def equivalent(left, right):
    """List ordering is presentation; identities, scope, quotes and counts must agree."""
    return (left.get("operator") == right.get("operator")
            and {qualified(r): r for r in left["source_evidence"]}
            == {qualified(r): r for r in right["source_evidence"]}
            and sorted(left["coverage"], key=lambda r: (r["snapshot"], r["file"]))
            == sorted(right["coverage"], key=lambda r: (r["snapshot"], r["file"]))
            and left.get("missing", []) == right.get("missing", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--operator")
    selection.add_argument("--family")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--emit", action="store_true")
    args = parser.parse_args()
    catalog = Catalog()
    selected = catalog.selected(args.operator, args.family)
    # Read all selected sources before writing anything: a missing git object must
    # not leave a partially refreshed set of operators.
    products = {op: extract_operator(catalog, op, args.aiter) for op in selected}
    stale = []
    for op, payload in products.items():
        path = catalog.operators[op]["path"] / "source_evidence.yaml"
        if args.emit:
            path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=105),
                            encoding="utf-8")
        elif not equivalent(payload, catalog.operators[op]["evidence"]):
            stale.append(op)
    if stale:
        parser.exit(1, "STALE source evidence: " + ", ".join(stale) + "\n")
    print(f"OK: {sum(len(p['source_evidence']) for p in products.values())} observations "
          f"across {len(products)} operators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
