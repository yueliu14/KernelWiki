#!/usr/bin/env python3
"""Validate operator-local knowledge and resolve its shared, snapshot-qualified references."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess

import yaml

import _extract_impl_facts as extractor
import _normalize_performance_decisions as normalizer
import _render_decisions as card_schema

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REF = re.compile(r"([a-z0-9_]+):(src_[0-9a-f]{16})")
SEED_REF = re.compile(r"([a-z0-9_]+):((?:cfg|sel)_[0-9a-f]{16})")
SOURCE_FIELDS = {"snapshot", "evidence_id", "category", "language", "file", "line", "end_line",
                 "match", "captured", "arch_scope", "evidence", "unreproducible"}


def read(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping: {path}")
    return data


def qualified(record):
    return record["snapshot"] + ":" + record.get("evidence_id", record.get("config_id", record.get("selection_id", "")))


def source_refs(card):
    refs = list(card.get("source_evidence", []))
    for solution in card.get("solutions", []):
        refs.extend(solution["source_evidence"])
    return refs


class Catalog:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.data = read(self.root / "corpus/catalog.yaml")
        if self.data.get("schema_version") != 2:
            raise ValueError("expected operator-local catalog schema_version 2")
        self.snapshots = self.data["snapshots"]
        self.operators = {}
        self.sources, self.cards, self.seeds = {}, {}, {}
        self.source_owners, self.card_owners, self.seed_owners = {}, {}, {}
        for entry in self.data["operators"]:
            op = entry["id"]
            if op in self.operators:
                raise ValueError(f"duplicate operator: {op}")
            directory = self.repo_path(entry["directory"])
            if directory != self.root / "knowledge/operator" / op / "corpus":
                raise ValueError(f"{op}: corpus is not under its operator")
            bundle = dict(entry, path=directory)
            for key, filename in (("scope", "scope.yaml"), ("evidence", "source_evidence.yaml"),
                                  ("decisions", "decisions.yaml"), ("axes", "performance_axes.yaml")):
                bundle[key] = read(directory / filename)
            seed_path = directory / "tuned_configs.yaml"
            bundle["tuning"] = read(seed_path) if seed_path.exists() else {"records": []}
            self.operators[op] = bundle
            for row in bundle["evidence"]["source_evidence"]:
                self._register(self.sources, self.source_owners, qualified(row), row, op)
            for card in bundle["decisions"]["cards"]:
                self._register(self.cards, self.card_owners, card["id"], card, op)
            for row in bundle["tuning"]["records"]:
                self._register(self.seeds, self.seed_owners, qualified(row), row, op)

    @staticmethod
    def _register(table, owners, key, value, owner):
        if key in table:
            raise ValueError(f"duplicate canonical record {key}: {owners[key]} / {owner}")
        table[key], owners[key] = value, owner

    def repo_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"path escapes repository: {relative}")
        return path

    def snapshot(self, name):
        if name not in self.snapshots:
            raise ValueError(f"unknown snapshot {name}")
        return self.snapshots[name]

    def resolve_source(self, ref):
        if not SOURCE_REF.fullmatch(ref) or ref not in self.sources:
            raise ValueError(f"unresolved source reference {ref}")
        return self.sources[ref]

    def source_url(self, row):
        snapshot = self.snapshot(row["snapshot"])
        repository = snapshot["repository"].removesuffix(".git").rstrip("/")
        suffix = f"#L{row['line']}" if row.get("line") else ""
        if row.get("end_line"):
            suffix += f"-L{row['end_line']}"
        return f"{repository}/blob/{snapshot['commit']}/{row['file']}{suffix}"

    def selected(self, operator=None, family=None):
        if operator:
            matches = [op for op, b in self.operators.items() if operator in [op, *b.get("aliases", [])]]
        elif family:
            matches = [op for op, b in self.operators.items() if b["family"] == family]
        else:
            matches = list(self.operators)
        if not matches:
            raise ValueError(f"unknown operator/family: {operator or family}")
        return matches

    def used_sources(self, op, include_shared=True):
        bundle = self.operators[op]
        cards = list(bundle["decisions"]["cards"])
        if include_shared:
            cards += [self.cards[ref] for ref in bundle["decisions"].get("shared_cards", [])]
        return {ref for card in cards for ref in source_refs(card)}

    def validate(self):
        vocabulary = set(self.data["authoring_languages"])
        family_owners = {}
        for family in self.data["families"]:
            for op in family["operators"]:
                if op in family_owners:
                    raise ValueError(f"ambiguous family for {op}")
                family_owners[op] = family["id"]
        known_ops = {p.name for p in (self.root / "knowledge/operator").iterdir() if p.is_dir()}
        if set(self.operators) != known_ops or set(family_owners) != known_ops:
            raise ValueError("catalog must cover every operator directory exactly once")
        for name, snapshot in self.snapshots.items():
            if not re.fullmatch(r"[0-9a-f]{40}", snapshot.get("commit", "")):
                raise ValueError(f"{name}: snapshot needs a full commit")
            if not snapshot.get("repository", "").startswith("https://github.com/"):
                raise ValueError(f"{name}: snapshot needs an upstream repository URL")
        for ref, row in self.sources.items():
            self.snapshot(row["snapshot"])
            if set(row) - SOURCE_FIELDS:
                raise ValueError(f"{ref}: unexpected source fields {sorted(set(row)-SOURCE_FIELDS)}")
            if row.get("unreproducible") or row["evidence_id"] != extractor.source_evidence_id(row):
                raise ValueError(f"{ref}: stale/unreproducible source ID")
            if row["language"] not in vocabulary:
                raise ValueError(f"{ref}: unknown source language")
            if not isinstance(row.get("line"), int) or row["line"] <= 0 or not row.get("evidence"):
                raise ValueError(f"{ref}: missing positive source line or quote")
            if row.get("end_line", row["line"]) < row["line"]:
                raise ValueError(f"{ref}: backwards source extent")
            self._upstream_path(row["file"])
        for ref, row in self.seeds.items():
            self.snapshot(row["snapshot"])
            if not SEED_REF.fullmatch(ref) or row["kind"] not in {"triton", "flydsl", "selection"}:
                raise ValueError(f"{ref}: invalid shipped seed")
            if set(row) & {"us", "ms", "tflops", "speedup", "bw", "errRatio"}:
                raise ValueError(f"{ref}: measurement in source-only seed")
            if not row.get("source_files"):
                raise ValueError(f"{ref}: seed without source files")
        source_view = [dict(row, evidence_id=ref) for ref, row in self.sources.items()]
        bound_files = {}
        for op, bundle in self.operators.items():
            if bundle["family"] != family_owners[op]:
                raise ValueError(f"{op}: family mismatch")
            scope, evidence, decisions, axes = [bundle[k] for k in ("scope", "evidence", "decisions", "axes")]
            if scope.get("operator") != op or evidence.get("operator") != op or axes.get("family") != op:
                raise ValueError(f"{op}: mismatched operator identity")
            if decisions.get("provenance", {}).get("operator_family") != op:
                raise ValueError(f"{op}: mismatched decision identity")
            if not decisions["cards"] or not evidence["source_evidence"]:
                raise ValueError(f"{op}: an empty directory is not operator coverage")
            local_files = set()
            for binding in scope["sources"]:
                self.snapshot(binding["snapshot"])
                self._upstream_path(binding["file"])
                key = (binding["snapshot"], binding["file"])
                if key in bound_files:
                    raise ValueError(f"duplicate source ownership {key}: {bound_files[key]} / {op}")
                bound_files[key] = op
                local_files.add(key)
            for row in evidence["source_evidence"]:
                if (row["snapshot"], row["file"]) not in local_files:
                    raise ValueError(f"{op}: source outside declared scope")
            for test in scope.get("validation_sources", []):
                self.snapshot(test["snapshot"])
                self._upstream_path(test["file"])
            normalizer.validate_schema(axes)
            errors = card_schema.development_steps(axes)[2]
            errors += card_schema.validate_traits(decisions.get("traits", {}))
            errors += card_schema.validate(decisions["cards"], source_view,
                                           card_schema.axis_ids_of(axes), self.seeds, decisions.get("traits", {}))
            if errors:
                raise ValueError(op + ": " + "\n".join(errors))
            mapped = {c for a in axes["axes"] for m in a.get("source_mappings", []) for c in m.get("categories", [])}
            if {r["category"] for r in evidence["source_evidence"]} - mapped:
                raise ValueError(f"{op}: unmapped source category")
            for card in decisions["cards"]:
                if card["status"] != "candidate_only" or op not in card["match"]["operator_families"]:
                    raise ValueError(f"{op}: invalid card status or ownership: {card['id']}")
                if set(card["match"]["operator_families"]) - known_ops:
                    raise ValueError(f"{card['id']}: unknown operator in match")
            refs = decisions.get("shared_cards", [])
            if len(refs) != len(set(refs)):
                raise ValueError(f"{op}: duplicate shared card reference")
            for ref in refs:
                if ref not in self.cards or self.card_owners[ref] == op:
                    raise ValueError(f"{op}: invalid shared card {ref}")
                if op not in self.cards[ref]["match"]["operator_families"]:
                    raise ValueError(f"{op}: shared card {ref} does not apply to this operator")
            for dependency in axes.get("dependencies", []):
                if set(dependency.get("decision_refs", [])) - self.cards.keys():
                    raise ValueError(f"{op}: unknown card in dependency")
        return self

    @staticmethod
    def _upstream_path(path):
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"invalid upstream path: {path}")

    def verify_source(self, checkout, operators=None):
        """Read pinned git objects, including references borrowed from another operator."""
        operators = list(self.operators) if operators is None else operators
        wanted = {ref for op in operators for ref in self.used_sources(op)}
        wanted |= {ref for ref, op in self.source_owners.items() if op in operators}
        blobs = {}
        def blob(snapshot, file):
            key = (snapshot, file)
            if key not in blobs:
                commit = self.snapshot(snapshot)["commit"]
                blobs[key] = subprocess.run(["git", "-C", str(checkout), "show", f"{commit}:{file}"],
                                            check=True, capture_output=True, text=True).stdout.splitlines()
            return blobs[key]
        for ref in sorted(wanted):
            row = self.resolve_source(ref)
            lines = blob(row["snapshot"], row["file"])
            for quote in row["evidence"]:
                number, text = quote.split(": ", 1)
                if not 1 <= int(number) <= len(lines) or lines[int(number)-1].rstrip() != text:
                    raise ValueError(f"{ref}: quote does not resolve at pinned source")
            if not 1 <= row["line"] <= row.get("end_line", row["line"]) <= len(lines):
                raise ValueError(f"{ref}: extent outside pinned source")
        for op in operators:
            for row in self.operators[op]["scope"].get("validation_sources", []):
                blob(row["snapshot"], row["file"])
            for row in self.operators[op]["tuning"]["records"]:
                for file in row["source_files"]:
                    blob(row["snapshot"], file)
        return len(wanted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator")
    parser.add_argument("--family")
    parser.add_argument("--verify-source", type=Path)
    args = parser.parse_args()
    catalog = Catalog().validate()
    selected = catalog.selected(args.operator, args.family)
    if args.verify_source:
        print(f"OK: {catalog.verify_source(args.verify_source, selected)} pinned source references")
    else:
        print(f"OK: {len(catalog.operators)} operators, {len(catalog.cards)} canonical cards, "
              f"{len(catalog.sources)} source observations, {len(catalog.seeds)} shipped groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
