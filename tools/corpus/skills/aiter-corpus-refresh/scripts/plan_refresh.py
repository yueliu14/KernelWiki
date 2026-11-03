#!/usr/bin/env python3
"""Read-only AITER change inventory and KernelWiki citation-impact report.

Requires Git, Python >= 3.10 and PyYAML. Reads committed git trees, never imports
AITER, fetches upstream, edits a scope or replaces evidence. JSON goes to stdout
or an explicitly supplied private report path.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import subprocess

import yaml

AITER = "https://github.com/ROCm/aiter"
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".s", ".asm"}
CONFIG_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".toml"}
BINARY_SUFFIXES = {".co", ".hsaco", ".so", ".a", ".o"}


def read_yaml(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"catalog path escapes KernelWiki: {relative}")
    return path


def upstream_path(path):
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError(f"invalid upstream path: {path!r}")
    return path


class Git:
    def __init__(self, directory):
        self.directory = str(directory)
        self.trees = {}

    def run(self, *args, allow_false=False):
        env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
        result = subprocess.run(["git", "-C", self.directory, *args], capture_output=True, env=env)
        if result.returncode and not (allow_false and result.returncode == 1):
            raise ValueError(f"git {args[0]} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
        return result

    def commit(self, ref):
        return self.run("rev-parse", "--verify", "--end-of-options", ref + "^{commit}").stdout.decode().strip()

    def ancestor(self, older, newer):
        return self.run("merge-base", "--is-ancestor", older, newer, allow_false=True).returncode == 0

    def tree(self, commit):
        if commit not in self.trees:
            entries = {}
            for raw in self.run("ls-tree", "-r", "-z", commit).stdout.split(b"\0"):
                if raw:
                    header, path = raw.split(b"\t", 1)
                    entries[path.decode("utf-8")] = tuple(header.decode().split())
            self.trees[commit] = entries
        return self.trees[commit]

    def changes(self, base, target):
        # Exact renames use object identity, avoiding blob downloads in partial clones.
        fields = self.run("diff", "--name-status", "-z", "--find-renames=100%", "--no-ext-diff",
                          "--no-textconv", base, target, "--").stdout.decode("utf-8").split("\0")
        index = 0
        while index < len(fields) and fields[index]:
            status, path = fields[index:index + 2]
            index += 2
            old_path = path
            if status.startswith(("R", "C")):
                path = fields[index]
                index += 1
            yield {"status": status, "path": path, "old_path": old_path if status[0] in "RCD" else None}


def classify(path):
    parts = Path(path).parts
    suffix = Path(path).suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return "binary"
    if parts and parts[0] == "op_tests":
        return "test_or_benchmark"
    if suffix in CONFIG_SUFFIXES:
        return "configuration"
    if suffix in SOURCE_SUFFIXES and parts and parts[0] in {"aiter", "csrc", "include", "hsa"}:
        return "implementation_or_helper"
    if suffix in {".md", ".rst", ".txt"}:
        return "documentation"
    return "other"


def family_hints(path):
    name = path.lower()
    hints = []
    if re.search(r"indexer|mqa_logits|index_score|topk_per_row", name):
        hints.append("indexer")
    if re.search(r"moe|fmoe|expert|moe_sort", name):
        hints.append("moe")
    if re.search(r"attention|attn|fmha|flash|mla|gqa|mqa|gdr|gated_delta|rope|kv_cache", name):
        hints.append("attention")
    if re.search(r"gemm|gemv|bmm|hgemm", name):
        hints.append("gemm")
    return hints


def load_inputs(root):
    catalog = read_yaml(root / "knowledge/operator/catalog.yaml")
    if catalog.get("schema_version") not in {2, 3}:
        raise ValueError("unsupported catalog: use operator-local schema 2 or 3")
    snapshots = catalog["snapshots"]
    for key, snapshot in snapshots.items():
        if snapshot["repository"].removesuffix(".git").rstrip("/") != AITER:
            raise ValueError(f"{key}: this helper inventories ROCm/aiter snapshots only")
        if not re.fullmatch(r"[0-9a-f]{40}", snapshot["commit"]):
            raise ValueError(f"{key}: a snapshot must use a full commit")
    watches, source_files, seed_files = defaultdict(list), {}, {}
    cards, consumers = {}, defaultdict(set)

    def watch(file, snapshot, owner, kind):
        upstream_path(file)
        if snapshot not in snapshots:
            raise ValueError(f"unknown snapshot: {snapshot}")
        row = {"snapshot": snapshot, "owner": owner, "kind": kind}
        if row not in watches[file]:
            watches[file].append(row)

    def card_input(owner, data):
        for card in data["cards"]:
            if card["id"] in cards:
                raise ValueError(f"duplicate canonical card: {card['id']}")
            cards[card["id"]] = card
            consumers[card["id"]].add(owner)
        for ref in data.get("shared_cards", []):
            consumers[ref].add(owner)

    for operator in catalog["operators"]:
        op = operator["id"]
        directory = inside(root, operator["directory"])
        scope = read_yaml(directory / "scope.yaml")
        for binding in scope["sources"]:
            watch(binding["file"], binding["snapshot"], op, "source")
        for binding in scope.get("validation_sources", []):
            watch(binding["file"], binding["snapshot"], op, "validation")
        for row in read_yaml(directory / "source_evidence.yaml")["source_evidence"]:
            source_files[row["snapshot"] + ":" + row["evidence_id"]] = upstream_path(row["file"])
        card_input(op, read_yaml(directory / "decisions.yaml"))
        for dep in read_yaml(directory / "performance_axes.yaml").get("dependencies", []):
            for ref in dep.get("decision_refs", []):
                consumers[ref].add(op)
        if (directory / "tuned_configs.yaml").exists():
            tuning = read_yaml(directory / "tuned_configs.yaml")
            for row in tuning["records"]:
                ref = row["snapshot"] + ":" + row.get("config_id", row.get("selection_id", ""))
                seed_files[ref] = [upstream_path(x) for x in row["source_files"]]
                for file in row["source_files"]:
                    watch(file, row["snapshot"], op, "shipped_configuration")
            for row in tuning.get("inventory", []):
                watch(row["file"], row["snapshot"], op, "shipped_configuration")
    if catalog.get("general_corpus"):
        card_input("general", read_yaml(inside(root, catalog["general_corpus"]) / "decisions.yaml"))
    missing = consumers.keys() - cards.keys()
    if missing:
        raise ValueError(f"unknown shared/dependency cards: {sorted(missing)}")
    cards_by_file = defaultdict(set)
    for ref, card in cards.items():
        sources = list(card.get("source_evidence", []))
        sources += [s for solution in card.get("solutions", []) for s in solution.get("source_evidence", [])]
        for source in sources:
            if source not in source_files:
                raise ValueError(f"{ref}: unresolved source reference: {source}")
            cards_by_file[source_files[source]].add(ref)
        for seed in card.get("shipped_evidence", []):
            if seed not in seed_files:
                raise ValueError(f"{ref}: unresolved shipped reference: {seed}")
            for file in seed_files[seed]:
                cards_by_file[file].add(ref)
    return catalog, watches, cards_by_file, consumers


def plan(kernelwiki, aiter, target_ref, since=None, full_inventory=False):
    root = Path(kernelwiki).resolve()
    catalog, watches, cards_by_file, consumers = load_inputs(root)
    git = Git(aiter)
    target = git.commit(target_ref)
    commits = {name: git.commit(row["commit"]) for name, row in catalog["snapshots"].items()}
    if since:
        base = git.commit(since)
        if not git.ancestor(base, target):
            raise ValueError("--since must be a verified ancestor of the pinned target; deepen a shallow cache "
                             "before treating the histories as divergent")
    else:
        ancestors = {commit for commit in commits.values() if git.ancestor(commit, target)}
        newest = [c for c in ancestors if not any(c != d and git.ancestor(c, d) for d in ancestors)]
        if len(newest) != 1:
            raise ValueError("no unique newest ancestral snapshot; check shallow history and supply an explicit --since commit")
        base = newest[0]
    target_tree = git.tree(target)
    result = []
    affected = set()
    for change in git.changes(base, target):
        names = {change["path"], change["old_path"]} - {None}
        refs = sorted({ref for name in names for ref in cards_by_file.get(name, [])})
        watched = []
        for name in sorted(names):
            for row in watches.get(name, []):
                commit = commits[row["snapshot"]]
                old_object = git.tree(commit).get(name)
                current_object = target_tree.get(change["path"])
                watched.append(dict(row, file=name, commit=commit, differs_from_pin=old_object != current_object))
        impacted = sorted({row["owner"] for row in watched} | {op for ref in refs for op in consumers[ref]})
        affected.update(impacted)
        kind = classify(change["path"])
        entry = dict(change, kind=kind, family_hints=family_hints(change["path"]),
                     training_name_hint=bool(re.search(r"(?:_bwd|backward|_wgrad|_train)", change["path"].lower())),
                     scoped_bindings=watched, cited_cards=refs, affected_consumers=impacted,
                     unscoped_candidate=not watched and kind in {
                         "implementation_or_helper", "configuration", "binary", "test_or_benchmark"})
        if change["path"] in target_tree:
            mode, object_type, oid = target_tree[change["path"]]
            entry["target_object"] = {"mode": mode, "type": object_type, "id": oid}
        result.append(entry)
    pinned_differences = []
    # A newest catalog snapshot is not a checkpoint for older owned bindings.
    # Audit them on the first pass and whenever a full review is explicitly requested.
    if since is None or full_inventory:
        for path, bindings in sorted(watches.items()):
            different = []
            current = target_tree.get(path)
            for row in bindings:
                commit = commits[row["snapshot"]]
                if git.tree(commit).get(path) != current:
                    different.append(dict(row, file=path, commit=commit, differs_from_pin=True))
            if not different:
                continue
            refs = sorted(cards_by_file.get(path, []))
            impacted = sorted({row["owner"] for row in different} | {op for ref in refs for op in consumers[ref]})
            affected.update(impacted)
            pinned_differences.append(dict(path=path, status="M" if current else "D", kind=classify(path),
                                          scoped_bindings=different, cited_cards=refs, affected_consumers=impacted))
    inventory = []
    if full_inventory:
        for path in sorted(target_tree):
            kind = classify(path)
            if path not in watches and kind in {"implementation_or_helper", "configuration", "binary", "test_or_benchmark"}:
                inventory.append({"path": path, "kind": kind, "family_hints": family_hints(path),
                                  "training_name_hint": bool(re.search(r"(?:_bwd|backward|_wgrad|_train)", path.lower()))})
    return {"schema_version": 2, "repository": AITER, "catalog_schema": catalog["schema_version"],
            "discovery_base": base, "target_commit": target,
            "summary": {"changed_files": len(result),
                        "changed_files_by_kind": dict(sorted(Counter(r["kind"] for r in result).items())),
                        "unscoped_candidates": sum(r["unscoped_candidate"] for r in result),
                        "inventory_candidates": len(inventory),
                        "inventory_by_kind": dict(sorted(Counter(r["kind"] for r in inventory).items())),
                        "pinned_different_files": len(pinned_differences),
                        "affected_consumers": sorted(affected)},
            "changes": result, "pinned_differences": pinned_differences, "unscoped_inventory": inventory,
            "limits": ["Family/training hints are lexical review aids, not ownership or eligibility decisions.",
                       "Only exact renames are detected; edited renames may appear as deletion plus addition.",
                       "The discovery window does not declare all older pinned sources obsolete or migrated.",
                       "No source was imported, executed, extracted, rewritten or performance-tested."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernelwiki", required=True, type=Path)
    parser.add_argument("--aiter", required=True, type=Path)
    parser.add_argument("--target", required=True, help="commit or fetched ref; resolved once to a full commit")
    parser.add_argument("--since", help="last fully reviewed AITER commit; defaults to newest ancestral catalog snapshot")
    parser.add_argument("--full-inventory", action="store_true", help="also list unscoped files already present at the target")
    parser.add_argument("--output", type=Path, help="private JSON report; stdout if omitted")
    args = parser.parse_args()
    try:
        report = plan(args.kernelwiki, args.aiter, args.target, args.since, args.full_inventory)
        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
            print(json.dumps(dict(report["summary"], target_commit=report["target_commit"])))
        else:
            print(payload, end="")
    except (ValueError, KeyError, OSError, yaml.YAMLError) as error:
        parser.exit(1, f"Cannot plan corpus refresh: {error}\n")


if __name__ == "__main__":
    main()
