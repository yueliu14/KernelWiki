#!/usr/bin/env python3
"""Validate and render workload decisions; optional source audit reads Git objects only."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]
ID = re.compile(r"[a-z][a-z0-9_-]*")
PIN = re.compile(r"(https://github\.com/[\w.-]+/[\w.-]+)/blob/([0-9a-f]{40})/([^#]+)(?:#L([1-9][0-9]*)(?:-L([1-9][0-9]*))?)?")
PHASES = {"prefill", "decode", "target_verify", "draft_decode", "draft_extend"}
FORMS = {"fusion", "staged", "persistent", "megakernel", "monokernel"}
AXES = {"fusion_boundary", "data_lifetime", "scheduling", "synchronization", "communication", "dispatch", "numerical_contract", "measurement", "compilation"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fields(obj, keys, label):
    require(isinstance(obj, dict) and set(obj) == set(keys.split()), f"{label}: unexpected or missing fields")


def strings(value, label, empty=False):
    require(isinstance(value, list) and (empty or value) and all(isinstance(x, str) and x.strip() for x in value), f"{label}: expected strings")
    require(len(value) == len(set(value)), f"{label}: duplicate entries")


def read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def safe(value):
    return escape(str(value)).replace("|", "&#124;").replace("\n", " ")


class Catalog:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.data = read(self.root / "knowledge/workload/catalog.yaml")
        self.chains, self.cases, self.sources, self.decisions, self.owners = {}, {}, {}, {}, {}

    def path(self, value):
        require(isinstance(value, str) and not Path(value).is_absolute() and ".." not in Path(value).parts, "invalid repository path")
        path = (self.root / value).resolve()
        require(path.is_relative_to(self.root), "path escapes repository")
        return path

    def validate(self):
        self.chains, self.cases, self.sources, self.decisions, self.owners = {}, {}, {}, {}, {}
        fields(self.data, "schema_version description source_sets chains cases", "workload catalog")
        require(self.data["schema_version"] == 1, "unsupported workload schema")
        require(isinstance(self.data["description"], str) and self.data["description"].strip(), "missing catalog description")
        require(isinstance(self.data["source_sets"], dict) and self.data["source_sets"], "missing source sets")
        for name, value in self.data["source_sets"].items():
            require(ID.fullmatch(name), "invalid source-set ID")
            path = self.path(value)
            require(path == self.root / f"knowledge/workload/_sources/{name}.yaml", "source set path mismatch")
            data = read(path)
            fields(data, "schema_version reviewed_on evidence_kind access sources", name)
            require(data["schema_version"] == 1 and data["sources"], "invalid source index")
            for row in data["sources"]:
                public = row.get("availability") == "public"
                fields(row, "id kind availability verification supports" + (" reference" if public else ""), name)
                require(isinstance(row["id"], str) and ID.fullmatch(row["id"]), "invalid evidence ID")
                require(row["kind"] in {"source_code", "documentation", "extraction_metadata", "environment_summary", "extraction_report"}, "invalid source kind")
                require(isinstance(row["supports"], str) and row["supports"].strip(), "empty source scope")
                if public:
                    require(isinstance(row["reference"], str) and PIN.fullmatch(row["reference"]), "source must pin a full Git commit")
                    require(row["verification"] == "source_linked", "source link cannot claim reproduction")
                else:
                    require(row["availability"] == "private" and row["verification"] == "curator_inspected", "invalid private source")
                key = f"{name}:{row['id']}"
                require(key not in self.sources, "duplicate evidence ID")
                self.sources[key] = row
        require(isinstance(self.data["chains"], list) and self.data["chains"], "missing chains")
        for chain in self.data["chains"]:
            fields(chain, "id directory operators", "chain")
            key = chain["id"]
            require(isinstance(key, str) and ID.fullmatch(key) and key not in self.chains, "duplicate/invalid chain")
            require(chain["directory"] == f"knowledge/workload/{key}", "chain directory mismatch")
            require((self.path(chain["directory"]) / "README.md").is_file(), "missing chain overview")
            strings(chain["operators"], "operators")
            for op in chain["operators"]:
                require(ID.fullmatch(op) and (self.root / f"knowledge/operator/{op}/README.md").is_file(), "unknown operator")
            self.chains[key] = chain
        require(isinstance(self.data["cases"], list) and self.data["cases"], "missing cases")
        paths = set()
        for case in self.data["cases"]:
            fields(case, "id chain path model phases backends execution_forms launch_summary source_refs", "case")
            key = case["id"]
            require(isinstance(key, str) and ID.fullmatch(key) and key not in self.cases, "duplicate/invalid case")
            require(case["chain"] in self.chains, "unknown case chain")
            path = self.path(case["path"])
            require(path.parent == self.root / self.chains[case["chain"]]["directory"] / "cases" and path.suffix == ".md" and path.is_file(), "case path/owner mismatch")
            require(path not in paths, "duplicate case path")
            paths.add(path)
            for key2, allowed in [("phases", PHASES), ("execution_forms", FORMS), ("backends", {"flydsl", "triton", "hip", "gluon", "asm", "ck", "tilelang"})]:
                strings(case[key2], key2, empty=key2 == "backends")
                require(set(case[key2]) <= allowed, f"invalid {key2}")
            require(all(isinstance(case[k], str) and case[k].strip() for k in ["model", "launch_summary"]), "missing case context")
            strings(case["source_refs"], "case sources")
            require(set(case["source_refs"]) <= self.sources.keys(), "unresolved case source")
            front = re.match(r"\A---\n(.*?)\n---(?:\n|$)", path.read_text(encoding="utf-8"), re.S)
            require(front is not None, "case needs front matter")
            meta = yaml.safe_load(front[1])
            require(isinstance(meta, dict) and meta.get("semantic_chain") == case["chain"] and meta.get("model") == case["model"], "case metadata disagrees with catalog")
            phases = meta.get("phase")
            phases = [phases] if isinstance(phases, str) else phases
            require(isinstance(phases, list) and all(isinstance(p, str) for p in phases), "invalid case phase metadata")
            require({"target_verify" if p == "decode_target_verify" else p for p in phases} == set(case["phases"]), "case phases disagree with catalog")
            require(isinstance(meta.get("source_set"), str), "case needs a source set")
            declared = (path.parent / meta["source_set"]).resolve()
            require(declared in {self.path(self.data["source_sets"][ref.split(":")[0]]) for ref in case["source_refs"]}, "case source set disagrees with catalog")
            self.cases[key] = case
        expected = set((self.root / "knowledge/workload").glob("*/cases/*.md"))
        require(expected == paths, "case catalog is incomplete or contains extra paths")
        for owner, chain in self.chains.items():
            path = self.path(chain["directory"]) / "corpus/decisions.yaml"
            if not path.exists():
                continue
            data = read(path)
            fields(data, "schema_version semantic_chain decisions", "decisions")
            require(data["schema_version"] == 1 and data["semantic_chain"] == owner, "decision owner/schema mismatch")
            require(isinstance(data["decisions"], list) and data["decisions"], "empty decision corpus")
            for decision in data["decisions"]:
                self._decision(owner, decision)
        actual = set((self.root / "knowledge/workload").glob("*/corpus/decisions.yaml"))
        require(actual == {self.path(c["directory"]) / "corpus/decisions.yaml" for c in self.chains.values() if (self.path(c["directory"]) / "corpus/decisions.yaml").exists()}, "unregistered decision corpus")
        return self

    def _decision(self, owner, d):
        fields(d, "id question axes status conditions actions rationale alternatives case_refs source_refs effect assessment validation_plan limitations", "decision")
        key = d["id"]
        require(isinstance(key, str) and ID.fullmatch(key) and key not in self.decisions, "duplicate/invalid decision ID")
        require(isinstance(d["question"], str) and d["question"].strip(), "missing question")
        require(d["status"] == "candidate_only", "source-only decision cannot claim validated applicability")
        for name in ["axes", "conditions", "actions", "rationale", "alternatives", "case_refs", "source_refs", "validation_plan", "limitations"]:
            strings(d[name], name)
        require(set(d["axes"]) <= AXES, "unknown decision axis")
        require(set(d["case_refs"]) <= self.cases.keys(), "unresolved decision case")
        require(any(self.cases[c]["chain"] == owner for c in d["case_refs"]), "decision needs an owning-chain case")
        cited = {s for c in d["case_refs"] for s in self.cases[c]["source_refs"]}
        require(set(d["source_refs"]) <= cited, "decision sources must belong to cited cases")
        fields(d["effect"], "outcome summary", "effect")
        require(d["effect"]["outcome"] == "not_assessed" and isinstance(d["effect"]["summary"], str) and d["effect"]["summary"].strip(), "v1 source-only decisions require unassessed effects")
        fields(d["assessment"], "implementation effect attribution", "assessment")
        for name, grade in d["assessment"].items():
            fields(grade, "level reasons missing_evidence", name)
            require(grade["level"] in {"unknown", "low", "medium", "high"}, "invalid confidence")
            strings(grade["reasons"], "confidence reasons")
            strings(grade["missing_evidence"], "missing evidence", empty=True)
            if name in {"effect", "attribution"}:
                require(grade["level"] == "unknown" and grade["missing_evidence"], "source inspection cannot establish measured effects or attribution")
        self.decisions[key], self.owners[key] = d, owner

    def verify_source(self, checkout):
        def git(*args):
            return subprocess.check_output(["git", "-C", str(checkout), *args], text=True)
        origin = git("remote", "get-url", "origin").strip().removesuffix(".git").rstrip("/")
        checked, skipped = 0, 0
        for row in self.sources.values():
            if row["availability"] != "public":
                continue
            match = PIN.fullmatch(row["reference"])
            repo, commit, file, start, end = match.groups()
            if repo != origin:
                skipped += 1
                continue
            require(not Path(file).is_absolute() and ".." not in Path(file).parts, "invalid upstream path")
            require(git("cat-file", "-t", commit).strip() == "commit", "source pin is not a commit")
            lines = git("show", f"{commit}:{file}").splitlines()
            if start:
                require(1 <= int(start) <= int(end or start) <= len(lines), "source line range does not resolve")
            checked += 1
        require(checked > 0, "checkout matches no public workload source")
        return checked, skipped


def link(catalog, path, target, label, anchor=""):
    return f"[{safe(label)}]({os.path.relpath(target, path.parent).replace(os.sep, '/')}{anchor})"


def sources(catalog, path, refs):
    out = []
    for ref in refs:
        row = catalog.sources[ref]
        target = row.get("reference")
        if target:
            out.append(f"- [{ref}]({target}) — {safe(row['supports'])}")
        else:
            source_set = ref.split(":")[0]
            out.append("- " + link(catalog, path, catalog.path(catalog.data["source_sets"][source_set]), ref) + " — private curator inspection; not publicly reproduced.")
    return out


def assessment_key(decision):
    return json.dumps([decision["assessment"], decision["effect"]], sort_keys=True)


def assessment_text(decision):
    out = []
    for name, grade in decision["assessment"].items():
        explanation = " ".join(grade["reasons"]).rstrip(". ")
        if grade["missing_evidence"]:
            explanation += "; missing: " + ", ".join(v.rstrip(". ") for v in grade["missing_evidence"])
        out += [f"**{name.capitalize()} confidence: {grade['level']}** — " + safe(explanation + "."), ""]
    return out + [f"**Effect: {decision['effect']['outcome']}** — {safe(decision['effect']['summary'])}", ""]


def pages(catalog):
    result = {}
    for name, value in catalog.data["source_sets"].items():
        data = read(catalog.path(value))
        if not all(row["availability"] == "public" for row in data["sources"]):
            continue
        path = catalog.path(value).with_suffix(".md")
        out = [f"# Workload source set: {name}", "",
               "GENERATED by `tools/workload/build.py`; edit the adjacent source index.", "",
               f"Reviewed: {data['reviewed_on']} · {data['evidence_kind']}", "", safe(data["access"]), "",
               "Each link pins the source revision. Tests listed here are test definitions, not results from executing them. The optional Git audit checks file/line locations, not the semantic truth of every conclusion.", ""]
        for row in data["sources"]:
            out += [f'<a id="{row["id"]}"></a>', f"## {row['id']}", "",
                    f"[Pinned source]({row['reference']}) — {safe(row['supports'])}", ""]
        result[path] = "\n".join(out)
    for owner, chain in catalog.chains.items():
        records = [d for k, d in catalog.decisions.items() if catalog.owners[k] == owner]
        if not records:
            continue
        path = catalog.path(chain["directory"]) / "corpus/decisions.md"
        out = [f"# {owner}: workload decisions", "", "GENERATED by `tools/workload/build.py`; edit the adjacent YAML.", "",
               "Source inspection establishes implementation precedents. Effect and independent-attribution confidence remain unknown; no speedup or reproduction is claimed.", "",
               link(catalog, path, path.parent.parent / "README.md", "Semantic boundary") + " · " + link(catalog, path, catalog.root / "knowledge/workload/CORPUS.md", "Workload index"), ""]
        counts = Counter(assessment_key(d) for d in records)
        shared = {}
        for d in records:
            key = assessment_key(d)
            if counts[key] > 1 and key not in shared:
                label = f"assessment-{len(shared) + 1}"
                shared[key] = label
                out += [f'<a id="{label}"></a>', f"## Shared assessment {len(shared)}", "",
                        "Applies only to the decisions linking here, whose assessment and effect records are identical.", ""]
                out += assessment_text(d)
        for d in records:
            out += [f'<a id="{d["id"]}"></a>', f"## {d['question']}", "", f"`{d['id']}` · candidate only · " + ", ".join(d["axes"]), ""]
            for field, title in [("conditions", "Use when"), ("actions", "Actions"), ("rationale", "Rationale"), ("alternatives", "Alternatives")]:
                out += [f"### {title}", ""] + ["- " + safe(v) for v in d[field]] + [""]
            out += ["### Evidence and assessment", ""]
            out += ["- " + link(catalog, path, catalog.path(catalog.cases[c]["path"]), catalog.cases[c]["model"]) for c in d["case_refs"]]
            out += [""] + sources(catalog, path, d["source_refs"]) + [""]
            key = assessment_key(d)
            if key in shared:
                levels = "; ".join(f"{name}: {grade['level']}" for name, grade in d["assessment"].items())
                out += [f"**Assessment:** {levels}; outcome: {d['effect']['outcome']}. "
                        f"[Reasons and missing evidence](#{shared[key]}).", ""]
            else:
                out += assessment_text(d)
            out += ["### Validation for a new workload", ""]
            out += ["- " + safe(v) for v in d["validation_plan"]] + ["", "### Limits", ""] + ["- " + safe(v) for v in d["limitations"]] + [""]
        result[path] = "\n".join(out).rstrip() + "\n"
    path = catalog.root / "knowledge/workload/CORPUS.md"
    out = ["# Workload cases and fusion decisions", "", "GENERATED by `tools/workload/build.py`; edit `catalog.yaml` and the owning decision YAML.", "",
           f"{len(catalog.chains)} semantic chains · {len(catalog.cases)} cases · {len(catalog.decisions)} decisions.", "",
           "Fusion, megakernel and monokernel are implementation labels, not mutually exclusive operators or guarantees of a single device launch. Launch summaries are source-inspected boundaries, not profiler counts or speedups.", "",
           "## Cases", "", "| Case | Semantic chain | Phases / backend | Execution forms | Launch boundary |", "|---|---|---|---|---|"]
    for case in catalog.cases.values():
        out.append("| " + link(catalog, path, catalog.path(case["path"]), case["model"] + " / " + case["id"]) + f" | {case['chain']} | {safe(', '.join(case['phases']))} / {safe(', '.join(case['backends']) or 'unspecified')} | {safe(', '.join(case['execution_forms']))} | {safe(case['launch_summary'])} |")
    out += ["", "## Decisions", "", "| Question | Owner | Axes | Effect confidence |", "|---|---|---|---|"]
    for key, d in catalog.decisions.items():
        owner = catalog.owners[key]
        target = catalog.path(catalog.chains[owner]["directory"]) / "corpus/decisions.md"
        out.append("| " + link(catalog, path, target, d["question"], "#" + key) + f" | {owner} | {safe(', '.join(d['axes']))} | {d['assessment']['effect']['level']} |")
    out += ["", "Use the " + link(catalog, path, catalog.root / "tools/workload/README.md", "maintenance guide") + " to audit pinned sources and add cases. Raw runs and private artifact locations remain outside this index.", ""]
    result[path] = "\n".join(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--verify-source", type=Path, metavar="CHECKOUT")
    args = parser.parse_args()
    try:
        catalog = Catalog().validate()
        if args.verify_source:
            checked, skipped = catalog.verify_source(args.verify_source)
            print(f"OK: {checked} pinned public source locations; {skipped} sources in other repositories not audited; no upstream code executed")
            return 0
        rendered = pages(catalog)
        stale = []
        for path, content in rendered.items():
            if args.emit:
                path.write_text(content, encoding="utf-8")
            elif not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(catalog.root)))
        require(not stale, "STALE: " + ", ".join(stale) + "; run tools/workload/build.py --emit")
        print(f"OK: {len(catalog.cases)} cases, {len(catalog.decisions)} workload decisions, {len(rendered)} generated pages")
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Cannot build workload knowledge: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
