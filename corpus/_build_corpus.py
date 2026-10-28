#!/usr/bin/env python3
"""Validate and render every catalogued family, without a GPU or consumer checkout.

    python3 corpus/_build_corpus.py --check
    python3 corpus/_build_corpus.py --family attention --emit
    python3 corpus/_build_corpus.py --family attention --verify-source /path/to/aiter

GEMM retains its existing seed-table presentation. Other families use the same card
schema and development-order renderer, with their own axes and source evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess

import _extract_impl_facts as extractor
import _normalize_performance_decisions as normalizer
import _render_decisions as decisions
import _render_facts as facts

HERE = Path(__file__).resolve().parent


def families():
    catalog = facts.load_yaml(str(HERE / "catalog.yaml"))
    result = catalog["families"]
    owners = {}
    for family in result:
        for pattern in set([family["id"], *family["patterns"]]):
            if pattern in owners:
                raise ValueError(f"ambiguous operator {pattern}: {owners[pattern]} / {family['id']}")
            owners[pattern] = family["id"]
    return result


def path_for(family, field):
    path = (HERE / family[field]).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f"{family['id']}.{field} escapes corpus/")
    return path


def load_inputs(family):
    def read(field):
        return facts.load_yaml(str(path_for(family, field))) if field in family else {}
    return tuple(read(field) for field in (
        "decisions", "source_evidence", "tuned_evidence", "tuned_db_evidence", "performance_axes"))


def validate_family(family, inputs):
    cards, source, tuned, database, axes = inputs
    name = family["id"]
    for data in (cards, source):
        if data.get("provenance", {}).get("operator_family") != name:
            raise ValueError(f"{name}: input provenance names another family")
    if axes.get("family") != name:
        raise ValueError(f"{name}: axes name another family")
    normalizer.validate_schema(axes)
    decisions._checked(*inputs)
    records = source.get("source_evidence", [])
    if not records or not cards.get("cards"):
        raise ValueError(f"{name}: empty source evidence or decision cards")
    provenance = source["provenance"]
    if not re.fullmatch(r"[0-9a-f]{40}", provenance.get("aiter_commit", "")):
        raise ValueError(f"{name}: source needs a full commit identity")
    if provenance.get("aiter_dirty_sources"):
        raise ValueError(f"{name}: dirty source evidence")
    vocabulary = set(facts.load_yaml(str(HERE / "catalog.yaml"))["authoring_languages"])
    mapped = {category for axis in axes["axes"] for mapping in axis.get("source_mappings", [])
              for category in mapping.get("categories", [])}
    source_fields = {"category", "language", "file", "line", "end_line", "match", "captured",
                     "arch_scope", "evidence", "evidence_id", "unreproducible"}
    for record in records:
        if set(record) - source_fields:
            raise ValueError(f"{name}: unexpected source fields {sorted(set(record) - source_fields)}")
        if record.get("unreproducible") or record["evidence_id"] != extractor.source_evidence_id(record):
            raise ValueError(f"{name}: stale/unreproducible source ID {record.get('evidence_id')}")
        if record["language"] not in vocabulary or record["category"] not in mapped:
            raise ValueError(f"{name}: unknown language or unmapped category in {record['evidence_id']}")
        if not isinstance(record.get("line"), int) or record["line"] <= 0:
            raise ValueError(f"{name}: invalid source line")
    ids = {card["id"] for card in cards["cards"]}
    if any(card.get("status") != "candidate_only" for card in cards["cards"]):
        raise ValueError(f"{name}: source-observed cards must remain candidate_only")
    for dependency in axes.get("dependencies", []):
        if set(dependency.get("decision_refs", [])) - ids:
            raise ValueError(f"{name}: dependency references an unknown card")


def source_url(source, record):
    origin = source["provenance"]["aiter_origin"].removesuffix(".git").rstrip("/")
    commit = source["provenance"]["aiter_commit"]
    extent = f"-L{record['end_line']}" if record.get("end_line") else ""
    return f"{origin}/blob/{commit}/{record['file']}#L{record['line']}{extent}"


def cell(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", " ")


def validation_sources(family, source):
    out = ["## Upstream correctness references", "",
           "These are pinned test sources, not test results. No GPU test was run to produce this corpus.", ""]
    for file in family.get("validation_sources", []):
        out.append(f"- [{file}]({source_url(source, dict(file=file, line=1))}).")
    return out + [""]


def render_source(family, source):
    rows = source["source_evidence"]
    out = [f"# {family['title']} — source evidence", "",
           "GENERATED by `corpus/_build_corpus.py`; do not hand-edit.", "",
           f"AITER commit: `{source['provenance']['aiter_commit']}`. "
           f"{len(source['coverage'])} scanned files; {len(rows)} observations.", "",
           "Counts describe source coverage, not performance or runtime engagement. "
           "A `symbol_declaration` locates a function and its full extent; its presence does not "
           "establish the behavior of every branch. Read the cited implementation and the card's conditions.",
           "", f"Start with [development decisions]({family['decision_document']}).", "",
           "## Source-evidence coverage", "", "| language | observations |", "|---|---:|"]
    out += [f"| {lang} | {count} |" for lang, count in sorted(Counter(r['language'] for r in rows).items())]
    out += ["", "## Scanned files", "",
            "The scope language is the file's backend context. In mixed Python files, explicitly "
            "decorated Triton/Gluon/FlyDSL functions and their nested bodies retain their own language.", "",
            "| scope language | source | observations |", "|---|---|---:|"]
    for row in source["coverage"]:
        out.append(f"| {row['language']} | [{row['file']}]({source_url(source, dict(row, line=1))}) | "
                   f"{row['evidence_records']} |")
    out += ["", "## Observations", ""]
    for category in sorted({r["category"] for r in rows}):
        out += [f"### {category}", "", "| ID | language | source | observed construct |", "|---|---|---|---|"]
        for row in rows:
            if row["category"] == category:
                out.append(f"| `{row['evidence_id']}` | {row['language']} | "
                           f"[{facts.location(row)}]({source_url(source, row)}) | "
                           f"{cell(row['match'])} |")
        out.append("")
    out += ["## Coverage limits", "", *family.get("coverage_notes", []), ""]
    out += [f"- {row['language']}: {row['subtree']} — {row['why']}" for row in source.get("missing", [])]
    out += [""] + validation_sources(family, source)
    out += ["", "## Sources", "", f"- [Extraction scope]({family['source_scope']}).",
            f"- [Verbatim evidence and provenance]({family['source_evidence']}).", ""]
    return "\n".join(out)


def render_decisions(family, inputs):
    card_data, source, tuned, database, axes = inputs
    cards, rows, shipped = decisions._checked(*inputs)
    by_id = {r["evidence_id"]: r for r in rows}
    out = [f"# {family['title']} — development decisions", "",
           "GENERATED by `corpus/_build_corpus.py`; edit the YAML inputs instead.", "",
           f"{len(cards)} source-observed cards, grounded at AITER "
           f"`{source['provenance']['aiter_commit']}`. No target-box speedup is claimed.", "",
           f"[Source evidence]({family['source_document']}) · [Scope]({family['source_scope']}) "
           f"· [Axes and dependencies]({family['performance_axes']})", "",
           "Machine matching covers only `match`; prose conditions still require review. "
           "Kernel declarations, registered host routes and measured implementations are different evidence.", ""]
    out += ["## Design traits", "",
            "`match.requires` names a property the implementation must have before a card's "
            "conditions can be evaluated. Missing design information does not establish applicability.", ""]
    out += [f"- `{name}` — {description}" for name, description in card_data.get("traits", {}).items()]
    out.append("")
    out += decisions.render_steps(cards, axes, by_id, shipped)
    out += validation_sources(family, source)
    out += ["## Coverage limits", "", *family.get("coverage_notes", []), "",
            "## Sources", "", f"- [Curated decisions]({family['decisions']}).",
            f"- [Source evidence]({family['source_evidence']}).", ""]
    text = "\n".join(out)
    for row in rows:
        location = facts.location(row)
        text = text.replace(f"`{location}`", f"[{location}]({source_url(source, row)})")
    return text


def pages(family):
    inputs = load_inputs(family)
    validate_family(family, inputs)
    if family["id"] == "gemm":
        return {path_for(family, "source_document"): facts.render(inputs[1], inputs[2]),
                path_for(family, "decision_document"): decisions.render(*inputs),
                HERE / family["seed_documents"][0]: decisions.render_triton_page(*inputs)}
    return {path_for(family, "source_document"): render_source(family, inputs[1]),
            path_for(family, "decision_document"): render_decisions(family, inputs)}


def verify_source(family, checkout):
    """Re-resolve quoted lines and match text against the named git objects, not HEAD."""
    source = load_inputs(family)[1]
    commit = source["provenance"]["aiter_commit"]
    cache = {}
    for record in source["source_evidence"]:
        file = record["file"]
        if file not in cache:
            cache[file] = subprocess.run(["git", "-C", str(checkout), "show", f"{commit}:{file}"],
                                        capture_output=True, text=True, check=True).stdout.splitlines()
        lines = cache[file]
        for quote in record["evidence"]:
            number, text = quote.split(": ", 1)
            if not 1 <= int(number) <= len(lines) or lines[int(number)-1].rstrip() != text:
                raise ValueError(f"{record['evidence_id']}: quote does not resolve at {commit}:{file}:{number}")
        if not 1 <= record["line"] <= record.get("end_line", record["line"]) <= len(lines):
            raise ValueError(f"{record['evidence_id']}: extent outside pinned source")
    for file in family.get("validation_sources", []):
        subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", f"{commit}:{file}"],
                       capture_output=True, text=True, check=True)
    return len(source["source_evidence"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", help="one catalog family; omitted means every family")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--verify-source", type=Path)
    args = parser.parse_args()
    selected = [f for f in families() if args.family is None or f["id"] == args.family]
    if not selected:
        parser.error(f"unknown corpus family: {args.family}")
    stale = []
    for family in selected:
        rendered = pages(family)
        if args.verify_source:
            print(f"OK: {family['id']}: {verify_source(family, args.verify_source)} pinned source citations")
            continue
        for path, text in rendered.items():
            if args.emit:
                path.write_text(text, encoding="utf-8")
            elif not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path.relative_to(HERE)))
        print(f"{family['id']}: validated {len(rendered)} pages")
    if stale:
        parser.exit(1, f"STALE: {', '.join(stale)}; run corpus/_build_corpus.py --emit\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
