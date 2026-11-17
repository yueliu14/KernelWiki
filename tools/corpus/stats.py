#!/usr/bin/env python3
"""Generate evidence coverage and scoped reported effects from the validated corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from html import escape
import json
import os

import upstream
from validate import Catalog

DIRECTORY = "knowledge/statistics"
OUTCOMES = ("beneficial", "regressed", "no_gain", "mixed", "constraint", "not_reported")


def counts(values):
    return dict(sorted(Counter(values).items()))


def coverage(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def result_key(pr_id, result_id):
    return f"{pr_id}:{result_id}"


def collect(catalog):
    """Use canonical IDs, never reference multiplicity, as the counting unit.

    The caller supplies a validated Catalog. Results are upstream reports; their
    metrics are normalized individually without pooling contexts or statistics.
    """
    results = {}
    for pr_id, pr in sorted(catalog.upstream.items()):
        for result in pr["results"]:
            key = result_key(pr_id, result["id"])
            results[key] = {
                "id": key, "pr_id": pr_id, "pr_number": pr["number"],
                "pr_state": pr["state"], "verification": "not_reproduced",
                "evidence_kind": "upstream_report",
                "evidence_page": f"{upstream.DIRECTORY}/{pr['number']}.md#{result['id']}",
                "outcome": result["outcome"], "summary": result["summary"],
                "context": deepcopy(result["context"]),
                "null_context_fields": sorted(k for k, v in result["context"].items() if v is None),
                "confidence": deepcopy(result["confidence"]),
                "limitations": list(result["limitations"]),
                "metrics": [dict(deepcopy(m), speedup=upstream.speedup(m)) for m in result["metrics"]],
                "linked_card_ids": [], "associations": [],
            }

    cards = []
    experience_kinds, experience_outcomes = [], []
    for card_id, card in sorted(catalog.cards.items()):
        entries = card.get("experience", [])
        refs = set()
        for entry in entries:
            kind = entry["provenance"]["kind"]
            experience_kinds.append(kind)
            experience_outcomes.append(entry["outcome"])
            if kind != "upstream_report":
                continue
            association = entry["upstream"]
            for ref in association["result_refs"]:
                key = result_key(association["pr"], ref)
                refs.add(key)
                row = results[key]
                row["linked_card_ids"].append(card_id)
                row["associations"].append({
                    "card_id": card_id, "conditions": list(entry["conditions"]),
                    "assessment": deepcopy(association["assessment"]),
                })
        numeric_refs = sorted(key for key in refs if results[key]["metrics"])
        cards.append({
            "id": card_id, "owner": catalog.card_owners[card_id],
            "page": str((catalog.card_directory(card_id) / "decisions.md").relative_to(catalog.root)) + f"#{card_id}",
            "type": card["type"], "evidence_level": card["evidence_level"],
            "axes": list(card["axes"]), "match": deepcopy(card["match"]),
            "upstream_review": card["upstream_review"]["status"],
            "experience_count": len(entries),
            "experience_by_kind": counts(e["provenance"]["kind"] for e in entries),
            "upstream_result_refs": sorted(refs), "numeric_upstream_result_refs": numeric_refs,
            "internally_validated_summary_count": sum(
                e["provenance"]["kind"] == "private_summary"
                and e["provenance"]["verification"] == "internally_validated" for e in entries),
        })

    rows = list(results.values())
    for row in rows:
        row["linked_card_ids"] = sorted(set(row["linked_card_ids"]))
    numeric = [r for r in rows if r["metrics"]]
    by_scope = {}
    for scope in sorted({r["context"]["scope"] for r in rows}):
        group = [r for r in rows if r["context"]["scope"] == scope]
        by_scope[scope] = {
            "result_count": len(group),
            "numeric_result_count": sum(bool(r["metrics"]) for r in group),
            "outcomes": counts(r["outcome"] for r in group),
            "result_refs": [r["id"] for r in group],
        }

    owners = []
    for owner in sorted(set(catalog.card_owners.values())):
        owned = [c for c in cards if c["owner"] == owner]
        owned_refs = sorted({ref for c in owned for ref in c["upstream_result_refs"]})
        owners.append({
            "owner": owner, "card_count": len(owned),
            "cards_with_experience": sum(c["experience_count"] > 0 for c in owned),
            "cards_with_numeric_upstream_results": sum(bool(c["numeric_upstream_result_refs"]) for c in owned),
            "cards_with_internal_validation": sum(c["internally_validated_summary_count"] > 0 for c in owned),
            "unique_linked_results": len(owned_refs), "result_refs": owned_refs,
        })

    summary = {
        "operators": len(catalog.operators), "canonical_cards": len(cards),
        "source_observations": len(catalog.sources), "shipped_groups": len(catalog.seeds),
        "cards_by_type": counts(c["type"] for c in cards),
        "cards_by_evidence_level": counts(c["evidence_level"] for c in cards),
        "cards_by_review_status": counts(c["upstream_review"] for c in cards),
        "experience_entries": len(experience_kinds),
        "experience_entries_by_kind": counts(experience_kinds),
        "experience_entries_by_outcome": counts(experience_outcomes),
        "canonical_pr_reports": len(catalog.upstream),
        "reports_by_lifecycle": counts(p["state"] for p in catalog.upstream.values()),
        "canonical_upstream_results": len(rows), "numeric_upstream_results": len(numeric),
        "metric_count": sum(len(r["metrics"]) for r in rows),
        "results_by_outcome": counts(r["outcome"] for r in rows),
        "results_by_effect_confidence": counts(r["confidence"]["level"] for r in rows),
        "results_by_gfx": counts(r["context"]["gfx"] or "unknown" for r in rows),
        "metrics_by_statistic": counts(m["statistic"] for r in rows for m in r["metrics"]),
        "null_context_field_counts": counts(k for r in rows for k in r["null_context_fields"]),
        "unlinked_upstream_results": sum(not r["linked_card_ids"] for r in rows),
        "coverage": {
            "cards_with_experience": coverage(sum(c["experience_count"] > 0 for c in cards), len(cards)),
            "cards_with_upstream_results": coverage(sum(bool(c["upstream_result_refs"]) for c in cards), len(cards)),
            "cards_with_numeric_upstream_results": coverage(sum(bool(c["numeric_upstream_result_refs"]) for c in cards), len(cards)),
            "cards_with_internal_validation": coverage(sum(c["internally_validated_summary_count"] > 0 for c in cards), len(cards)),
            "upstream_results_with_metrics": coverage(len(numeric), len(rows)),
        },
    }
    return {"schema_version": 1, "population": "catalog_operator_and_general_cards_and_aiter_pr_reports",
            "summary": summary, "owners": owners, "results_by_scope": by_scope,
            "cards": cards, "results": rows}


def safe(value):
    return escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def relative_link(label, target, page, root):
    path, sep, fragment = target.partition("#")
    relative = os.path.relpath(root / path, page.parent).replace(os.sep, "/")
    return f"[{safe(label)}]({relative}{sep}{fragment})"


def fraction_text(value):
    suffix = f" ({value['fraction']:.1%})" if value["fraction"] is not None else " (undefined)"
    return f"{value['numerator']}/{value['denominator']}" + suffix


def render_overview(catalog, data):
    page = catalog.root / DIRECTORY / "README.md"
    summary = data["summary"]
    out = ["# Corpus evidence statistics", "",
           "GENERATED by `tools/corpus/stats.py` (also included in `build.py`). Edit canonical YAML inputs and regenerate.", "",
           "[Reported effects](reported_effects.md) · [Machine-readable data](statistics.json) · "
           "[Definitions and maintenance](../../tools/corpus/statistics.md)", "",
           "Population: canonical operator/general cards and canonical AITER PR reports. Workload case pages, "
           "top-model histories and unstructured Markdown are outside this report. Counts describe the evidence "
           "recorded in the corpus, not all experiments performed upstream or the effectiveness of an agent.", "",
           "## Inventory", "", "| Record | Count |", "|---|---:|"]
    for key in ("operators", "canonical_cards", "source_observations", "shipped_groups", "experience_entries",
                "canonical_pr_reports", "canonical_upstream_results", "numeric_upstream_results", "metric_count"):
        out.append(f"| {key} | {summary[key]} |")
    out += ["", "## Evidence coverage", "", "| Coverage | Numerator / denominator |", "|---|---:|"]
    out += [f"| {key} | {fraction_text(value)} |" for key, value in summary["coverage"].items()]
    out += ["", "Numerical coverage means a linked report contains at least one encoded metric. It does not "
            "mean that every claim on the card has been quantified or that the card caused that improvement. "
            "Internal validation counts public summaries claiming internal validation; it is not public reproduction. "
            "Missing measurements are unknown, not 1×. No attempt-level success rate or pooled speedup is inferred.", ""]
    for title, key in (("Card types", "cards_by_type"), ("Upstream review coverage", "cards_by_review_status"),
                       ("Experience provenance", "experience_entries_by_kind"),
                       ("PR lifecycle", "reports_by_lifecycle"),
                       ("Reported result outcomes", "results_by_outcome"),
                       ("Reported effect confidence", "results_by_effect_confidence"),
                       ("Reported metric statistics", "metrics_by_statistic"),
                       ("Explicitly null comparison fields", "null_context_field_counts")):
        out += [f"## {title}", "", "| Category | Count |", "|---|---:|"]
        out += [f"| {safe(name)} | {n} |" for name, n in summary[key].items()]
        if not summary[key]:
            out.append("| None recorded | 0 |")
        out.append("")
    out += ["Null-field counts are only a syntactic missing-data audit. Non-null text may still disclose unknown "
            "versions, incomplete shapes or unspecified repetitions; it is not a completeness or comparability certificate. "
            "See each result's context, missing-evidence list and limitations.", "",
            "## Results by measurement scope", "",
            "Each cell counts canonical result IDs. A result may contain multiple metrics, including different statistics.", "",
            "| Scope | Results | With metrics | " + " | ".join(OUTCOMES) + " |",
            "|---|---:|---:|" + "---:|" * len(OUTCOMES)]
    for scope, group in data["results_by_scope"].items():
        out.append(f"| {scope} | {group['result_count']} | {group['numeric_result_count']} | " +
                   " | ".join(str(group["outcomes"].get(o, 0)) for o in OUTCOMES) + " |")
    out += ["", "## Coverage by canonical owner", "",
            "Cards are assigned to one owner. Results are deduplicated within each owner; a result linked by "
            "several owners appears in several rows, so result columns must not be summed across owners.", "",
            "| Owner | Cards | With experience | With numerical PR evidence | With internal validation | Unique linked results |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in data["owners"]:
        out.append("| " + " | ".join(str(row[k]) for k in (
            "owner", "card_count", "cards_with_experience", "cards_with_numeric_upstream_results",
            "cards_with_internal_validation", "unique_linked_results")) + " |")
    out += ["", "## Card evidence index", "",
            "Result references establish an association. Follow them for conditions, reported effects and "
            "bundle/isolated attribution; an associated bundle gain is not an individual card's gain.", "",
            "| Card | Type | Review | Experience entries | Numerical result refs / all result refs | Internal summaries |",
            "|---|---|---|---:|---|---:|"]
    for card in data["cards"]:
        refs = ", ".join(relative_link(ref, next(r["evidence_page"] for r in data["results"] if r["id"] == ref), page, catalog.root)
                         for ref in card["upstream_result_refs"]) or "—"
        out.append(f"| {relative_link(card['id'], card['page'], page, catalog.root)} | {card['type']} | "
                   f"{card['upstream_review']} | {card['experience_count']} | "
                   f"{len(card['numeric_upstream_result_refs'])}/{len(card['upstream_result_refs'])}: {refs} | "
                   f"{card['internally_validated_summary_count']} |")
    return page, "\n".join(out) + "\n"


def render_effects(catalog, data):
    page = catalog.root / DIRECTORY / "reported_effects.md"
    out = ["# Scoped upstream reported effects", "",
           "GENERATED by `tools/corpus/stats.py` (also included in `build.py`).", "",
           "[Coverage overview](README.md) · [Full contexts and attribution](statistics.json)", "",
           "Each row is one reported metric, or a result with no encoded metric. All are upstream reports, "
           "not reproduced measurements. Ratios normalize one metric only: latency before/after, throughput "
           "after/before, latency reduction 1/(1−p/100), throughput increase 1+p/100. "
           "Reported mean, median, maximum and other statistics remain distinct. No cross-result average, "
           "ranking, component credit or success probability is computed.", "",
           "The result link contains full shape/layout, software, numerical checks, timing protocol, measured "
           "revision gaps and source quotations. PR lifecycle does not establish runtime engagement. "
           "Rounded equality does not prove statistical equivalence. Unknown scope remains separate.", ""]
    for scope in data["results_by_scope"]:
        out += [f"## {scope}", "",
                "| Result / metric | PR state | Hardware | Baseline → candidate | Reported outcome | Statistic | Raw metric | Speedup | Effect confidence | Attribution scopes |",
                "|---|---|---|---|---|---|---|---:|---|---|"]
        for row in data["results"]:
            context = row["context"]
            if context["scope"] != scope:
                continue
            attribution = ", ".join(sorted({a["assessment"]["attribution"]["scope"] for a in row["associations"]})) or "unlinked"
            for i, metric in enumerate(row["metrics"] or [None]):
                if metric is None:
                    statistic, raw, ratio = "—", "No encoded metric", "—"
                else:
                    statistic = metric["statistic"]
                    values = (f"{metric['before']:g} → {metric['after']:g}" if "before" in metric else f"{metric['value']:g}")
                    raw = f"{metric['kind']}: {values} {metric['unit']}"
                    ratio = f"{metric['speedup']:.4g}×"
                label = row["id"] + (f" / {i + 1}" if metric else "")
                out.append("| " + " | ".join([
                    relative_link(label, row["evidence_page"], page, catalog.root), row["pr_state"],
                    safe(context["hardware"] or "unknown"),
                    safe(f"{context['baseline'] or 'unknown'} → {context['candidate'] or 'unknown'}"),
                    row["outcome"], statistic, safe(raw), ratio, row["confidence"]["level"], attribution,
                ]) + " |")
        out.append("")
    return page, "\n".join(out)


def pages(catalog):
    data = collect(catalog)
    outputs = dict([render_overview(catalog, data), render_effects(catalog, data)])
    outputs[catalog.root / DIRECTORY / "statistics.json"] = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog = Catalog().validate()
    stale = []
    for path, text in pages(catalog).items():
        if args.emit:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        elif not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(str(path.relative_to(catalog.root)))
    if stale:
        parser.exit(1, "STALE: " + ", ".join(stale) + "; run tools/corpus/stats.py --emit\n")
    print("OK: corpus evidence statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
