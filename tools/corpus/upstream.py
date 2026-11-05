"""Reviewed PR reports and their links to decisions; never measured priors for a new workload."""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
from pathlib import PurePosixPath
import re
import subprocess

import yaml

DIRECTORY = "knowledge/_sources/aiter/pull_requests"
REPOSITORY = "https://github.com/ROCm/aiter"
SHA = re.compile(r"[0-9a-f]{40}")
ID = re.compile(r"[a-z][a-z0-9_]*")
LEVELS = {"unknown", "low", "medium", "high"}
OUTCOMES = {"beneficial", "no_gain", "regressed", "mixed", "constraint", "not_reported"}
CONTEXT = {"hardware", "gfx", "dtype", "shape_layout", "software", "baseline", "candidate",
           "baseline_commit", "candidate_commit", "scope", "protocol", "correctness"}
UNITS = {"latency": {"ns", "us", "ms", "s"}, "throughput": {"TFLOP/s", "GB/s", "tokens/s"},
         "speedup": {"x"}, "latency_reduction": {"percent"}, "throughput_increase": {"percent"}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, required, optional=(), label="record"):
    require(isinstance(value, dict) and set(required) <= value.keys()
            and not value.keys() - set(required) - set(optional), f"{label}: invalid fields")


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def strings(value, empty=False):
    return isinstance(value, list) and (empty or bool(value)) and all(nonempty(v) for v in value)


def timestamp(value):
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).utcoffset() is not None
    except ValueError:
        return False


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def confidence(value, label):
    fields(value, {"level", "reasons", "missing_evidence"}, label=label)
    require(isinstance(value["level"], str) and value["level"] in LEVELS
            and strings(value["reasons"]) and strings(value["missing_evidence"], empty=True),
            f"{label}: confidence needs a level, reasons and missing_evidence")


def speedup(metric):
    """Normalize one declared statistic only; never aggregate across shapes or scopes."""
    kind = metric["kind"]
    if kind == "latency":
        return metric["before"] / metric["after"]
    if kind == "throughput":
        return metric["after"] / metric["before"]
    if kind == "speedup":
        return metric["value"]
    if kind == "latency_reduction":
        return 1 / (1 - metric["value"] / 100)
    return 1 + metric["value"] / 100


def validate_metric(metric):
    fields(metric, {"kind", "unit", "statistic"}, {"before", "after", "value", "source"}, "metric")
    kind = metric["kind"]
    require(isinstance(kind, str) and kind in UNITS, "metric: unknown kind")
    require(isinstance(metric["unit"], str) and metric["unit"] in UNITS[kind], "metric: invalid unit")
    require(metric["statistic"] in {"single_case", "mean", "median", "min", "max", "geomean", "unspecified"},
            "metric: invalid statistic")
    values = {"before", "after"} if kind in {"latency", "throughput"} else {"value"}
    require(set(metric) - {"source"} == {"kind", "unit", "statistic"} | values, "metric: use before/after OR value")
    require(all(type(metric[k]) in {float, int} and math.isfinite(metric[k]) for k in values),
            "metric: values must be finite numbers, not missing data or booleans")
    if kind == "latency_reduction":
        require(metric["value"] < 100, "metric: latency reduction must be below 100 percent")
    elif kind == "throughput_increase":
        require(metric["value"] > -100, "metric: throughput increase must exceed -100 percent")
    else:
        require(all(metric[k] > 0 for k in values), "metric: timings, throughput and ratios must be positive")


def validate_record(record):
    fields(record, {"schema_version", "id", "number", "repository", "title", "state", "base_commit",
                    "head_commit", "merge_commit", "captured_at", "updated_at", "sources", "changes", "results"},
           label="PR")
    n = record["number"]
    require(type(n) is int and n > 0 and record["id"] == f"aiter_pr_{n}", "PR: invalid number/ID")
    require(record["schema_version"] == 1 and record["repository"] == REPOSITORY, "PR: unsupported schema/repository")
    require(nonempty(record["title"]) and record["state"] in {"open", "closed", "merged"}, "PR: invalid title/state")
    require(timestamp(record["captured_at"]) and timestamp(record["updated_at"]), "PR: timestamps need timezone")
    for key in ("base_commit", "head_commit", "merge_commit"):
        value = record[key]
        require((key == "merge_commit" and value is None) or isinstance(value, str) and SHA.fullmatch(value),
                f"PR: {key} needs a full commit")
    require((record["state"] == "merged") == (record["merge_commit"] is not None), "PR: merge state disagrees with merge commit")
    url = f"{REPOSITORY}/pull/{n}"
    sources = {}
    require(isinstance(record["sources"], list) and record["sources"], "PR: missing sources")
    for source in record["sources"]:
        fields(source, {"id", "url", "excerpt", "sha256", "captured_at", "updated_at"}, label="PR source")
        key = source["id"]
        require(isinstance(key, str) and ID.fullmatch(key) and key not in sources, "PR: duplicate/invalid source ID")
        require(source["url"] == url or (isinstance(source["url"], str) and source["url"].startswith(url + "#")),
                "PR source: use this PR or a specific comment anchor")
        require(nonempty(source["excerpt"]) and source["sha256"] == digest(source["excerpt"]),
                "PR source: excerpt fingerprint mismatch")
        require(timestamp(source["captured_at"]) and timestamp(source["updated_at"]), "PR source: timestamps need timezone")
        sources[key] = source
    changes = {}
    require(isinstance(record["changes"], list) and record["changes"], "PR: missing reviewed changes")
    for change in record["changes"]:
        fields(change, {"id", "commit", "parent_commit", "file", "before", "after", "summary"}, label="PR change")
        key = change["id"]
        require(isinstance(key, str) and ID.fullmatch(key) and key not in changes, "PR: duplicate/invalid change ID")
        for field in ("commit", "parent_commit"):
            require(isinstance(change[field], str) and SHA.fullmatch(change[field]), "PR change: full commits required")
        path = change["file"]
        require(nonempty(path) and not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts,
                "PR change: invalid upstream path")
        require(isinstance(change["before"], str) and nonempty(change["after"])
                and change["before"] != change["after"] and nonempty(change["summary"]), "PR change: needs distinct before/after excerpts")
        changes[key] = change
    results = {}
    require(isinstance(record["results"], list) and record["results"], "PR: needs a result or explicit not_reported record")
    for result in record["results"]:
        fields(result, {"id", "outcome", "summary", "source_refs", "context", "metrics", "confidence", "limitations"}, label="PR result")
        key = result["id"]
        require(isinstance(key, str) and ID.fullmatch(key) and key not in results, "PR: duplicate/invalid result ID")
        require(result["outcome"] in OUTCOMES and nonempty(result["summary"]), "PR result: invalid outcome/summary")
        require(strings(result["source_refs"]) and set(result["source_refs"]) <= sources.keys(), "PR result: unresolved source")
        fields(result["context"], CONTEXT, label="PR comparison context")
        context = result["context"]
        require(all(v is None or nonempty(v) for v in context.values()), "PR context: unknown fields must be null")
        require(context["scope"] in {"kernel", "operator", "end_to_end", "unknown"}, "PR context: invalid measurement scope")
        for field in ("baseline_commit", "candidate_commit"):
            require(context[field] is None or SHA.fullmatch(context[field]), "PR context: measured revisions must be full commits or null")
        require(isinstance(result["metrics"], list) and strings(result["limitations"], empty=True), "PR result: invalid metrics/limitations")
        if result["outcome"] == "not_reported":
            require(not result["metrics"], "PR result: not_reported cannot have metrics")
        for metric in result["metrics"]:
            validate_metric(metric)
            locator = metric.get("source")
            fields(locator, {"ref", "quote"}, label="metric source")
            require(isinstance(locator["ref"], str) and locator["ref"] in result["source_refs"]
                    and nonempty(locator["quote"]) and locator["quote"] in sources[locator["ref"]]["excerpt"],
                    "metric source: quote does not occur in referenced excerpt")
            # Detect transcription errors. Meaning, baseline identity and units still require review.
            tokens = re.findall(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?",
                                locator["quote"].replace("−", "-"))
            numbers = [float(token.replace(",", "")) for token in tokens]
            require(all(any(math.isclose(metric[k], number, rel_tol=1e-12) for number in numbers)
                        for k in ("before", "after", "value") if k in metric),
                    "metric source: numeric value does not occur in quoted report")
            ratio = speedup(metric)
            if result["outcome"] == "beneficial":
                require(ratio > 1, "PR result: beneficial outcome contradicts metric")
            elif result["outcome"] == "regressed":
                require(ratio < 1, "PR result: regressed outcome contradicts metric")
        confidence(result["confidence"], "PR effect")
        level = result["confidence"]["level"]
        # A public report alone cannot establish internal or independent reproduction.
        require(level != "high", "PR effect: high confidence requires separate reproduced experience; a PR report is insufficient")
        if result["outcome"] == "not_reported":
            require(not result["metrics"] and level == "unknown", "PR result: not_reported cannot have metrics or known effect confidence")
        else:
            require(level in {"low", "medium"}, "PR result: reported effects need low or medium confidence")
        if level == "medium":
            require(result["metrics"] and all(context[k] is not None for k in
                    ("hardware", "gfx", "dtype", "shape_layout", "baseline", "candidate", "protocol", "correctness"))
                    and context["scope"] != "unknown", "PR effect: medium confidence needs quantitative comparison context")
        missing = {k for k, v in context.items() if v is None}
        require(missing <= set(result["confidence"]["missing_evidence"]), "PR effect: missing context must be listed in missing_evidence")
        results[key] = result
    return record


def load(root):
    records = {}
    for path in sorted((root / DIRECTORY).glob("*.yaml")):
        record = validate_record(yaml.safe_load(path.read_text(encoding="utf-8")))
        require(path.stem == str(record["number"]), "PR record filename must match number")
        require(record["id"] not in records, "duplicate PR record")
        records[record["id"]] = record
    return records


def validate_entry(label, entry):
    """Local shape checks also run through the existing experience validator."""
    fields(entry, {"outcome", "conditions", "summary", "provenance", "upstream"}, label=label)
    require(entry["outcome"] in OUTCOMES and strings(entry["conditions"]) and nonempty(entry["summary"]),
            f"{label}: invalid upstream conclusion")
    require(entry["provenance"] == {"kind": "upstream_report", "references": [], "artifacts": "not_published",
                                    "verification": "not_reproduced"}, f"{label}: upstream reports cannot claim reproduced validation")
    link = entry["upstream"]
    fields(link, {"pr", "change_refs", "result_refs", "assessment"}, label=label)
    require(nonempty(link["pr"]) and strings(link["change_refs"]) and strings(link["result_refs"]), f"{label}: missing PR references")
    assessment = link["assessment"]
    fields(assessment, {"association", "attribution", "applies_to"}, label=label)
    confidence(assessment["association"], label + " association")
    require(assessment["association"]["level"] != "unknown", f"{label}: unresolved associations belong in review backlog")
    fields(assessment["applies_to"], {"axes", "backend"}, label=label)
    require(strings(assessment["applies_to"]["axes"]) and nonempty(assessment["applies_to"]["backend"]), f"{label}: missing target direction")
    attribution = assessment["attribution"]
    fields(attribution, {"scope", "confidence", "isolation_source_refs"}, label=label)
    require(attribution["scope"] in {"isolated", "bundle", "unclear"}
            and strings(attribution["isolation_source_refs"], empty=True), f"{label}: invalid attribution")
    confidence(attribution["confidence"], label + " attribution")
    if attribution["scope"] != "isolated":
        require(attribution["confidence"]["level"] in {"unknown", "low"}, f"{label}: bundle effects cannot establish independent attribution")
    else:
        require(attribution["isolation_source_refs"], f"{label}: isolated attribution needs explicit evidence")


def validate_cards(cards, records):
    for record in records.values():
        validate_record(record)
    for card in cards.values():
        label = card["id"]
        review = card.get("upstream_review")
        fields(review, {"status"}, {"checked_at", "summary"}, label + " upstream_review")
        require(review["status"] in {"unreviewed", "reviewed", "unresolved", "blocked"}, f"{label}: invalid PR review status")
        if review["status"] != "unreviewed":
            require(timestamp(review.get("checked_at")) and nonempty(review.get("summary")), f"{label}: reviewed/blocked state needs date and scope")
        entries = [e for e in card.get("experience", []) if e.get("provenance", {}).get("kind") == "upstream_report"]
        require(not entries or review["status"] in {"reviewed", "blocked"}, f"{label}: PR evidence requires a review state")
        require(review["status"] != "reviewed" or entries, f"{label}: reviewed state needs linked PR evidence")
        for entry in entries:
            validate_entry(label, entry)
            link = entry["upstream"]
            require(link["pr"] in records, f"{label}: unresolved PR reference")
            record = records[link["pr"]]
            changes = {c["id"] for c in record["changes"]}
            results = {r["id"]: r for r in record["results"]}
            require(set(link["change_refs"]) <= changes and set(link["result_refs"]) <= results.keys(), f"{label}: unresolved change/result reference")
            target = link["assessment"]["applies_to"]
            require(set(target["axes"]) <= set(card["axes"]) and target["backend"] in card["match"]["target_languages"], f"{label}: PR assessment targets an unrelated axis/backend")
            outcomes = {results[r]["outcome"] for r in link["result_refs"]}
            require(outcomes == {entry["outcome"]} or (entry["outcome"] == "mixed" and len(outcomes) > 1), f"{label}: conclusion contradicts referenced outcomes")
            attribution = link["assessment"]["attribution"]
            require(set(attribution["isolation_source_refs"]) <= {s["id"] for s in record["sources"]}, f"{label}: unresolved isolation evidence")
    return records


def verify_sources(records, checkout):
    """Audit reviewed before/after excerpts and parent links using Git objects, without upstream execution."""
    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True).stdout
    count = 0
    for record in records.values():
        for change in record["changes"]:
            # cat-file preserves real parents even at a shallow boundary.
            commit_text = git("cat-file", "-p", change["commit"])
            require(f"parent {change['parent_commit']}\n" in commit_text, "PR change: parent does not match commit")
            after = git("show", f"{change['commit']}:{change['file']}")
            require(change["after"] in after, "PR change: after excerpt does not resolve")
            tree = git("ls-tree", change["parent_commit"], "--", change["file"])
            before = git("show", f"{change['parent_commit']}:{change['file']}") if tree else ""
            require(change["before"] in before if change["before"] else not tree, "PR change: before excerpt does not resolve")
            require(change["after"] not in before and (not change["before"] or change["before"] not in after),
                    "PR change: excerpts do not identify a changed construct")
            count += 1
    return count
