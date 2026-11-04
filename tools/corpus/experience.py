"""Public conclusion summaries, deliberately separate from raw measurements and run history."""
import re

import upstream

FIELDS = {"outcome", "conditions", "summary", "provenance"}
CONTEXT_FIELDS = {"gfx", "software", "dtype", "shape_layout", "baseline", "protocol", "correctness"}
PROVENANCE_FIELDS = {"kind", "references", "artifacts", "verification"}
OUTCOMES = {"beneficial", "no_gain", "regressed", "mixed", "constraint"}
PINNED_SUMMARY = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/blob/[0-9a-f]{40}/[^\s#]+\.md")


def validate(label, entries):
    errors = []
    if not isinstance(entries, list):
        return [f"{label}: experience must be a list"]
    for i, entry in enumerate(entries):
        name = f"{label}.experience[{i}]"
        if isinstance(entry, dict) and isinstance(entry.get("provenance"), dict) and entry["provenance"].get("kind") == "upstream_report":
            try:
                upstream.validate_entry(name, entry)
            except ValueError as error:
                errors.append(str(error))
            continue
        if not isinstance(entry, dict) or not FIELDS <= set(entry) or set(entry) - FIELDS - {"context"}:
            errors.append(f"{name}: expected public summary fields {sorted(FIELDS)}")
            continue
        if not isinstance(entry["outcome"], str) or entry["outcome"] not in OUTCOMES:
            errors.append(f"{name}: unknown experience outcome")
        conditions = entry["conditions"]
        if not isinstance(conditions, list) or not conditions or not all(isinstance(x, str) and x.strip() for x in conditions):
            errors.append(f"{name}: experience needs explicit conditions")
        if not isinstance(entry["summary"], str) or not entry["summary"].strip():
            errors.append(f"{name}: experience needs a conclusion summary")
        provenance = entry["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_FIELDS:
            errors.append(f"{name}: invalid experience provenance")
            continue
        refs = provenance["references"]
        if not isinstance(refs, list) or not all(isinstance(x, str) and PINNED_SUMMARY.fullmatch(x) for x in refs):
            errors.append(f"{name}: historical summaries need pinned public references")
        if provenance["artifacts"] != "not_published":
            errors.append(f"{name}: raw run artifacts are not part of the public corpus")
        if provenance["kind"] == "historical_summary":
            if not refs or provenance["verification"] != "not_reproduced":
                errors.append(f"{name}: historical summaries need attribution and cannot claim reproduced validation")
        elif provenance["kind"] == "private_summary":
            if provenance["verification"] != "internally_validated":
                errors.append(f"{name}: private summaries require internal validation")
            context = entry.get("context")
            if (not isinstance(context, dict) or set(context) != CONTEXT_FIELDS
                    or not all(isinstance(v, str) and v.strip() for v in context.values())):
                errors.append(f"{name}: internally validated summaries need a complete public comparison context")
        else:
            errors.append(f"{name}: unknown experience provenance kind")
    return errors
