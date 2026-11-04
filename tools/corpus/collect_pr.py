#!/usr/bin/env python3
"""Capture public AITER PR evidence for review, or discover PRs associated with a commit.

Writes only to an explicitly supplied private path. It does not select directions,
infer benchmark revisions, assign confidence, execute AITER or update a checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request

import yaml

import upstream

API = "https://api.github.com/repos/ROCm/aiter"
ROOT = Path(__file__).resolve().parents[2]


def request(endpoint):
    req = urllib.request.Request(API + "/" + endpoint, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "KernelWiki-corpus",
        "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def paginate(endpoint, fetch=request):
    rows, page = [], 1
    while True:
        batch = fetch(f"{endpoint}?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise ValueError("GitHub did not return a collection")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
        page += 1


def capture(number, fetch=request):
    if type(number) is not int or number <= 0:
        raise ValueError("PR number must be positive")
    pr = fetch(f"pulls/{number}")
    result = {"schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(), "pr": pr}
    for key, endpoint in (("comments", f"issues/{number}/comments"), ("reviews", f"pulls/{number}/reviews"),
                          ("review_comments", f"pulls/{number}/comments"), ("commits", f"pulls/{number}/commits"),
                          ("files", f"pulls/{number}/files")):
        result[key] = paginate(endpoint, fetch)
    if len(result["files"]) != pr["changed_files"] or len(result["commits"]) != pr["commits"]:
        raise ValueError("GitHub returned incomplete files/commits; do not treat this capture as a complete review")
    return result


def verify_capture(record, captured):
    """Check current metadata and possibly historical quotations against preserved captures."""
    upstream.validate_record(record)
    captures = captured if isinstance(captured, list) else [captured]
    if any(c["pr"]["number"] != record["number"] for c in captures):
        raise ValueError("capture belongs to another PR")
    current = next((c for c in captures if c["captured_at"] == record["captured_at"]), None)
    if current is None:
        raise ValueError("missing capture for current PR metadata")
    pr = current["pr"]
    expected_state = "merged" if pr["merged_at"] else pr["state"]
    expected = {"number": pr["number"], "state": expected_state, "head_commit": pr["head"]["sha"],
                "base_commit": pr["base"]["sha"], "merge_commit": pr["merge_commit_sha"] if pr["merged_at"] else None,
                "updated_at": pr["updated_at"], "captured_at": current["captured_at"], "title": pr["title"]}
    if any(record[key] != value for key, value in expected.items()):
        raise ValueError("PR metadata does not match the captured revision")
    texts, memberships = {}, []
    for snapshot in captures:
        item = snapshot["pr"]
        texts[(snapshot["captured_at"], item["html_url"])] = (item.get("body") or "", item["updated_at"])
        for collection in ("comments", "reviews", "review_comments"):
            for row in snapshot.get(collection, []):
                texts[(snapshot["captured_at"], row["html_url"])] = (
                    row.get("body") or "", row.get("updated_at") or row["submitted_at"])
        known_commits = {row["sha"] for row in snapshot["commits"]}
        if item["merged_at"]:
            known_commits.add(item["merge_commit_sha"])
        memberships.append((known_commits, {row["filename"] for row in snapshot["files"]}))
    for source in record["sources"]:
        key = (source["captured_at"], source["url"])
        if key not in texts or source["excerpt"] not in texts[key][0].replace("\r\n", "\n"):
            raise ValueError("PR quotation does not occur in the captured source")
        if source["updated_at"] != texts[key][1]:
            raise ValueError("PR source timestamp does not match the captured revision")
    for change in record["changes"]:
        if not any(change["commit"] in commits and change["file"] in files for commits, files in memberships):
            raise ValueError("PR change does not belong to captured commits/files")
    return len(record["sources"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pr", type=int)
    modes.add_argument("--commit", help="full change commit; associations are candidates, not causal proof")
    modes.add_argument("--check-evidence", type=Path, help="check canonical YAML against --capture offline")
    parser.add_argument("--output", type=Path, help="private JSON destination outside this repository")
    parser.add_argument("--capture", type=Path, action="append", help="saved API JSON; repeat to audit retained historical excerpts")
    args = parser.parse_args()
    try:
        if args.check_evidence:
            if args.capture is None:
                raise ValueError("--check-evidence requires --capture")
            count = verify_capture(yaml.safe_load(args.check_evidence.read_text()),
                                   [json.loads(path.read_text()) for path in args.capture])
            print(f"OK: {count} captured PR quotations and metadata")
            return 0
        if args.output is None or args.output.resolve().is_relative_to(ROOT):
            raise ValueError("provide --output outside the public repository")
        if args.commit:
            if not upstream.SHA.fullmatch(args.commit):
                raise ValueError("--commit requires a full commit")
            result = {"commit": args.commit, "pull_requests": paginate(f"commits/{args.commit}/pulls")}
        else:
            result = capture(args.pr)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Captured review inputs in {args.output}; no corpus or checkpoint was changed.")
        return 0
    except (ValueError, OSError, KeyError, urllib.error.URLError) as error:
        parser.exit(1, f"Cannot collect PR evidence: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
