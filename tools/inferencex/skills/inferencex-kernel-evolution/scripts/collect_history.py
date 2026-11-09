#!/usr/bin/env python3
"""Collect public AgentX history and compare adjacent Docker tags without interpolation."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ORIGIN = "https://inferencex.semianalysis.com"
REPO = "SemiAnalysisAI/InferenceX"
CONFIG_FIELDS = (
    "hardware", "framework", "model", "precision", "spec_method", "disagg",
    "is_multinode", "prefill_tp", "prefill_ep", "prefill_dp_attention",
    "prefill_num_workers", "decode_tp", "decode_ep", "decode_dp_attention",
    "decode_num_workers", "num_prefill_gpu", "num_decode_gpu", "benchmark_type",
    "offload_mode", "isl", "osl", "conc",
)
METRIC_CONFIG_FIELDS = ("pp", "dcp_size", "pcp_size", "spec_algorithm", "spec_draft_model", "kv_offloading")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(url, destination, offline=False):
    meta = destination.with_suffix(destination.suffix + ".meta.json")
    if not destination.exists():
        if offline:
            raise FileNotFoundError(f"Offline input unavailable: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if url.startswith("https://api.github.com/"):
            data = subprocess.run(["gh", "api", url], check=True, capture_output=True).stdout
        else:
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
        destination.write_bytes(data)
        save_json(meta, {"url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                         "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    elif not meta.exists():
        # Pre-existing exploratory inputs have only a filesystem timestamp, not a verified fetch time.
        data = destination.read_bytes()
        save_json(meta, {"url": url, "retrieved_at_utc": None,
                         "cache_mtime_utc": datetime.fromtimestamp(destination.stat().st_mtime, timezone.utc).isoformat(),
                         "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    receipt = json.loads(meta.read_text())
    if receipt["url"] != url or receipt["sha256"] != hashlib.sha256(destination.read_bytes()).hexdigest():
        raise ValueError(f"Cache URL/hash mismatch: {destination}")
    return json.loads(destination.read_text())


def normalize(row):
    metrics = row["metrics"]
    full_response = any(k.endswith("_full_response_itl") and isinstance(v, (int, float))
                        for k, v in metrics.items())
    field = "p90_full_response_itl" if full_response else "p90_itl"
    itl = metrics.get(field)
    x = 1 / itl if isinstance(itl, (float, int)) and itl > 0 else None
    y = metrics.get("output_tput_per_gpu")
    if row["disagg"]:
        raise ValueError("This pilot only supports aggregate deployments; select an aggregate series")
    assert row["num_prefill_gpu"] == row["num_decode_gpu"], row["id"]
    chips = row["num_decode_gpu"]
    assert chips > 0
    if metrics.get("output_tput_tps") is not None and y is not None:
        assert math.isclose(metrics["output_tput_tps"] / chips, y, abs_tol=1e-4, rel_tol=1e-5), row["id"]
    match = re.search(r"/actions/runs/(\d+)", row["run_url"])
    if not match:
        raise ValueError(f"No GitHub run ID: {row['run_url']}")
    image = row["image"]
    if not image:
        raise ValueError(f"Image missing: {row['id']}")
    config = {key: row.get(key) for key in CONFIG_FIELDS}
    config.update({key: metrics.get(key) for key in METRIC_CONFIG_FIELDS})
    return {
        "id": row["id"], "date": row["date"], "run_started_at": row["run_started_at"],
        "curve_date": row.get("curve_date"), "curve_workflow_run_id": row.get("curve_workflow_run_id"),
        "github_run_id": match.group(1), "run_url": row["run_url"],
        "github_run_attempt": int(row["run_url"].split("/attempts/")[1].split("/")[0]) if "/attempts/" in row["run_url"] else None,
        "image_name": image.split("@")[0], "image_reference": image,
        "image_digest": image.split("@", 1)[1] if "@" in image else None,
        "recipe_fingerprint": row.get("recipe_fingerprint"), "config": config,
        "p90_interactivity_tok_s_user": x, "interactivity_source_field": field,
        "p90_itl_seconds": itl, "output_tok_s_chip": y, "physical_chips": chips,
        "dataset_revision": None, "model_revision": None,
        "comparison_evidence": "dashboard_configuration_only; CI flags and revisions require separate verification",
    }


def config_key(point):
    return json.dumps(point["config"], sort_keys=True)


def ordering(point):
    return (point["date"], point["run_started_at"] or "", int(point["github_run_id"]), int(point["id"]))


def build_history(rows):
    # History can carry one measured result into multiple published curve snapshots.
    unique = {}
    for row in rows:
        point = normalize(row)
        if point["id"] not in unique:
            unique[point["id"]] = point
    points = sorted(unique.values(), key=ordering)
    groups = defaultdict(list)
    for point in points:
        groups[point["image_name"]].append(point)
    images = []
    representatives = {}
    for name, group in groups.items():
        by_config = defaultdict(list)
        for point in group:
            by_config[config_key(point)].append(point)
        def valid(point):
            return all(isinstance(point[field], (int, float)) and math.isfinite(point[field]) and point[field] > 0
                       for field in ["p90_interactivity_tok_s_user", "output_tok_s_chip"])
        reps = {key: min([point for point in values if valid(point)] or values, key=ordering)
                for key, values in by_config.items()}
        representatives[name] = reps
        repetitions = []
        for key, values in by_config.items():
            if len(values) < 2:
                continue
            x = [p["p90_interactivity_tok_s_user"] for p in values if p["p90_interactivity_tok_s_user"] is not None]
            y = [p["output_tok_s_chip"] for p in values if p["output_tok_s_chip"] is not None]
            repetitions.append({"representative_id": reps[key]["id"], "ids": [p["id"] for p in values],
                                "x_min_max": [min(x), max(x)] if x else None,
                                "y_min_max": [min(y), max(y)] if y else None,
                                "interpretation": "observed spread; recipe changes may contribute, not a noise estimate"})
        images.append({"image_name": name, "first_date": group[0]["date"], "last_date": group[-1]["date"],
                       "digests": sorted({p["image_digest"] for p in group}, key=str),
                       "run_ids": list(dict.fromkeys(p["github_run_id"] for p in group)),
                       "point_count": len(group), "representative_ids": [p["id"] for p in reps.values()],
                       "repetitions": repetitions})
    images.sort(key=lambda entry: ordering(groups[entry["image_name"]][0]))
    comparisons, transitions = [], []
    for old, new in zip(images, images[1:]):
        before, after = representatives[old["image_name"]], representatives[new["image_name"]]
        common = sorted(before.keys() & after.keys())
        edge = {"from_image": old["image_name"], "to_image": new["image_name"],
                "from_first_date": old["first_date"], "to_first_date": new["first_date"],
                "matching_points": len(common), "removed_configs": [before[k]["config"] for k in sorted(before.keys() - after.keys())],
                "added_configs": [after[k]["config"] for k in sorted(after.keys() - before.keys())],
                "kernel_evidence": "not_evaluated_by_performance_collector"}
        ambiguous = len(old["digests"]) != 1 or len(new["digests"]) != 1
        edge["ambiguous_tag_digest"] = ambiguous
        for key in common:
            a, b = before[key], after[key]
            x1, x2 = a["p90_interactivity_tok_s_user"], b["p90_interactivity_tok_s_user"]
            y1, y2 = a["output_tok_s_chip"], b["output_tok_s_chip"]
            valid = not ambiguous and all(isinstance(v, (int, float)) and v > 0 for v in [x1, x2, y1, y2])
            comparisons.append({"from_image": old["image_name"], "to_image": new["image_name"],
                                "from_id": a["id"], "to_id": b["id"], "from_date": a["date"], "to_date": b["date"],
                                "tp": a["config"]["decode_tp"], "ep": a["config"]["decode_ep"],
                                "conc": a["config"]["conc"], "offload": a["config"]["offload_mode"],
                                "x1": x1, "x2": x2, "y1": y1, "y2": y2,
                                "interactivity_change_pct": 100 * (x2 / x1 - 1) if valid else None,
                                "throughput_change_pct": 100 * (y2 / y1 - 1) if valid else None,
                                "recipe_changed": a["recipe_fingerprint"] != b["recipe_fingerprint"],
                                "status": "descriptive_same_dashboard_config" if valid else "incomparable",
                                "from_run_url": a["run_url"], "to_run_url": b["run_url"]})
        transitions.append(edge)
    return points, images, comparisons, transitions


def collect_ci(run_id, cache, offline, model_keys):
    base = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}"
    run = fetch(base, cache / "ci" / f"{run_id}.run.json", offline)
    jobs, artifacts = [], []
    for endpoint, target in [("jobs", jobs), ("artifacts", artifacts)]:
        page = 1
        while True:
            suffix = "&filter=all" if endpoint == "jobs" else ""
            cache_kind = "jobs.all" if endpoint == "jobs" else endpoint
            data = fetch(f"{base}/{endpoint}?per_page=100&page={page}{suffix}",
                         cache / "ci" / f"{run_id}.{cache_kind}.{page}.json", offline)
            target.extend(data[endpoint])
            if len(target) >= data["total_count"]:
                break
            if not data[endpoint]:
                raise ValueError("Pagination ended before total_count")
            page += 1
    return {"run_id": run_id, "name": run["name"], "head_sha": run["head_sha"],
            "created_at": run["created_at"], "head_branch": run["head_branch"], "path": run["path"],
            "run_url": run["html_url"], "conclusion": run["conclusion"],
            "model_jobs": [{key: job.get(key) for key in ["id", "name", "html_url", "started_at", "completed_at", "conclusion", "run_attempt"]}
                           for job in jobs if any(re.search(rf"\b{re.escape(model)}\b", job["name"], re.I) for model in model_keys)],
            "artifacts": [{key: item.get(key) for key in ["id", "name", "size_in_bytes", "expired", "archive_download_url"]}
                          for item in artifacts]}


def discover_framework(raw, latest, hardware, requested):
    coverage = []
    for framework in sorted({r["framework"] for r in raw if r["hardware"] == hardware}):
        rows = [r for r in raw if r["hardware"] == hardware and r["framework"] == framework]
        current = [r for r in latest if r["hardware"] == hardware and r["framework"] == framework]
        coverage.append({"framework": framework, "history_rows": len(rows),
                         "unique_points": len({r['id'] for r in rows}),
                         "first_date": min(r["date"] for r in rows), "last_date": max(r["date"] for r in rows),
                         "current_points": len(current), "current_latest_date": max((r["date"] for r in current), default=None)})
    if requested != "auto":
        selected, reason = requested, "explicit framework selection"
    else:
        eligible = [c for c in coverage if c["framework"] in ("vllm", "sglang")]
        if not eligible:
            raise ValueError("No MI355X vLLM/SGLang history found; inspect framework coverage explicitly")
        chosen = max(eligible, key=lambda c: (c["current_latest_date"] or c["last_date"], c["unique_points"], c["framework"]))
        selected, reason = chosen["framework"], "newest published curve; historical point coverage breaks ties"
    return selected, {"hardware": hardware, "requested_framework": requested, "selected_framework": selected,
                      "selection_reason": reason, "available_frameworks": coverage}


def matching_jobs(point, jobs):
    config = point["config"]
    candidates = []
    for job in jobs:
        name = job["name"]
        aliases = {"sglang": ("sglang", "sgl"), "vllm": ("vllm",)}.get(config["framework"], (config["framework"],))
        framework_matches = any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", name.lower()) for alias in aliases)
        tp = re.search(r"\bTP(\d+)(?:/EP(\d+))?\b", name)
        if ("eval" in name.lower() or job["conclusion"] != "success" or not tp
            or int(tp[1]) != config["decode_tp"] or int(tp[2] or 1) != config["decode_ep"]
            or not re.search(rf" c{config['conc']}$", name)
            or config["hardware"] not in name
            or not framework_matches
            or ("dram KV offload" in name) != (config["offload_mode"] == "on")):
            continue
        if point["github_run_attempt"] and job.get("run_attempt") and job["run_attempt"] > point["github_run_attempt"]:
            continue
        candidates.append(job)
    if candidates:
        newest = max(j.get("run_attempt") or 1 for j in candidates)
        candidates = [j for j in candidates if (j.get("run_attempt") or 1) == newest]
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Muse-Spark-Open")
    parser.add_argument("--hardware", default="mi355x")
    parser.add_argument("--framework", default="auto", help="auto selects MI355X vLLM/SGLang from published coverage; an explicit name overrides")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"model": args.model})
    history_url = ORIGIN + "/api/v1/benchmarks/history?" + query + "&benchmarkType=agentic_traces"
    raw = fetch(history_url, args.cache / "history.json", args.offline)
    latest = fetch(ORIGIN + "/api/v1/benchmarks?" + query, args.cache / "benchmarks.json", args.offline)
    framework, discovery = discover_framework(raw, latest, args.hardware, args.framework)
    save_json(args.output / "framework_discovery.json", discovery)
    selected = [r for r in raw if r["hardware"] == args.hardware and r["framework"] == framework]
    if not selected:
        raise ValueError("No selected history rows")
    points, images, comparisons, transitions = build_history(selected)
    selected_latest = [r for r in latest if r["hardware"] == args.hardware and r["framework"] == framework]
    assert {r["id"] for r in selected_latest} <= {p["id"] for p in points}
    for name, value in [("raw_selected_history.json", selected), ("raw_selected_latest.json", selected_latest),
                        ("points.json", points), ("images.json", images), ("transitions.json", transitions)]:
        save_json(args.output / name, value)
    with (args.output / "point_comparisons.csv").open("w") as stream:
        if comparisons:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(comparisons)
    if args.ci:
        run_ids = list(dict.fromkeys(p["github_run_id"] for p in points))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            manifests = list(pool.map(lambda run: collect_ci(run, args.cache, args.offline, {r["model"] for r in selected}), run_ids))
        save_json(args.output / "ci_manifest.json", manifests)
        by_run = {run["run_id"]: run for run in manifests}
        for point in points:
            config = point["config"]
            candidates = matching_jobs(point, by_run[point["github_run_id"]]["model_jobs"])
            point["github_job_match"] = "unique" if len(candidates) == 1 else "unresolved"
            if len(candidates) == 1:
                job = candidates[0]
                point.update({"github_job_id": job["id"], "job_url": job["html_url"],
                              "job_started_at": job["started_at"], "job_completed_at": job["completed_at"]})
        save_json(args.output / "points.json", points)
    receipts = [json.loads(p.read_text()) | {"cache_file": str(p).removesuffix(".meta.json")}
                for p in sorted(args.cache.rglob("*.meta.json"))]
    save_json(args.output / "retrieval_manifest.json", receipts)
    summary = {"model": args.model, "hardware": args.hardware, "framework": framework,
               "history_total_rows": len(raw), "selected_snapshot_rows": len(selected), "unique_points": len(points),
               "image_versions": len(images), "run_count": len({p['github_run_id'] for p in points}),
               "first_date": points[0]["date"], "latest_date": max(p["date"] for p in points),
               "adjacent_matched_pairs": len(comparisons), "latest_point_count": len(selected_latest),
               "source": history_url, "coverage": "currently retained published AgentX curve snapshots, not all CI runs ever executed",
               "policy": "earliest valid observation per image name and dashboard configuration; preserve every repetition",
               "causal_kernel_attribution": "not established"}
    save_json(args.output / "coverage.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
