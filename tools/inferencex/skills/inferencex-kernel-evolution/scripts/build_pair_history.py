#!/usr/bin/env python3
"""Render the full configuration matrix and per-pair kernel evidence from reviewed rules."""
import argparse
import copy
import json
import math
import shlex
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def config_key(point):
    return json.dumps(point["config"], sort_keys=True)


def label(point):
    c = point["config"]
    parallelism = f"TEP{c['decode_tp']}" if c['decode_tp'] == c['decode_ep'] else f"TP{c['decode_tp']}/EP{c['decode_ep']}"
    return f"{parallelism} conc{c['conc']} / " + ("HiCache" if c["offload_mode"] == "on" else "GPU")


def flags(command):
    tokens = shlex.split(command)
    parsed = {}
    for i, token in enumerate(tokens):
        if token.startswith("--"):
            key, sep, value = token[2:].partition("=")
            parsed[key] = value if sep else tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else True
    return parsed


def deltas(a, b):
    return [{"key": k, "before": a.get(k), "after": b.get(k)} for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]


def endpoint(point, launch):
    return {key: point.get(key) for key in ["id", "date", "image_name", "image_digest", "job_url", "github_job_id", "recipe_fingerprint", "p90_interactivity_tok_s_user", "output_tok_s_chip"]} | {"launch": launch}


def source_ref(reference, old_files, new_files):
    result = copy.deepcopy(reference)
    for side, inventory in [("before", old_files), ("after", new_files)]:
        path = reference.get(side + "_path")
        if path is not None and path not in inventory:
            raise ValueError(f"Reviewed source path missing from inventory: {path}")
        result[side + "_sha256"] = inventory[path]["sha256"] if path else None
        if path:
            for key in ["original_path", "repository_path", "source_kind", "source_url"]:
                if key in inventory[path]:
                    result[side + "_" + key] = inventory[path][key]
    a, b = result.get("before_path"), result.get("after_path")
    result["path_relation"] = "added" if a is None else "removed" if b is None else "same_path" if a == b else "different_paths"
    result["body_relation"] = "identical_file_bytes" if a and b and result["before_sha256"] == result["after_sha256"] else "different_file_bytes"
    if a and b and result.get("before_repository_path") and result.get("after_repository_path"):
        result["path_relation"] = "same_repository_path" if result["before_repository_path"] == result["after_repository_path"] else "different_repository_paths"
    return result


def pair_document(pair):
    a, b = pair["before"], pair["after"]
    text = f"# {pair['configuration_label']}: {a['date']} → {b['date']}\n\n"
    text += "[Configuration matrix](../configuration_history.md) · [All pairs](../pair_changes.md)\n\n"
    text += "| Endpoint | Image | Point / CI | p90 interactivity (tok/s/user) | Output tok/s/chip |\n| --- | --- | --- | ---: | ---: |\n"
    for side, point in [("Before", a), ("After", b)]:
        text += f"| {side}: {point['date']} | `{point['image_name']}` | [{point['id']}]({point['job_url']}) | {point['p90_interactivity_tok_s_user']:.6f} | {point['output_tok_s_chip']:.5f} |\n"
    text += f"\nInteractivity **{pair['interactivity_change_pct']:+.2f}%**; output throughput/chip **{pair['throughput_change_pct']:+.2f}%**.\n"
    if pair["skipped_images"]:
        text += "\nThis is the next observed point for this configuration; intermediate images without a matching point: " + ", ".join(f"`{name}`" for name in pair["skipped_images"]) + ". The endpoint delta spans that gap.\n"
    text += "\n## Kernel changes for this pair\n\nThese are source-verified candidates and configuration eligibility checks. GPU-time ranking and realized benchmark-phase launch counts are unavailable. Missing CLI flags mean the option was not explicitly passed; defaults may differ between images.\n"
    if pair.get("source_provenance"):
        text += "\n**Source provenance:** " + "; ".join(f"{side}: {info['source_kind']} ({info.get('source_revision') or info.get('digest') or 'see inventory'})" for side, info in pair["source_provenance"].items()) + ". Upstream-commit source is corroborated by CI version logs but is not a recovered historical container. A current tag snapshot alone does not establish historical file identity.\n"
    for kernel in pair["kernel_changes"]:
        text += f"\n### {kernel['pipeline']}\n\n"
        for title, key in [("Before", "before"), ("After", "after"), ("Device language", "language"), ("Code/body change", "body_change"), ("Fusion", "fusion"), ("Applicability to this pair", "applicability")]:
            text += f"**{title}:** {kernel[key]}\n\n"
        text += "| Source role | Before path / file SHA256 | After path / file SHA256 | Relation |\n| --- | --- | --- | --- |\n"
        for source in kernel["sources"]:
            def cell(side):
                if not source.get(side + "_path"):
                    return "Absent / separate existing operations"
                path = source.get(side + "_original_path", source[side + "_path"])
                where = f"[{path}]({source[side+'_source_url']})" if source.get(side + "_source_url") else f"`{path}`"
                return f"{where}<br>`{source[side+'_sha256']}`"
            text += f"| {source['role']} | {cell('before')} | {cell('after')} | {source['path_relation']}; {source['body_relation']} |\n"
        text += "\nEvidence: " + ", ".join(f"[{ref}](../{ref})" for ref in kernel["evidence"]) + ".\n"
    if pair["excluded_attributions"]:
        text += "\n### Changes not attributed to this configuration\n\n" + "\n".join("- " + note for note in pair["excluded_attributions"]) + "\n"
    text += "\n## Exact CI configuration changes\n\n| Explicit launch flag | Before | After |\n| --- | --- | --- |\n"
    for delta in pair["launch_flag_changes"]:
        text += f"| `{delta['key']}` | `{delta['before'] if delta['before'] is not None else 'not explicitly passed'}` | `{delta['after'] if delta['after'] is not None else 'not explicitly passed'}` |\n"
    text += "\n| Endpoint | Client concurrency | Server admission | Monokernel CLI | Page size | Dataset |\n| --- | ---: | ---: | --- | ---: | --- |\n"
    for side, point in [("Before", a), ("After", b)]:
        opts = flags(point["launch"]["launch_command"])
        admission = opts.get('max-running-requests', opts.get('max-num-seqs', 'not passed'))
        mono = opts.get('monokernel-backend', opts.get('monokernel', point['launch'].get('server_env', {}).get('VLLM_ROCM_MONO_DECODE', 'not passed')))
        page = opts.get('page-size', opts.get('block-size', 'not passed'))
        text += f"| {side} | {pair['config']['conc']} | {admission} | {mono} | {page} | {', '.join(point['launch']['datasets'])} |\n"
    text += "\nBoth complete commands, environment changes, image digests and source hashes are in [pair_records.json](../data/pair_records.json). Endpoint CI logs are linked through [launch configurations](../evidence/ci_launch_configs.json) and [log extracts](../evidence/ci_log_extracts.json).\n"
    if pair.get("runtime_log_evidence"):
        report = pair["runtime_log_evidence"]["report"]
        text += f"\n## Recovered-container kernel and log comparison\n\n[Detailed endpoint log/source comparison](../{report}) adds exact active/HIT/JIT messages, measured-phase versus warmup alignment, symbol-body hashes, unchanged AITER paths and expanded fusion review. [Recovered endpoint artifacts](../evidence/recovered_endpoint_logs.json) preserve the exact file hashes and upload links. Diagnostic occurrence counts are not GPU kernel invocation counts.\n"
    text += "\n## Runtime and attribution status\n\n"
    if pair["matched_server_artifacts"]:
        text += "Exact endpoint server artifacts already inspected: " + ", ".join(f"{x['side']} `{x['artifact_id']}`" for x in pair["matched_server_artifacts"]) + ". See [server-log extracts](../evidence/server_log_extracts.json). Initialization and fast-path messages may include warmup.\n"
        if pair.get("additional_server_log_evidence"):
            text += "Additional exact endpoint archives: [recovered endpoint log evidence](../evidence/recovered_endpoint_logs.json).\n"
    else:
        text += "No nested server artifact for these exact endpoints has been inspected in this pilot; source guards and their own CI launch logs provide the current evidence.\n"
    text += "\nNo per-kernel causal speedup is assigned. Model/dataset revisions are not fully pinned at both endpoints, recipe fingerprints differ, and the launch changes above are potential contributors. GPU profiles would be new reproduced evidence.\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", type=Path, required=True)
    args = parser.parse_args()
    root = args.knowledge
    points = read(root / "data/points.json")
    images = read(root / "data/images.json")
    by_id = {point["id"]: point for point in points}
    launches = {row["job_id"]: row for row in read(root / "evidence/ci_launch_configs.json")}
    inventory_rows = read(root / "evidence/image_source_inventory.json")
    inventories = {row["image_name"]: {f["path"]: f for f in row["relevant_files"]} for row in inventory_rows}
    snapshot_info = {row["image_name"]: {k: row[k] for k in ["source_kind", "source_revision", "digest", "historical_api_digests"] if k in row} for row in inventory_rows}
    rules = read(root / "evidence/kernel_pair_rules.json")
    server = read(root / "evidence/server_log_extracts.json")
    recovered_ids = set()
    recovered_logs = root / "evidence/recovered_endpoint_logs.json"
    if recovered_logs.exists():
        recovered_rows = read(recovered_logs)
        recovered_ids = {r["artifact_id"] for r in recovered_rows}
        server.extend(recovered_rows)
        server = list({(row["artifact_id"], row.get("job_id")): row for row in server}.values())
    recovered_links = root / "evidence/recovered_pair_links.json"
    log_links = {r["pair_id"]: r for r in read(recovered_links)} if recovered_links.exists() else {}
    groups = defaultdict(list)
    for index, image in enumerate(images):
        for id in image["representative_ids"]:
            point = by_id[id]
            groups[config_key(point)].append((index, point))
    pairs = []
    for group in groups.values():
        for (old_index, a), (new_index, b) in zip(group, group[1:]):
            assert a["config"] == b["config"]
            if any(len(images[index]["digests"]) != 1 for index in [old_index, new_index]):
                raise ValueError("Split ambiguous image tags before constructing pair evidence")
            for point in [a, b]:
                if point.get("github_job_id") not in launches:
                    raise ValueError(f"Fetch the exact endpoint CI launch log first: {point.get('job_url')}")
                if not all(isinstance(point[k], (int, float)) and math.isfinite(point[k]) and point[k] > 0 for k in ["p90_interactivity_tok_s_user", "output_tok_s_chip"]):
                    raise ValueError(f"Requested pair metric unavailable: {point['id']}")
            cfg = a["config"]
            applicable = [r for r in rules if
                          ((r.get("from_image_name") == a["image_name"] and r.get("to_image_name") == b["image_name"])
                           if r.get("from_image_name") else
                           (r["from_digest"] == a["image_digest"] and r["to_digest"] == b["image_digest"]))
                          and all(cfg.get(key) == value for key, value in r["config_selector"].items())]
            if len(applicable) != 1:
                raise ValueError(f"Review source applicability for {a['id']} → {b['id']}; found {len(applicable)} rules")
            rule = applicable[0]
            ac, bc = launches[a["github_job_id"]], launches[b["github_job_id"]]
            af, bf = flags(ac["launch_command"]), flags(bc["launch_command"])
            for side, parsed in [("before", af), ("after", bf)]:
                for key, value in rule.get("required_cli", {}).get(side, {}).items():
                    if parsed.get(key) != value:
                        raise ValueError(f"Applicability rule disagrees with {side} CLI: {key}")
            for side, launch in [("before", ac), ("after", bc)]:
                for key, value in rule.get("required_env", {}).get(side, {}).items():
                    if launch["server_env"].get(key) != value:
                        raise ValueError(f"Applicability rule disagrees with {side} environment: {key}")
            parallel = f"tep{cfg['decode_tp']}" if cfg['decode_tp'] == cfg['decode_ep'] else f"tp{cfg['decode_tp']}_ep{cfg['decode_ep']}"
            slug = f"{parallel}_conc{cfg['conc']}_{cfg['offload_mode']}_{a['date']}_to_{b['date']}"
            kernels = copy.deepcopy(rule["kernel_changes"])
            for kernel in kernels:
                kernel["evidence_level"] = "source_verified"
                kernel["sources"] = [source_ref(ref, inventories[a["image_name"]], inventories[b["image_name"]]) for ref in kernel["sources"]]
            matched_server = []
            for side, point in [("before", a), ("after", b)]:
                for artifact in server:
                    if artifact.get("job_id") == point["github_job_id"] and artifact.get("exact_job_link_verified"):
                        matched_server.append({"side": side, "artifact_id": artifact["artifact_id"]})
                    elif artifact.get("run_id") == point["github_run_id"] and f"_tp{cfg['decode_tp']}_conc{cfg['conc']}_" in artifact.get("name", ""):
                        matched_server.append({"side": side, "artifact_id": artifact["artifact_id"]})
            pair = {"id": slug, "configuration_label": label(a), "config": cfg,
                    "before": endpoint(a, ac), "after": endpoint(b, bc),
                    "pairing": "next_observed_image_for_same_configuration",
                    "skipped_images": [i["image_name"] for i in images[old_index + 1:new_index]],
                    "interactivity_change_pct": 100 * (b["p90_interactivity_tok_s_user"] / a["p90_interactivity_tok_s_user"] - 1),
                    "throughput_change_pct": 100 * (b["output_tok_s_chip"] / a["output_tok_s_chip"] - 1),
                    "launch_flag_changes": deltas(af, bf), "environment_changes": deltas(ac["server_env"], bc["server_env"]),
                    "kernel_changes": kernels, "excluded_attributions": rule["excluded_attributions"],
                    "matched_server_artifacts": matched_server, "top_kernels": None, "gpu_profile_status": "unavailable",
                    "causal_attribution": "not_established", "report": "pairs/" + slug + ".md"}
            if all("source_kind" in snapshot_info[p["image_name"]] for p in [a, b]):
                pair["source_provenance"] = {"before": snapshot_info[a["image_name"]], "after": snapshot_info[b["image_name"]]}
            if any(x["artifact_id"] in recovered_ids for x in matched_server):
                pair["additional_server_log_evidence"] = "evidence/recovered_endpoint_logs.json"
            if slug in log_links:
                pair["runtime_log_evidence"] = log_links[slug]
            pairs.append(pair)
    pairs.sort(key=lambda p: (p["config"]["decode_tp"], p["config"]["offload_mode"], p["config"]["conc"], p["before"]["date"]))
    (root / "pairs").mkdir(exist_ok=True)
    for pair in pairs:
        (root / pair["report"]).write_text(pair_document(pair))
    write(root / "data/pair_records.json", pairs)
    write(root / "data/pair_coverage.json", {"pairing_policy": "next observed image per full configuration key",
          "configuration_rows": len(groups), "representative_points": sum(len(g) for g in groups.values()),
          "pair_records": len(pairs), "gap_pairs": sum(bool(p["skipped_images"]) for p in pairs),
          "endpoint_jobs": len({p[s]["github_job_id"] for p in pairs for s in ["before", "after"]}),
          "source_rules": "evidence/kernel_pair_rules.json", "gpu_kernel_time_ranking": "unavailable"})
    index = "# Point-to-point kernel changes\n\nEach row links to the exact configuration's dates, performance, CI commands and kernel/fusion changes. Pair the next observed image in each configuration row; skipped image columns remain explicit. Singleton rows remain in the [configuration matrix](configuration_history.md).\n\n| Configuration | Dates | Δ interactivity | Δ output/chip | Skipped images | Kernel record |\n| --- | --- | ---: | ---: | ---: | --- |\n"
    for p in pairs:
        index += f"| {p['configuration_label']} | {p['before']['date']} → {p['after']['date']} | {p['interactivity_change_pct']:+.2f}% | {p['throughput_change_pct']:+.2f}% | {len(p['skipped_images'])} | [Paths, bodies, fusion and CI]({p['report']}) |\n"
    index += "\nThe older [adjacent-image CSV](data/point_comparisons.csv) remains a narrower view. [Structured pair records](data/pair_records.json) include all rows above. Kernel annotations are pinned to reviewed endpoint image/source identities and configuration/CLI guards in [kernel_pair_rules.json](evidence/kernel_pair_rules.json); a new image requires a new source review.\n"
    (root / "pair_changes.md").write_text(index)
    matrix = "# Performance history by configuration\n\nEach cell is **(p90 interactivity in tok/s/user, output throughput in tok/s/chip)** and links to its CI job. Columns are image versions; dates are first observed dashboard dates. The earliest valid observation per image/configuration is shown; [all measured points](performance_points.md) preserve repeats. `—` means no retained observation.\n\nThe last column links each available point-to-point pair to its kernel record. Pairs connect successive observed versions within the row, including across blank columns. Singleton rows have no pair. [Pair index](pair_changes.md) · [Structured pair records](data/pair_records.json).\n\n"
    compact = len(images) > 6
    if compact:
        matrix += "Image names and provenance: [version index](README.md#image-versions).\n\n"
    matrix += "| Configuration | " + " | ".join((f"V{n:02d} / {i['first_date']}" if compact else f"{i['first_date']} / `{i['image_name'].split(':')[-1]}`") for n, i in enumerate(images, 1)) + " | Kernel changes for pairs |\n"
    matrix += "| --- | " + " | ".join("---:" for _ in images) + " | --- |\n"
    for key, group in sorted(groups.items(), key=lambda item: (item[1][0][1]["config"]["decode_tp"], item[1][0][1]["config"]["offload_mode"], item[1][0][1]["config"]["conc"])):
        by_image = dict(group)
        cells = []
        for i in range(len(images)):
            point = by_image.get(i)
            cells.append(f"[({point['p90_interactivity_tok_s_user']:.3f}, {point['output_tok_s_chip']:.3f})]({point['job_url']})" if point else "—")
        links = [f"[{p['before']['date'][5:]} → {p['after']['date'][5:]}]({p['report']})" for p in pairs if json.dumps(p["config"], sort_keys=True) == key]
        matrix += f"| {label(group[0][1])} | " + " | ".join(cells) + " | " + ("; ".join(links) or "—") + " |\n"
    if log_links:
        matrix += "\n[Recovered-container kernel/log supplement](recovered_image_history.md): 16 pairs across the four recovered images, including phase-aligned logs for ten pairs above and six longer baseline comparisons.\n"
    (root / "configuration_history.md").write_text(matrix)
    print(json.dumps({"configuration_rows": len(groups), "pair_records": len(pairs), "gap_pairs": sum(bool(p["skipped_images"]) for p in pairs), "endpoint_jobs": len({p[s]["github_job_id"] for p in pairs for s in ["before", "after"]})}))


if __name__ == "__main__":
    main()
