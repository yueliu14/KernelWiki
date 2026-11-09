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
    a, b = result.get("before_path"), result.get("after_path")
    result["path_relation"] = "added" if a is None else "removed" if b is None else "same_path" if a == b else "different_paths"
    result["body_relation"] = "identical_file_bytes" if a and b and result["before_sha256"] == result["after_sha256"] else "different_file_bytes"
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
    for kernel in pair["kernel_changes"]:
        text += f"\n### {kernel['pipeline']}\n\n"
        for title, key in [("Before", "before"), ("After", "after"), ("Device language", "language"), ("Code/body change", "body_change"), ("Fusion", "fusion"), ("Applicability to this pair", "applicability")]:
            text += f"**{title}:** {kernel[key]}\n\n"
        text += "| Source role | Before path / file SHA256 | After path / file SHA256 | Relation |\n| --- | --- | --- | --- |\n"
        for source in kernel["sources"]:
            def cell(side):
                if not source.get(side + '_path'):
                    return 'Absent / separate existing operations'
                text = f"`{source[side+'_path']}`<br>`{source[side+'_sha256']}`"
                implementation = source.get(side + '_implementation')
                if implementation:
                    path = '../' + implementation['snapshot_path']
                    text += f'<br>[Complete source]({path})'
                    for entry in implementation['entrypoints']:
                        text += f"<br>[{entry['name']}]({path}#L{entry['line']})"
                return text
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
        text += f"| {side} | {pair['config']['conc']} | {opts.get('max-running-requests','not passed')} | {opts.get('monokernel-backend','not passed')} | {opts.get('page-size','not passed')} | {', '.join(point['launch']['datasets'])} |\n"
    text += "\nBoth complete commands, environment changes, image digests and source hashes are in [pair_records.json](../data/pair_records.json). Endpoint CI logs are linked through [launch configurations](../evidence/ci_launch_configs.json) and [log extracts](../evidence/ci_log_extracts.json).\n"
    text += "\n## Runtime and attribution status\n\n"
    if pair.get("verified_runtime"):
        text += "Both exact endpoint server artifacts were retrieved by the upload IDs in their own CI job logs. "
        text += "[Runtime recheck](../runtime_verification.md) · [Phase-aligned evidence](../evidence/verified_runtime_facts.json) · [Source/body audit](../evidence/verified_source_audit.json).\n\n"
        text += "| Endpoint / artifact | Dispatch diagnostic family | Setup/graph warmup rows | Benchmark warmup rows | Measurement rows |\n| --- | --- | --- | --- | --- |\n"
        boundary_notes = []
        for side in ['before', 'after']:
            fact = pair['verified_runtime'][side]
            dispatch = [f for f in fact['families'] if f['kind'] == 'dispatch']
            for family in dispatch:
                def rows(phase):
                    value = family['phases'].get(phase)
                    return ', '.join(map(str, value['observed_rows'])) if value else 'no diagnostic'
                text += f"| {side} / `{fact['artifact_id']}` | {family['family']} | {rows('server_setup_or_graph_warmup')} | {rows('benchmark_warmup')} | {rows('e2e_measurement')} |\n"
            if not dispatch:
                text += f"| {side} / `{fact['artifact_id']}` | No selected dispatch diagnostics | — | — | — |\n"
            boundary = [f['family'] for f in dispatch if 'phase_boundary_uncertain' in f['phases']]
            if boundary:
                boundary_notes.append(f"{side}: {', '.join(boundary)} also has events within one second of the measurement start; these remain boundary-uncertain.")
        if boundary_notes:
            text += '\n' + '\n\n'.join(boundary_notes) + '\n'
        text += "\nRows describe the first logged shape occurrences, not sustained coverage. Once-only diagnostics suppress later uses. Initialization, dispatch and GPU timing are separate evidence levels; no log count is used as a launch count.\n"
        if pair['id'] == 'tep8_conc1_off_2026-10-22_to_2026-10-30':
            text += "\nThe [bounded GPU validation attempt](../gpu_validation/README.md) obtained no allocation, so this pair still has no reproduced GPU trace.\n"
    elif pair["matched_server_artifacts"]:
        text += "Exact endpoint server artifacts already inspected: " + ", ".join(f"{x['side']} `{x['artifact_id']}`" for x in pair["matched_server_artifacts"]) + ". See [server-log extracts](../evidence/server_log_extracts.json). Initialization and fast-path messages may include warmup.\n"
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
    inventories = {row["digest"]: {f["path"]: f for f in row["relevant_files"]} for row in read(root / "evidence/image_source_inventory.json")}
    rules = read(root / "evidence/kernel_pair_rules.json")
    server = read(root / "evidence/server_log_extracts.json")
    verified_path = root / 'evidence/verified_runtime_facts.json'
    verified = {f['point_id']: f for f in read(verified_path)} if verified_path.exists() else {}
    source_index_path = root / 'kernel_sources/manifest.json'
    source_index = {s['image_digest']: {f['container_path']: f for f in s['files']} for s in read(source_index_path)['snapshots']} if source_index_path.exists() else {}
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
            applicable = [r for r in rules if r["from_digest"] == a["image_digest"] and r["to_digest"] == b["image_digest"] and all(cfg.get(key) == value for key, value in r["config_selector"].items())]
            if len(applicable) != 1:
                raise ValueError(f"Review source applicability for {a['id']} → {b['id']}; found {len(applicable)} rules")
            rule = applicable[0]
            ac, bc = launches[a["github_job_id"]], launches[b["github_job_id"]]
            af, bf = flags(ac["launch_command"]), flags(bc["launch_command"])
            for side, parsed in [("before", af), ("after", bf)]:
                for key, value in rule.get("required_cli", {}).get(side, {}).items():
                    if parsed.get(key) != value:
                        raise ValueError(f"Applicability rule disagrees with {side} CLI: {key}")
            slug = f"tep{cfg['decode_tp']}_conc{cfg['conc']}_{cfg['offload_mode']}_{a['date']}_to_{b['date']}"
            kernels = copy.deepcopy(rule["kernel_changes"])
            for kernel in kernels:
                kernel["evidence_level"] = "source_verified"
                kernel["sources"] = [source_ref(ref, inventories[a["image_digest"]], inventories[b["image_digest"]]) for ref in kernel["sources"]]
                for source in kernel['sources']:
                    for side, point in [('before', a), ('after', b)]:
                        path = source.get(side + '_path')
                        if source_index and path:
                            implementation = source_index[point['image_digest']][path]
                            assert implementation['sha256'] == source[side + '_sha256']
                            source[side + '_implementation'] = {k: implementation[k] for k in ['snapshot_path', 'entrypoints']}
            matched_server = []
            for side, point in [("before", a), ("after", b)]:
                for artifact in server:
                    if artifact["run_id"] == point["github_run_id"] and f"_tp{cfg['decode_tp']}_conc{cfg['conc']}_" in artifact["name"]:
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
            if a['id'] in verified and b['id'] in verified:
                pair['verified_runtime'] = {}
                pair['matched_server_artifacts'] = []
                for side, point in [('before', a), ('after', b)]:
                    fact = verified[point['id']]
                    pair['matched_server_artifacts'].append({'side': side, 'artifact_id': fact['artifact_id'], 'exact_job_link_verified': True})
                    pair['verified_runtime'][side] = {'artifact_id': fact['artifact_id'], 'point_id': point['id'],
                        'evidence': 'evidence/verified_runtime_facts.json',
                        'families': [{'kind': f['kind'], 'family': f['family'], 'phases': {phase: {'observed_rows': info['observed_rows']} for phase, info in f['phases'].items()}} for f in fact['families']]}
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
    index += "\nThe older [adjacent-image CSV](data/point_comparisons.csv) remains a narrower view. [Structured pair records](data/pair_records.json) include all rows above. Kernel annotations are pinned to reviewed endpoint digests and configuration/CLI guards in [kernel_pair_rules.json](evidence/kernel_pair_rules.json); a new image requires a new source review.\n"
    (root / "pair_changes.md").write_text(index)
    matrix = "# Performance history by configuration\n\nEach cell is **(p90 interactivity in tok/s/user, output throughput in tok/s/chip)** and links to its CI job. Columns are image versions; dates are first observed dashboard dates. The earliest valid observation per image/configuration is shown; [all measured points](performance_points.md) preserve repeats. `—` means no retained observation.\n\nThe last column links each available point-to-point pair to its kernel record. Pairs connect successive observed versions within the row, including across blank columns. Singleton rows have no pair. [Pair index](pair_changes.md) · [Structured pair records](data/pair_records.json).\n\n"
    matrix += "| Configuration | " + " | ".join(f"{i['first_date']} / `{i['image_name'].split(':')[-1]}`" for i in images) + " | Kernel changes for pairs |\n"
    matrix += "| --- | " + " | ".join("---:" for _ in images) + " | --- |\n"
    for key, group in sorted(groups.items(), key=lambda item: (item[1][0][1]["config"]["decode_tp"], item[1][0][1]["config"]["offload_mode"], item[1][0][1]["config"]["conc"])):
        by_image = dict(group)
        cells = []
        for i in range(len(images)):
            point = by_image.get(i)
            cells.append(f"[({point['p90_interactivity_tok_s_user']:.3f}, {point['output_tok_s_chip']:.3f})]({point['job_url']})" if point else "—")
        links = [f"[{p['before']['date'][5:]} → {p['after']['date'][5:]}]({p['report']})" for p in pairs if json.dumps(p["config"], sort_keys=True) == key]
        matrix += f"| {label(group[0][1])} | " + " | ".join(cells) + " | " + ("; ".join(links) or "—") + " |\n"
    (root / "configuration_history.md").write_text(matrix)
    print(json.dumps({"configuration_rows": len(groups), "pair_records": len(pairs), "gap_pairs": sum(bool(p["skipped_images"]) for p in pairs), "endpoint_jobs": len({p[s]["github_job_id"] for p in pairs for s in ["before", "after"]})}))


if __name__ == "__main__":
    main()
