#!/usr/bin/env python3
"""Normalize DeepSeek Flash container/commit source evidence without conflating provenance."""
import argparse
import ast
import difflib
import hashlib
import json
import re
import tarfile
from pathlib import Path


def selected(path):
    model = ("/deepseek_v4_1/" in path or "/deepseek_v41/" in path) and path.startswith("vllm/models/") and not any(x in path for x in ["/nvidia/", "/cpu/", "/xpu/"])
    shared = path.startswith(("vllm/models/deepseek_v4/amd/", "vllm/models/deepseek_v4/common/ops/"))
    support = path.startswith("vllm/") and ("aiter" in path.lower() or path in ["vllm/envs.py", "vllm/config/kernel.py", "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"])
    return path.endswith(".py") and (model or shared or support or path.startswith(("vllm/model_executor/kernels/mhc/", "vllm/model_executor/layers/mhc.py")))


def symbols(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--knowledge", type=Path, required=True)
    args = parser.parse_args()
    images = json.loads((args.knowledge / "data/images.json").read_text())
    server = json.loads((args.knowledge / "evidence/server_log_extracts.json").read_text())
    snapshots, contents = [], []
    for number, image in enumerate(images, 1):
        version = f"v{number:02d}"
        directory = args.cache / "images" / version
        receipt = json.loads((directory / "receipt.json").read_text())
        files, data = {}, {}
        metadata = {"version": number, "image_name": image["image_name"], "historical_api_digests": image["digests"], "docker_receipt": receipt}
        startup = "\n".join(e["text"] for item in server if item["version"] == number for e in item["extracts"])
        metadata["ci_reported_versions"] = sorted(set(re.findall(r"\bversion\s+([0-9][\w.+]*\+g[0-9a-f]+)", startup)))
        if receipt["status"] == "captured":
            manifest = json.loads((directory / "source_manifest.json").read_text())
            inspect = json.loads((directory / "image_inspect.json").read_text())[0]
            metadata.update({"source_kind": "container", "digest": inspect["RepoDigests"][0].split("@", 1)[1],
                             "packages": {k: v["version"] for k, v in manifest["packages"].items()},
                             "archive_sha256": manifest["archive_sha256"], "external_archive": str(directory / "sources.tar.gz")})
            # Prefer the installed runtime package over build-context copies.
            with tarfile.open(directory / "sources.tar.gz") as archive:
                for entry in archive:
                    match = re.search(r"/(?:site|dist)-packages/(vllm/.*)$", "/" + entry.name)
                    if not entry.isfile() or not match or not selected(match[1]):
                        continue
                    repo_path = match[1]
                    text = archive.extractfile(entry).read().decode()
                    key = repo_path.replace("deepseek_v4_1", "deepseek_v41")
                    data[key] = text
                    files[key] = {"path": key, "original_path": "/" + entry.name, "repository_path": repo_path,
                                  "sha256": hashlib.sha256(text.encode()).hexdigest(), "source_kind": "container"}
        else:
            upstream = args.cache / "upstream" / version
            selection = json.loads((upstream / "selection.json").read_text())
            metadata.update({"source_kind": "upstream_commit", "digest": None, "source_revision": selection["commit"],
                             "historical_image_status": "tag unavailable; upstream source corroborated by CI package commit, not container bytes"})
            assert any(selection["commit"].startswith(v.split("+g")[1]) for v in metadata["ci_reported_versions"]), version
            for entry in json.loads((upstream / "source_manifest.json").read_text()):
                if "error" in entry or not selected(entry["path"]):
                    continue
                text = (upstream / "sources" / entry["path"]).read_text()
                assert hashlib.sha256(text.encode()).hexdigest() == entry["sha256"]
                key = entry["path"].replace("deepseek_v4_1", "deepseek_v41")
                data[key] = text
                files[key] = {"path": key, "original_path": entry["path"], "repository_path": entry["path"],
                              "sha256": entry["sha256"], "source_kind": "upstream_commit", "source_url": entry["url"]}
        assert data, version
        destination = args.cache / "normalized_sources" / version
        for key, text in data.items():
            f = destination / key
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
            files[key]["declared_device_languages"] = [name for token, name in [("@triton.jit", "Triton"), ("flydsl.compiler", "FlyDSL"), ("triton.experimental.gluon", "Gluon")] if token in text]
        metadata["relevant_files"] = [files[k] for k in sorted(files)]
        snapshots.append(metadata)
        contents.append(data)
    evidence = args.knowledge / "evidence"
    (evidence / "image_source_inventory.json").write_text(json.dumps(snapshots, indent=2) + "\n")
    (evidence / "source_diffs").mkdir(exist_ok=True)
    changes = []
    pairs = {(i, i + 1) for i in range(len(images) - 1)} | {(8, 10)}
    for i, j in sorted(pairs):
        old, new = contents[i], contents[j]
        before = {x["path"]: x for x in snapshots[i]["relevant_files"]}
        after = {x["path"]: x for x in snapshots[j]["relevant_files"]}
        edits = []
        filename = f"v{i+1:02d}_to_v{j+1:02d}.diff"
        with (evidence / "source_diffs" / filename).open("w") as output:
            for key in sorted(old.keys() | new.keys()):
                if old.get(key) == new.get(key):
                    continue
                sa, sb = symbols(old.get(key, "")), symbols(new.get(key, ""))
                edits.append({"canonical_path": key, "before": before.get(key), "after": after.get(key),
                              "changed_symbols": [name for name in sorted(sa.keys() | sb.keys()) if sa.get(name) != sb.get(name)]})
                # Keep selected model diffs reviewable; shared support inventories remain hashed.
                if "models/deepseek_v41/" in key or "rocm_aiter_mla_sparse" in key or "/mhc/" in key or key.endswith("/mhc.py"):
                    output.writelines(difflib.unified_diff(old.get(key, "").splitlines(True), new.get(key, "").splitlines(True),
                                      fromfile=f"V{i+1:02d}:" + before.get(key, {}).get("original_path", key),
                                      tofile=f"V{j+1:02d}:" + after.get(key, {}).get("original_path", key), n=3))
        changes.append({"from_version": i + 1, "to_version": j + 1, "from_image": images[i]["image_name"], "to_image": images[j]["image_name"],
                        "source_kinds": [snapshots[i]["source_kind"], snapshots[j]["source_kind"]], "changes": edits,
                        "diff": "evidence/source_diffs/" + filename})
    (evidence / "source_changes.json").write_text(json.dumps(changes, indent=2) + "\n")
    print(json.dumps({"source_snapshots": len(snapshots), "files": [len(x["relevant_files"]) for x in snapshots], "comparison_edges": len(changes)}))


if __name__ == "__main__":
    main()
