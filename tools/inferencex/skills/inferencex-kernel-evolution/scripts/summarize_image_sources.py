#!/usr/bin/env python3
"""Compare captured image source hashes and emit reviewable kernel diffs."""
import argparse
import difflib
import hashlib
import json
import tarfile
from pathlib import Path


def relevant(path):
    return any(term in path for term in (
        "/rocm_muse_spark/", "/sgl_amd_mono/muse_spark/", "/srt/monokernels/",
        "/ops/attention/muse_", "/ops/attention/qk_norm_rope_cache.py",
        "/ops/attention/verify_reduce_norm_gate.py", "/ops/attention/gqa_verify_splitkv_2kv.py",
        "/ops/attention/swa_verify_splitkv.py", "/ops/moe/mxfp4_compact_a4_moe.py",
        "/ops/moe/router_pooler_topk.py", "/ops/gemm/bf16_tiled_small_m_gemm.py",
        "/ops/flydsl/muse_moe", "/srt/layers/attention/aiter_backend.py", "/srt/environ.py",
    ))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timeline = json.loads(args.image_timeline.read_text())
    inventories, sources = [], []
    for image in timeline:
        assert len(image["digests"]) == 1, "Split mutable tags before source comparison"
        digest = image["digests"][0]
        directory = args.images / digest.split(":")[1][:12]
        manifest = json.loads((directory / "source_manifest.json").read_text())
        inspection = json.loads((directory / "image_inspect.json").read_text())[0]
        assert any(value.endswith("@" + digest) for value in inspection["RepoDigests"]), digest
        wanted = {entry["path"]: entry for entry in manifest["files"] if relevant(entry["path"])}
        content = {}
        with tarfile.open(directory / "sources.tar.gz") as archive:
            for member in archive:
                path = "/" + member.name
                if path in wanted:
                    data = archive.extractfile(member).read()
                    assert hashlib.sha256(data).hexdigest() == wanted[path]["sha256"]
                    content[path] = data.decode()
        assert content.keys() == wanted.keys()
        sources.append(content)
        git = [{"path": entry["path"], "head": entry["head"],
                "tracked_dirty_lines": len((entry["status"] or "").splitlines()),
                "status_sha256": hashlib.sha256((entry["status"] or "").encode()).hexdigest()}
               for entry in manifest["git"]]
        inventories.append({"image_name": image["image_name"], "digest": digest,
                            "image_id": inspection["Id"], "repo_digests": inspection["RepoDigests"],
                            "platform": inspection["Os"] + "/" + inspection["Architecture"],
                            "image_created": inspection["Created"], "image_size_bytes": inspection["Size"],
                            "capture_utc": manifest["captured_at_utc"], "git": git,
                            "packages": {name: entry["version"] for name, entry in manifest["packages"].items()},
                            "captured_source_count": len(manifest["files"]),
                            "archive_sha256": manifest["archive_sha256"],
                            "external_archive": str(directory / "sources.tar.gz"),
                            "relevant_files": list(wanted.values())})
    write_json(args.output / "image_source_inventory.json", inventories)
    changes = []
    diffs = args.output / "source_diffs"
    diffs.mkdir(exist_ok=True)
    for index in range(len(inventories) - 1):
        a, b = inventories[index:index + 2]
        aa, bb = sources[index:index + 2]
        ha = {entry["path"]: entry["sha256"] for entry in a["relevant_files"]}
        hb = {entry["path"]: entry["sha256"] for entry in b["relevant_files"]}
        delta = [{"path": path, "before_sha256": ha.get(path), "after_sha256": hb.get(path),
                  "change": "added" if path not in ha else "removed" if path not in hb else "changed"}
                 for path in sorted(ha.keys() | hb.keys()) if ha.get(path) != hb.get(path)]
        patch_name = f"{index + 1:02d}_to_{index + 2:02d}.diff"
        with (diffs / patch_name).open("w") as output:
            for entry in delta:
                path = entry["path"]
                # New monokernel bodies have their own cited excerpts; avoid duplicating an entire package.
                if "/sgl_amd_mono/" in path or path.endswith("/environ.py"):
                    continue
                output.writelines(difflib.unified_diff(aa.get(path, "").splitlines(True),
                                                      bb.get(path, "").splitlines(True),
                                                      fromfile=f"{a['digest']}{path}",
                                                      tofile=f"{b['digest']}{path}", n=3))
        changes.append({"from_image": a["image_name"], "to_image": b["image_name"],
                        "relevant_file_changes": delta, "unchanged_relevant_files": len(ha.keys() & hb.keys()) - sum(p in ha and p in hb for p in [e['path'] for e in delta]),
                        "diff_file": "source_diffs/" + patch_name})
    write_json(args.output / "source_changes.json", changes)
    print(json.dumps({"images_verified": len(inventories),
                      "relevant_file_counts": [len(i["relevant_files"]) for i in inventories],
                      "changed_file_counts": [len(c["relevant_file_changes"]) for c in changes]}))


if __name__ == "__main__":
    main()
