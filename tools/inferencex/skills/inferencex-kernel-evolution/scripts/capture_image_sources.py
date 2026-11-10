#!/usr/bin/env python3
"""Run inside a pinned image on CPU; collect source bytes without importing GPU packages."""
import hashlib
import importlib.metadata
import json
import os
import site
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
roots = []
access_gaps = []
for parent in [Path("/sgl-workspace"), Path("/workspace"), Path("/app"), Path("/opt"), Path("/root")]:
    if parent.is_dir():
        try:
            children = list(parent.iterdir())
        except PermissionError:
            access_gaps.append(str(parent))
            continue
        for child in children:
            if any(k in child.name.lower() for k in ["sglang", "vllm", "aiter", "flydsl", "mono", "triton"]):
                roots.append(child)
for direct in ["/sglang", "/vllm", "/aiter", "/sgl-amd-mono"]:
    if Path(direct).is_dir():
        roots.append(Path(direct))
for directory in site.getsitepackages():
    for package in ["sglang", "vllm", "aiter", "flydsl", "sgl_kernel", "sgl_amd_mono"]:
        candidate = Path(directory) / package
        if candidate.is_dir():
            roots.append(candidate)
roots = sorted(set(p.resolve() for p in roots))
roots = [p for p in roots if not any(q != p and q in p.parents for q in roots)]
extensions = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cuh", ".cu", ".hip", ".mlir", ".s", ".S", ".yaml", ".yml", ".json", ".toml", ".cmake", ".md"}
files, git = [], []
for root in roots:
    if (root / ".git").exists():
        def git_read(*args):
            result = subprocess.run(["git", "-c", f"safe.directory={root}", "-C", str(root), *args], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        git.append({"path": str(root), "head": git_read("rev-parse", "HEAD"),
                    "status": git_read("status", "--porcelain", "--untracked-files=no")})
    for current, directories, names in os.walk(root):
        directories[:] = [d for d in directories if d not in {".git", "__pycache__", ".venv", "node_modules", "build", "dist", ".cache"}]
        for name in names:
            path = Path(current) / name
            if path.is_symlink() or path.suffix not in extensions or path.stat().st_size > 10_000_000:
                continue
            data = path.read_bytes()
            files.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
with tarfile.open(out / "sources.tar.gz", "w:gz") as archive:
    for entry in files:
        archive.add(entry["path"], arcname=entry["path"].lstrip("/"), recursive=False)
packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name", "")
    if any(term in name.lower() for term in ["sglang", "vllm", "aiter", "flydsl", "triton", "torch", "mono"]):
        packages[name] = {"version": distribution.version, "direct_url": distribution.read_text("direct_url.json")}
manifest = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "roots": list(map(str, roots)), "inaccessible_roots": access_gaps, "git": git, "packages": packages, "files": files,
            "method": "CPU-only source export; no framework imports, model execution, or GPU profiling",
            "archive_sha256": hashlib.sha256((out / "sources.tar.gz").read_bytes()).hexdigest()}
(out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"roots": manifest["roots"], "git": git, "packages": packages, "source_files": len(files), "archive_bytes": (out / "sources.tar.gz").stat().st_size}))
