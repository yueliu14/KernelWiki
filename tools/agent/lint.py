#!/usr/bin/env python3
"""Agent-view lint for the kernel-level branch. Exit 1 on any failure.

- every relative markdown link resolves (slim.py --check);
- every cards.md is up to date with its corpus/decisions.yaml (build_cards.py --check);
- AGENTS.md stays a router (<= 12 KB);
- no HTML entity escapes in markdown (they cost tokens and read badly);
- no path into material removed from this branch (top_model/, workload/, _sources/).
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ENTITY = re.compile(r"&(#x?[0-9a-fA-F]+|quot|lt|gt|amp|nbsp);")
REMOVED = re.compile(r"\]\([^)]*knowledge/(top_model|workload|_sources)/|\]\((\.\./)+(top_model|workload|_sources)/")
fails = []

for tool in ("slim.py", "build_cards.py"):
    r = subprocess.run([sys.executable, str(HERE / tool), "--check"], capture_output=True, text=True)
    if r.returncode:
        fails.append(f"{tool} --check: {r.stdout.strip()}")

size = (ROOT / "AGENTS.md").stat().st_size
if size > 12_000:
    fails.append(f"AGENTS.md is {size} bytes (> 12000): keep it a router")

for p in sorted(ROOT.glob("**/*.md")):
    if ".git" in p.parts:
        continue
    text = p.read_text()
    rel = p.relative_to(ROOT)
    if ENTITY.search(text):
        fails.append(f"{rel}: HTML entity escapes")
    if REMOVED.search(text):
        fails.append(f"{rel}: link into removed material")

print("\n".join(fails) if fails else "lint ok")
sys.exit(1 if fails else 0)
