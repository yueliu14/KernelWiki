#!/usr/bin/env python3
"""Idempotent slimming pass over the kernel-level knowledge tree (agent/kernel-level-lean branch).

1. Backend source walkthroughs (`operator/*/backend/*_aiter.md`): a `Symbols:` line lists every symbol as a pinned
   GitHub link (~40% of those pages' bytes). Keep the names, drop the per-symbol URLs; the file's own `[Source]` link
   on the line above still points at the pinned revision.
2. Operator READMEs: point the corpus links at the rendered `cards.md` instead of the removed generated pages, and
   drop the imported GEAK front-matter `<details>` block (metadata, not knowledge).
3. Every markdown file: a relative link whose target no longer exists on this branch (model/workload/serving
   material, generated corpus pages) is reduced to its text, so no link leads an agent into a dead end.

    python3 tools/agent/slim.py          # rewrite in place
    python3 tools/agent/slim.py --check  # exit 1 if a rewrite would change anything
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = re.compile(r"^(Symbols: )(.*)$", re.M)
SYMLINK = re.compile(r"\[(`[^`]+`)\]\([^)]*\)")
DETAILS = re.compile(r"\n<details>\n<summary>Original GEAK metadata</summary>\n.*?</details>\n", re.S)
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
CORPUS_LINKS = [
    (re.compile(r"\[Corpus decisions\]\(corpus/decisions\.md\) · \[Source evidence\]\(corpus/source_evidence\.md\)"
                r"( · \[Extraction scope\]\(corpus/scope\.yaml\))?"), "[Decision cards](cards.md)"),
    (re.compile(r"\[Corpus decisions\]\(corpus/decisions\.md\)"), "[Decision cards](cards.md)"),
]


def slim_symbols(text):
    return SYMBOLS.sub(lambda m: m.group(1) + SYMLINK.sub(r"\1", m.group(2)), text)


def fix_links(text, path):
    def repl(m):
        label, target = m.group(1), m.group(2)
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            return m.group(0)
        dest = target.split("#", 1)[0]
        if not dest or (path.parent / dest).resolve().exists():
            return m.group(0)
        return label
    return LINK.sub(repl, text)


def process(path):
    text = path.read_text()
    new = text
    if path.match("knowledge/operator/*/backend/*_aiter.md"):
        new = slim_symbols(new)
    if path.match("knowledge/operator/*/README.md"):
        for pat, rep in CORPUS_LINKS:
            new = pat.sub(rep, new)
        new = DETAILS.sub("\n", new)
    new = fix_links(new, path)
    return text, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    changed = saved = 0
    for p in sorted(ROOT.glob("**/*.md")):
        if ".git" in p.parts:
            continue
        old, new = process(p)
        if old != new:
            changed += 1
            saved += len(old) - len(new)
            if not a.check:
                p.write_text(new)
    print(f"{changed} files {'would change' if a.check else 'changed'}, {saved:,} bytes removed")
    if a.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
