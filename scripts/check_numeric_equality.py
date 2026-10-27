"""Check that regenerated outputs carry exactly the same numbers as a base commit.

For every generated Markdown file (results/tables/*.md, results/figures/captions.md,
results/runs_anomalies.md) the file on disk is compared with its version at the base
commit:
  - table rows (lines starting with "|"): the numeric tokens of every row, in order,
    must be identical, and so must the number of table rows. Any difference fails.
  - other lines (prose): numeric tokens are compared per file; differences are listed
    for review, because a reworded label can legitimately remove a number such as a
    section number.
Figures (results/figures/*.png, *.pdf) must be byte-identical.
The full unified text diff of every Markdown file that differs is printed.

Exit status 1 if any table cell or figure differs, else 0.

Usage:  python scripts/check_numeric_equality.py <base-commit>
"""
import difflib
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?(?:[eE][-+−]?\d+)?")
SEPARATOR = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


def at_commit(rev: str, path: str) -> bytes | None:
    """The file's bytes at `rev`, or None if it did not exist there."""
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def split_numbers(text: str) -> tuple[list[list[str]], list[str]]:
    """(numeric tokens per table row, numeric tokens of all prose lines)."""
    rows, prose = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            if not SEPARATOR.match(line):
                rows.append(NUMBER.findall(line))
        else:
            prose += NUMBER.findall(line)
    return rows, prose


def main() -> int:
    """Compare every generated file with the base commit and report."""
    base = sys.argv[1]
    md = sorted(ROOT.glob("results/tables/*.md")) + [ROOT / "results/figures/captions.md",
                                                     ROOT / "results/runs_anomalies.md"]
    figures = sorted(ROOT.glob("results/figures/*.png")) + sorted(ROOT.glob("results/figures/*.pdf"))
    failures, notes, diffs, n_cells = [], [], [], 0
    for p in md:
        rel = p.relative_to(ROOT).as_posix()
        old = at_commit(base, rel)
        if old is None:
            notes.append(f"{rel}: new file (not at {base}); not compared")
            continue
        old_t, new_t = old.decode("utf-8"), p.read_text(encoding="utf-8")
        (old_rows, old_prose), (new_rows, new_prose) = split_numbers(old_t), split_numbers(new_t)
        n_cells += sum(len(r) for r in new_rows)
        if len(old_rows) != len(new_rows):
            failures.append(f"{rel}: {len(old_rows)} table rows at {base}, {len(new_rows)} now")
        for i, (a, b) in enumerate(zip(old_rows, new_rows)):
            if a != b:
                failures.append(f"{rel}: table row {i + 1} numbers differ: {a} -> {b}")
        if old_prose != new_prose:
            gone, added = Counter(old_prose) - Counter(new_prose), Counter(new_prose) - Counter(old_prose)
            notes.append(f"{rel}: prose numbers differ: removed {dict(gone) or 'none'}, "
                         f"added {dict(added) or 'none'}"
                         + ("" if gone or added else " (same numbers, different order)"))
        if old_t != new_t:
            diffs += difflib.unified_diff(old_t.splitlines(), new_t.splitlines(),
                                          f"{base}:{rel}", f"now:{rel}", lineterm="", n=0)
    for p in figures:
        rel = p.relative_to(ROOT).as_posix()
        if at_commit(base, rel) != p.read_bytes():
            failures.append(f"{rel}: not byte-identical to {base}")
    print(f"Compared {len(md)} Markdown outputs ({n_cells} numeric table cells) and "
          f"{len(figures)} figure files against {base}.")
    print(f"Table-cell or figure differences: {len(failures)}")
    for f in failures:
        print("  FAIL", f)
    for n in notes:
        print("  NOTE", n)
    print("\nFull text diff of Markdown outputs:" + ("" if diffs else " none"))
    print("\n".join(diffs))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
