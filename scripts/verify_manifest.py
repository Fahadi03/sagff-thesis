"""Verify the split files on disk against data/splits/manifest.json (CLAUDE.md R3).

Checks, for every (track, k, seed) in the manifest:
  - train/val/test.jsonl exist and their sha256 matches the manifest
  - the uid order inside each file matches the manifest's uid list
and globally:
  - the pinned upstream commits in configs/data.yaml match the manifest
  - every raw file's sha256 matches configs/data.yaml

Exits non-zero on any mismatch.

Reproduction:  python scripts/download_data.py
               python scripts/make_splits.py
               python scripts/verify_manifest.py
"""
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    problems = []

    for ds in ("bnintent30", "banking77"):
        if cfg[ds]["commit"] != manifest["sources"][ds]["commit"]:
            problems.append(f"{ds}: configs/data.yaml commit {cfg[ds]['commit']} != manifest "
                            f"{manifest['sources'][ds]['commit']}")
        for name, digest in cfg[ds]["files_sha256"].items():
            p = ROOT / "data" / "raw" / ds / name
            if not p.exists():
                problems.append(f"raw file missing: {p.relative_to(ROOT)}")
            elif sha256_file(p) != digest:
                problems.append(f"raw file changed: {p.relative_to(ROOT)}")

    for e in manifest["splits"]:
        d = SPLITS_DIR / e["path"]
        for s in ("train", "val", "test"):
            p = d / f"{s}.jsonl"
            if not p.exists():
                problems.append(f"missing: {p.relative_to(ROOT)}")
                continue
            if sha256_file(p) != e["sha256"][s]:
                problems.append(f"sha256 mismatch: {p.relative_to(ROOT)}")
            uids = [json.loads(line)["uid"] for line in p.read_text(encoding="utf-8").splitlines()]
            if uids != e["uids"][s]:
                problems.append(f"uid order mismatch: {p.relative_to(ROOT)}")

    n_files = 3 * len(manifest["splits"])
    if problems:
        print(f"FAIL: {len(problems)} problem(s) across {n_files} split files:")
        for p in problems[:50]:
            print("  -", p)
        return 1
    print(f"OK: {n_files} split files ({len(manifest['splits'])} (track, k, seed) entries) "
          "match the manifest; upstream pins and raw checksums match configs/data.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
