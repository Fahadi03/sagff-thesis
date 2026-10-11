"""Verify split files on disk against their manifests (CLAUDE.md R3).

Both conditions are checked: data/splits/manifest.json (clean) and
data/splits_dirty/manifest.json (leaky condition, §5). For every
(track, k, seed) entry:
  - every split file listed in the entry exists and its sha256 matches
  - the uid order inside each file matches the manifest's uid list
and globally:
  - the pinned upstream commits in configs/data.yaml match each manifest
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
MANIFESTS = (ROOT / "data" / "splits" / "manifest.json",
             ROOT / "data" / "splits_dirty" / "manifest.json")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    problems, n_files, n_entries = [], 0, 0

    for ds in ("bnintent30", "banking77"):
        for name, digest in cfg[ds]["files_sha256"].items():
            p = ROOT / "data" / "raw" / ds / name
            if not p.exists():
                problems.append(f"raw file missing: {p.relative_to(ROOT)}")
            elif sha256_file(p) != digest:
                problems.append(f"raw file changed: {p.relative_to(ROOT)}")

    for mpath in MANIFESTS:
        if not mpath.exists():
            problems.append(f"manifest missing: {mpath.relative_to(ROOT)}")
            continue
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        for ds in ("bnintent30", "banking77"):
            if cfg[ds]["commit"] != manifest["sources"][ds]["commit"]:
                problems.append(f"{mpath.parent.name}: {ds} commit in configs/data.yaml "
                                f"{cfg[ds]['commit']} != manifest {manifest['sources'][ds]['commit']}")
        for e in manifest["splits"]:
            n_entries += 1
            d = mpath.parent / e["path"]
            for name, digest in e["sha256"].items():
                n_files += 1
                p = d / f"{name}.jsonl"
                if not p.exists():
                    problems.append(f"missing: {p.relative_to(ROOT)}")
                    continue
                if sha256_file(p) != digest:
                    problems.append(f"sha256 mismatch: {p.relative_to(ROOT)}")
                uids = [json.loads(line)["uid"]
                        for line in p.read_text(encoding="utf-8").splitlines()]
                if uids != e["uids"][name]:
                    problems.append(f"uid order mismatch: {p.relative_to(ROOT)}")

    if problems:
        print(f"FAIL: {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  -", p)
        return 1
    print(f"OK: {n_files} split files in {n_entries} (track, k, seed) entries across "
          f"{len(MANIFESTS)} manifests match; upstream pins and raw checksums match "
          "configs/data.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
