"""Generate the fixed k-shot splits for every (track, k, seed) (CLAUDE.md R3, R5, R9).

For each (track, k, seed):
  train  at most k rows per class from the track's cleaned train portion
         (k = "full": all of it)
  val    the track's full val portion (fixed; identical for every k and seed)
  test   the track's full test portion, ALL classes (R5)

Sampling is done on uids. For each class, train uids are ranked by
sha256(seed, uid) and the first k are taken. Consequences:
  - deterministic and independent of any RNG library or library version;
  - nested: for a fixed seed, the k=5 set is a subset of k=10, which is a
    subset of k=20 - so the k-trend compares growing training sets, not
    unrelated resamples;
  - R9: the two BNIntent30 tracks share uids and labels, so they get identical
    uid lists in identical order automatically.
The train file is ordered by sha256(seed, "order", uid); val/test by uid.

Manifest policy: data/splits/manifest.json is written only if it does not
exist (or with --rewrite-manifest). Otherwise the freshly generated splits are
checked against it and the script exits non-zero on any mismatch.

Usage:  python scripts/make_splits.py [--rewrite-manifest]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.data import SPLITS, TRACKS, build_tracks, hash_key  # noqa: E402

SEEDS = (13, 42, 87, 100, 2026)  # R4
KS = (5, 10, 20, "full")          # R4
SPLITS_DIR = ROOT / "data" / "splits"
MANIFEST = SPLITS_DIR / "manifest.json"


def split_dir(base: Path, track: str, k, seed: int) -> Path:
    return base / track / f"k{k}_seed{seed}"


def sample_train(rows: list[dict], k, seed: int) -> list[dict]:
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label_id"]].append(r)
    chosen = []
    for lid in sorted(by_class):
        ranked = sorted(by_class[lid], key=lambda r: hash_key(str(seed), r["uid"]))
        chosen += ranked if k == "full" else ranked[:k]
    return sorted(chosen, key=lambda r: hash_key(str(seed), "order", r["uid"]))


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps({"uid": r["uid"], "text": r["text"], "label_id": r["label_id"]},
                              ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def generate(out_dir: Path, tracks: dict | None = None) -> dict:
    """Write all split files under out_dir and return the manifest dict."""
    if tracks is None:
        tracks, _ = build_tracks()
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    manifest = {
        "generator": "scripts/make_splits.py",
        "sources": {
            "bnintent30": {"repo": data_cfg["bnintent30"]["repo"],
                           "commit": data_cfg["bnintent30"]["commit"],
                           "archive_url": data_cfg["bnintent30"]["archive_url"]},
            "banking77": {"repo": data_cfg["banking77"]["repo"],
                          "commit": data_cfg["banking77"]["commit"],
                          "urls": data_cfg["banking77"]["urls"],
                          "files_sha256": data_cfg["banking77"]["files_sha256"]},
        },
        "seeds": list(SEEDS), "ks": [str(k) for k in KS],
        "label_names": {t: tracks[t]["label_names"] for t in TRACKS},
        "splits": [],
    }
    for track in TRACKS:
        t = tracks[track]
        portion = {s: [r for r in t["rows"] if r["split"] == s] for s in SPLITS}
        for k in KS:
            for seed in SEEDS:
                files = {"train": sample_train(portion["train"], k, seed),
                         "val": portion["val"], "test": portion["test"]}
                d = split_dir(out_dir, track, k, seed)
                d.mkdir(parents=True, exist_ok=True)
                sha = {}
                for s, rows in files.items():
                    b = jsonl_bytes(rows)
                    (d / f"{s}.jsonl").write_bytes(b)
                    sha[s] = hashlib.sha256(b).hexdigest()
                names = t["label_names"]
                meta = {
                    "track": track, "k": str(k), "seed": seed,
                    "label_names": names,
                    "totals": {s: len(files[s]) for s in SPLITS},
                    "per_class": {s: {names[i]: c for i, c in
                                      sorted(Counter(r["label_id"] for r in files[s]).items())}
                                  for s in SPLITS},
                    "sha256": sha,
                    "sampling": "per class: first k train uids ranked by sha256(seed, uid); "
                                "val/test: full fixed portions",
                }
                (d / "meta.json").write_bytes(
                    (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
                manifest["splits"].append({
                    "track": track, "k": str(k), "seed": seed,
                    "path": d.relative_to(out_dir).as_posix(),
                    "sha256": sha,
                    "uids": {s: [r["uid"] for r in files[s]] for s in SPLITS},
                })
    return manifest


def manifest_bytes(m: dict) -> bytes:
    """Header fields indented; one compact line per (track, k, seed) entry."""
    head = {key: v for key, v in m.items() if key != "splits"}
    body = json.dumps(head, ensure_ascii=False, indent=1)[:-2]  # drop closing "\n}"
    entries = ",\n  ".join(json.dumps(e, ensure_ascii=False, separators=(",", ":"))
                           for e in m["splits"])
    return (body + ',\n "splits": [\n  ' + entries + "\n ]\n}\n").encode("utf-8")


def compare(new: dict, old: dict) -> list[str]:
    problems = []
    if new["sources"] != old.get("sources"):
        problems.append("upstream sources (commit/URL/checksum) differ from the manifest")
    old_by = {(e["track"], e["k"], e["seed"]): e for e in old.get("splits", [])}
    for e in new["splits"]:
        key = (e["track"], e["k"], e["seed"])
        o = old_by.pop(key, None)
        if o is None:
            problems.append(f"{key}: not in manifest")
            continue
        for s in SPLITS:
            if e["sha256"][s] != o["sha256"][s]:
                problems.append(f"{key} {s}: sha256 {e['sha256'][s][:12]}.. != manifest "
                                f"{o['sha256'][s][:12]}..")
            if e["uids"][s] != o["uids"][s]:
                problems.append(f"{key} {s}: uid list differs from manifest")
    problems += [f"{key}: in manifest but not generated" for key in old_by]
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite-manifest", action="store_true",
                    help="deliberately overwrite data/splits/manifest.json")
    args = ap.parse_args()
    manifest = generate(SPLITS_DIR)
    n = len(manifest["splits"])
    if not MANIFEST.exists() or args.rewrite_manifest:
        MANIFEST.write_bytes(manifest_bytes(manifest))
        print(f"Generated {n} (track, k, seed) splits; wrote {MANIFEST.relative_to(ROOT)}")
        return
    problems = compare(manifest, json.loads(MANIFEST.read_text(encoding="utf-8")))
    if problems:
        print(f"Generated {n} splits, but they do NOT match the committed manifest:")
        for p in problems[:50]:
            print("  -", p)
        sys.exit(1)
    print(f"Generated {n} splits; all match {MANIFEST.relative_to(ROOT)} (manifest not modified)")


if __name__ == "__main__":
    main()
