"""Generate the fixed k-shot splits for every (track, k, seed) (CLAUDE.md R3, R5, R9, §3.5).

Two conditions, written to separate trees with separate manifests so they
cannot be confused:
  clean  data/splits/        all three tracks, cleaned per §3.3
  dirty  data/splits_dirty/  bnintent30-bn only, official assignment after
                             step 1, steps 2-4 NOT applied (§5 leaky condition)

For each (track, k, seed), four files:
  train     at most k rows per class from the train portion (k="full": all)
  val       at most k rows per class from the val portion (k-matched, §3.5).
            The ONLY split model selection may use.
  test      the full test portion, ALL classes (R5)
  val_full  the full val portion, for the val_full_diagnostic metric only
            ("used_for_selection": false)

Sampling (R3, requirements): for each class, uids are ranked by
sha256(seed, uid) and the first k are taken - for train and val alike.
  - no RNG library, so no library upgrade can change the splits;
  - nested: for a fixed (track, seed), k=5 is a subset of k=10, of k=20, of full,
    for train AND val, so training-set size is the only thing that varies with k;
  - R9: the two BNIntent30 tracks share uids and labels, so they get identical
    uid lists in identical order automatically.
train.jsonl is ordered by sha256(seed, "order", uid); the other files by uid.

Manifest policy: a manifest is written only if it does not exist (or with
--rewrite-manifest). Otherwise freshly generated splits are checked against it
and the script exits non-zero on any mismatch.

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

from src.data import build_dirty_tracks, build_tracks, hash_key  # noqa: E402

SEEDS = (13, 42, 87, 100, 2026)  # R4
KS = (5, 10, 20, "full")          # R4
FILES = ("train", "val", "test", "val_full")
CONDITIONS = {
    "clean": ROOT / "data" / "splits",
    "dirty": ROOT / "data" / "splits_dirty",
}
SPLITS_DIR = CONDITIONS["clean"]
MANIFEST = SPLITS_DIR / "manifest.json"


def split_dir(base: Path, track: str, k, seed: int) -> Path:
    return base / track / f"k{k}_seed{seed}"


def sample_kshot(rows: list[dict], k, seed: int) -> list[dict]:
    """First k uids per class by sha256(seed, uid) rank. Nested in k by construction."""
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label_id"]].append(r)
    chosen = []
    for lid in sorted(by_class):
        ranked = sorted(by_class[lid], key=lambda r: hash_key(str(seed), r["uid"]))
        chosen += ranked if k == "full" else ranked[:k]
    return chosen


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps({"uid": r["uid"], "text": r["text"], "label_id": r["label_id"]},
                              ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def sources() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    return {
        "bnintent30": {"repo": cfg["bnintent30"]["repo"], "commit": cfg["bnintent30"]["commit"],
                       "archive_url": cfg["bnintent30"]["archive_url"]},
        "banking77": {"repo": cfg["banking77"]["repo"], "commit": cfg["banking77"]["commit"],
                      "urls": cfg["banking77"]["urls"],
                      "files_sha256": cfg["banking77"]["files_sha256"]},
    }


def generate(out_dir: Path, tracks: dict, condition: str = "clean") -> dict:
    """Write all split files under out_dir and return the manifest dict."""
    manifest = {
        "generator": "scripts/make_splits.py",
        "condition": condition,
        "description": {
            "clean": "cleaned per CLAUDE.md §3.3 (steps 1-4)",
            "dirty": "official BNIntent30 assignment, step 1 applied, steps 2-4 NOT applied "
                     "(leaky condition, CLAUDE.md §5)",
        }[condition],
        "sources": sources(),
        "seeds": list(SEEDS), "ks": [str(k) for k in KS],
        "files": {"train": "k-shot train", "val": "k-matched val, the only split used for "
                  "selection", "test": "full test", "val_full": "full val, diagnostic only, "
                  "used_for_selection=false"},
        "label_names": {t: tracks[t]["label_names"] for t in tracks},
        "splits": [],
    }
    for track, t in tracks.items():
        portion = {s: [r for r in t["rows"] if r["split"] == s] for s in ("train", "val", "test")}
        for k in KS:
            for seed in SEEDS:
                train = sample_kshot(portion["train"], k, seed)
                files = {
                    "train": sorted(train, key=lambda r: hash_key(str(seed), "order", r["uid"])),
                    "val": sorted(sample_kshot(portion["val"], k, seed), key=lambda r: r["uid"]),
                    "test": portion["test"],
                    "val_full": portion["val"],
                }
                d = split_dir(out_dir, track, k, seed)
                d.mkdir(parents=True, exist_ok=True)
                sha = {}
                for name in FILES:
                    b = jsonl_bytes(files[name])
                    (d / f"{name}.jsonl").write_bytes(b)
                    sha[name] = hashlib.sha256(b).hexdigest()
                names = t["label_names"]
                meta = {
                    "track": track, "condition": condition, "k": str(k), "seed": seed,
                    "label_names": names,
                    "totals": {f: len(files[f]) for f in FILES},
                    "per_class": {f: {names[i]: c for i, c in
                                      sorted(Counter(r["label_id"] for r in files[f]).items())}
                                  for f in FILES},
                    "sha256": sha,
                    "val_full_used_for_selection": False,
                    "sampling": "train and val: per class, first k uids ranked by "
                                "sha256(seed, uid), nested across k; test and val_full: full "
                                "fixed portions",
                }
                (d / "meta.json").write_bytes(
                    (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
                manifest["splits"].append({
                    "track": track, "k": str(k), "seed": seed,
                    "path": d.relative_to(out_dir).as_posix(),
                    "sha256": sha,
                    "uids": {f: [r["uid"] for r in files[f]] for f in FILES},
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
    for field in ("condition", "sources", "label_names"):
        if new[field] != old.get(field):
            problems.append(f"{field} differs from the manifest")
    old_by = {(e["track"], e["k"], e["seed"]): e for e in old.get("splits", [])}
    for e in new["splits"]:
        key = (e["track"], e["k"], e["seed"])
        o = old_by.pop(key, None)
        if o is None:
            problems.append(f"{key}: not in manifest")
            continue
        for f in FILES:
            if e["sha256"][f] != o["sha256"].get(f):
                problems.append(f"{key} {f}: sha256 {e['sha256'][f][:12]}.. != manifest "
                                f"{str(o['sha256'].get(f))[:12]}..")
            if e["uids"][f] != o["uids"].get(f):
                problems.append(f"{key} {f}: uid list differs from manifest")
    problems += [f"{key}: in manifest but not generated" for key in old_by]
    return problems


def build_condition_tracks(condition: str) -> dict:
    return build_tracks()[0] if condition == "clean" else build_dirty_tracks()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite-manifest", action="store_true",
                    help="deliberately overwrite the manifest(s)")
    args = ap.parse_args()
    failed = False
    for condition, out_dir in CONDITIONS.items():
        manifest = generate(out_dir, build_condition_tracks(condition), condition)
        path = out_dir / "manifest.json"
        rel = path.relative_to(ROOT)
        n = len(manifest["splits"])
        if not path.exists() or args.rewrite_manifest:
            path.write_bytes(manifest_bytes(manifest))
            print(f"[{condition}] generated {n} (track, k, seed) splits; wrote {rel}")
            continue
        problems = compare(manifest, json.loads(path.read_text(encoding="utf-8")))
        if problems:
            failed = True
            print(f"[{condition}] generated {n} splits, but they do NOT match {rel}:")
            for p in problems[:50]:
                print("  -", p)
        else:
            print(f"[{condition}] generated {n} splits; all match {rel} (manifest not modified)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
