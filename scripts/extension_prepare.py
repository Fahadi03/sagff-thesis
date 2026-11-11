"""§12 extension, step 1: es/hi split files and the collision diagnostics (no model, no run).

Splits. For every (k, seed) entry of bnintent30-en in data/splits/manifest.json, the four
split files are read (sha256 verified against that manifest) and rewritten for es and hi
with ONLY the text replaced by the reference translation (hash verified against §12):
same uids, same order, same label ids. They go to data/extension/splits/ with their own
manifest.json, in the format of data/splits/manifest.json. The existing split
manifest and files are only read. Idempotent: an existing extension manifest must match.

Collision diagnostics (descriptive; nothing is removed, §12 "Rows"):
  - on the 4,096 clean rows, per language (en and bn as reference), with the §3.3 key:
    label conflicts, cross-split and within-split duplicates (rows and keys involved),
    and what §3.3 steps 2-4 would remove if they were applied to that language alone;
  - per run (k, seed): test rows colliding with the run's own training subset
    (§12 amendment 2026-11-09), mean and range over seeds.

Writes results/extension/collisions.md (numbers only).

Usage (from the repository root):  python scripts/extension_prepare.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import STEPS, dedup_key  # noqa: E402
from src.extension import (KS, LANGS, SEEDS, SOURCE_TRACK, SPLIT_BASE, TRACK,  # noqa: E402
                           colliding_test_uids, load_translations, read_jsonl, sha256_bytes)

MAIN = ROOT / "data" / "splits"
FILES = ("train", "val", "test", "val_full")
OUT = ROOT / "results" / "extension" / "collisions.md"


def jsonl_bytes(rows: list[dict]) -> bytes:
    """Same serialisation as scripts/make_splits.py."""
    return "".join(json.dumps({"uid": r["uid"], "text": r["text"], "label_id": r["label_id"]},
                              ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def manifest_bytes(m: dict) -> bytes:
    """Same layout as scripts/make_splits.py: header indented, one line per entry."""
    head = {key: v for key, v in m.items() if key != "splits"}
    body = json.dumps(head, ensure_ascii=False, indent=1)[:-2]
    entries = ",\n  ".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in m["splits"])
    return (body + ',\n "splits": [\n  ' + entries + "\n ]\n}\n").encode("utf-8")


def build_splits(tr: dict[str, dict[str, str]]) -> dict:
    main = json.loads((MAIN / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "generator": "scripts/extension_prepare.py",
        "condition": "clean",
        "description": "docs/PROTOCOL.md §12: the clean bnintent30-en splits (same uids, order and labels) "
                       "with the text replaced by the reference es/hi translations",
        "source_manifest": "data/splits/manifest.json",
        "source_track": SOURCE_TRACK,
        "translations_sha256": {lang: sha256_bytes((ROOT / "data" / "extension" / f"{lang}.jsonl").read_bytes())
                                for lang in LANGS},
        "seeds": main["seeds"], "ks": main["ks"], "files": main["files"],
        "label_names": {TRACK[lang]: main["label_names"][SOURCE_TRACK] for lang in LANGS},
        "splits": [],
    }
    for lang in LANGS:
        for e in (e for e in main["splits"] if e["track"] == SOURCE_TRACK):
            d = SPLIT_BASE / TRACK[lang] / Path(e["path"]).name
            d.mkdir(parents=True, exist_ok=True)
            sha, uids = {}, {}
            for name in FILES:
                raw = (MAIN / e["path"] / f"{name}.jsonl").read_bytes()
                if sha256_bytes(raw) != e["sha256"][name]:
                    raise RuntimeError(f"{e['path']}/{name}.jsonl does not match data/splits/manifest.json")
                rows = [json.loads(l) for l in raw.decode("utf-8").splitlines()]
                b = jsonl_bytes([{**r, "text": tr[lang][r["uid"]]} for r in rows])
                (d / f"{name}.jsonl").write_bytes(b)
                sha[name], uids[name] = sha256_bytes(b), [r["uid"] for r in rows]
            manifest["splits"].append({"track": TRACK[lang], "k": e["k"], "seed": e["seed"],
                                       "path": d.relative_to(SPLIT_BASE).as_posix(), "sha256": sha, "uids": uids})
    path = SPLIT_BASE / "manifest.json"
    new = manifest_bytes(manifest)
    if path.exists() and path.read_bytes() != new:
        sys.exit(f"STOP: regenerated splits do not match the existing {path.relative_to(ROOT)}")
    path.write_bytes(new)
    print(f"{len(manifest['splits'])} (track, k, seed) entries -> {path.relative_to(ROOT)}")
    return manifest


def diagnostics(rows: list[dict], lang: str) -> dict:
    """Collision counts for one language's text over the clean rows (nothing removed)."""
    key = {r["uid"]: dedup_key(r[lang]) for r in rows}
    intents, splits, per_split = defaultdict(set), defaultdict(set), defaultdict(int)
    for r in rows:
        intents[key[r["uid"]]].add(r["intent"])
        splits[key[r["uid"]]].add(r["split"])
        per_split[(r["split"], key[r["uid"]])] += 1
    conflict_keys = {k for k, v in intents.items() if len(v) > 1}
    cross_keys = {k for k, v in splits.items() if len(v) > 1}
    d = {"conflict_keys": len(conflict_keys),
         "conflict_rows": sum(key[r["uid"]] in conflict_keys for r in rows),
         "cross_keys": len(cross_keys),
         "cross_rows": sum(key[r["uid"]] in cross_keys for r in rows),
         "within_keys": sum(1 for n in per_split.values() if n > 1),
         "within_rows": sum(n - 1 for n in per_split.values() if n > 1)}
    # what §3.3 steps 2-4 would remove if applied to this language alone (diagnostic only)
    kept = [{"uid": r["uid"], "intent": r["intent"], "split": r["split"], "text": r[lang]} for r in rows]
    for step, _, fn in STEPS:
        kept, dropped = fn(kept, {lang: "text"})
        d[f"step{step}"] = len(dropped)
    return d


def main() -> None:
    tr = {lang: load_translations(lang) for lang in LANGS}
    manifest = build_splits(tr)
    clean = {lang: {r["uid"]: r for r in read_jsonl(ROOT / "data" / "clean" / f"bnintent30-{lang}.jsonl")}
             for lang in ("en", "bn")}
    names = json.loads((ROOT / "data" / "clean" / "bnintent30-en.label_names.json").read_text(encoding="utf-8"))
    rows = [{"uid": u, "intent": names[r["label_id"]], "split": r["split"], "en": r["text"],
             "bn": clean["bn"][u]["text"], "es": tr["es"][u], "hi": tr["hi"][u]}
            for u, r in sorted(clean["en"].items())]
    assert len(rows) == 4096

    L = ["# Extension: deduplication-key collisions on the translated text (docs/PROTOCOL.md §12)", "",
         "Generated by `python scripts/extension_prepare.py`. Do not edit by hand. Numbers only. Descriptive: "
         "no row is removed or re-cleaned; the extension uses exactly the clean rows and splits of "
         "`bnintent30-en` / `bnintent30-bn`.", "",
         f"Rows: the {len(rows)} clean uids (official splits train/val/test). Key: the §3.3 dedup key "
         "(sorted multiset of normalised tokens), computed on each language's own text. en and bn are shown as "
         "a reference: their rows were cleaned on the union of both languages, so all their counts are 0.", "",
         "## On the clean rows", "",
         "| language | label-conflict keys | rows in conflicts | cross-split keys | rows in cross-split keys "
         "| within-split duplicate keys | within-split extra rows | §3.3 step 2 would drop | step 3 | step 4 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lang in ("en", "bn") + LANGS:
        d = diagnostics(rows, lang)
        L.append(f"| {lang} | {d['conflict_keys']} | {d['conflict_rows']} | {d['cross_keys']} | {d['cross_rows']} | "
                 f"{d['within_keys']} | {d['within_rows']} | {d['step2']} | {d['step3']} | {d['step4']} |")

    L += ["", "## Per run: test rows colliding with the run's own training subset", "",
          "§12 amendment 2026-11-09: a test row collides if its key equals the key of any row in that run's "
          "training subset (k- and seed-specific). These rows are removed from scoring in the rescoring "
          "sensitivity analysis. Mean (min-max) over the five seeds; test split of 856 rows.", "",
          "| language | k=5 | k=10 | k=20 | k=full |", "| --- | --- | --- | --- | --- |"]
    text = {lang: {r["uid"]: r[lang] for r in rows} for lang in ("en", "bn") + LANGS}
    by = {(e["k"], e["seed"]): e for e in manifest["splits"] if e["track"] == TRACK["es"]}
    for lang in ("en", "bn") + LANGS:
        cells = []
        for k in KS:
            n = [len(colliding_test_uids([text[lang][u] for u in by[(k, s)]["uids"]["train"]],
                                         [(u, text[lang][u]) for u in by[(k, s)]["uids"]["test"]]))
                 for s in SEEDS]
            cells.append(f"{sum(n) / len(n):.1f} ({min(n)}-{max(n)})" if max(n) else "0")
        L.append(f"| {lang} | " + " | ".join(cells) + " |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
