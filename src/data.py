"""Load, clean and assign uids for all three tracks (CLAUDE.md §3).

Tracks (§3.1):
    bnintent30-en  BNIntent30 `text` field     (English)
    bnintent30-bn  BNIntent30 `bn_text` field  (Bangla)
    banking77      BANKING77 CSVs              (English)

The two BNIntent30 tracks are built from ONE shared cleaned row set (R9), so
they contain the same uids, labels and split assignment; only the text differs.

Text comparison key: every "same text" decision in this module (conflicts,
cross-split duplicates) and in the tests uses `norm_text` = whitespace
collapsed, nothing else. Stored text is never edited.

Usage:  python -m src.data      # writes data/clean/ and results/tables/data_cleaning.md
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REPORT = ROOT / "results" / "tables" / "data_cleaning.md"

TRACKS = ("bnintent30-en", "bnintent30-bn", "banking77")
SPLITS = ("train", "val", "test")
# §3.3 step 3: when a text occurs in several splits, the occurrence in the
# highest-priority split is kept. Keep the measuring instrument intact.
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
FAILED_MARKER = "TRANSLATION_FAILED"
# §3.3 step 1 expected totals (published paper). Asserted, never adjusted.
STEP1_EXPECTED = {"train": 2952, "val": 595, "test": 886}
LANG_FIELDS = {"bn": "bn_text", "en": "text"}

# BANKING77 has no official val split. We carve 1/6 of each class's training
# rows as val. Why 1/6: BNIntent30's official split is 100 train : 20 val per
# class, i.e. val = 1/6 of the train+val pool. Using the same proportion keeps
# the role of the val set comparable across tracks. The carve is fixed (one
# carve for all k and seeds, exactly like BNIntent30's fixed official val) and
# group-aware: identical texts always land on the same side, so the carve itself
# can never create train/val overlap.
B77_VAL_FRACTION = 1 / 6
B77_VAL_SALT = "banking77-val-carve-v1"


def norm_text(s: str) -> str:
    return " ".join(s.split())


def hash_key(*parts: str) -> str:
    """Version-independent deterministic ordering key (no RNG library involved)."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# BNIntent30
# --------------------------------------------------------------------------- #
def load_bnintent30_raw(raw_dir: Path = RAW / "bnintent30") -> list[dict]:
    """All 4,500 raw rows. uid = source file stem + row index in that file (§3.3 step 4)."""
    files = sorted(raw_dir.glob("*.json"))
    if len(files) != 30:
        raise FileNotFoundError(
            f"Expected 30 BNIntent30 JSON files in {raw_dir}, found {len(files)}. "
            "Run: python scripts/download_data.py")
    rows = []
    for f in files:
        for i, r in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            rows.append({"uid": f"bn30.{f.stem}.{i:04d}", "text": r["text"],
                         "bn_text": r["bn_text"], "intent": r["intent"], "split": r["split"]})
    return rows


def _split_counts(rows: list[dict]) -> dict:
    c = Counter(r["split"] for r in rows)
    return {s: c.get(s, 0) for s in SPLITS}


def _texts_with_conflict(rows: list[dict], field: str) -> set[str]:
    labels = defaultdict(set)
    for r in rows:
        labels[norm_text(r[field])].add(r["intent"])
    return {t for t, ls in labels.items() if len(ls) > 1}


def _cross_split_losers(rows: list[dict], field: str) -> set[str]:
    """uids of lower-priority occurrences of texts that appear in >1 split."""
    by_text = defaultdict(list)
    for r in rows:
        by_text[norm_text(r[field])].append(r)
    losers = set()
    for group in by_text.values():
        if len({r["split"] for r in group}) < 2:
            continue
        best = min(SPLIT_PRIORITY[r["split"]] for r in group)
        losers |= {r["uid"] for r in group if SPLIT_PRIORITY[r["split"]] != best}
    return losers


def clean_bnintent30(rows: list[dict]) -> tuple[list[dict], dict]:
    """Apply §3.3 steps 1-3 in order, at row level, to the union of problems in
    either language (R9). Returns (clean rows, report dict)."""
    rep = {"raw": _split_counts(rows), "steps": []}

    # Step 1 - drop TRANSLATION_FAILED rows, then assert the published totals.
    failed = [r for r in rows if r["bn_text"].strip() == FAILED_MARKER]
    rows = [r for r in rows if r["bn_text"].strip() != FAILED_MARKER]
    after1 = _split_counts(rows)
    rep["steps"].append({"step": 1, "name": f"bn_text is literally {FAILED_MARKER}",
                         "removed": _split_counts(failed), "remaining": after1,
                         "by_language": {"bn": len(failed), "en": 0, "both": 0}})
    if after1 != STEP1_EXPECTED:
        raise AssertionError(
            f"§3.3 step 1 totals do not match the published paper. Expected "
            f"{STEP1_EXPECTED} (total 4,433), got {after1} (total {sum(after1.values())}). "
            "Something upstream changed. STOP and report; do not adjust the expectation.")
    rep["step1_total"] = sum(after1.values())

    # Step 2 - drop EVERY row whose text (in either language) carries >1 intent.
    conflict = {lang: _texts_with_conflict(rows, f) for lang, f in LANG_FIELDS.items()}
    hit = {lang: {r["uid"] for r in rows if norm_text(r[f]) in conflict[lang]}
           for lang, f in LANG_FIELDS.items()}
    drop = hit["bn"] | hit["en"]
    removed = [r for r in rows if r["uid"] in drop]
    rows = [r for r in rows if r["uid"] not in drop]
    rep["steps"].append({"step": 2, "name": "label conflict (same text, >1 intent; all rows dropped)",
                         "removed": _split_counts(removed), "remaining": _split_counts(rows),
                         "conflict_texts": {lang: len(v) for lang, v in conflict.items()},
                         "by_language": _by_language(hit)})

    # Step 3 - cross-split duplicates: keep test > val > train occurrence.
    hit = {lang: _cross_split_losers(rows, f) for lang, f in LANG_FIELDS.items()}
    cross_texts = {lang: _n_cross_split_texts(rows, f) for lang, f in LANG_FIELDS.items()}
    drop = hit["bn"] | hit["en"]
    removed = [r for r in rows if r["uid"] in drop]
    rows = [r for r in rows if r["uid"] not in drop]
    rep["steps"].append({"step": 3, "name": "cross-split duplicate (priority test > val > train)",
                         "removed": _split_counts(removed), "remaining": _split_counts(rows),
                         "cross_split_texts": cross_texts, "by_language": _by_language(hit)})

    rep["final"] = _split_counts(rows)
    return rows, rep


def _by_language(hit: dict[str, set]) -> dict:
    return {"bn": len(hit["bn"] - hit["en"]), "en": len(hit["en"] - hit["bn"]),
            "both": len(hit["bn"] & hit["en"])}


def _n_cross_split_texts(rows: list[dict], field: str) -> int:
    splits = defaultdict(set)
    for r in rows:
        splits[norm_text(r[field])].add(r["split"])
    return sum(len(s) > 1 for s in splits.values())


# --------------------------------------------------------------------------- #
# BANKING77
# --------------------------------------------------------------------------- #
def load_banking77_raw(raw_dir: Path = RAW / "banking77") -> list[dict]:
    rows = []
    for split in ("train", "test"):
        path = raw_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/download_data.py")
        # dtype=str + keep_default_na=False: never let pandas turn text like "NA" into NaN
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        for i, (text, cat) in enumerate(zip(df["text"], df["category"])):
            rows.append({"uid": f"b77.{split}.{i:05d}", "text": text, "intent": cat, "split": split})
    return rows


def carve_banking77_val(rows: list[dict]) -> list[dict]:
    """Move round(n_c / 6) training rows per class into val, group-aware by text."""
    out = [dict(r) for r in rows]
    train_by_class = defaultdict(lambda: defaultdict(list))
    for r in out:
        if r["split"] == "train":
            train_by_class[r["intent"]][norm_text(r["text"])].append(r)
    for intent, groups in train_by_class.items():
        n_class = sum(len(g) for g in groups.values())
        target = round(n_class * B77_VAL_FRACTION)
        ordered = sorted(groups.values(),
                         key=lambda g: hash_key(B77_VAL_SALT, min(r["uid"] for r in g)))
        taken = 0
        for g in ordered:
            if taken >= target:
                break
            for r in g:
                r["split"] = "val"
            taken += len(g)
    return out


# --------------------------------------------------------------------------- #
# Tracks
# --------------------------------------------------------------------------- #
def _to_track(rows: list[dict], text_field: str) -> dict:
    label_names = sorted({r["intent"] for r in rows})
    lid = {n: i for i, n in enumerate(label_names)}
    track_rows = sorted(({"uid": r["uid"], "text": r[text_field], "label_id": lid[r["intent"]],
                          "split": r["split"]} for r in rows), key=lambda r: r["uid"])
    return {"label_names": label_names, "rows": track_rows}


def build_tracks() -> tuple[dict, dict]:
    """Return ({track: {"label_names", "rows"}}, report). Rows carry uid, text,
    label_id, split. Deterministic: same raw files -> identical output."""
    bn_raw = load_bnintent30_raw()
    bn_clean, bn_rep = clean_bnintent30(bn_raw)
    b77_raw = load_banking77_raw()
    b77 = carve_banking77_val(b77_raw)
    tracks = {
        "bnintent30-en": _to_track(bn_clean, "text"),
        "bnintent30-bn": _to_track(bn_clean, "bn_text"),
        "banking77": _to_track(b77, "text"),
    }
    report = {"bnintent30": bn_rep, "bn_raw": bn_raw, "bn_clean": bn_clean,
              "b77_raw": b77_raw, "b77": b77}
    return tracks, report


def write_clean(tracks: dict) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    for name, t in tracks.items():
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in t["rows"])
        (CLEAN / f"{name}.jsonl").write_bytes(lines.encode("utf-8"))
        (CLEAN / f"{name}.label_names.json").write_bytes(
            (json.dumps(t["label_names"], ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- #
# Cleaning report (§3.3 step 5). Every number below is computed here (R8).
# Contains counts and intent names only - never utterance text (§3.4).
# --------------------------------------------------------------------------- #
def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _per_class(rows: list[dict]) -> dict:
    c = Counter((r["intent"], r["split"]) for r in rows)
    return c


def write_report(tracks: dict, rep: dict, data_cfg: dict | None = None) -> str:
    br = rep["bnintent30"]
    L = ["# Data cleaning report", "",
         "Generated by `python -m src.data` (src/data.py). Do not edit by hand.", ""]
    if data_cfg:
        L += [f"- BNIntent30 source: `{data_cfg['bnintent30']['repo']}` @ "
              f"`{data_cfg['bnintent30']['commit']}`",
              f"- BANKING77 source: `{data_cfg['banking77']['repo']}` @ "
              f"`{data_cfg['banking77']['commit']}`", ""]
    L += ["Text comparison key for every \"same text\" decision: whitespace-collapsed string, "
          "otherwise exact (case and punctuation preserved). Rules are applied at row level to "
          "the union of problems found in either language (R9), so `bnintent30-en` and "
          "`bnintent30-bn` share one cleaned row set.", "",
          "## BNIntent30: rows removed per step", "",
          _row(["Step", "Rule", "train", "val", "test", "removed", "remaining (train/val/test)",
                "triggered by bn only / en only / both"]),
          _row(["---"] * 8),
          _row(["0", "raw", "", "", "", "", f"{sum(br['raw'].values())} "
                f"({br['raw']['train']}/{br['raw']['val']}/{br['raw']['test']})", ""])]
    for s in br["steps"]:
        rm, rem, bl = s["removed"], s["remaining"], s["by_language"]
        L.append(_row([s["step"], s["name"], rm["train"], rm["val"], rm["test"],
                       sum(rm.values()),
                       f"{sum(rem.values())} ({rem['train']}/{rem['val']}/{rem['test']})",
                       f"{bl['bn']} / {bl['en']} / {bl['both']}"]))
    s2, s3 = br["steps"][1], br["steps"][2]
    L += ["",
          f"- Step 1 check: {br['step1_total']:,} rows = {br['steps'][0]['remaining']['train']:,} "
          f"train / {br['steps'][0]['remaining']['val']} val / {br['steps'][0]['remaining']['test']} "
          "test, matching the published paper (asserted in code).",
          f"- Step 2: distinct conflicting texts: Bangla {s2['conflict_texts']['bn']}, "
          f"English {s2['conflict_texts']['en']}.",
          f"- Step 3: distinct texts occurring in more than one split (before step 3): Bangla "
          f"{s3['cross_split_texts']['bn']}, English {s3['cross_split_texts']['en']}. The Bangla "
          "excess is machine translation mapping distinct English sentences onto one Bangla sentence.",
          f"- Final: {sum(br['final'].values()):,} rows = {br['final']['train']:,} train / "
          f"{br['final']['val']} val / {br['final']['test']} test.", ""]

    # Per-class table
    raw_c, s1_rows = _per_class(rep["bn_raw"]), [r for r in rep["bn_raw"]
                                                 if r["bn_text"].strip() != FAILED_MARKER]
    s1_c, fin_c = _per_class(s1_rows), _per_class(rep["bn_clean"])
    L += ["## BNIntent30: per-class counts (train / val / test)", "",
          _row(["intent", "raw", "after step 1", "final", "train removed by steps 2-3",
                "test removed by steps 2-3"]),
          _row(["---"] * 6)]
    for intent in sorted({r["intent"] for r in rep["bn_raw"]}):
        f = lambda c: "/".join(str(c.get((intent, s), 0)) for s in SPLITS)
        L.append(_row([intent, f(raw_c), f(s1_c), f(fin_c),
                       s1_c.get((intent, "train"), 0) - fin_c.get((intent, "train"), 0),
                       s1_c.get((intent, "test"), 0) - fin_c.get((intent, "test"), 0)]))
    fin = Counter(r["intent"] for r in rep["bn_clean"] if r["split"] == "train")
    L += ["", f"Final train rows per class: min {min(fin.values())}, max {max(fin.values())} "
          f"(k=20 needs 20).", ""]

    # Observations not acted on
    L += ["## BNIntent30: observations NOT acted on (flagged for decision)", ""]
    for lang, field in LANG_FIELDS.items():
        strict = defaultdict(set)
        for r in rep["bn_clean"]:
            strict[norm_text(r[field]).casefold().strip(" ?.!,।")].add(r["split"])
        n = sum(len(v) > 1 for v in strict.values())
        L.append(f"- {lang}: {n} distinct texts still occur in more than one split if case and "
                 "leading/trailing punctuation (` ?.!,।`) are also ignored. Exact-text rule per "
                 "§3.3 leaves these in place.")
    no_bn = sum(not any("ঀ" <= ch <= "৿" for ch in r["bn_text"]) for r in rep["bn_clean"])
    L.append(f"- {no_bn} surviving rows have a `bn_text` with no Bangla-script character "
             "(digits/symbols only). Kept: no rule covers them.")
    for lang, field in LANG_FIELDS.items():
        dup = defaultdict(int)
        for r in rep["bn_clean"]:
            dup[(r["split"], norm_text(r[field]))] += 1
        L.append(f"- {lang}: {sum(v - 1 for v in dup.values() if v > 1)} rows are exact "
                 "duplicates of another row *within the same split* (not leakage; kept).")
    L.append("")

    # BANKING77
    b77, b77_raw = rep["b77"], rep["b77_raw"]
    bc = _split_counts(b77)
    per = Counter((r["intent"], r["split"]) for r in b77)
    intents = sorted({r["intent"] for r in b77})
    vals = [per[(i, "val")] for i in intents]
    by_split = defaultdict(set)
    for r in b77:
        by_split[r["split"]].add(norm_text(r["text"]))
    L += ["## BANKING77", "",
          f"- Raw: {sum(1 for r in b77_raw if r['split'] == 'train'):,} train / "
          f"{sum(1 for r in b77_raw if r['split'] == 'test'):,} test, {len(intents)} intents. "
          "No cleaning rule is specified for BANKING77 in CLAUDE.md; none was applied.",
          f"- Val carved from train: fraction {B77_VAL_FRACTION:.4f} (1/6) per class, group-aware "
          f"by text -> {bc['train']:,} train / {bc['val']:,} val / {bc['test']:,} test; "
          f"val per class min {min(vals)}, max {max(vals)}.",
          f"- **Flag:** {len(by_split['train'] & by_split['test'])} distinct texts occur in both "
          f"train and test, {len(by_split['val'] & by_split['test'])} in both val and test, "
          f"{len(by_split['train'] & by_split['val'])} in both train and val (exact text). "
          "These are upstream overlaps in the official BANKING77 split; leakage test 2 will "
          "report them until a cleaning decision is made.", ""]

    text = "\n".join(L)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_bytes(text.encode("utf-8"))
    return text


def main() -> None:
    import yaml
    tracks, rep = build_tracks()
    write_clean(tracks)
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    write_report(tracks, rep, cfg)
    for name, t in tracks.items():
        c = Counter(r["split"] for r in t["rows"])
        print(f"{name:15s} {len(t['label_names'])} intents  "
              + "  ".join(f"{s}={c[s]}" for s in SPLITS))
    print(f"wrote {CLEAN.relative_to(ROOT)}/ and {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
