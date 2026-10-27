"""Load, clean and assign uids for all three tracks (docs/PROTOCOL.md §3).

Tracks (§3.1):
    bnintent30-en  BNIntent30 `text` field     (English)
    bnintent30-bn  BNIntent30 `bn_text` field  (Bangla)
    banking77      BANKING77 CSVs              (English)

The two BNIntent30 tracks are built from ONE shared cleaned row set (R9), so
they contain the same uids, labels and split assignment; only the text differs.

Every "same text" decision in this repository - cleaning, the BANKING77 val
carve, the leakage-dose table and the tests - goes through `dedup_key` below.
Nothing computes its own variant. Stored text is never edited.

Usage:  python -m src.data      # writes data/clean/ and results/tables/data_cleaning.md
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REPORT = ROOT / "results" / "tables" / "data_cleaning.md"

TRACKS = ("bnintent30-en", "bnintent30-bn", "banking77")
SPLITS = ("train", "val", "test")
# §3.3 step 3: the occurrence in the highest-priority split is kept.
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
FAILED_MARKER = "TRANSLATION_FAILED"
# §3.3 step 1 expected totals (published paper). Asserted, never adjusted.
STEP1_EXPECTED = {"train": 2952, "val": 595, "test": 886}
MIN_TRAIN_PER_CLASS = 20  # the largest k (§3.3 post-cleaning assertion)
BN30_FIELDS = {"bn": "bn_text", "en": "text"}
B77_FIELDS = {"en": "text"}

# BANKING77 has no official val split. 1/6 of each class's (cleaned) training
# rows becomes val, matching BNIntent30's 100:20 train:val ratio (R3). The carve
# is fixed for all k and seeds, like BNIntent30's official val, and group-aware
# by dedup_key so it cannot itself create train/val overlap. It runs AFTER
# cleaning steps 2-4, so val is exactly 1/6 of clean rows.
B77_VAL_FRACTION = 1 / 6
B77_VAL_SALT = "banking77-val-carve-v1"


# --------------------------------------------------------------------------- #
# Text normalization and the deduplication key (§3.3).
# --------------------------------------------------------------------------- #
def norm_key(s: str) -> str:
    """Token-level normalization shared by the dedup key and the TF-IDF tokenizer:
    NFC, lowercased, every Unicode punctuation character (category P*:
    ? . ! , ; : ' " ( ) - and the Bangla danda U+0964 / double danda U+0965 ...)
    replaced by a space, whitespace collapsed.

    Punctuation becomes a space rather than vanishing because that is what
    TfidfVectorizer's tokenizer does: it splits on punctuation, it does not glue
    the neighbours together.
    """
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return " ".join(s.split())


def dedup_key(s: str) -> str:
    """THE deduplication key (§3.3): the sorted multiset of normalized tokens.

    It is the coarsest equivalence induced by any model component. All three
    handcrafted families are word-order-insensitive (TF-IDF is a bag of words,
    char_wb n-grams never cross word boundaries, the statistics are counts), so
    two utterances that differ only in word order are one input to that branch.
    The token multiset is used instead of the feature vector because the vector
    depends on a fitted vocabulary, hence on the split, hence on deduplication;
    the multiset induces the same classes and is split-independent.
    The frozen encoder IS order-sensitive, so this is stricter than the encoder
    alone needs: conservative in the safe direction (evaluation only gets harder).

    This is the ONLY function that decides whether two texts are "the same".
    """
    return " ".join(sorted(norm_key(s).split()))


def exact_key(s: str) -> str:
    """Exact string, whitespace collapsed only. Used ONLY to report the
    counterfactual counts that §3.3 asks for; never decides anything. Whitespace is
    collapsed because raw BANKING77 has duplicates that differ only by a trailing
    space."""
    return " ".join(s.split())


def byte_key(s: str) -> str:
    """Byte-for-byte identical string. Reporting only; never decides anything."""
    return s


# The comparison-key levels reported side by side, coarsening left to right.
# Only the last one is ever used to decide anything.
REPORT_KEYS = (("byte-for-byte exact", byte_key),
               ("whitespace-collapsed exact", exact_key),
               ("normalized, order-sensitive", norm_key),
               ("normalized token multiset (dedup_key, applied)", dedup_key))


def hash_key(*parts: str) -> str:
    """Version-independent deterministic ordering key (no RNG library involved)."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Loading (§3.3 step 5: uid = source file + row position, fixed at load time,
# so it is independent of any later processing order)
# --------------------------------------------------------------------------- #
def load_bnintent30_raw(raw_dir: Path = RAW / "bnintent30") -> list[dict]:
    """All 4,500 raw BNIntent30 rows with uids fixed from source file and row position."""
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


def load_banking77_raw(raw_dir: Path = RAW / "banking77") -> list[dict]:
    """All raw BANKING77 rows (official train and test) with uids fixed at load time."""
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


# --------------------------------------------------------------------------- #
# Cleaning steps 2-4. Each takes the rows, the language->field map and a key
# function, and returns (kept rows, {uid: set of languages that triggered the drop}).
# Union over languages (R9) is built in: a row goes if ANY language flags it.
# --------------------------------------------------------------------------- #
def step2_conflicts(rows, fields, key=dedup_key):
    """§3.3 step 2: drop every row whose key carries more than one intent, in any language."""
    labels = {lang: defaultdict(set) for lang in fields}
    for r in rows:
        for lang, f in fields.items():
            labels[lang][key(r[f])].add(r["intent"])
    dropped = {}
    for r in rows:
        why = {lang for lang, f in fields.items() if len(labels[lang][key(r[f])]) > 1}
        if why:
            dropped[r["uid"]] = why
    return [r for r in rows if r["uid"] not in dropped], dropped


def step3_cross_split(rows, fields, key=dedup_key):
    """Greedy in (split priority, uid) order: a row is dropped iff one of its
    keys is already held by a KEPT row of a higher-priority split. Testing
    against kept rows (not all rows) means a row is never dropped because of a
    row that was itself dropped."""
    held = {lang: defaultdict(set) for lang in fields}  # key -> priorities of kept rows
    kept, dropped = [], {}
    for r in sorted(rows, key=lambda r: (SPLIT_PRIORITY[r["split"]], r["uid"])):
        p = SPLIT_PRIORITY[r["split"]]
        why = {lang for lang, f in fields.items() if any(q < p for q in held[lang][key(r[f])])}
        if why:
            dropped[r["uid"]] = why
            continue
        kept.append(r)
        for lang, f in fields.items():
            held[lang][key(r[f])].add(p)
    return sorted(kept, key=lambda r: r["uid"]), dropped


def step4_within_split(rows, fields, key=dedup_key):
    """Greedy in uid order within each split: keep the lowest uid of each key."""
    held = {lang: set() for lang in fields}  # (split, key) of kept rows
    kept, dropped = [], {}
    for r in sorted(rows, key=lambda r: r["uid"]):
        why = {lang for lang, f in fields.items() if (r["split"], key(r[f])) in held[lang]}
        if why:
            dropped[r["uid"]] = why
            continue
        kept.append(r)
        for lang, f in fields.items():
            held[lang].add((r["split"], key(r[f])))
    return kept, dropped


STEPS = (
    (2, "label conflict (one key, >1 intent; every row with that key dropped)", step2_conflicts),
    (3, "cross-split duplicate (priority test > val > train)", step3_cross_split),
    (4, "within-split duplicate (lowest uid kept)", step4_within_split),
)


def _split_counts(rows) -> dict:
    c = Counter(r["split"] for r in rows)
    return {s: c.get(s, 0) for s in SPLITS}


def _counts_by_split(uids, rows) -> dict:
    split_of = {r["uid"]: r["split"] for r in rows}
    c = Counter(split_of[u] for u in uids)
    return {s: c.get(s, 0) for s in SPLITS}


def _trigger_counts(dropped: dict, fields) -> dict:
    langs = list(fields)
    if len(langs) == 1:
        return {langs[0]: len(dropped)}
    return {"bn": sum(w == {"bn"} for w in dropped.values()),
            "en": sum(w == {"en"} for w in dropped.values()),
            "both": sum(w == {"bn", "en"} for w in dropped.values())}


def apply_steps_2_to_4(rows, fields) -> tuple[list[dict], list[dict]]:
    """Run steps 2-4 with dedup_key, recording for each step the same step run
    with exact_key on the same input (counterfactual, for the report only)."""
    records = []
    for num, name, fn in STEPS:
        kept, dropped = fn(rows, fields, dedup_key)
        _, dropped_exact = fn(rows, fields, exact_key)
        records.append({
            "step": num, "name": name, "_input": rows, "_fn": fn,
            "removed": _counts_by_split(dropped, rows),
            "removed_exact_key": _counts_by_split(dropped_exact, rows),
            "by_language": _trigger_counts(dropped, fields),
            "remaining": _split_counts(kept),
        })
        rows = kept
    return rows, records


# --------------------------------------------------------------------------- #
# Per-dataset pipelines
# --------------------------------------------------------------------------- #
def bnintent30_step1(rows: list[dict]) -> tuple[list[dict], dict]:
    """§3.3 step 1: drop TRANSLATION_FAILED rows and assert the published totals."""
    failed = [r for r in rows if r["bn_text"].strip() == FAILED_MARKER]
    kept = [r for r in rows if r["bn_text"].strip() != FAILED_MARKER]
    after1 = _split_counts(kept)
    if after1 != STEP1_EXPECTED:
        raise AssertionError(
            f"§3.3 step 1 totals do not match the published paper. Expected "
            f"{STEP1_EXPECTED} (total 4,433), got {after1} (total {sum(after1.values())}). "
            "Something upstream changed; the expectation is never adjusted to match.")
    record = {"step": 1, "name": f"bn_text is literally {FAILED_MARKER}",
              "removed": _split_counts(failed), "removed_exact_key": _split_counts(failed),
              "by_language": {"bn": len(failed), "en": 0, "both": 0}, "remaining": after1}
    return kept, record


def clean_bnintent30(rows: list[dict]) -> tuple[list[dict], dict]:
    """§3.3 steps 1-4 at row level, union of both languages (R9)."""
    rep = {"raw": _split_counts(rows)}
    rows, rec1 = bnintent30_step1(rows)
    rows, recs = apply_steps_2_to_4(rows, BN30_FIELDS)
    rep["steps"] = [rec1] + recs
    rep["final"] = _split_counts(rows)
    return rows, rep


def clean_banking77(rows: list[dict]) -> tuple[list[dict], dict]:
    """§3.3 steps 2-4 on the official train/test, then the group-aware val carve."""
    rep = {"raw": _split_counts(rows)}
    rows, rep["steps"] = apply_steps_2_to_4(rows, B77_FIELDS)
    rep["before_carve"] = _split_counts(rows)
    rows = carve_banking77_val(rows)
    rep["final"] = _split_counts(rows)
    return rows, rep


def carve_banking77_val(rows: list[dict]) -> list[dict]:
    """Move round(n_c / 6) training rows per class into val, group-aware by dedup_key."""
    out = [dict(r) for r in rows]
    train_by_class = defaultdict(lambda: defaultdict(list))
    for r in out:
        if r["split"] == "train":
            train_by_class[r["intent"]][dedup_key(r["text"])].append(r)
    for groups in train_by_class.values():
        target = round(sum(len(g) for g in groups.values()) * B77_VAL_FRACTION)
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


def smallest_train_classes(track: dict, n: int = 3) -> list[tuple[str, int]]:
    """The n classes with the fewest training rows, as (name, count)."""
    c = Counter(r["label_id"] for r in track["rows"] if r["split"] == "train")
    names = track["label_names"]
    return sorted(((names[i], c.get(i, 0)) for i in range(len(names))), key=lambda x: (x[1], x[0]))[:n]


def assert_min_train(tracks: dict) -> None:
    """§3.3: every class in every track keeps >= 20 train rows. Never reduce k instead."""
    bad = []
    for name, t in tracks.items():
        bad += [f"{name}: {cls} has {n}" for cls, n in smallest_train_classes(t, len(t["label_names"]))
                if n < MIN_TRAIN_PER_CLASS]
    if bad:
        raise AssertionError(f"Classes below {MIN_TRAIN_PER_CLASS} training rows after cleaning "
                             f"(k is never reduced to fit): {bad}")


def build_tracks() -> tuple[dict, dict]:
    """Return ({track: {"label_names", "rows"}}, report). Deterministic."""
    bn_raw = load_bnintent30_raw()
    bn_clean, bn_rep = clean_bnintent30(bn_raw)
    b77_raw = load_banking77_raw()
    b77_clean, b77_rep = clean_banking77(b77_raw)
    tracks = {
        "bnintent30-en": _to_track(bn_clean, "text"),
        "bnintent30-bn": _to_track(bn_clean, "bn_text"),
        "banking77": _to_track(b77_clean, "text"),
    }
    assert_min_train(tracks)
    report = {"bnintent30": bn_rep, "banking77": b77_rep, "bn_raw": bn_raw,
              "bn_clean": bn_clean, "b77_raw": b77_raw, "b77_clean": b77_clean}
    return tracks, report


def build_dirty_tracks() -> dict:
    """Leaky condition (§5): official BNIntent30 assignment, step 1 applied,
    steps 2-4 NOT applied. Bangla only."""
    rows, _ = bnintent30_step1(load_bnintent30_raw())
    return {"bnintent30-bn": _to_track(rows, "bn_text")}


def write_clean(tracks: dict) -> None:
    """Write each cleaned track to data/clean/ (rows and label names)."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    for name, t in tracks.items():
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in t["rows"])
        (CLEAN / f"{name}.jsonl").write_bytes(lines.encode("utf-8"))
        (CLEAN / f"{name}.label_names.json").write_bytes(
            (json.dumps(t["label_names"], ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- #
# Cleaning report (§3.3 step 6). Every number is computed here (R8). Counts
# and intent names only - never utterance text (§3.4).
# --------------------------------------------------------------------------- #
def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _fmt(c: dict, splits=SPLITS) -> str:
    return "/".join(str(c.get(s, 0)) for s in splits)


def _step_table(rep: dict, langs: str) -> list[str]:
    L = [_row(["Step", "Rule", "split", "removed (dedup_key, applied)",
               "would be removed under whitespace-collapsed exact key", f"triggered by {langs}"]),
         _row(["---"] * 6)]
    for s in rep["steps"]:
        bl = s["by_language"]
        trig = " / ".join(str(v) for v in bl.values())
        for sp in SPLITS:
            L.append(_row([s["step"], s["name"] if sp == "train" else "", sp,
                           s["removed"][sp], s["removed_exact_key"][sp], trig if sp == "train" else ""]))
        L.append(_row(["", "", "**total**", f"**{sum(s['removed'].values())}**",
                       sum(s["removed_exact_key"].values()),
                       f"remaining {sum(s['remaining'].values())} ({_fmt(s['remaining'])})"]))
    return L


def _per_class_table(raw, clean, extra=None) -> list[str]:
    rc = Counter((r["intent"], r["split"]) for r in raw)
    fc = Counter((r["intent"], r["split"]) for r in clean)
    head = ["intent", "raw (train/val/test)"] + (["after step 1"] if extra else []) + \
           ["final (train/val/test)", "train removed", "val removed", "test removed"]
    L = [_row(head), _row(["---"] * len(head))]
    base = extra or rc
    for intent in sorted({r["intent"] for r in raw}):
        g = lambda c, s: c.get((intent, s), 0)
        cells = [intent, "/".join(str(g(rc, s)) for s in SPLITS)]
        if extra:
            cells.append("/".join(str(g(extra, s)) for s in SPLITS))
        cells.append("/".join(str(g(fc, s)) for s in SPLITS))
        cells += [g(base, s) - g(fc, s) for s in SPLITS]
        L.append(_row(cells))
    return L


def _b77_per_class_table(raw, clean) -> list[str]:
    official = lambda r: r["uid"].split(".")[1]  # uid records the official split
    rc = Counter((r["intent"], official(r)) for r in raw)
    oc = Counter((r["intent"], official(r)) for r in clean)
    fc = Counter((r["intent"], r["split"]) for r in clean)
    head = ["intent", "raw (train/test)", "after steps 2-4 (train/test)",
            "removed (train/test)", "final after carve (train/val/test)"]
    L = [_row(head), _row(["---"] * len(head))]
    for intent in sorted({r["intent"] for r in raw}):
        g = lambda c, s: c.get((intent, s), 0)
        L.append(_row([intent, f"{g(rc, 'train')}/{g(rc, 'test')}",
                       f"{g(oc, 'train')}/{g(oc, 'test')}",
                       f"{g(rc, 'train') - g(oc, 'train')}/{g(rc, 'test') - g(oc, 'test')}",
                       "/".join(str(g(fc, s)) for s in SPLITS)]))
    return L


def _key_level_table(rep: dict) -> list[str]:
    """Rows that step 3 / step 4 would remove under each key level, using
    ONE track's text only, evaluated on exactly the input the applied step got."""
    per_track = (("bnintent30-en", rep["bnintent30"], {"en": "text"}),
                 ("bnintent30-bn", rep["bnintent30"], {"bn": "bn_text"}),
                 ("banking77", rep["banking77"], {"en": "text"}))
    head = ["track", "step"] + [f"key {i + 1}: {name}" for i, (name, _) in enumerate(REPORT_KEYS)]
    L = [_row(head), _row(["---"] * len(head))]
    for track, drep, fields in per_track:
        for s in drep["steps"]:
            if s["step"] not in (3, 4):
                continue
            label = "3 cross-split" if s["step"] == 3 else "4 within-split"
            cells = []
            for _, key in REPORT_KEYS:
                _, dropped = s["_fn"](s["_input"], fields, key)
                c = _counts_by_split(dropped, s["_input"])
                cells.append(f"{sum(c.values())} ({_fmt(c)})")
            L.append(_row([track, label] + cells))
    return L


def _test_removal_note(tracks: dict, rep: dict) -> list[str]:
    """Which test rows were removed, and why; final test per-class range."""
    L = ["## Test rows removed, and why", ""]
    for name, drep in (("BNIntent30 (both tracks)", rep["bnintent30"]),
                       ("BANKING77", rep["banking77"])):
        t = {s["step"]: s["removed"]["test"] for s in drep["steps"]}
        parts = []
        if 1 in t:
            parts.append(f"step 1 removed {t[1]} (untranslated placeholders)")
        parts += [f"step 2 removed {t[2]} (label conflicts: unanswerable items, every "
                  "prediction would be scored wrong)",
                  f"step 3 removed {t[3]} (by design: priority test > val > train keeps the "
                  "evaluation copy)",
                  f"step 4 removed {t[4]} (duplicate test items: the lowest-uid copy of each "
                  "key is kept)"]
        L.append(f"- **{name}:** " + "; ".join(parts) + ".")
    _, bn_only = step4_within_split(next(s["_input"] for s in rep["bnintent30"]["steps"]
                                         if s["step"] == 4), {"en": "text"})
    en_test = sum(1 for u in bn_only if u in {r["uid"] for r in rep["bn_raw"] if r["split"] == "test"})
    L.append(f"- **Cost of R9 on the English test set:** using English text alone, step 4 would "
             f"remove {en_test} test rows. The BNIntent30 step-4 test removals are triggered by "
             "the Bangla side; because both tracks must hold identical rows (R9), the paired "
             "English rows go too, although their English texts are distinct. So the English test "
             "set loses distinct items to keep the comparison parallel.")
    L += ["", _row(["track", "test rows", "test rows per class (min-max)"]), _row(["---"] * 3)]
    for name, tr in tracks.items():
        c = Counter(r["label_id"] for r in tr["rows"] if r["split"] == "test")
        L.append(_row([name, sum(c.values()), f"{min(c.values())}-{max(c.values())}"]))
    return L + [""]


def write_report(tracks: dict, rep: dict, data_cfg: dict | None = None) -> str:
    """Write results/tables/data_cleaning.md (counts and intent names only, never text)."""
    br, b7 = rep["bnintent30"], rep["banking77"]
    L = ["# Data cleaning report", "",
         "Generated by `python -m src.data` (src/data.py). Do not edit by hand.", ""]
    if data_cfg:
        L += [f"- BNIntent30 source: `{data_cfg['bnintent30']['repo']}` @ "
              f"`{data_cfg['bnintent30']['commit']}`",
              f"- BANKING77 source: `{data_cfg['banking77']['repo']}` @ "
              f"`{data_cfg['banking77']['commit']}`", ""]
    L += ["**Deduplication key** (`src.data.dedup_key`, used by every decision): the sorted "
          "multiset of normalized tokens. Tokens come from `norm_key`: Unicode NFC, lowercased, "
          "every punctuation character (Unicode category P, incl. the Bangla danda) replaced by a "
          "space, whitespace collapsed. Sorting makes the key word-order-insensitive, because all "
          "three handcrafted feature families are (docs/PROTOCOL.md §3.3). The \"exact key\" column "
          "re-runs the same step on the same input with the exact string (whitespace collapsed "
          "only); it is reported for comparison only and decides nothing.", "",
          "Steps 2-4 are applied at row level to the union of problems found in either language "
          "(R9), so `bnintent30-en` and `bnintent30-bn` share one cleaned row set. Steps 3 and 4 "
          "are greedy in a fixed order (step 3: test, val, train, then uid; step 4: uid), and a row "
          "is dropped only for colliding with a row that is *kept*.", "",
          "## BNIntent30: rows removed per step", "",
          f"Raw: {sum(br['raw'].values())} rows ({_fmt(br['raw'])} train/val/test).", ""]
    L += _step_table(br, "bn only / en only / both")
    s1 = br["steps"][0]["remaining"]
    L += ["",
          f"- Step 1 check: {sum(s1.values()):,} = {s1['train']:,} train / {s1['val']} val / "
          f"{s1['test']} test, matching the published paper (asserted in code).",
          f"- Final: {sum(br['final'].values()):,} rows = {br['final']['train']:,} train / "
          f"{br['final']['val']} val / {br['final']['test']} test.",
          "- Step 5 (uids) removes nothing: uids are fixed at load time from source file and row "
          "position, so they are independent of processing order.", ""]
    step1_rows = [r for r in rep["bn_raw"] if r["bn_text"].strip() != FAILED_MARKER]
    s1c = Counter((r["intent"], r["split"]) for r in step1_rows)
    L += ["## BNIntent30: per-class counts", "",
          "\"removed\" columns count steps 2-4 (relative to the after-step-1 counts).", ""]
    L += _per_class_table(rep["bn_raw"], rep["bn_clean"], extra=s1c)

    L += ["", "## BANKING77: rows removed per step", "",
          f"Raw: {sum(b7['raw'].values()):,} rows ({b7['raw']['train']:,} train / "
          f"{b7['raw']['test']:,} test; no official val). Step 1 does not apply "
          "(no translation placeholders).", ""]
    L += _step_table(b7, "en")
    bc = b7["before_carve"]
    vals = Counter(r["intent"] for r in rep["b77_clean"] if r["split"] == "val")
    L += ["",
          f"- After steps 2-4: {bc['train']:,} train / {bc['test']:,} test.",
          f"- Val carve (after cleaning): fraction {B77_VAL_FRACTION:.4f} (1/6) per class, "
          f"group-aware by dedup_key -> {b7['final']['train']:,} train / {b7['final']['val']:,} "
          f"val / {b7['final']['test']:,} test; val per class min {min(vals.values())}, "
          f"max {max(vals.values())}.", "",
          "## BANKING77: per-class counts", "",
          "\"removed\" counts steps 2-4 per official split. The val carve happens after "
          "cleaning and moves rows, it removes none.", ""]
    L += _b77_per_class_table(rep["b77_raw"], rep["b77_clean"])

    L += ["", "## Duplicate counts under four comparison-key levels", "",
          "Rows that step 3 (cross-split) and step 4 (within-split) would remove, as total "
          "(train/val/test). Keys coarsen left to right. Key 1 = byte-for-byte identical string "
          "(`byte_key`); key 2 = identical after collapsing whitespace only (`exact_key`); "
          "key 3 = identical normalized token SEQUENCE (`norm_key`: NFC, lowercase, "
          "punctuation -> space, whitespace collapsed); key 4 = identical "
          "normalized token MULTISET (`dedup_key`, the key actually applied). Each row uses ONE "
          "track's text only, on exactly the input the applied step received, so the BNIntent30 "
          "rows differ from the applied union in the step tables above. Keys 1-3 decide nothing.",
          ""]
    L += _key_level_table(rep)
    L += [""] + _test_removal_note(tracks, rep)
    L += ["## Smallest training classes after cleaning", "",
          f"Assertion: every class in every track keeps >= {MIN_TRAIN_PER_CLASS} training rows "
          "(passes, or this report would not exist).", "",
          _row(["track", "1st", "2nd", "3rd"]), _row(["---"] * 4)]
    for name, t in tracks.items():
        L.append(_row([name] + [f"{c} ({n})" for c, n in smallest_train_classes(t)]))

    no_bn = sum(not any("ঀ" <= ch <= "৿" for ch in r["bn_text"]) for r in rep["bn_clean"])
    L += ["", "## Observations not acted on", "",
          f"- {no_bn} surviving BNIntent30 rows have a `bn_text` with no Bangla-script character "
          "(digits/symbols only). Kept: no rule covers them.", ""]

    text = "\n".join(L)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_bytes(text.encode("utf-8"))
    return text


def main() -> None:
    """Build the tracks, write data/clean/ and the cleaning report."""
    import yaml
    tracks, rep = build_tracks()
    write_clean(tracks)
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    write_report(tracks, rep, cfg)
    for name, t in tracks.items():
        c = Counter(r["split"] for r in t["rows"])
        print(f"{name:15s} {len(t['label_names'])} intents  "
              + "  ".join(f"{s}={c[s]}" for s in SPLITS)
              + "  smallest train: " + ", ".join(f"{a}={b}" for a, b in smallest_train_classes(t)))
    print(f"wrote {CLEAN.relative_to(ROOT)}/ and {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
