"""§3.3 assertions. Cleaning applies to ALL tracks; every comparison uses dedup_key."""
from collections import Counter, defaultdict

import pytest

from conftest import check
from src.data import (FAILED_MARKER, MIN_TRAIN_PER_CLASS, TRACKS, clean_bnintent30,
                      load_bnintent30_raw, dedup_key)


@pytest.fixture(scope="module")
def cleaned():
    return clean_bnintent30(load_bnintent30_raw())


PUBLISHED_AFTER_STEP1 = {"train": 2952, "val": 595, "test": 886}  # BNIntent30 paper, 4,433 rows


def test_step1_totals_match_published_paper():
    """Counted here directly from the raw rows, independently of src.data's own
    step-1 assertion, so this test can fail even if that assertion were removed."""
    raw = load_bnintent30_raw()
    after1 = Counter(r["split"] for r in raw if r["bn_text"].strip() != FAILED_MARKER)
    assert dict(after1) == PUBLISHED_AFTER_STEP1, \
        f"§3.3 step 1 must leave 2,952/595/886 rows; got {dict(after1)}. Upstream data changed."
    assert sum(after1.values()) == 4433, f"step 1 total {sum(after1.values())} != 4,433"


def test_cleaning_report_records_the_step1_totals(cleaned):
    _, rep = cleaned
    assert rep["steps"][0]["remaining"] == PUBLISHED_AFTER_STEP1


def test_no_translation_failed_survives(cleaned):
    rows, _ = cleaned
    bad = [r["uid"] for r in rows
           if FAILED_MARKER in r["bn_text"] or FAILED_MARKER in r["text"]]
    check(bad, f"{FAILED_MARKER} rows survived cleaning")


@pytest.mark.parametrize("track", TRACKS)
def test_no_key_has_two_labels(built, track):
    labels = defaultdict(set)
    for r in built[0][track]["rows"]:
        labels[dedup_key(r["text"])].add(r["label_id"])
    bad = [f"{k!r} -> labels {sorted(ls)}" for k, ls in labels.items() if len(ls) > 1]
    check(bad, f"[{track}] dedup keys carrying more than one label")


@pytest.mark.parametrize("track", TRACKS)
def test_no_key_in_two_splits(built, track):
    splits = defaultdict(set)
    for r in built[0][track]["rows"]:
        splits[dedup_key(r["text"])].add(r["split"])
    bad = [f"{k!r} in {sorted(s)}" for k, s in splits.items() if len(s) > 1]
    check(bad, f"[{track}] dedup keys occurring in more than one split")


@pytest.mark.parametrize("track", TRACKS)
def test_no_key_twice_within_a_split(built, track):
    seen = Counter((r["split"], dedup_key(r["text"])) for r in built[0][track]["rows"])
    bad = [f"{s}: {k!r} x{n}" for (s, k), n in seen.items() if n > 1]
    check(bad, f"[{track}] dedup keys repeated inside one split")


@pytest.mark.parametrize("track", TRACKS)
def test_every_class_keeps_min_train_rows(built, track):
    t = built[0][track]
    c = Counter(r["label_id"] for r in t["rows"] if r["split"] == "train")
    bad = [f"{name}: {c.get(i, 0)}" for i, name in enumerate(t["label_names"])
           if c.get(i, 0) < MIN_TRAIN_PER_CLASS]
    check(bad, f"[{track}] classes below {MIN_TRAIN_PER_CLASS} train rows (k is never reduced to fit)")


def test_uids_unique_and_stable(cleaned):
    rows, _ = cleaned
    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids)), "duplicate uids after cleaning"
    again, _ = clean_bnintent30(load_bnintent30_raw())
    assert uids == [r["uid"] for r in again], "uids are not deterministic across re-runs"
