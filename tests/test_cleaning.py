"""§3.3 assertions for BNIntent30 cleaning."""
from collections import defaultdict

import pytest

from conftest import check
from src.data import FAILED_MARKER, clean_bnintent30, load_bnintent30_raw, norm_text

FIELDS = {"bn": "bn_text", "en": "text"}


@pytest.fixture(scope="module")
def cleaned():
    return clean_bnintent30(load_bnintent30_raw())


def test_step1_totals_match_published_paper(cleaned):
    _, rep = cleaned
    after1 = rep["steps"][0]["remaining"]
    assert after1 == {"train": 2952, "val": 595, "test": 886}, \
        f"§3.3 step 1 must leave 2,952/595/886 rows; got {after1}. Upstream data changed - STOP."
    assert sum(after1.values()) == 4433, f"step 1 total {sum(after1.values())} != 4,433"


def test_no_translation_failed_survives(cleaned):
    rows, _ = cleaned
    bad = [r["uid"] for r in rows
           if FAILED_MARKER in r["bn_text"] or FAILED_MARKER in r["text"]]
    check(bad, f"{FAILED_MARKER} rows survived cleaning")


@pytest.mark.parametrize("lang", FIELDS)
def test_no_text_has_two_labels(cleaned, lang):
    rows, _ = cleaned
    labels = defaultdict(set)
    for r in rows:
        labels[norm_text(r[FIELDS[lang]])].add(r["intent"])
    bad = [f"{t!r} -> {sorted(ls)}" for t, ls in labels.items() if len(ls) > 1]
    check(bad, f"[{lang}] texts carrying more than one intent")


@pytest.mark.parametrize("lang", FIELDS)
def test_no_text_in_two_splits(cleaned, lang):
    rows, _ = cleaned
    splits = defaultdict(set)
    for r in rows:
        splits[norm_text(r[FIELDS[lang]])].add(r["split"])
    bad = [f"{t!r} in {sorted(s)}" for t, s in splits.items() if len(s) > 1]
    check(bad, f"[{lang}] texts occurring in more than one split")


def test_uids_unique_and_stable(cleaned):
    rows, _ = cleaned
    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids)), "duplicate uids after cleaning"
    again, _ = clean_bnintent30(load_bnintent30_raw())
    assert uids == [r["uid"] for r in again], "uids are not deterministic across re-runs"
