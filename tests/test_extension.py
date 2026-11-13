"""§12 extension: uid identity of the es/hi tracks with en and bn (docs/PROTOCOL.md §12).

For every (k, seed) and every split, bnintent30-es and bnintent30-hi must hold exactly the
uids of bnintent30-en and bnintent30-bn, in the same order and with the same labels; only
the text differs. scripts/run_extension.py refuses to start unless this module PASSES
(a skip does not count).

These tests SKIP, with a reason, when the extension data is absent (data/extension/ is
gitignored and the translations are not redistributable), so that the main grid's test
suite does not depend on the extension. Skipping never lets an extension run start.
"""
import json
from pathlib import Path

import pytest

from conftest import FILES, MANIFEST, check, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
EXT_MANIFEST = ROOT / "data" / "extension" / "splits" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not (EXT_MANIFEST.exists() and (ROOT / "data/extension/es.jsonl").exists()
         and (ROOT / "data/extension/hi.jsonl").exists()),
    reason="§12 extension data absent (data/extension/); run scripts/extension_prepare.py")


@pytest.fixture(scope="module")
def manifests():
    main = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ext = json.loads(EXT_MANIFEST.read_text(encoding="utf-8"))
    return main, ext


def _entries(m, track):
    return {(e["k"], e["seed"]): e for e in m["splits"] if e["track"] == track}


def test_translations_match_recorded_hashes():
    from src.extension import EXPECTED_SHA256, LANGS, load_translations
    for lang in LANGS:
        tr = load_translations(lang)  # raises if the sha256 differs from §12
        assert len(tr) == 4500 and EXPECTED_SHA256[lang]


def test_uid_identity_across_en_bn_es_hi(manifests):
    """Same (k, seed) entries, and per split the same uid list in the same order."""
    main, ext = manifests
    en, bn = _entries(main, "bnintent30-en"), _entries(main, "bnintent30-bn")
    es, hi = _entries(ext, "bnintent30-es"), _entries(ext, "bnintent30-hi")
    problems = []
    for name, d in (("bn", bn), ("es", es), ("hi", hi)):
        if set(d) != set(en):
            problems.append(f"{name}: (k, seed) entries differ from en")
    for key in sorted(en):
        for f in FILES:
            ref = en[key]["uids"][f]
            for name, d in (("bn", bn), ("es", es), ("hi", hi)):
                if key in d and d[key]["uids"][f] != ref:
                    problems.append(f"{name} k={key[0]} seed={key[1]} {f}: uid list differs from en")
    check(problems, "uid identity en/bn/es/hi")
    assert len(en) == 20


def test_split_files_match_manifest_and_en_labels(manifests):
    """Every es/hi split file matches its sha256, its uid list, and en's labels row by row;
    the text is non-empty and differs from the English file (it is the translation)."""
    import hashlib
    main, ext = manifests
    en = _entries(main, "bnintent30-en")
    problems = []
    for e in ext["splits"]:
        for f in FILES:
            raw = (EXT_MANIFEST.parent / e["path"] / f"{f}.jsonl").read_bytes()
            if hashlib.sha256(raw).hexdigest() != e["sha256"][f]:
                problems.append(f"{e['path']}/{f}: sha256 differs from the extension manifest")
            rows = [json.loads(l) for l in raw.decode("utf-8").splitlines()]
            ref = read_jsonl(ROOT / "data" / "splits" / en[(e["k"], e["seed"])]["path"] / f"{f}.jsonl")
            if [r["uid"] for r in rows] != e["uids"][f]:
                problems.append(f"{e['path']}/{f}: uids differ from the manifest")
            if [(r["uid"], r["label_id"]) for r in rows] != [(r["uid"], r["label_id"]) for r in ref]:
                problems.append(f"{e['path']}/{f}: (uid, label) rows differ from en")
            if any(not r["text"].strip() for r in rows):
                problems.append(f"{e['path']}/{f}: empty text")
            if sum(a["text"] == b["text"] for a, b in zip(rows, ref)) > len(rows) // 10:
                problems.append(f"{e['path']}/{f}: more than 10% of rows identical to English")
    check(problems, "extension split files")
    assert len(ext["splits"]) == 40


# --------------------------------------------------------------------------- #
# Bangla translation-method control, bnintent30-bn-cl (§12, amendment 2026-11-12).
# Its own split directory and manifest; same uid identity requirement. Skipped, with a
# reason, when its data is absent; scripts/run_extension.py bn-cl refuses to start unless
# this module passed with no skip.
# --------------------------------------------------------------------------- #
BNCL_MANIFEST = ROOT / "data" / "extension" / "splits_bn_cl" / "manifest.json"
bncl_present = pytest.mark.skipif(
    not (BNCL_MANIFEST.exists() and (ROOT / "data/extension/bn_cl.jsonl").exists()),
    reason="§12 bn-cl control data absent; run scripts/extension_prepare.py bn-cl")


@bncl_present
def test_bn_cl_translation_matches_recorded_hash():
    from src.extension import CONTROL, load_translations
    tr = load_translations(CONTROL)  # raises if the sha256 differs from §12
    assert len(tr) == 4096


@bncl_present
def test_bn_cl_uid_identity_with_en_bn_es_hi(manifests):
    """bn-cl holds, per (k, seed) and split, the uid list of en, bn, es and hi, in the same order."""
    main, ext = manifests
    cl = json.loads(BNCL_MANIFEST.read_text(encoding="utf-8"))
    refs = {"en": _entries(main, "bnintent30-en"), "bn": _entries(main, "bnintent30-bn"),
            "es": _entries(ext, "bnintent30-es"), "hi": _entries(ext, "bnintent30-hi")}
    bncl = _entries(cl, "bnintent30-bn-cl")
    problems = [] if set(bncl) == set(refs["en"]) else ["bn-cl: (k, seed) entries differ from en"]
    for key in sorted(refs["en"]):
        for f in FILES:
            for name, d in refs.items():
                if key in bncl and bncl[key]["uids"][f] != d[key]["uids"][f]:
                    problems.append(f"bn-cl k={key[0]} seed={key[1]} {f}: uid list differs from {name}")
    check(problems, "uid identity bn-cl vs en/bn/es/hi")
    assert len(bncl) == 20


@bncl_present
def test_bn_cl_split_files_match_manifest_and_en_labels():
    """Every bn-cl split file matches its sha256 and uid list, and en's (uid, label) rows; text non-empty."""
    import hashlib
    main = json.loads(MANIFEST.read_text(encoding="utf-8"))
    en = _entries(main, "bnintent30-en")
    cl = json.loads(BNCL_MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for e in cl["splits"]:
        for f in FILES:
            raw = (BNCL_MANIFEST.parent / e["path"] / f"{f}.jsonl").read_bytes()
            if hashlib.sha256(raw).hexdigest() != e["sha256"][f]:
                problems.append(f"{e['path']}/{f}: sha256 differs from the bn-cl manifest")
            rows = [json.loads(l) for l in raw.decode("utf-8").splitlines()]
            ref = read_jsonl(ROOT / "data" / "splits" / en[(e["k"], e["seed"])]["path"] / f"{f}.jsonl")
            if [r["uid"] for r in rows] != e["uids"][f]:
                problems.append(f"{e['path']}/{f}: uids differ from the manifest")
            if [(r["uid"], r["label_id"]) for r in rows] != [(r["uid"], r["label_id"]) for r in ref]:
                problems.append(f"{e['path']}/{f}: (uid, label) rows differ from en")
            if any(not r["text"].strip() for r in rows):
                problems.append(f"{e['path']}/{f}: empty text")
    check(problems, "bn-cl split files")
    assert len(cl["splits"]) == 20
