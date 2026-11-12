"""Spanish and Hindi extension (docs/PROTOCOL.md §12, with its amendments): constants and
helpers shared by the extension scripts and tests.

Tracks bnintent30-es and bnintent30-hi are the clean rows and splits of bnintent30-en
(same uids, same order, same labels) with the text replaced by the reference
translations data/extension/{es,hi}.jsonl, which must match the SHA-256 hashes recorded
in §12 (amendment 2026-11-10). Nothing here touches an existing track, split, cache or
result: the extension has its own split directory, embedding cache and run directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.data import dedup_key

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "extension"
SPLIT_BASE = EXT / "splits"              # own manifest.json; data/splits/ is never modified
CACHE_DIR = EXT / "cache"                # own embedding cache; data/cache/ is never modified
RUN_DIR = ROOT / "results" / "extension" / "runs"   # never results/runs/ (§12)
SOURCE_TRACK = "bnintent30-en"           # rows, splits and labels are taken from here
LANGS = ("es", "hi")
TRACK = {lang: f"bnintent30-{lang}" for lang in LANGS}
HEADS = ("sagff", "baseline")            # in src.train.CONDITIONS order
KS = ("5", "10", "20", "full")
K_RANK = {"5": 1, "10": 2, "20": 3, "full": 4}
SEEDS = (13, 42, 87, 100, 2026)
E1_KS = ("5", "10")
EXPECTED_SHA256 = {  # docs/PROTOCOL.md §12, amendment 2026-11-10
    "es": "26587bc208e24739b203c1c3ac8f25bfccbf9ea4ad10c6e0fee7b925cc4eb727",
    "hi": "c4b272ccaab6e76c902ffecce36285f357ddd0ad108813d686fbb2dfc5feed7a",
}

# Bangla translation-method control (§12, amendment 2026-11-12): the same rows with Bangla
# written by Claude from English. It has its own split directory and manifest, so the es/hi
# manifest and every es/hi output stay byte-identical.
CONTROL = "bn-cl"
TRACK[CONTROL] = "bnintent30-bn-cl"
CONTROL_SPLIT_BASE = EXT / "splits_bn_cl"
EXPECTED_SHA256[CONTROL] = "f01ed4bc007771dae2088bf3889fd6600cbb93a9545c17a8f950d75f7852c40e"  # §12 note 2026-11-13
TRANSLATION_FILE = {"es": "es.jsonl", "hi": "hi.jsonl", CONTROL: "bn_cl.jsonl"}
# group -> (languages, split directory, suffix of its files under results/extension/)
GROUPS = {"es-hi": (LANGS, SPLIT_BASE, ""), CONTROL: ((CONTROL,), CONTROL_SPLIT_BASE, "_bn_cl")}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_translations(lang: str) -> dict[str, str]:
    """{uid: text} of the reference translation; refuses a file whose hash differs from §12."""
    name = TRANSLATION_FILE[lang]
    raw = (EXT / name).read_bytes()
    got = sha256_bytes(raw)
    if got != EXPECTED_SHA256[lang]:
        raise RuntimeError(f"data/extension/{name} sha256 {got} != {EXPECTED_SHA256[lang]} "
                           "recorded in docs/PROTOCOL.md §12")
    return {r["uid"]: r["text"] for r in map(json.loads, raw.decode("utf-8").splitlines())}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def colliding_test_uids(train_texts: list[str], test_rows: list[tuple[str, str]]) -> list[str]:
    """§12 amendment 2026-11-09: a test row collides if its dedup key (§3.3) equals the key
    of any row in the run's own training subset. test_rows: (uid, text) pairs."""
    keys = {dedup_key(t) for t in train_texts}
    return [u for u, t in test_rows if dedup_key(t) in keys]
