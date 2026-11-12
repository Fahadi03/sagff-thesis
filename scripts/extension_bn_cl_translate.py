"""Bangla translation-method control (docs/PROTOCOL.md §12, amendment 2026-11-12):
batch export and merge for the track bnintent30-bn-cl. Translates nothing itself.

  export   Reads ONLY data/clean/bnintent30-en.jsonl (uid, English text, intent, split)
           and writes one batch file per intent under data/extension/provenance/bn_cl/:
               in_<intent>.tsv     idx <TAB> uid <TAB> English sentence   (rows sorted by uid)
           It never opens bn_text, data/clean/bnintent30-bn.jsonl or any Bangla split.
  merge    Reads the translator's files  out_<intent>.txt  (one line "idx|Bangla" per row),
           checks that every idx of the batch has exactly one non-empty line containing
           Bengali script, and writes data/extension/bn_cl.jsonl, one {"uid", "text"} line
           per clean uid (4,096), sorted by uid. Prints its SHA-256.

The translator (Claude, model recorded in §12) reads the idx and English columns of the
in_ files and writes the out_ files. All files here are gitignored: they hold dataset text.

Usage:  python scripts/extension_bn_cl_translate.py export
        python scripts/extension_bn_cl_translate.py merge
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN_EN = ROOT / "data" / "clean" / "bnintent30-en.jsonl"
NAMES = ROOT / "data" / "clean" / "bnintent30-en.label_names.json"
PROV = ROOT / "data" / "extension" / "provenance" / "bn_cl"
OUT = ROOT / "data" / "extension" / "bn_cl.jsonl"
BENGALI = re.compile(r"[ঀ-৿]")


def batches() -> dict[str, list[dict]]:
    names = json.loads(NAMES.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for line in CLEAN_EN.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by[names[r["label_id"]]].append({"uid": r["uid"], "en": r["text"]})
    return {intent: sorted(rows, key=lambda r: r["uid"]) for intent, rows in sorted(by.items())}


def export() -> None:
    PROV.mkdir(parents=True, exist_ok=True)
    total = 0
    for intent, rows in batches().items():
        assert len(rows) <= 150, (intent, len(rows))
        for r in rows:
            assert "\t" not in r["en"] and "\n" not in r["en"]
        (PROV / f"in_{intent}.tsv").write_text(
            "".join(f"{i}\t{r['uid']}\t{r['en']}\n" for i, r in enumerate(rows, 1)), encoding="utf-8", newline="\n")
        total += len(rows)
        print(f"{intent}: {len(rows)} rows")
    print(f"{total} rows in {len(batches())} batch files -> {PROV.relative_to(ROOT)}")


def merge() -> None:
    out, problems = {}, []
    for intent, rows in batches().items():
        f = PROV / f"out_{intent}.txt"
        if not f.exists():
            problems.append(f"{intent}: {f.name} missing")
            continue
        got = {}
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            m = re.fullmatch(r"(\d+)\|(.*)", line)
            if not m:
                problems.append(f"{intent} line {n}: not 'idx|text'")
                continue
            i, text = int(m.group(1)), " ".join(m.group(2).split())
            if i in got:
                problems.append(f"{intent}: idx {i} appears twice")
            got[i] = text
        want = set(range(1, len(rows) + 1))
        if set(got) != want:
            problems.append(f"{intent}: missing idx {sorted(want - set(got))[:10]}, extra {sorted(set(got) - want)[:10]}")
        for i, r in enumerate(rows, 1):
            t = got.get(i, "")
            if not t or not BENGALI.search(t):
                problems.append(f"{intent} idx {i}: empty or no Bengali script")
            out[r["uid"]] = t
    if problems:
        sys.exit("STOP, nothing written:\n  " + "\n  ".join(problems[:40]))
    assert len(out) == 4096, len(out)
    b = "".join(json.dumps({"uid": u, "text": out[u]}, ensure_ascii=False) + "\n" for u in sorted(out)).encode("utf-8")
    OUT.write_bytes(b)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows, sha256 {hashlib.sha256(b).hexdigest()}")


if __name__ == "__main__":
    {"export": export, "merge": merge}[sys.argv[1]]()
