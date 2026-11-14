"""Exploratory, post hoc text diagnostics: the original Bangla (Google) against bn-cl (Claude),
with en, es and hi for reference (docs/PROTOCOL.md §12, note 2026-11-15).

Added after the results of E1 and E4 were seen; descriptive only, no test, no training run.
Numbers only: no dataset text is written. For each of the five BNIntent30 tracks, on the
4,096 clean rows (identical uids in every track):

  1. XLM-R subword pieces per whitespace word (mean and median over utterances), and mean
     utterance length in pieces (the pinned tokenizer of src/embeddings.py, no special tokens).
  2. At k = 5, per seed: share of test word tokens absent from the train-fit TF-IDF vocabulary,
     and share of test items with an all-zero TF-IDF vector. Same definitions and code as
     scripts/feature_report.py (results/tables/tfidf_coverage.md), which this script checks
     for en and bn.
  3. In the frozen XLM-R embedding space (the cached mean-pooled embeddings every run uses):
     mean cosine similarity over all same-intent pairs and over all different-intent pairs,
     and their ratio (class separability).
  4. Type-token ratio and number of distinct word types (word tokens as in 2).

Writes results/extension/bangla_text_diagnostics.md.

Usage (from the repository root):  python scripts/bangla_text_diagnostics.py
"""
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import ENCODER, REVISION, load_embeddings  # noqa: E402
from src.extension import (CACHE_DIR, CONTROL, CONTROL_SPLIT_BASE, SEEDS, SPLIT_BASE,  # noqa: E402
                           load_translations)
from src.features import featurize_run, word_tokens  # noqa: E402
from src.train import RunSpec, load_split  # noqa: E402

OUT = ROOT / "results" / "extension" / "bangla_text_diagnostics.md"
# track -> (label, split directory or None for data/splits/, embedding cache directory or None)
TRACKS = {"bnintent30-bn": ("bn (Google, original)", None, None),
          "bnintent30-bn-cl": ("bn-cl (Claude)", CONTROL_SPLIT_BASE, CACHE_DIR),
          "bnintent30-en": ("en (source)", None, None),
          "bnintent30-es": ("es (Claude)", SPLIT_BASE, CACHE_DIR),
          "bnintent30-hi": ("hi (Claude)", SPLIT_BASE, CACHE_DIR)}


def splits(track: str, k: str, seed: int) -> dict:
    """The run's split files, sha256-verified against their manifest (src.train.load_split)."""
    base = TRACKS[track][1]
    return load_split(RunSpec(track=track, k=k, seed=seed, condition="sagff", split_base=base))[0]


def pair_similarity(E: np.ndarray, y: np.ndarray, centre: bool = False) -> tuple[float, float]:
    """Mean cosine similarity over all same-label pairs and over all different-label pairs.
    centre=True subtracts the track's mean embedding first (removes the common direction that
    makes raw cosines of mean-pooled XLM-R vectors nearly constant)."""
    if centre:
        E = E - E.mean(0, keepdims=True)
    Z = E / np.linalg.norm(E, axis=1, keepdims=True)
    n = len(Z)
    tot = (float(Z.sum(0) @ Z.sum(0)) - n) / 2
    same_sum, same_n = 0.0, 0
    for c in np.unique(y):
        S, m = Z[y == c].sum(0), int((y == c).sum())
        same_sum += (float(S @ S) - m) / 2
        same_n += m * (m - 1) // 2
    diff_n = n * (n - 1) // 2 - same_n
    return same_sum / same_n, (tot - same_sum) / diff_n


def coverage(track: str, seed: int) -> tuple[float, float]:
    """As scripts/feature_report.py: test-token OOV against the train-fit vocabulary (%), and
    the share of test items whose TF-IDF row is all zero (%)."""
    sp = splits(track, "5", seed)
    pipe, _ = featurize_run(sp)
    vocab = set(pipe._tfidf.vocabulary_)
    toks = [t for r in sp["test"] for t in word_tokens(r["text"])]
    empty = pipe.report()["transform_log"]["test"]["tfidf_empty_rows"]
    return 100 * sum(t not in vocab for t in toks) / len(toks), 100 * empty / len(sp["test"])


def main() -> None:
    from transformers import AutoTokenizer
    for lang in ("es", "hi", CONTROL):
        load_translations(lang)             # stops if a reference file differs from §12
    tok = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION)
    rows, cov = {}, {}
    for track, (label, _, cache) in TRACKS.items():
        full = splits(track, "full", SEEDS[0])
        recs = sorted((r for n in ("train", "val_full", "test") for r in full[n]), key=lambda r: r["uid"])
        assert len(recs) == 4096 and len({r["uid"] for r in recs}) == 4096, track
        texts, uids = [r["text"] for r in recs], [r["uid"] for r in recs]
        y = np.array([r["label_id"] for r in recs])
        pieces = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
        ratio = [p / max(1, len(t.split())) for p, t in zip(pieces, texts)]
        words = [w for t in texts for w in word_tokens(t)]
        E = load_embeddings(track, uids, cache)
        same, diff = pair_similarity(E, y)
        csame, cdiff = pair_similarity(E, y, centre=True)
        rows[track] = {"ppw_mean": statistics.mean(ratio), "ppw_median": statistics.median(ratio),
                       "pieces": statistics.mean(pieces), "same": same, "diff": diff, "sep": same / diff,
                       "csame": csame, "cdiff": cdiff,
                       "types": len(set(words)), "tokens": len(words), "ttr": len(set(words)) / len(words)}
        cov[track] = {s: coverage(track, s) for s in SEEDS}
        print(f"{track}: done")

    # consistency with the committed tfidf_coverage.md for the two main BNIntent30 tracks
    ref = (ROOT / "results" / "tables" / "tfidf_coverage.md").read_text(encoding="utf-8")
    for track in ("bnintent30-bn", "bnintent30-en"):
        m = re.search(rf"\| {track} \| 5 \| \d+ \| [\d.]+ \([^)]*\) \| ([\d.]+) \([^)]*\) \| [\d.]+ \| ([\d.]+) \(", ref)
        z, o = float(m.group(1)), float(m.group(2))
        zz = statistics.mean(v[1] for v in cov[track].values())
        oo = statistics.mean(v[0] for v in cov[track].values())
        assert abs(zz - z) < 0.051 and abs(oo - o) < 0.051, (track, zz, z, oo, o)
    print("k=5 coverage matches results/tables/tfidf_coverage.md for bn and en")

    lab = {t: v[0] for t, v in TRACKS.items()}
    L = ["# Bangla text diagnostics: original bn (Google) against bn-cl (Claude), with en, es, hi", "",
         "Generated by `python scripts/bangla_text_diagnostics.py`. Do not edit by hand. Numbers only: no "
         "dataset text. **Exploratory and post hoc** (docs/PROTOCOL.md §12, note 2026-11-15): added after the "
         "results of E1 and E4 were seen; descriptive only, no test, no new training run.", "",
         "All five tracks hold the same 4,096 clean rows (same uids, intents and splits). Word tokens are the "
         "TF-IDF tokens of the pipeline (`word_tokens` = `norm_key(text).split()`). Subword pieces come from the "
         f"frozen encoder's tokenizer (`{ENCODER}` at revision `{REVISION}`), without special tokens. Embeddings "
         "are the cached mean-pooled XLM-R vectors every run uses.", "",
         "## 1. Subword segmentation and length (all clean rows)", "",
         "| track | pieces per whitespace word, mean | median | utterance length in pieces, mean |",
         "| --- | --- | --- | --- |"]
    for t, r in rows.items():
        L.append(f"| {lab[t]} | {r['ppw_mean']:.3f} | {r['ppw_median']:.3f} | {r['pieces']:.2f} |")
    for title, idx in (("## 2a. Test-token OOV against the training vocabulary at k = 5 (% of test word tokens)", 0),
                       ("## 2b. Test items with an all-zero TF-IDF vector at k = 5 (% of 856 test items)", 1)):
        L += ["", title, "",
              "Per seed; same definition and code as `results/tables/tfidf_coverage.md` (the en and bn means "
              "are checked against it by the script).", "",
              "| track | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean |",
              "| --- | " + " | ".join("---" for _ in SEEDS) + " | --- |"]
        for t in rows:
            v = [cov[t][s][idx] for s in SEEDS]
            L.append(f"| {lab[t]} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {statistics.mean(v):.1f} |")
    L += ["", "## 3. Class separability in the frozen XLM-R embedding space (all clean rows)", "",
          "Mean cosine similarity over all pairs of utterances with the same intent and over all pairs with "
          "different intents; ratio = same / different (higher: intents more separated).", "",
          "| track | same-intent pairs | different-intent pairs | ratio |", "| --- | --- | --- | --- |"]
    for t, r in rows.items():
        L.append(f"| {lab[t]} | {r['same']:.4f} | {r['diff']:.4f} | {r['sep']:.4f} |")
    L += ["", "Raw cosines of mean-pooled XLM-R vectors are close to 1 for all pairs (the vectors share a dominant "
          "common direction), so the ratio varies little. As a supplementary view, not part of the request that "
          "defined this file, the same quantities after subtracting the track's mean embedding (centred "
          "cosine). The different-intent mean is then near zero, so the gap (same minus different) is reported "
          "instead of a ratio.", "",
          "| track | same-intent pairs, centred | different-intent pairs, centred | gap (same - different) |",
          "| --- | --- | --- | --- |"]
    for t, r in rows.items():
        L.append(f"| {lab[t]} | {r['csame']:.4f} | {r['cdiff']:.4f} | {r['csame'] - r['cdiff']:.4f} |")
    L += ["", "## 4. Lexical variety (all clean rows)", "",
          "| track | word tokens | distinct word types | type-token ratio |", "| --- | --- | --- | --- |"]
    for t, r in rows.items():
        L.append(f"| {lab[t]} | {r['tokens']:,} | {r['types']:,} | {r['ttr']:.4f} |")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
