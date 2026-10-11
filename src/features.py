"""Handcrafted feature families (CLAUDE.md §4), fit per run on the TRAIN split only (R1).

    tfidf      word TF-IDF, chi-squared selected to TFIDF_DIM columns
    chargram   character 3/4/5-grams (within word boundaries), hashed to CHARGRAM_DIM
    stats      6 utterance statistics, standardized

R1 is enforced by the class, not by convention:
  - transform() before fit() raises FeaturesNotFittedError;
  - fit() a second time raises FeaturesAlreadyFittedError: one instance per
    run, never refit (so the test-time object IS the train-time object, R1/§8.5);
  - fit() takes `train_texts`, `train_labels` and a mandatory keyword
    `split="train"`; any other split name raises. A caller cannot fit on val or
    test without writing the word "val"/"test" next to the call and seeing it fail.

FIXED OUTPUT DIMENSION (identical for every k, track and seed, so W_p has one shape):
  - tfidf:    chi2 keeps min(TFIDF_DIM, train vocabulary) terms; if the train
              vocabulary is smaller than TFIDF_DIM (e.g. every k=5 split) the
              block is ZERO-PADDED on the right to TFIDF_DIM. Padded columns are
              always 0 and are logged as `tfidf_padded_columns`.
  - chargram: hashing has a fixed output size by construction.
  - stats:    always 6 columns.
  TOTAL_DIM = TFIDF_DIM + CHARGRAM_DIM + 6. The dimensions are fixed constants,
  chosen once from the measured k=5..full vocabulary range (335-2,220 words),
  not tuned on any result.

NORMALIZATION APPLIED BY EACH FAMILY, compared with the dedup key (src.data.norm_key:
NFC, lowercase, punctuation -> space, whitespace collapsed). No family may be
MORE aggressive than the key, or two texts the key keeps apart could become one
input to the feature branch and the dedup would leave leakage (CLAUDE.md §3.3).
  - tfidf:    tokens = norm_key(text).split(). EXACTLY the key - equal, not more
              aggressive. (sklearn's default token_pattern is not used: it
              splits Bangla words at vowel signs, e.g. "আমার" -> "আম".)
              Single-character tokens are kept (the default pattern drops them),
              so "don't" -> ["don", "t"] and "dont" -> ["dont"] stay distinct.
  - chargram: NFC + lowercase; whitespace collapsed by char_wb; punctuation KEPT.
              LESS aggressive than the key (it distinguishes "pin?" from "pin").
  - stats:    computed on the NFC text with case, punctuation and digits intact
              (they are what the statistics measure). LESS aggressive than the key.
Feature-space information loss (out-of-vocabulary words, unselected terms, hash
collisions) is not text normalization; scripts/feature_report.py measures
whether it ever makes two distinct keys produce identical feature vectors.
"""
from __future__ import annotations

import hashlib
import unicodedata

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, normalize

from src.data import norm_key

TFIDF_DIM = 1000
CHARGRAM_DIM = 2 ** 12  # 4096
CHARGRAM_NGRAMS = (3, 5)
STATS_NAMES = ("token_count", "char_count", "mean_token_length",
               "digit_ratio", "punctuation_ratio", "uppercase_ratio")
STATS_DIM = len(STATS_NAMES)
TOTAL_DIM = TFIDF_DIM + CHARGRAM_DIM + STATS_DIM

NORMALIZATION = {
    "tfidf": "norm_key(text).split(): NFC, lowercase, punctuation -> space, whitespace "
             "collapsed (identical to the dedup key)",
    "chargram": "NFC, lowercase, whitespace collapsed (char_wb); punctuation kept "
                "(less aggressive than the dedup key)",
    "stats": "NFC only; case, punctuation and digits kept (less aggressive than the dedup key)",
}


class FeaturesNotFittedError(NotFittedError):
    pass


class FeaturesAlreadyFittedError(RuntimeError):
    pass


# Module-level so the fitted object stays picklable.
def word_tokens(text: str) -> list[str]:
    return norm_key(text).split()


def nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def utterance_stats(text: str) -> list[float]:
    t = unicodedata.normalize("NFC", text)
    tokens = t.split()
    n_chars = len(t)
    letters = [ch for ch in t if ch.isalpha()]
    return [
        float(len(tokens)),
        float(n_chars),
        float(np.mean([len(w) for w in tokens])) if tokens else 0.0,
        sum(ch.isdigit() for ch in t) / n_chars if n_chars else 0.0,      # incl. Bangla ০-৯
        sum(unicodedata.category(ch).startswith("P") for ch in t) / n_chars if n_chars else 0.0,
        sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0,  # 0 for Bangla
    ]


def _fingerprint(texts: list[str]) -> str:
    return hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()


class HandcraftedFeatures:
    """The three §4 families. One instance per run: fit once on train, transform anything."""

    def __init__(self) -> None:
        self._fitted = False
        self.n_fits = 0
        self.transform_log: dict[str, dict] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train_texts: list[str], train_labels: list[int], *, split: str) -> "HandcraftedFeatures":
        """Fit every learned component on the TRAIN split of ONE run (R1).

        `split` is mandatory and must be "train". Fitting on val, test, or any
        union of splits is a leakage bug, so it is refused rather than allowed.
        """
        if self._fitted:
            raise FeaturesAlreadyFittedError(
                "HandcraftedFeatures is already fitted. Create a new instance per run; "
                "never refit (R1, CLAUDE.md §8 assertion 5).")
        if split != "train":
            raise ValueError(f"fit() must receive the TRAIN split only (R1); got split={split!r}.")
        train_texts, train_labels = list(train_texts), list(train_labels)
        if not train_texts or len(train_texts) != len(train_labels):
            raise ValueError("fit() needs a non-empty train split with one label per text.")

        # tfidf: vocabulary + idf from train, then chi2 against train labels.
        self._tfidf = TfidfVectorizer(analyzer=word_tokens, min_df=1, norm=None)
        counts = self._tfidf.fit_transform(train_texts)
        self.tfidf_vocab_size = counts.shape[1]
        self.tfidf_selected = min(TFIDF_DIM, self.tfidf_vocab_size)
        self._chi2 = SelectKBest(chi2, k=self.tfidf_selected).fit(counts, train_labels)
        self.tfidf_padded_columns = TFIDF_DIM - self.tfidf_selected
        self.tfidf_unselected_terms = self.tfidf_vocab_size - self.tfidf_selected
        vocab = self._tfidf.get_feature_names_out()
        self.selected_terms = vocab[self._chi2.get_support()]

        # chargram: hashing is stateless; the idf weighting is learned from train.
        self._hasher = HashingVectorizer(analyzer="char_wb", ngram_range=CHARGRAM_NGRAMS,
                                         n_features=CHARGRAM_DIM, alternate_sign=False,
                                         norm=None, preprocessor=nfc_lower)
        self._char_idf = TfidfTransformer(sublinear_tf=True).fit(self._hasher.transform(train_texts))

        # stats: scaler learned from train.
        self._scaler = StandardScaler().fit(np.array([utterance_stats(t) for t in train_texts]))

        self.train_size = len(train_texts)
        self.train_fingerprint = _fingerprint(train_texts)
        self._fitted = True
        self.n_fits += 1
        return self

    # ------------------------------------------------------------ transform
    def transform(self, texts: list[str], *, split: str | None = None) -> sp.csr_matrix:
        """Apply the train-fitted components unchanged. Returns (n, TOTAL_DIM) float32 CSR.
        `split` only labels the log entry."""
        if not self._fitted:
            raise FeaturesNotFittedError("HandcraftedFeatures.transform() called before fit() (R1).")
        texts = list(texts)
        tfidf = normalize(self._chi2.transform(self._tfidf.transform(texts)), norm="l2")
        if self.tfidf_padded_columns:
            tfidf = sp.hstack([tfidf, sp.csr_matrix((len(texts), self.tfidf_padded_columns))])
        chars = self._char_idf.transform(self._hasher.transform(texts))
        stats = sp.csr_matrix(self._scaler.transform(np.array([utterance_stats(t) for t in texts])))
        X = sp.hstack([tfidf, chars, stats], format="csr", dtype=np.float32)
        assert X.shape[1] == TOTAL_DIM, f"feature dimension {X.shape[1]} != fixed {TOTAL_DIM}"

        if split is not None:
            self.transform_log[split] = {
                "n": len(texts),
                "tfidf_empty_rows": int((tfidf.getnnz(axis=1) == 0).sum()),
                "chargram_empty_rows": int((chars.getnnz(axis=1) == 0).sum()),
                "transformed_by": id(self),
            }
        return X

    # ------------------------------------------------------------- logging
    def report(self) -> dict:
        """Everything a run config needs about the realised features (R6)."""
        if not self._fitted:
            raise FeaturesNotFittedError("report() called before fit().")
        return {
            "dims": {"tfidf": TFIDF_DIM, "chargram": CHARGRAM_DIM, "stats": STATS_DIM,
                     "total": TOTAL_DIM},
            "tfidf_vocab_size": int(self.tfidf_vocab_size),
            "tfidf_selected": int(self.tfidf_selected),
            "tfidf_padded_columns": int(self.tfidf_padded_columns),
            "tfidf_unselected_terms": int(self.tfidf_unselected_terms),
            "chargram_ngrams": list(CHARGRAM_NGRAMS),
            "stats_names": list(STATS_NAMES),
            "normalization": NORMALIZATION,
            "fit": {"split": "train", "n": self.train_size,
                    "texts_sha256": self.train_fingerprint, "n_fits": self.n_fits},
            "transform_log": {k: {kk: vv for kk, vv in v.items() if kk != "transformed_by"}
                              for k, v in self.transform_log.items()},
        }


def featurize_run(splits: dict[str, list[dict]]) -> tuple[HandcraftedFeatures, dict]:
    """The ONLY way a run should build features: fit one fresh instance on
    splits["train"], then transform every split (train included) with that same
    object. Returns (fitted pipeline, {split name: matrix})."""
    pipeline = HandcraftedFeatures().fit([r["text"] for r in splits["train"]],
                                         [r["label_id"] for r in splits["train"]], split="train")
    X = {name: pipeline.transform([r["text"] for r in rows], split=name)
         for name, rows in splits.items()}
    return pipeline, X
