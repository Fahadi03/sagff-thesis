"""Translation-quality checks for the Spanish and Hindi extension (docs/PROTOCOL.md §12).

Descriptive only: no row is excluded and no model of the main grid is trained.
Over the 4,096 clean BNIntent30 uids, for bn (the dataset's own translation), es and
hi (written by Claude; §12 amendment 2026-11-10):

  1. LaBSE similarity: cosine between the English sentence and its translation.
  2. Intent preservation: a logistic regression on LaBSE embeddings of the ENGLISH
     TRAIN split only (mirroring the main protocol), applied to each language's test
     split; accuracy and Macro-F1, overall and per intent.
  3. A review sheet of 5 uids per intent (150) for human rating.

Outputs
  results/extension/translation_quality.md      numbers only (no dataset text)
  results/extension/low_similarity_uids.txt     uids with similarity < 0.70, per language
  results/extension/review_sample_uids.txt      the 150 sampled uids
  data/extension/review_sheet.csv               WITH dataset text: gitignored, never committed
                                                (UTF-8 with BOM, so Excel shows bn/hi correctly)
  data/extension/labse/*.npy                    embedding cache (gitignored)

The script stops if es.jsonl or hi.jsonl differs from the SHA-256 recorded in §12.

Usage (from the repository root):  python scripts/translation_quality.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import sklearn
import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import hash_key  # noqa: E402  (same sha256 ranking as the splits)

EXT = ROOT / "data" / "extension"
OUT = ROOT / "results" / "extension"
CLEAN = ROOT / "data" / "clean"
EXPECTED_SHA256 = {  # docs/PROTOCOL.md §12, amendment 2026-11-10
    "es": "26587bc208e24739b203c1c3ac8f25bfccbf9ea4ad10c6e0fee7b925cc4eb727",
    "hi": "c4b272ccaab6e76c902ffecce36285f357ddd0ad108813d686fbb2dfc5feed7a",
}
LABSE = "sentence-transformers/LaBSE"
LABSE_REVISION = "836121a0533e5664b21c7aacc5d22951f2b8b25b"
LABSE_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "modules.json", "sentence_bert_config.json",
               "1_Pooling/config.json", "2_Dense/config.json", "2_Dense/model.safetensors"]
MAX_LEN = 256          # LaBSE's sentence_bert_config.json
BATCH = 64
LOW_SIM = 0.70
F1_GAP = 0.15          # flag an intent whose F1 is more than this below English
REVIEW_PER_INTENT = 5
REVIEW_SEED = "13"
LANGS = ("bn", "es", "hi")
THREADS = 6


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_rows() -> tuple[list[dict], list[str]]:
    for lang, sha in EXPECTED_SHA256.items():
        got = sha256_file(EXT / f"{lang}.jsonl")
        if got != sha:
            sys.exit(f"STOP: data/extension/{lang}.jsonl sha256 {got} != recorded {sha} (PROTOCOL §12).")
    en = read_jsonl(CLEAN / "bnintent30-en.jsonl")
    bn = {r["uid"]: r for r in read_jsonl(CLEAN / "bnintent30-bn.jsonl")}
    names = json.loads((CLEAN / "bnintent30-en.label_names.json").read_text(encoding="utf-8"))
    tr = {lang: {r["uid"]: r["text"] for r in read_jsonl(EXT / f"{lang}.jsonl")} for lang in ("es", "hi")}
    rows = []
    for r in en:
        u = r["uid"]
        assert bn[u]["label_id"] == r["label_id"] and bn[u]["split"] == r["split"], u
        assert tr["es"].get(u, "").strip() and tr["hi"].get(u, "").strip(), f"missing translation {u}"
        rows.append({"uid": u, "label_id": r["label_id"], "split": r["split"],
                     "en": r["text"], "bn": bn[u]["text"], "es": tr["es"][u], "hi": tr["hi"][u]})
    assert len(rows) == 4096, len(rows)
    return rows, names


# --------------------------------------------------------------------------- #
# LaBSE: CLS pooling -> Dense(768, 768) + tanh -> L2 normalisation (modules.json)
# --------------------------------------------------------------------------- #
class Labse:
    def __init__(self):
        self.path = Path(snapshot_download(LABSE, revision=LABSE_REVISION, allow_patterns=LABSE_FILES))
        self.tok = transformers.AutoTokenizer.from_pretrained(self.path)
        self.model = transformers.AutoModel.from_pretrained(self.path).eval()
        dense = load_file(str(self.path / "2_Dense" / "model.safetensors"))
        self.w, self.b = dense["linear.weight"], dense["linear.bias"]

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[np.ndarray, int]:
        out, truncated = [], 0
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            truncated += sum(len(ids) > MAX_LEN for ids in self.tok(batch)["input_ids"])
            enc = self.tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            cls = self.model(**enc).last_hidden_state[:, 0]
            v = torch.tanh(cls @ self.w.T + self.b)
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(out).astype(np.float32), truncated


def embeddings(rows: list[dict], langs: tuple = ("en",) + LANGS, model: Labse | None = None
               ) -> tuple[dict[str, np.ndarray], dict[str, int], Labse]:
    cache = EXT / "labse"
    cache.mkdir(parents=True, exist_ok=True)
    model = model or Labse()
    E, trunc = {}, {}
    order = hashlib.sha256("\n".join(r["uid"] for r in rows).encode()).hexdigest()
    for lang in langs:
        texts = [r[lang] for r in rows]
        key = hashlib.sha256((LABSE_REVISION + order + "\x1f".join(texts)).encode("utf-8")).hexdigest()[:16]
        f = cache / f"{lang}_{key}.npy"
        if f.exists():
            E[lang] = np.load(f)
            trunc[lang] = int(json.loads((cache / f"{lang}_{key}.json").read_text())["truncated"])
        else:
            E[lang], trunc[lang] = model.encode(texts)
            np.save(f, E[lang])
            (cache / f"{lang}_{key}.json").write_text(json.dumps({"truncated": trunc[lang]}))
        print(f"  {lang}: {E[lang].shape} ({f.name})")
    return E, trunc, model


# --------------------------------------------------------------------------- #
# report helpers
# --------------------------------------------------------------------------- #
def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def bn_cl_report(rows, names, E, trunc, model, sim, by_intent, clf, y, split, acc, mf1, per, preds, sample) -> None:
    """Bangla translation-method control (docs/PROTOCOL.md §12, amendment 2026-11-12): the same
    checks for the track bn-cl (Bangla written by Claude), next to the original bn, plus the
    share of rows whose bn-cl text is identical to the original bn text. Descriptive only.
    Writes results/extension/translation_quality_bn_cl.md and low_similarity_uids_bn_cl.txt
    (numbers and uids only) and data/extension/review_sheet_bn_cl.csv (with text, gitignored)."""
    from src.data import dedup_key, norm_key
    from src.extension import CONTROL, EXPECTED_SHA256 as EXT_SHA, load_translations

    tr = load_translations(CONTROL)             # stops if the sha256 differs from §12
    for r in rows:
        r[CONTROL] = tr[r["uid"]]
    Ecl, tcl, _ = embeddings(rows, (CONTROL,), model)
    Ecl, uids = Ecl[CONTROL], [r["uid"] for r in rows]
    n_int = len(names)
    test = split == "test"
    yt = y[test]
    s_cl = (E["en"] * Ecl).sum(axis=1)
    s_pair = (E["bn"] * Ecl).sum(axis=1)
    by_cl = {lid: float(s_cl[y == lid].mean()) for lid in range(n_int)}
    p_cl = clf.predict(Ecl[test])
    acc_cl = accuracy_score(yt, p_cl)
    mf1_cl = f1_score(yt, p_cl, average="macro", labels=range(n_int), zero_division=0)
    per_cl = f1_score(yt, p_cl, average=None, labels=range(n_int), zero_division=0)
    same = {name: sum(f(r["bn"]) == f(r[CONTROL]) for r in rows)
            for name, f in (("exact string (whitespace-stripped)", str.strip),
                            ("normalised, order-sensitive (norm_key)", norm_key),
                            ("§3.3 dedup key (normalised token multiset)", dedup_key))}
    same_split = {s: sum(r["bn"].strip() == r[CONTROL].strip() for r in rows if r["split"] == s)
                  for s in ("train", "val", "test")}

    low = [u for u, s in zip(uids, s_cl) if s < LOW_SIM]
    (OUT / "low_similarity_uids_bn_cl.txt").write_text(
        f"# uids with LaBSE similarity (English vs bn-cl) below {LOW_SIM:.2f}.\n"
        "# Generated by scripts/translation_quality.py --bn-cl. uids only, no text. Descriptive: no row is excluded.\n\n"
        f"## bn-cl ({len(low)} uids)\n" + "".join(u + "\n" for u in low), encoding="utf-8", newline="\n")
    idx = {u: i for i, u in enumerate(uids)}
    with open(EXT / "review_sheet_bn_cl.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "intent", "split", "en", "bn (original)", "bn-cl (Claude)", "labse_bn", "labse_bn_cl",
                    "adequacy_bn (1-5)", "adequacy_bn_cl (1-5)", "intent_preserved_bn (y/n)",
                    "intent_preserved_bn_cl (y/n)", "comment"])
        for r in sample:
            i = idx[r["uid"]]
            w.writerow([r["uid"], names[r["label_id"]], r["split"], r["en"], r["bn"], r[CONTROL],
                        f3(sim["bn"][i]), f3(s_cl[i])] + [""] * 5)

    def sim_row(name, s, by):
        lo = min(by, key=by.get)
        return (f"| {name} | {len(s)} | {f3(s.mean())} | {f3(np.median(s))} | {f3(np.percentile(s, 5))} | "
                f"{f3(np.percentile(s, 25))} | {pct((s < 0.5).mean())} | {pct((s < LOW_SIM).mean())} | "
                f"{f3(by[lo])} ({names[lo]}) |")

    T = ["# Translation quality: bn-cl (Bangla written by Claude) next to the original bn", "",
         "Generated by `python scripts/translation_quality.py --bn-cl`. Do not edit by hand. Numbers only: no "
         "dataset text. Descriptive (docs/PROTOCOL.md §12, amendment 2026-11-12): no row is excluded on the basis "
         "of these checks. Same encoder, classifier and rows as `translation_quality.md`, which is unchanged.", "",
         "## Provenance", "",
         f"- Date: {date.today().isoformat()}",
         f"- Rows: the {len(rows)} clean uids (train {int((split == 'train').sum())}, val "
         f"{int((split == 'val').sum())}, test {int(test.sum())}).",
         "- bn: the dataset's own translation (Google Translate, reviewed by its authors). bn-cl: written by "
         "Claude (model Fable 5.2) from English only, not reviewed (§12 note 2026-11-13).",
         f"- `data/extension/bn_cl.jsonl` sha256 `{EXT_SHA[CONTROL]}` (verified)",
         f"- Encoder: `{LABSE}` at revision `{LABSE_REVISION}`, as in `translation_quality.md`; max length {MAX_LEN}.",
         f"- Sentences truncated at the max length: bn {trunc['bn']}, bn-cl {tcl[CONTROL]}.", "",
         "## 1. LaBSE similarity (cosine, all clean uids)", "",
         f"| pair | n | mean | median | 5th pct | 25th pct | share < 0.50 | share < {LOW_SIM:.2f} | "
         "lowest per-intent mean (intent) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
         sim_row("English vs bn (reference)", sim["bn"], by_intent["bn"]),
         sim_row("English vs bn-cl", s_cl, by_cl),
         sim_row("bn vs bn-cl", s_pair, {lid: float(s_pair[y == lid].mean()) for lid in range(n_int)}), "",
         f"uids with English-vs-bn-cl similarity below {LOW_SIM:.2f}: "
         f"`results/extension/low_similarity_uids_bn_cl.txt` ({len(low)}).", "",
         "## 2. Intent preservation", "",
         "The logistic regression of `translation_quality.md` (fit on LaBSE embeddings of the English TRAIN split "
         f"only), applied unchanged to the test split ({int(test.sum())} rows).", "",
         "| language | accuracy | Macro-F1 | Macro-F1 minus English | prediction agrees with English prediction |",
         "| --- | --- | --- | --- | --- |",
         f"| en | {f3(acc['en'])} | {f3(mf1['en'])} | -- | -- |",
         f"| bn (reference) | {f3(acc['bn'])} | {f3(mf1['bn'])} | {mf1['bn'] - mf1['en']:+.3f} | "
         f"{pct(float((preds['bn'] == preds['en']).mean()))} |",
         f"| bn-cl | {f3(acc_cl)} | {f3(mf1_cl)} | {mf1_cl - mf1['en']:+.3f} | "
         f"{pct(float((p_cl == preds['en']).mean()))} |", "",
         "Per intent: mean English-vs-translation similarity (all rows) and F1 on the test split.", "",
         "| intent | similarity bn | similarity bn-cl | test n | F1 en | F1 bn | F1 bn-cl |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for lid in range(n_int):
        T.append(f"| {names[lid]} | {f3(by_intent['bn'][lid])} | {f3(by_cl[lid])} | {int((yt == lid).sum())} | "
                 f"{f3(per['en'][lid])} | {f3(per['bn'][lid])} | {f3(per_cl[lid])} |")
    T += ["", f"Intents whose F1 is more than {F1_GAP:.2f} below English:", ""]
    for name, p in (("bn (reference)", per["bn"]), ("bn-cl", per_cl)):
        flagged = [(names[lid], p[lid] - per["en"][lid]) for lid in range(n_int) if per["en"][lid] - p[lid] > F1_GAP]
        T.append(f"- {name}: " + ("; ".join(f"{n} ({d:+.3f})" for n, d in sorted(flagged, key=lambda x: x[1]))
                                  if flagged else "none"))
    T += ["", "## 3. Rows where bn-cl is identical to the original bn text", "",
          "Computed by script on the two texts of each clean uid; not by reading (§12, blindness check).", "",
          "| comparison | identical rows | share of 4,096 |", "| --- | --- | --- |"]
    T += [f"| {name} | {n} | {pct(n / len(rows))} |" for name, n in same.items()]
    T += ["", "Exact-string identical rows by split: " + ", ".join(f"{s} {n}" for s, n in same_split.items()) + ".", "",
          "## 4. Review sheet", "",
          f"The {len(sample)} uids of `results/extension/review_sample_uids.txt`, with English, the original bn, bn-cl, "
          "both LaBSE scores and empty rating columns: `data/extension/review_sheet_bn_cl.csv` (UTF-8 with BOM; "
          "gitignored, not committed).", ""]
    (OUT / "translation_quality_bn_cl.md").write_text("\n".join(T), encoding="utf-8", newline="\n")
    print("wrote results/extension/translation_quality_bn_cl.md, low_similarity_uids_bn_cl.txt; "
          "data/extension/review_sheet_bn_cl.csv")


def main() -> None:
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    rows, names = load_rows()
    print(f"{len(rows)} clean uids; es/hi hashes match PROTOCOL §12")
    E, trunc, model = embeddings(rows)
    uids = [r["uid"] for r in rows]
    y = np.array([r["label_id"] for r in rows])
    split = np.array([r["split"] for r in rows])

    # 1. similarity ------------------------------------------------------------ #
    sim = {lang: (E["en"] * E[lang]).sum(axis=1) for lang in LANGS}
    by_intent = {lang: {lid: float(sim[lang][y == lid].mean()) for lid in range(len(names))} for lang in LANGS}

    # 2. intent preservation ---------------------------------------------------- #
    train, test = split == "train", split == "test"
    clf = LogisticRegression(max_iter=5000, random_state=0).fit(E["en"][train], y[train])
    yt = y[test]
    preds = {lang: clf.predict(E[lang][test]) for lang in ("en",) + LANGS}
    acc = {lang: accuracy_score(yt, p) for lang, p in preds.items()}
    mf1 = {lang: f1_score(yt, p, average="macro", labels=range(len(names)), zero_division=0) for lang, p in preds.items()}
    per = {lang: f1_score(yt, p, average=None, labels=range(len(names)), zero_division=0) for lang, p in preds.items()}
    agree_en = {lang: float((preds[lang] == preds["en"]).mean()) for lang in LANGS}

    # 3. review sample ------------------------------------------------------------ #
    by_lid = defaultdict(list)
    for r in rows:
        by_lid[r["label_id"]].append(r)
    sample = []
    for lid in sorted(by_lid):
        sample += sorted(by_lid[lid], key=lambda r: hash_key(REVIEW_SEED, "review", r["uid"]))[:REVIEW_PER_INTENT]
    idx = {u: i for i, u in enumerate(uids)}

    if "--bn-cl" in sys.argv:   # Bangla control only: the files written below are left untouched
        bn_cl_report(rows, names, E, trunc, model, sim, by_intent, clf, y, split, acc, mf1, per, preds, sample)
        return

    # --- write outputs ------------------------------------------------------------ #
    OUT.mkdir(parents=True, exist_ok=True)
    low = {lang: [u for u, s in zip(uids, sim[lang]) if s < LOW_SIM] for lang in LANGS}
    L = [f"# uids with LaBSE similarity (English vs translation) below {LOW_SIM:.2f}, per language.",
         "# Generated by scripts/translation_quality.py. uids only, no text. Descriptive: no row is excluded."]
    for lang in LANGS:
        L += ["", f"## {lang} ({len(low[lang])} uids)"] + low[lang]
    (OUT / "low_similarity_uids.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "review_sample_uids.txt").write_text(
        f"# {REVIEW_PER_INTENT} uids per intent ({len(sample)}), ranked by sha256('{REVIEW_SEED}', 'review', uid) "
        "within each intent over all clean rows. Generated by scripts/translation_quality.py. "
        "The sheet with text is data/extension/review_sheet.csv (gitignored).\n"
        + "\n".join(r["uid"] for r in sample) + "\n", encoding="utf-8")
    with open(EXT / "review_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "intent", "split", "en", "bn", "es", "hi", "labse_bn", "labse_es", "labse_hi",
                    "adequacy_bn (1-5)", "adequacy_es (1-5)", "adequacy_hi (1-5)",
                    "intent_preserved_bn (y/n)", "intent_preserved_es (y/n)", "intent_preserved_hi (y/n)", "comment"])
        for r in sample:
            i = idx[r["uid"]]
            w.writerow([r["uid"], names[r["label_id"]], r["split"], r["en"], r["bn"], r["es"], r["hi"]]
                       + [f3(sim[lang][i]) for lang in LANGS] + [""] * 7)

    weights_sha = sha256_file(model.path / "model.safetensors")
    dense_sha = sha256_file(model.path / "2_Dense" / "model.safetensors")
    T = ["# Translation quality: bn, es, hi against English (BNIntent30 clean rows)", "",
         "Generated by `python scripts/translation_quality.py`. Do not edit by hand. Numbers only: no dataset "
         "text. Descriptive (docs/PROTOCOL.md §12): no row is excluded on the basis of these checks.", "",
         "## Provenance", "",
         f"- Date: {date.today().isoformat()}",
         f"- Rows: the {len(rows)} clean uids of `bnintent30-en` / `bnintent30-bn` "
         f"(train {int(train.sum())}, val {int((split == 'val').sum())}, test {int(test.sum())}).",
         "- Translations: bn is the dataset's own translation (Google Translate, reviewed by its authors); "
         "es and hi were written by Claude from English only, not reviewed (§12 amendment 2026-11-10).",
         f"- `data/extension/es.jsonl` sha256 `{EXPECTED_SHA256['es']}` (verified)",
         f"- `data/extension/hi.jsonl` sha256 `{EXPECTED_SHA256['hi']}` (verified)",
         f"- Encoder: `{LABSE}` at revision `{LABSE_REVISION}`; CLS pooling, Dense 768-768 + tanh, "
         f"L2 normalisation (the model's `modules.json`); max length {MAX_LEN}. "
         f"Weights sha256 `{weights_sha}`; dense layer sha256 `{dense_sha}`.",
         "- Sentences truncated at the max length: " + ", ".join(f"{l} {trunc[l]}" for l in ("en",) + LANGS) + ".",
         f"- Python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__}, "
         f"scikit-learn {sklearn.__version__}, numpy {np.__version__}; {THREADS} threads.", "",
         "## 1. LaBSE similarity (cosine, English vs translation, all clean uids)", "",
         f"| language | n | mean | median | 5th pct | 25th pct | share < 0.50 | share < {LOW_SIM:.2f} | "
         "lowest per-intent mean (intent) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lang in LANGS:
        s = sim[lang]
        lo = min(by_intent[lang], key=by_intent[lang].get)
        T.append(f"| {lang} | {len(s)} | {f3(s.mean())} | {f3(np.median(s))} | {f3(np.percentile(s, 5))} | "
                 f"{f3(np.percentile(s, 25))} | {pct((s < 0.5).mean())} | {pct((s < LOW_SIM).mean())} | "
                 f"{f3(by_intent[lang][lo])} ({names[lo]}) |")
    T += ["", f"uids below {LOW_SIM:.2f}: `results/extension/low_similarity_uids.txt` "
          f"(bn {len(low['bn'])}, es {len(low['es'])}, hi {len(low['hi'])}).", "",
          "Mean similarity per intent:", "",
          "| intent | bn | es | hi |", "| --- | --- | --- | --- |"]
    for lid in range(len(names)):
        T.append(f"| {names[lid]} | " + " | ".join(f3(by_intent[l][lid]) for l in LANGS) + " |")
    T += ["", "## 2. Intent preservation", "",
          f"Logistic regression (scikit-learn defaults, lbfgs, max_iter 5000, random_state 0) fit on LaBSE "
          f"embeddings of the English TRAIN split only ({int(train.sum())} rows), applied unchanged to the "
          f"test split ({int(test.sum())} rows) of each language. English is the reference ceiling.", "",
          "| language | accuracy | Macro-F1 | Macro-F1 minus English | prediction agrees with English prediction |",
          "| --- | --- | --- | --- | --- |",
          f"| en | {f3(acc['en'])} | {f3(mf1['en'])} | -- | -- |"]
    for lang in LANGS:
        T.append(f"| {lang} | {f3(acc[lang])} | {f3(mf1[lang])} | {mf1[lang] - mf1['en']:+.3f} | {pct(agree_en[lang])} |")
    T += ["", "Per-intent F1 on the test split:", "",
          "| intent | test n | en | bn | es | hi |", "| --- | --- | --- | --- | --- | --- |"]
    for lid in range(len(names)):
        T.append(f"| {names[lid]} | {int((yt == lid).sum())} | " + " | ".join(f3(per[l][lid]) for l in ("en",) + LANGS) + " |")
    T += ["", f"Intents whose F1 is more than {F1_GAP:.2f} below English:", ""]
    any_flag = False
    for lang in LANGS:
        flagged = [(names[lid], per[lang][lid] - per["en"][lid]) for lid in range(len(names))
                   if per["en"][lid] - per[lang][lid] > F1_GAP]
        any_flag |= bool(flagged)
        T.append(f"- {lang}: " + ("; ".join(f"{n} ({d:+.3f})" for n, d in sorted(flagged, key=lambda x: x[1]))
                                  if flagged else "none"))
    # 2b. Spanish 'weather' vs 'time' (counts only): 'tiempo' means both weather and time
    wl, tl = names.index("weather"), names.index("time")
    T += ["", "### Spanish `weather` and `time` (counts only)", "",
          "Spanish *tiempo* means both 'weather' and 'time'. Test items of the two intents classified as the "
          "other, per language, and the number of Spanish items containing the whole words *tiempo*, *hora* "
          "and *clima* (case-insensitive):", "",
          "| language | weather test items predicted as time | time test items predicted as weather |",
          "| --- | --- | --- |"]
    for lang in ("en",) + LANGS:
        p = preds[lang]
        T.append(f"| {lang} | {int(((yt == wl) & (p == tl)).sum())} of {int((yt == wl).sum())} | "
                 f"{int(((yt == tl) & (p == wl)).sum())} of {int((yt == tl).sum())} |")
    T += ["", "| es items | n | contain tiempo | contain hora | contain clima |", "| --- | --- | --- | --- | --- |"]
    words = {w: re.compile(rf"\b{w}\b", re.IGNORECASE) for w in ("tiempo", "hora", "clima")}
    for lid in (wl, tl):
        for sp in ("train", "test"):
            texts = [r["es"] for r in rows if r["label_id"] == lid and r["split"] == sp]
            T.append(f"| {names[lid]}, {sp} | {len(texts)} | "
                     + " | ".join(str(sum(bool(words[w].search(t)) for t in texts)) for w in words) + " |")

    T += ["", "## 3. Review sheet", "",
          f"{len(sample)} uids ({REVIEW_PER_INTENT} per intent; sha256 ranking with seed {REVIEW_SEED} and salt "
          "'review' over all clean rows): `results/extension/review_sample_uids.txt`. The sheet with the four "
          "texts, the LaBSE scores and empty rating columns (adequacy 1-5, intent preserved y/n, comment, per "
          "language) is `data/extension/review_sheet.csv` (UTF-8 with BOM; gitignored, not committed).", ""]
    (OUT / "translation_quality.md").write_text("\n".join(T), encoding="utf-8")
    print("wrote results/extension/translation_quality.md, low_similarity_uids.txt, review_sample_uids.txt; "
          "data/extension/review_sheet.csv")


if __name__ == "__main__":
    main()
