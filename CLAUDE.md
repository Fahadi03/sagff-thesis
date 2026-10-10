# SAGFF thesis experiments — repository rules (v2)

Paper: *"When Do Handcrafted Features Help? Scarcity-Aware Feature Fusion for
Low-Resource Intent Classification in English and Bangla"*

**This is v2. It supersedes the v1 CLAUDE.md completely.** v1 assumed a local
GPU and a BANKING77-vs-BNIntent30 comparison. Both have changed. If anything in
the repo still reflects v1, fix it to match this file.

This repository produces the numbers in a journal manuscript. Every number must
be reproducible and leakage-free. The rules below are not style preferences —
they are what makes the results publishable.

**If a rule blocks you, STOP and report it in plain language. Do not work around
it, do not "temporarily" relax it, do not silently change a config to make a run
pass.** A failed run that is reported is useful. A passing run that broke a rule
is worse than useless, because it looks fine.

---

## 0. What changed in v2, and why (read this first)

**No local GPU.** Training runs on CPU. Full fine-tuning of `xlm-roberta-base`
is therefore off the table — it would take weeks.

**So: the encoder is FROZEN.** Sentence representations are computed once per
track with a frozen pretrained encoder, cached to disk, and reused by every run.
Each run trains only the fusion head (projection + gate + classifier). This is
feature-based transfer in the sense of Peters, Ruder & Smith (2019), and it is a
deliberate methodological choice, not a shortcut: few-sample full fine-tuning is
unstable across seeds (Dodge et al. 2020), and that instability would swamp the
gate trend this paper measures. State it that way in the paper.

**The language comparison is now parallel.** BNIntent30 contains BOTH the
English source (`text`) and the Bangla translation (`bn_text`) for every row, with
the same intent label. So the English and Bangla tracks are the *same rows in two
languages* — same intents, same content, same splits, same encoder. Only the
language differs. This removes the domain/label-space confound that a
BANKING77-vs-BNIntent30 comparison would have had.

**POS features are dropped** from all tracks. See §4.

---

## 1. NON-NEGOTIABLE rules

### R1 — Per-run feature extractors are fit on the TRAIN split only
`TfidfVectorizer`, the chi-squared selector, the char n-gram hasher, and every
scaler are fit **inside each run, on that run's train split only**, then applied
unchanged to val and test.

Never fit on the full dataset. Never fit on train+val. Never fit before
splitting. If a small k-shot train split produces a tiny vocabulary, that is a
real finding about low-resource settings — handle it with `min_df=1`,
`max_features`, or hashing, and report what you did. **Do not widen the fitting
data.**

### R1b — The frozen encoder cache is NOT a violation of R1
Encoding all utterances (train, val and test) once with the frozen pretrained
encoder is **not** fitting on test data: no parameter is estimated from this
dataset at any point. The encoder's weights come from pretraining and never
change.

The distinction that matters:
- **Frozen encoder → embed everything once. Allowed.** Nothing is learned here.
- **TF-IDF / chi2 / scalers → fit per run on train only. R1 applies.** These
  *are* learned from the data.

Write this distinction as a comment in the caching script, and state it in the
paper's methodology section. A reviewer will check it.

### R2 — Identical backbone and identical head hyperparameters across compared models
The `baseline`, `concat` and `sagff` heads use the **same cached embeddings,
same optimizer, same LR, same schedule, same epochs, same batch size, same
seed, same early-stopping rule**. The only difference is the head's structure.

If a model needs a different hyperparameter to train at all, report it instead of
changing one side only.

### R3 — Splits are fixed, reproducible, and committed as manifests (not as data)
Splits are generated once per (track, k, seed) by `scripts/make_splits.py` and
written to `data/splits/`. Training code **reads** splits; it never samples them.

Because the Bangla data has no license (§3), **the split files themselves are
gitignored.** What gets committed is `data/splits/manifest.json`:
- for each (track, k, seed): the ordered list of row `uid`s per split
- the sha256 of each generated split file
- the resolved upstream commit SHA / download URL the rows came from

Reproduction = run the download script, run `make_splits.py`, verify every
sha256 against the manifest. `scripts/verify_manifest.py` must do this check and
exit non-zero on any mismatch.

### R4 — The seed list, k list, metrics and tests are fixed
- seeds: `13, 42, 87, 100, 2026` (all five — CPU runs are cheap now)
- k: `5, 10, 20, full`
- primary metric: **Macro-F1** on test
- model-vs-model test: **Wilcoxon signed-rank**, paired over (track, k, seed)
- **trend test for the signature hypothesis: Spearman rank correlation between
  `k` and `gate_mean`**, computed per track, with p-value and n

That last one matters: Wilcoxon compares models to each other, it says nothing
about whether gate reliance changes with k. The paper's central claim needs its
own statistic. Do not omit it.

Do not drop a seed because a run is slow. Do not add one because results look
noisy. Do not switch metrics because Macro-F1 looks bad.

### R5 — Full-way evaluation
Test evaluation always uses **all** intents of the track. Never sample N classes
for evaluation. `k` restricts only the number of training examples per class.

### R6 — One JSON log per run, written even on failure
Every run writes exactly one file to `results/runs/<run_id>.json` with the full
config, git commit hash, library versions, encoder name and pooling mode, split
sha256s, per-split sizes, realised feature dimensions, all metrics, wall-clock
time, and the gate statistics (R7). On an exception, write the JSON with
`"status": "failed"` and the traceback. Never overwrite an existing run file.

### R7 — Gate statistics are the paper's main result, not a debug print
For every `sagff` run, record over the **test** split:
- `gate_mean` — mean of `g` over all examples and all dimensions
- `gate_std`
- `gate_mean_per_dim` — length-`d_model` list
- `gate_hist` — 20-bin histogram over [0,1] of all gate values

These go in the run JSON. Not optional.

### R8 — Never report a number that did not come out of `results/runs/`
Tables and figures are generated by a script that reads the JSON logs. No
hand-typed numbers anywhere, ever.

### R9 — The two BNIntent30 tracks must contain IDENTICAL rows
`bnintent30-en` and `bnintent30-bn` are the same rows in two languages. Every
cleaning step (§3) is applied **at row level, to the union of problems found in
either language**, producing one shared cleaned row set with stable `uid`s.
Split generation then runs on `uid`s, so for a given (k, seed) both tracks
contain exactly the same `uid`s in exactly the same order.

`tests/test_parallel.py` must assert this for every (k, seed). If the two tracks
ever diverge by a single row, the paper's central English-vs-Bangla comparison
is no longer controlled.

---

## 2. The model spec (implement exactly)

Input per example:
- `e` — cached frozen-encoder embedding, dimension `d_model` (768 for
  `xlm-roberta-base`)
- `h_feat` — handcrafted feature vector (§4), fit per run on train only

```
h_enc    = LayerNorm(e)                         # (B, d_model)
h_feat_p = LayerNorm(W_p @ h_feat)              # (B, d_model), W_p learned
g        = sigmoid(W_g @ [h_enc ; h_feat_p])    # (B, d_model)  <- VECTOR gate
h        = g * h_enc + (1 - g) * h_feat_p       # elementwise
logits   = Classifier(Dropout(h))
```

Things that are easy to get wrong and must not be:
- **`g` is a VECTOR** of size `d_model`, one gate value per dimension. Not a
  scalar. If your gate tensor is `(B, 1)` you have built the wrong model.
- **Both branches are LayerNorm'd before the gate.** Not a training trick:
  without it the gate absorbs the scale mismatch between dense encoder
  activations and sparse TF-IDF values, and `gate_mean` stops meaning anything.
  The whole paper rests on `gate_mean` being interpretable.
- **Gate convention: `g` high = relying on the encoder, `g` low = relying on
  handcrafted features.** Never flip this. Every figure and sentence depends on
  the direction.

### The three heads (one cached embedding set, three heads)
1. `baseline` — `Classifier(Dropout(LayerNorm(e)))`, no handcrafted features
2. `concat` — `Classifier(Dropout([h_enc ; h_feat_p]))`, the fusion ablation
3. `sagff` — as above

`concat` exists to answer the first question any reviewer asks: *was the gate
actually necessary, or would plain concatenation have done the same?* It is not
optional.

---

## 3. Data: sources, cleaning, licensing

### 3.1 Tracks

| track id | source | language | intents |
|---|---|---|---|
| `bnintent30-en` | BNIntent30 `text` field | English | 30 |
| `bnintent30-bn` | BNIntent30 `bn_text` field | Bangla | 30 |
| `banking77` | BANKING77 | English | 77 |

`bnintent30-en` vs `bnintent30-bn` is the **primary** comparison (parallel,
controlled). `banking77` is a **generality check** — a different domain and a
much larger label space — and runs last.

### 3.2 Download, pinned
- **BNIntent30:** from `JahidBasher/GAN-BnBERT`, path `dataset/BNIntent30/`.
  Resolve `main` to a **specific commit SHA once**, record it in
  `configs/data.yaml` and in the manifest, and download that SHA from then on.
  Never download `main` again.
- **BANKING77:** the two CSVs from PolyAI's GitHub. Download once into
  `data/raw/banking77/`, record the URLs and sha256 in `configs/data.yaml`.
  **Do not depend on the `datasets` loading script or pin `datasets==3.6.0`** —
  read the CSVs with pandas. One less thing to break in six months.
- BANKING77's license should be recorded in the README — verify it rather than
  assuming.

### 3.3 BNIntent30 cleaning — apply in this order, at row level (R9)
1. **Drop rows whose `bn_text` is literally `TRANSLATION_FAILED`** (67 rows).
   After this the totals must be exactly **4,433 = 2,952 train / 595 val / 886
   test**, which matches the published paper. Assert this. If it does not match,
   stop and report — something upstream changed.
2. **Drop all rows involved in a label conflict.** A conflict is one `bn_text`
   string carrying more than one distinct `intent` (4 known cases, e.g. `হ্যাঁ`
   labelled both `yes` and `no`). Drop **every** row with that text, not just
   one. Rationale: such a row is unlearnable in training and unanswerable in
   test — whatever the model predicts is scored wrong.
3. **Resolve cross-split duplicates with priority `test > val > train`.** If the
   same `bn_text` appears in more than one official split, keep the occurrence in
   the *evaluation* split and drop the training (or val) occurrence.

   Rationale, which must be in the paper: Macro-F1 weights all 30 classes
   equally, so removing test rows destabilises the metric, whereas losing ~76
   rows from a 2,952-row training pool costs almost nothing — and the k-shot
   regimes sample only k per class anyway. **Keep the measuring instrument
   intact; spend from the training material.**
4. Assign a stable `uid` to every surviving row and freeze it. `uid`s must be
   deterministic across re-runs (derive from the source file and row index, not
   from iteration order).
5. Write `results/tables/data_cleaning.md`: counts removed at each step, broken
   down per split, plus the per-split and per-class totals before and after.

**This cleaning report is a paper contribution, not bookkeeping.** Finding
genuine train/test overlap in a published Bangla benchmark — caused by machine
translation collapsing distinct English sentences onto one Bangla sentence — is
a result worth its own paragraph. Also produce the "dirty" comparison runs
described in §5.

### 3.4 Licensing — code public, data not
GAN-BnBERT has no license file and the paper states no data license. Therefore:
- the repo may be public, but **no BNIntent30 text, cleaned or raw, is
  committed** — `data/` is gitignored except `data/splits/manifest.json`
- reproducibility comes from the download script + manifest + sha256 checks (R3)
- draft a short email to the authors requesting permission to redistribute
  derived splits, and save it as `docs/data_license_request.md`

---

## 4. Feature families

Three families, **identical across all three tracks**:

1. **TF-IDF** — word-level, chi-squared selected to a fixed `max_features`
2. **Character n-gram hashes** — n = 3,4,5, hashed to a fixed dimension
3. **Utterance statistics** — token count, char count, mean token length, digit
   ratio, punctuation ratio, uppercase ratio

The total feature dimension must be **identical for every k**, so that `W_p` has
a stable shape across runs. Log the realised dimension per family in every run
config.

### POS features are DROPPED — do not add them back
A Bangla POS tagger exists (`bnlp_toolkit`'s CRF `BengaliPOS`) and roughly works,
but using it would require mapping bnlp's BIS-style tags and the English tagger's
Penn Treebank tags onto a shared UPOS set — otherwise the two languages'
feature vectors are not comparable, which breaks R9 in spirit — plus pinning a
model file that bnlp downloads from a GitHub `master` branch at runtime, which
breaks R3.

That is roughly a day of work for the weakest-contributing family. Dropped from
**all** tracks, which preserves the control.

The paper frames this honestly and usefully: *the absence of a reliable Bangla
POS tagger with a tag set comparable to English is itself an instance of the
low-resource problem this paper studies.* Record it in limitations/future work.

---

## 5. Scope — do not expand

| Dimension | Value |
|---|---|
| Encoder | `xlm-roberta-base`, **frozen**, mean-pooled (§6) |
| Tracks | `bnintent30-en`, `bnintent30-bn`, `banking77` |
| k | 5, 10, 20, full |
| seeds | 13, 42, 87, 100, 2026 |
| heads | baseline, concat, sagff |

3 x 4 x 5 x 3 = **180 runs**. On cached embeddings these are seconds each.

**Run order matters** (the grid must be resumable and must follow this order, so
that if time runs out the core claim is already complete):
1. `bnintent30-bn` and `bnintent30-en` — the primary parallel comparison
2. `banking77` — the generality check

**Plus one small extra set:** repeat `bnintent30-bn`, `sagff` + `baseline`, all
k, all seeds, on the **uncleaned** (leaky) official splits, tagged
`dirty=true` in the run JSON. 40 extra runs, seconds each. This produces the
before/after table that turns the leakage finding into a measured result.

**Out of scope — do not build, do not suggest building:** CLINC150, HWU64,
MASSIVE, the Banglish/code-mixed subset, scalar-vs-vector gate ablation,
prompt/LLM baselines, hyperparameter search, POS features.

**Optional, only if everything above is finished and verified:** a
full-fine-tuning supplementary on free cloud GPU — `bnintent30-en` and
`bnintent30-bn`, `baseline` + `sagff`, all k, **one seed (13)** = 16 runs, to
show the trend survives fine-tuning. Ask before starting this.

---

## 6. Encoder caching

`scripts/cache_embeddings.py`:
- loads `xlm-roberta-base` once, `model.eval()`, inside `torch.no_grad()`
- **mean pooling over tokens with the attention mask**, not the `<s>`/CLS token.
  This matters: CLS is only a good sentence representation *after* fine-tuning.
  With a frozen encoder, mean pooling is substantially better. Record
  `pooling: "mean"` in every run JSON.
- fixed `max_length` (state the value; truncation count must be logged)
- writes `data/cache/<track>_<encoder>_<pooling>.npy` plus a sidecar JSON with
  the `uid` order, encoder name, revision, pooling, max_length and sha256
- is idempotent: if the cache and sidecar exist and match, do nothing
- training code loads embeddings **by `uid`**, never by positional index

Expect roughly 20 minutes on CPU for all three tracks, once.

---

## 7. Layout

```
├── CLAUDE.md
├── requirements.txt
├── .gitignore              # data/ except data/splits/manifest.json
├── configs/
│   ├── data.yaml           # pinned SHAs, URLs, checksums
│   └── default.yaml        # head hyperparameters, shared by all three heads
├── data/
│   ├── raw/                # downloaded, never edited      (gitignored)
│   ├── clean/              # after §3.3, with uids          (gitignored)
│   ├── cache/              # frozen embeddings .npy         (gitignored)
│   └── splits/
│       ├── manifest.json   # COMMITTED
│       └── <track>/k<k>_seed<seed>/{train,val,test}.jsonl   (gitignored)
├── src/
│   ├── data.py             # load, clean, uid assignment
│   ├── features.py         # the three families, fit/transform
│   ├── models.py           # baseline / concat / sagff heads
│   ├── train.py            # single run -> one JSON log
│   └── analysis.py         # JSON logs -> tables + figures
├── scripts/
│   ├── download_data.py
│   ├── make_splits.py
│   ├── cache_embeddings.py
│   ├── verify_manifest.py
│   ├── smoke_test.py
│   └── run_all.py
├── tests/
│   ├── test_leakage.py     # R1 enforcement
│   ├── test_parallel.py    # R9 enforcement
│   └── test_cleaning.py    # §3.3 assertions
├── results/
│   ├── runs/               # one JSON per run
│   ├── tables/
│   └── figures/
└── README.md               # reproduction commands
```

---

## 8. The tests

`tests/test_leakage.py` must assert at minimum:
1. A vectorizer fit on train, applied to test, yields no feature index outside
   the train-fit vocabulary size.
2. Train, val and test splits for a given (track, k, seed) are disjoint by
   utterance text — zero overlap. (This is what the §3.3 dedup makes possible.)
3. A k-shot train split has at most `k` examples per class; the test split
   contains **all** classes of the track.
4. Re-running `make_splits.py` with the same (track, k, seed) reproduces
   byte-identical files matching the manifest sha256.
5. The fitted feature pipeline used at test time is the *same object* from
   training — an identity check, not a refit.

`tests/test_parallel.py` must assert R9 for every (k, seed).

`tests/test_cleaning.py` must assert the §3.3 step-1 count (4,433 =
2,952/595/886), that no `TRANSLATION_FAILED` survives, that no text has two
labels, and that no text appears in two splits.

`scripts/run_all.py` runs the whole test suite first and **refuses to start the
grid if anything fails.**

---

## 9. Working style

- **Smoke test before the grid.** One track, k=5, one seed, one head. Confirm a
  valid JSON log. Only then run the grid.
- **Determinism:** seed `random`, `numpy`, `torch`; set
  `torch.use_deterministic_algorithms(True)` where possible; log anything you
  cannot make deterministic.
- **CPU-aware:** set thread counts explicitly and log them. Head training on
  cached embeddings should take seconds — if a run takes minutes, something is
  recomputing embeddings. Stop and find out why.
- **Report blockers early, in plain language.** A surprising number, a count
  that does not match §3.3, a test that cannot be satisfied — surface it. Never
  paper over it.
- Commit after each working stage with a clear message.
- Do not add dependencies a stage does not need. Do not refactor working code
  that is not in the way.
