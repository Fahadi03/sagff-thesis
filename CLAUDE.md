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
All five heads (`baseline`, `concat`, `baseline-cap`, `concat-cap`, `sagff`) and
the two permutation controls (`sagff-noise-perepoch`, `sagff-permuted-fixed`) use
the **same cached embeddings, same optimizer, same LR, same schedule, same
epochs, same batch size, same seed, same early-stopping rule**. The only
difference is the head's structure (and, for the controls, the permuted feature
mapping). **No per-head or per-regime
tuning was performed:** the training hyperparameters (§2, "Training") are one
set, applied unchanged to every head, condition, track and k.

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

R3 also fixes, as requirements rather than implementation details:
- **Nested k-sets.** For a given (track, seed), the k=5 train set is a subset of
  the k=10 set, which is a subset of k=20, which is a subset of full. Validation
  sets nest identically (3.5). This makes training-set SIZE the only quantity
  that varies along the x-axis of the paper's main figure; with independent
  resamples per k, a change in the gate could equally be explained by having
  drawn different examples.
- **Deterministic sampling without an RNG.** Selection ranks sha256(seed, uid),
  not any random-number library, so a library upgrade cannot change the splits.
- **Group-aware val carve.** Where a val split is carved from train (BANKING77),
  rows sharing a normalized key always land on the same side, so the carve
  cannot itself create overlap. BANKING77's val fraction is 1/6 per class,
  matching BNIntent30's 100:20 train:val ratio.
- **`.gitattributes` pins LF line endings,** so committed files and their sha256
  do not differ between Windows and other systems.

### R4 — The seed list, k list, metrics and tests are fixed
- seeds: `13, 42, 87, 100, 2026` (all five — CPU runs are cheap now)
- k: `5, 10, 20, full`
- primary metric: **Macro-F1** on test
- model-vs-model test: **Wilcoxon signed-rank**, paired over (track, k, seed)
- **trend test for the signature hypothesis: Spearman rank correlation between
  `k` and the gate DISPLACEMENT** (`gate_displacement.mean` = `gate_mean` -
  `gate_init_mean`, see R7), computed per track, with p-value and n
- every confirmatory statistic is fixed in §10, "Pre-specified analyses", which
  was committed before any grid run

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

Every run JSON also records:
- the trainable parameter count of its head (all five heads are tabulated in
  `results/tables/head_capacity.md`);
- the number of all-zero TF-IDF rows in its test split. For those items the
  TF-IDF family carries no information at all, which is itself relevant to when
  features help; the paper reports it;
- the degenerate (constant-on-train) utterance statistics, from
  `HandcraftedFeatures.report()["degenerate_stats"]`;
- `dirty` and `permuted` flags and `permutation_mode` (`dirty` true only for the
  leaky condition; `permuted` true for both controls, with mode `per-epoch` or
  `fixed`), and for permuted runs the permutation's per-split size and
  fixed-point count.

### R7 — Gate statistics are the paper's main result, not a debug print
For every `sagff` run, record over the **test** split:
- `gate_mean` — mean of `g` over all examples and all dimensions
- `gate_std`
- `gate_mean_per_dim` — length-`d_model` list
- `gate_hist` — 20-bin histogram over [0,1] of all gate values

The same four statistics are recorded on the same test split BEFORE training,
at initialisation, under `gate_init_mean`, `gate_init_std`,
`gate_init_mean_per_dim`, `gate_init_hist`. The quantity the paper analyses is
the DISPLACEMENT from initialisation, not the final level: the run JSON carries
the derived field `gate_displacement` (`mean` = `gate_mean` - `gate_init_mean`;
`mean_abs_per_dim` = mean over dimensions of |final - init|), and the analysis
uses it. Reason: with 150 training examples the gate can move little from its
starting point, with 2,672 it can move a great deal, so the final level alone
would change with k for reasons of optimisation, not feature reliance.

The same four statistics are also recorded at the FINAL trained epoch (the last
epoch before early stopping ended the run) under `gate_final_mean`,
`gate_final_std`, `gate_final_mean_per_dim`, `gate_final_hist`, with the derived
`gate_displacement_final_epoch` (including `best_minus_final_gate_mean`). The
paper's quantity is the displacement at the selected best epoch; the final-epoch
values show whether it depends on the stopping point
(`results/tables/stopping_diagnostic.md`).

These go in the run JSON. Not optional.

### R8 — Never report a number that did not come out of `results/runs/`
Tables and figures are generated by a script that reads the JSON logs. No
hand-typed numbers anywhere, ever. This applies to prose too: no sentence in
CLAUDE.md or in any draft may state something the generated numbers contradict.

**Never quote a single-seed figure anywhere.** Always the mean and range across
seeds. (Cautionary example: a "2 leaked test rows at k=5" figure was really seed
13's minimum; the mean over seeds is 7.0, range 2-11.)

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

### The five heads (one cached embedding set, five heads)
| head | handcrafted features | gate | definition |
|---|---|---|---|
| `baseline` | no | no | `Classifier(Dropout(LayerNorm(e)))` |
| `concat` | yes | no | `Classifier(Dropout([h_enc ; h_feat_p]))` |
| `baseline-cap` | no | no | `Classifier(Dropout(LayerNorm(W2 @ gelu(W1 @ h_enc))))` |
| `concat-cap` | yes | no | `Classifier(Dropout(LayerNorm(W2 @ gelu(W1 @ [h_enc ; h_feat_p]))))` |
| `sagff` | yes | yes | as above |

`baseline-cap` and `concat-cap` are parameter-matched to `sagff`
(`results/tables/head_capacity.md`; widths frozen in `configs/default.yaml`):
`baseline-cap` within 20%, `concat-cap` within 1%.

**DESIGNATED PRIMARY (CONFIRMATORY) COMPARISONS — both at matched capacity:**
1. **`baseline-cap` vs `concat-cap`** — does handcrafted information help?
2. **`concat-cap` vs `sagff`** — does gating help?

`baseline` and `concat` remain as reference points to prior-work-style setups;
comparisons involving them are descriptive, not confirmatory. The analysis must
lead with the two primary comparisons and the signature hypothesis, so it does
not drown in five heads.

`concat` exists to answer the first question any reviewer asks: *was the gate
actually necessary, or would plain concatenation have done the same?* It is not
optional.

Why the capped heads exist: `sagff`'s head is far larger than `baseline`'s, and
several times larger than `concat`'s, because W_g alone exceeds W_p (exact
ratios: `results/tables/head_capacity.md`). Without matched controls, a weak `sagff` result at low k
could be blamed on a large head trained on few examples, and a `sagff` > `concat`
gap on the gate's extra parameters. The capped heads make both comparisons
tests of information and of gating rather than of capacity.

### Gate initialisation and the permuted control (the signature hypothesis)
- **Initialisation.** W_g's bias is set explicitly to `heads.gate_bias_init` in
  `configs/default.yaml` (0.0, so g is approximately 0.5 at initialisation); the
  W_g weights keep PyTorch's default initialisation under the run seed. The value
  lives in the config, not in the code.
- **Two permutation controls.** Both are identical to `sagff` in every respect
  (same head, same real, train-fitted features), except for the mapping from
  utterance to handcrafted feature vector, which is permuted deterministically
  (`src.train.permutation`, RNG-free). Both preserve every marginal property of
  the feature branch — sparsity, scale, dimension, distribution. Both are tagged
  `permuted=true`; all tracks, all k, all seeds: 60 runs each.
  - **`sagff-noise-perepoch`** (control I-A): the training mapping is re-permuted
    every epoch from (seed, epoch); evaluation splits use one fixed permutation.
    The branch carries no stable content and cannot be memorized. Expected: the
    gate shuts the branch off and g rises. Tests whether the gate can reject a
    worthless branch.
  - **`sagff-permuted-fixed`** (control I-B): one fixed permutation per split. A
    fixed random vector per training example is a perfect training-set
    identifier: memorizable, non-generalizing. Expected: the gate moves toward the
    features and test Macro-F1 falls to chance. Tests whether the gate is fooled by
    exploitability. Keep it as is; its failure is the result.
- **What the gate measures — a limitation established before the grid, stated in
  the paper as known, not hidden.** A sanity run of the fixed-permutation design
  showed the gate moving toward memorizable noise. So `gate_mean` does not measure
  how INFORMATIVE a branch is; it measures how EXPLOITABLE the branch is during
  training — the cheapest route to zero training loss. Under real features the two
  coincide because TF-IDF and character n-grams capture genuine regularities;
  under memorizable noise they come apart. I-A shows whether the gate is blind,
  I-B whether it is infallible, and §10 S3 tests whether the gate's reliance on
  real features is EARNED (accompanied by a test-set gain).

### Training (identical for every head, condition, track and k)
AdamW, weight decay 0.01, batch size 32, dropout `heads.dropout`, gradient
clipping 1.0, constant learning rate, up to 200 epochs, early stopping with
patience 20 on Macro-F1 of the k-matched validation split (§3.5); the best
epoch is restored before testing. The learning rate was chosen ONCE from
a half-decade grid, extended at whichever edge the winner sat until it was
bracketed, on validation using the `baseline` head ONLY
(`scripts/select_lr.py`, `results/tables/lr_selection.md`), and applied
unchanged to all seven conditions; its value and provenance are frozen in
`configs/default.yaml` (`training.lr`, `training.lr_selection`). Selecting on the
baseline removes any suggestion that hyperparameters were tuned in favour of
the proposed model; if it disadvantages anything, it disadvantages `sagff`. No
per-head or per-regime tuning was performed.

**No regularisation tuning was performed.** Weight decay is 0.01 and dropout is
`heads.dropout` for every head, untuned. Tuning them on the baseline head would
tune them in the wrong place: at roughly 25k parameters the baseline barely
overfits. The capacity-matched design absorbs the overfitting confound instead,
because the three large heads (`baseline-cap`, `concat-cap`, `sagff`) are equally
exposed to it.

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

### 3.3 Cleaning — applies to ALL tracks, at row level (R9)

**THE COMPARISON KEY.** Every "same text" decision in this section uses the
deduplication key `src.data.dedup_key`: the text is normalized (Unicode NFC,
lowercased, all punctuation — including ? . ! , ; : ' " ( ) and the Bangla danda
and its variant — replaced by a space, whitespace collapsed), split into tokens,
and the tokens are sorted, so the key is the multiset of normalized tokens.
Rationale, which belongs in the paper: TfidfVectorizer lowercases and
discards punctuation by default, so to the TF-IDF family "How do I
unblock my PIN?" and "how do i unblock my pin" are the same input. A dedup key
weaker than the model's own normalization would leave leakage in precisely the
branch this paper measures. Contamination is only active to the extent that a
leaked row's training counterpart is actually sampled, so the contaminated
fraction of the test set grows with k: in the leaky condition, 7.0 on average
(range 2-11 across the five seeds) of 886 BNIntent30-bn test rows have a
training counterpart at k=5, rising to 89 at k=full
(`results/tables/leakage_dose.md`).

The deduplication key is the COARSEST equivalence relation induced by any model
component: two utterances are duplicates if ANY component cannot distinguish
them. All three handcrafted families are order-insensitive — TF-IDF is a bag of
words, char_wb n-grams never cross word boundaries, and the statistics are
counts — so the key is the sorted multiset of normalized tokens. We use the token
multiset rather than the feature vector itself because the feature vector depends
on a fitted vocabulary, which depends on the split, which depends on
deduplication; the token multiset induces the same equivalence classes while
remaining split-independent.

The frozen encoder is order-sensitive, so this key is stricter than the encoder
branch alone would require: it removes some pairs the encoder would distinguish.
The deduplication is therefore conservative in the safe direction — it can only
make evaluation harder, never easier.

Report
duplicate counts under BOTH exact-string and normalized keys so the difference
is visible in the paper.

Apply in this order, at row level, to the union of problems found in either
language (R9):

1. **Step 1.** Drop rows whose `bn_text` is literally `TRANSLATION_FAILED`.
   After this, BNIntent30 totals must be exactly 4,433 = 2,952 train / 595 val /
   886 test, matching the published paper. Assert it. (Already verified as
   matching.)
2. **Step 2.** Drop all rows involved in a label conflict — one normalized key
   carrying more than one distinct intent. Drop EVERY row bearing a conflicted
   key. Such a row is unlearnable in training and unanswerable in test: any
   prediction is scored wrong.
3. **Step 3.** Resolve cross-split duplicates with priority test > val > train.
   If one normalized key appears in more than one split, keep the occurrence in
   the EVALUATION split and drop the training or val occurrence. Rationale for
   the direction, which must be in the paper: Macro-F1 weights all classes
   equally, so removing test rows destabilises the metric, whereas losing a
   small fraction of the much larger training pool costs almost nothing, and the
   k-shot regimes sample only k per class anyway. Keep the measuring instrument intact;
   spend from the training material.
4. **Step 4.** Resolve within-split duplicates. If one normalized key appears
   more than once inside the same split, keep the lowest uid and drop the rest.
   Rationale: machine translation collapses distinct English sentences onto one
   Bangla sentence ("please set an alarm" and "set an alarm" both become the
   same Bangla string), so Bangla has far more within-split duplicates than
   English (counts in `results/tables/data_cleaning.md`). Left in place, a Bangla k-shot train set would hold fewer DISTINCT
   utterances than its English counterpart at the same k — then k does not mean
   the same thing in the two tracks, so the two lines in the paper's main figure
   no longer share an x-axis, and any English-vs-Bangla difference would be
   confounded by effective training-set size.
5. **Step 5.** Assign a stable uid to every surviving row and freeze it. uids
   must be deterministic and independent of processing order.
6. **Step 6.** Write `results/tables/data_cleaning.md`: counts removed per step
   per split, per-class before/after, both duplicate counts (exact and
   normalized), and which language triggered each removal.

**STEPS 2, 3 AND 4 APPLY TO BANKING77 AS WELL.** No dataset is exempt.

**ASSERT AFTER ALL CLEANING:** every class in every track retains at least 20
training rows (the largest k). Report the three smallest classes per track. If
any class falls below 20, STOP and report — do not silently reduce k.

This cleaning report is a paper contribution, not bookkeeping. Genuine
train/test overlap in a published Bangla benchmark, caused by translation
collapse, deserves its own paragraph and its own before/after table.

### 3.4 Licensing — code public, data not
GAN-BnBERT has no license file and the paper states no data license. Therefore:
- the repo may be public, but **no BNIntent30 text, cleaned or raw, is
  committed** — `data/` is gitignored except `data/splits/manifest.json`
- reproducibility comes from the download script + manifest + sha256 checks (R3)
- draft a short email to the authors requesting permission to redistribute
  derived splits, and save it as `docs/data_license_request.md`

### 3.5 Validation sets are k-matched

The validation set scales with k: for a run at k, the validation split holds at
most k examples per class, drawn from that track's val portion by the same
sha256(seed, uid) ranking used for train, and NESTED the same way (the k=5 val
set is a subset of the k=10 val set, and so on). k="full" uses the full val
portion.

Model selection — early stopping, epoch choice, anything that touches the
weights or picks a checkpoint — uses ONLY the k-matched validation set.

Rationale: a run that trains on 150 rows but early-stops on 577 labelled rows is
not a 5-shot result. In a real 5-shot setting those 577 labels do not exist; if
they did, they would be training data. Reviewers raise this, and correctly.

Additionally, each run logs metrics on the FULL val portion under the key
`val_full_diagnostic`, explicitly marked `"used_for_selection": false`. This
answers "how much would a larger validation set have helped?" without
contaminating the headline claim.

The k-matching constraint binds at k=5 and k=10. BNIntent30 holds at most 20
validation rows per class, so at k=20 and k=full the k-matched validation set
equals the full validation portion. At k=20 the validation set is therefore of
comparable size to the training set, which matches established few-shot practice
of using a development set the same size as the training set (Gao et al., 2021 —
VERIFIED: Gao, Fisch & Chen, "Making Pre-trained Language Models Better Few-shot
Learners", ACL 2021, §3: "we assume a development set ... of the same size as the
few-shot training set". Do not re-check; do not carry this note into the
manuscript).


---

## 4. Feature families

Three families, **identical across all three tracks**:

1. **TF-IDF** — word-level, chi-squared selected to a fixed `max_features`
2. **Character n-gram hashes** — n = 3,4,5, hashed to a fixed dimension
3. **Utterance statistics** — token count, char count, mean token length, digit
   ratio, punctuation ratio, uppercase ratio

The total feature dimension must be **identical for every k**, so that `W_p` has
a stable shape across runs. Log the realised dimension per family in every run
config. The dimensions (TF-IDF columns, hash buckets) were selected ONCE on
k-matched validation data only, with the handcrafted branch alone, by the rule
"smallest configuration within one SD of the best", with the grid extended
downward in any dimension whose chosen value sat on the grid's lower edge, until
the choice was bracketed, and are frozen in `configs/default.yaml`
(`scripts/select_feature_dims.py`, `results/tables/feature_dim_selection.md`,
which also states how flat performance is across the grid).

`uppercase_ratio` stays in every track. As measured
(`results/tables/feature_overview.md`), the column is degenerate (constant on the
train split) in every `bnintent30-en` split because the CLINC150 source text is
lowercased, and in only some `bnintent30-bn` k=5 splits because embedded Latin
script supplies uppercase. It is a dataset preprocessing artifact, not a property
of either language. It is logged per run as `degenerate_stats`. It is kept
because removing it would change F and invalidate the frozen dimension selection.

**All-zero TF-IDF rows are a result, not a caveat**
(`results/tables/tfidf_coverage.md`). The char n-gram vectors are never empty,
so the handcrafted branch is never silent — only its word-level component is.
At k=5, on parallel data at identical k, the TF-IDF family is silent for 9.0% of
Bangla test items versus 4.5% of English (means over five seeds). Together with
the higher Bangla test-token OOV rates, this is the second measured piece of
evidence for retaining the char n-gram family. Report the table.

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
| heads | baseline, concat, baseline-cap, concat-cap, sagff |
| run conditions | the five heads + `sagff-noise-perepoch` + `sagff-permuted-fixed` |

3 tracks x 4 k x 5 seeds x 7 conditions = **420 runs**, plus 40 leaky-condition
runs = **460**. On cached embeddings these are seconds to tens of seconds each.

**Run order matters** (the grid must be resumable and must follow this order, so
that if time runs out the core claim is already complete):
1. `bnintent30-bn` and `bnintent30-en` — the primary parallel comparison
2. `banking77` — the generality check

Within each track the conditions run in the order `sagff`, `baseline-cap`,
`concat-cap`, `sagff-noise-perepoch`, `sagff-permuted-fixed`, `baseline`, `concat`
(`src.train.CONDITIONS`), and the leaky-condition set runs after all clean runs.
Our own claims come first: `baseline-cap` is half of primary comparison S4 and is
never dropped. If time runs out, what gets dropped is the comparison to
prior-work-style setups (`baseline`, `concat`), not our own claims.

**Plus the leaky-condition set.** Repeat `bnintent30-bn`, heads `sagff` and
`baseline`, all k, all seeds, on splits built from the official assignment with
step 1 applied and steps 2-4 NOT applied, tagged `dirty=true` in the run JSON.
40 runs, seconds each.

The dirty condition starts after step 1 on purpose. `TRANSLATION_FAILED` rows
are corrupt placeholder strings, not leakage; leaving them in would make the
dirty-vs-clean comparison measure two things at once (leakage AND garbage text),
and neither effect could then be attributed. Starting after step 1 isolates the
effect of cross-split duplication alone — one variable.

For the paper: "the leaky condition retains the official splits after removal of
untranslated placeholder rows, isolating the effect of cross-split duplication."

The dirty condition keeps its own k-matched val sets per 3.5, built from its own
uncleaned val portion.

**Out of scope — do not build, do not suggest building:** CLINC150, HWU64,
MASSIVE, the Banglish/code-mixed subset, scalar-vs-vector gate ablation,
prompt/LLM baselines, hyperparameter search (beyond the two one-off,
validation-only selections already made and frozen: feature dimensions §4 and
learning rate §2), POS features.

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
- fixed `max_length` (state the value; truncation count must be logged).
  Implemented: `FacebookAI/xlm-roberta-base` pinned to its hub revision in
  `src/embeddings.py`; `max_length` = 128; the cache per track covers every uid
  of the clean and leaky conditions; timings, truncation counts and shapes are in
  each sidecar JSON
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
│   └── default.yaml        # frozen feature dims + head config, shared by all heads
├── data/
│   ├── raw/                # downloaded, never edited      (gitignored)
│   ├── clean/              # after §3.3, with uids          (gitignored)
│   ├── cache/              # frozen embeddings .npy         (gitignored)
│   ├── splits/
│   │   ├── manifest.json   # COMMITTED
│   │   └── <track>/k<k>_seed<seed>/{train,val,test,val_full}.jsonl + meta.json  (gitignored)
│   └── splits_dirty/       # leaky condition (§5), same layout, own manifest.json (COMMITTED)
├── src/
│   ├── data.py             # load, clean, uid assignment
│   ├── features.py         # the three families, fit/transform
│   ├── embeddings.py       # frozen-encoder cache: encode once, load by uid
│   ├── models.py           # baseline / concat / sagff / baseline-cap heads
│   ├── train.py            # single run -> one JSON log
│   └── analysis.py         # JSON logs -> tables + figures
├── scripts/
│   ├── download_data.py
│   ├── make_splits.py
│   ├── cache_embeddings.py
│   ├── verify_manifest.py
│   ├── leakage_dose.py     # dose-response table for the leaky condition
│   ├── select_lr.py        # one-off, validation-only, baseline head only, then frozen
│   ├── select_feature_dims.py  # one-off, validation-only, then frozen
│   ├── head_capacity.py    # baseline-cap width + parameter table
│   ├── feature_report.py
│   ├── stopping_diagnostic.py  # best-epoch vs final-epoch gate (Task M)
│   ├── smoke_test.py
│   └── run_all.py
├── tests/
│   ├── test_leakage.py     # R1 enforcement
│   ├── test_parallel.py    # R9 enforcement
│   ├── test_cleaning.py    # §3.3 assertions
│   └── test_models.py      # §2 head spec (vector gate, capacity match)
├── results/
│   ├── runs/               # one JSON per run
│   ├── selection/          # JSON logs of the one-off selection runs (not results)
│   ├── smoke/              # smoke-test JSON
│   ├── diagnostics/        # diagnostic runs (not results)
│   ├── tables/
│   └── figures/
└── README.md               # reproduction commands
```

---

## 8. The tests

`tests/test_leakage.py` must assert at minimum the following. For each, what it
PROVES and what it does NOT prove — a test must not be cited for more than it
establishes.
1. A pipeline fit on train, applied to val/test, yields no TF-IDF index outside
   the train-fit vocabulary, and the vocabulary does not grow at transform time.
   PROVES test-time indices lie inside the train-fit vocabulary. Does NOT prove
   the vectorizer was fit on the right data (a vectorizer fit on train+test would
   pass too); the train-text fingerprint in assertion 5 establishes that.
2. Train, val and test splits for a given (track, k, seed) share no dedup key.
   PROVES no utterance is duplicated across splits under `dedup_key`. Does NOT
   prove the absence of near-paraphrases that differ in tokens.
3. A k-shot train split, and its k-matched val split, hold at most `k` examples
   per class; the test split contains **all** classes of the track. PROVES the
   sampling sizes and full-way evaluation. Does NOT prove the sample is
   representative.
4. Re-running `make_splits.py` reproduces byte-identical files matching the
   manifest sha256. PROVES determinism on this machine and library set. Does NOT
   prove identical results on another OS/library version (the manifest check on
   that machine does).
5. The fitted feature pipeline used at test time is the *same object* from
   training, fit exactly once, on exactly the train texts (fingerprint), and its
   fitted state is unchanged by transforming test. PROVES no refit and no
   fit on the wrong data inside `featurize_run`. Does NOT prove that future
   training code calls `featurize_run` — train.py must, and its run JSON records
   the fingerprint.

`tests/test_parallel.py` must assert R9 for every (k, seed). PROVES identical uid
sets, order and labels across the two BNIntent30 tracks. Does NOT prove the two
texts of a uid are good translations of each other.

`tests/test_cleaning.py` must assert the §3.3 step-1 count (4,433 =
2,952/595/886), that no `TRANSLATION_FAILED` survives, that no dedup key has two
labels, that no dedup key appears in two splits or twice within a split, and
that every class keeps at least 20 train rows. PROVES the cleaned row sets meet
§3.3. Does NOT prove the official labels are correct.

`tests/test_models.py` asserts the §2 spec: the gate is a `(B, d_model)` vector
in (0, 1); the gate bias comes from the config and the mean gate at
initialisation is ~0.5; feature-free heads ignore `h_feat`; feature heads use
it; `baseline-cap` is within 20% and `concat-cap` within 1% of `sagff`'s
parameter count; logged counts match the modules; the grid order is the
supervisor's ruling; both controls use the `sagff` head with the right
permutation mode; the permutation is deterministic, seed- and split-dependent
and a true permutation; the per-epoch permutation differs between epochs and is
reproducible from (seed, epoch). PROVES the architecture and the control's
construction. Does NOT prove anything about training.

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

---

## 10. Pre-specified analyses

**Fixed before any grid run.** This section was committed before a single grid
run existed; the paper names that commit (`git log --diff-filter=A -S
"## 10. Pre-specified analyses" -- CLAUDE.md`). Choosing statistics after
seeing results is p-hacking; fixing them beforehand is this paper's strongest
integrity claim. Nothing here may be changed after the grid starts. Any analysis
not listed here is reported as exploratory and labelled as such.

**Common rules.**
- Data: the clean condition only (`dirty=false`), test split, the epoch selected
  by early stopping. Displacement = `gate_displacement.mean`; Macro-F1 =
  `metrics.test.macro_f1`.
- Pairing unit: (track, k, seed). Tracks `bnintent30-bn`, `bnintent30-en`,
  `banking77`; k 5, 10, 20, full; seeds 13, 42, 87, 100, 2026.
- Tests: `scipy.stats.spearmanr` and `scipy.stats.wilcoxon` (`zero_method="wilcox"`),
  all TWO-SIDED at alpha = 0.05. The predicted direction is stated below in
  advance; a significant result in the opposite direction is reported as
  contradicting the prediction, never re-framed.
- Multiplicity: the family is the nine tests below (S1 x 3 tracks, S2 x 2, S3,
  S4, S5). Each p-value is reported unadjusted AND Holm-adjusted across the nine.
- Reported for every test: the statistic, p, Holm-adjusted p, n (and, for Wilcoxon,
  n after dropping zero differences), the median paired difference and the
  matched-pairs rank-biserial correlation; for Spearman, rho.
- Failed runs: reported; their pairs are excluded and the reduced n stated. No run
  is repeated with changed settings.
- The final-epoch displacement (`gate_displacement_final_epoch.mean`) is analysed
  with the same S1-S3 as a pre-specified SENSITIVITY analysis, not as a
  replacement.

**S1 — signature hypothesis.** Spearman rank correlation between k (5 < 10 < 20
< full, as ranks) and the `sagff` gate displacement, computed separately per
track; n = 4 k x 5 seeds = 20 per track. Report rho, p, n.
*Predicted direction: rho > 0* — the gate leans further toward the handcrafted
branch (more negative displacement) when training data is scarce, and back toward
the encoder as k grows. The same correlation is reported for both controls,
descriptively.

**S2 — real versus control trends.** Paired Wilcoxon signed-rank on the gate
displacement, `sagff` against each control, paired over (track, k, seed);
n = 60 per comparison.
- S2a, `sagff` vs `sagff-noise-perepoch`. *Predicted: `sagff` displacement LOWER*
  (the gate moves toward real features and away from per-epoch noise, which it
  should shut off, so g rises under noise).
- S2b, `sagff` vs `sagff-permuted-fixed`. *No directional prediction is
  registered.* The fixed control is memorizable, so the gate may move toward it as
  far as, or further than, toward real features (the sanity run suggested so, but
  a single run is not a basis for a prediction). S2b quantifies exploitability; it
  does not test the paper's claim.

**S3 — is the reliance earned?** Spearman correlation, over (track, k, seed),
n = 60, between the `sagff` gate displacement and the test Macro-F1 difference
(`sagff` minus `baseline-cap`, same track, k, seed). *Predicted sign: rho < 0* —
the gate moves toward the handcrafted features (more negative displacement)
precisely where the features deliver a test-set gain. This is what separates
earned reliance from the memorization established by control I-B.

**S4 — does handcrafted information help? (PRIMARY model comparison.)**
Wilcoxon signed-rank on test Macro-F1, `concat-cap` vs `baseline-cap`, paired over
(track, k, seed), n = 60. Capacity matched, no gate on either side.
*Predicted: `concat-cap` HIGHER.*

**S5 — does gating help? (PRIMARY model comparison.)** Wilcoxon signed-rank on
test Macro-F1, `sagff` vs `concat-cap`, paired over (track, k, seed), n = 60.
Capacity matched, features on both sides. *Predicted: `sagff` HIGHER.*

S4 and S5 are the designated primary model comparisons. `baseline` and `concat`
are reference points only: comparisons involving them, the leaky condition, and
per-k breakdowns are descriptive.
