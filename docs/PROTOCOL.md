# SAGFF thesis experiments — research protocol and repository rules (v2)

> **Name of this file.** This file is `docs/PROTOCOL.md`. It was originally named
> `CLAUDE.md` at the repository root: the configuration file read by the AI coding
> assistant (Claude Code) that implemented the pipeline. See
> `docs/ai_use_disclosure.md` and the git history.

Paper: *"When Do Handcrafted Features Help? Scarcity-Aware Feature Fusion for
Low-Resource Intent Classification in English and Bangla"*

**This is version 2 of the protocol. It supersedes version 1 (then named
`CLAUDE.md`) completely.** Version 1 assumed a local GPU and a
BANKING77-vs-BNIntent30 comparison; both have changed. Where the repository and
this file disagree, this file is authoritative.

This repository produces the numbers in a journal manuscript. Every number must
be reproducible and leakage-free. The rules below are not style preferences —
they are what makes the results publishable.

**A rule that blocks a step is reported as a blocker; it is not worked around,
relaxed "temporarily", or bypassed by changing a config so that a run passes.** A
failed run that is reported is useful. A passing run that broke a rule is worse
than useless, because it looks fine.

---

## 0. What changed in version 2, and why

**No local GPU.** Training runs on CPU. Full fine-tuning of `xlm-roberta-base`
is therefore off the table — it would take weeks.

**So: the encoder is FROZEN.** Sentence representations are computed once per
track with a frozen pretrained encoder, cached to disk, and reused by every run.
Each run trains only the fusion head (projection + gate + classifier). This is
feature-based transfer in the sense of Peters, Ruder & Smith (2019), and it is a
deliberate methodological choice, not a shortcut: few-sample full fine-tuning is
unstable across seeds (Dodge et al. 2020), and that instability would swamp the
gate trend this paper measures. The paper states it in these terms.

**The language comparison is now parallel.** BNIntent30 contains BOTH the
English source (`text`) and the Bangla translation (`bn_text`) for every row, with
the same intent label. So the English and Bangla tracks are the *same rows in two
languages* — same intents, same content, same splits, same encoder. Only the
language differs. This removes the domain/label-space confound that a
BANKING77-vs-BNIntent30 comparison would have had.

**POS features are dropped** from all tracks. See §4.

---

## 1. Non-negotiable rules

### R1 — Per-run feature extractors are fit on the TRAIN split only
`TfidfVectorizer`, the chi-squared selector, the char n-gram hasher, and every
scaler are fit **inside each run, on that run's train split only**, then applied
unchanged to val and test.

They are never fit on the full dataset, on train+val, or before splitting. If a
small k-shot train split produces a tiny vocabulary, that is a real finding about
low-resource settings; it is handled with `min_df=1`, `max_features`, or hashing,
and the handling is reported. **The fitting data is never widened.**

### R1b — The frozen encoder cache is NOT a violation of R1
Encoding all utterances (train, val and test) once with the frozen pretrained
encoder is **not** fitting on test data: no parameter is estimated from this
dataset at any point. The encoder's weights come from pretraining and never
change.

The distinction that matters:
- **Frozen encoder → embed everything once. Allowed.** Nothing is learned here.
- **TF-IDF / chi2 / scalers → fit per run on train only. R1 applies.** These
  *are* learned from the data.

This distinction is written as a comment in the caching code
(`src/embeddings.py`) and is stated in the paper's methodology section.

### R2 — Identical backbone and identical head hyperparameters across compared models
All five heads (`baseline`, `concat`, `baseline-cap`, `concat-cap`, `sagff`) and
the two permutation controls (`sagff-noise-perepoch`, `sagff-permuted-fixed`) use
the **same cached embeddings, same optimizer, same LR, same schedule, same
epochs, same batch size, same seed, same early-stopping rule**. The only
difference is the head's structure (and, for the controls, the permuted feature
mapping). **No per-head or per-regime
tuning was performed:** the training hyperparameters (§2, "Training") are one
set, applied unchanged to every head, condition, track and k.

If a model needs a different hyperparameter to train at all, that is reported
instead of changing one side only.

### R3 — Splits are fixed, reproducible, and committed as manifests (not as data)
Splits are generated once per (track, k, seed) by `scripts/make_splits.py` and
written to `data/splits/`. Training code **reads** splits; it never samples them.

Because the Bangla data has no license (§3), **the split files themselves are
gitignored.** What is committed is `data/splits/manifest.json`:
- for each (track, k, seed): the ordered list of row `uid`s per split
- the sha256 of each generated split file
- the resolved upstream commit SHA / download URL the rows came from

Reproduction = run the download script, run `make_splits.py`, verify every
sha256 against the manifest. `scripts/verify_manifest.py` performs this check and
exits non-zero on any mismatch.

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
own statistic, and it is reported.

No seed is dropped because a run is slow, none is added because results look
noisy, and the metric is not switched because Macro-F1 looks bad.

### R5 — Full-way evaluation
Test evaluation always uses **all** intents of the track. N classes are never
sampled for evaluation. `k` restricts only the number of training examples per
class.

### R6 — One JSON log per run, written even on failure
Every run writes exactly one file to `results/runs/<run_id>.json` with the full
config, git commit hash, library versions, encoder name and pooling mode, split
sha256s, per-split sizes, realised feature dimensions, all metrics, wall-clock
time, and the gate statistics (R7). On an exception, the JSON is written with
`"status": "failed"` and the traceback. An existing run file is never overwritten.

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
For every `sagff` run, the following are recorded over the **test** split:
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

Two further gate records, added because of the observed stopping behaviour:
- **Trajectory:** `gate_mean` and `gate_std` (scalars only) at EVERY epoch, inside
  `training_log.curve`, on the same test split.
- **Convergence-anchored readout:** the four statistics at the first epoch whose
  mean training loss falls below `training.converged_loss_threshold` (0.01), under
  `gate_converged_*`, with the derived `gate_displacement_converged` (null if the
  loss never fell below the threshold). It is deterministic, free of validation
  noise and comparable across regimes.

These are part of every run JSON, not optional.

### R8 — No number is reported that did not come out of `results/runs/`
Tables and figures are generated by a script that reads the JSON logs. No
hand-typed numbers anywhere. This applies to prose too: no sentence in this
protocol or in any draft may state something the generated numbers contradict.

**No single-seed figure is quoted anywhere**; always the mean and range across
seeds. (Cautionary example: a "2 leaked test rows at k=5" figure was really seed
13's minimum; the mean over seeds is 7.0, range 2-11.)

### R9 — The two BNIntent30 tracks contain IDENTICAL rows
`bnintent30-en` and `bnintent30-bn` are the same rows in two languages. Every
cleaning step (§3) is applied **at row level, to the union of problems found in
either language**, producing one shared cleaned row set with stable `uid`s.
Split generation then runs on `uid`s, so for a given (k, seed) both tracks
contain exactly the same `uid`s in exactly the same order.

`tests/test_parallel.py` asserts this for every (k, seed). If the two tracks
ever diverged by a single row, the paper's central English-vs-Bangla comparison
would no longer be controlled.

---

## 2. The model spec

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

Three properties are essential:
- **`g` is a VECTOR** of size `d_model`, one gate value per dimension, not a
  scalar. A gate tensor of shape `(B, 1)` would be a different model.
- **Both branches are LayerNorm'd before the gate.** Not a training trick:
  without it the gate absorbs the scale mismatch between dense encoder
  activations and sparse TF-IDF values, and `gate_mean` stops meaning anything.
  The whole paper rests on `gate_mean` being interpretable.
- **Gate convention: `g` high = relying on the encoder, `g` low = relying on
  handcrafted features.** This direction is fixed; every figure and sentence
  depends on it.

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
comparisons involving them are descriptive, not confirmatory. The analysis
leads with the two primary comparisons and the signature hypothesis, so it does
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
  W_g weights keep PyTorch's default initialisation under the run seed. Training
  code reads the value from the config (`scripts/head_capacity.py` wrote it there
  when the head configuration was frozen).
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
    exploitability. It is kept as designed; its failure is the result.
- **What the gate measures — a limitation established before the grid, stated in
  the paper as known, not hidden.** A sanity run of the fixed-permutation design,
  made before the analysis plan was registered and not logged in this repository,
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

**Measured fact about this regime, which the paper reports**
(`results/tables/stopping_diagnostic.md`, `sagff`, five seeds): mean training
loss falls below 0.01 by epoch 4.2 (range 4-5) for `bnintent30-bn` and 4.0 (4-4)
for `bnintent30-en` at k=5, and by epoch 7.2 (6-8) for `banking77` at k=5. Early
stopping at k=5 therefore chooses among functionally converged models on a
150-row validation set.

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

Dataset citations (verified, `docs/citation_checklist.md`):
- **BNIntent30:** Hasan, Basher & Shawon, "Bengali Intent Classification with
  Generative Adversarial BERT", 2023 26th International Conference on Computer and
  Information Technology (ICCIT), IEEE, pp. 1-6, doi:10.1109/ICCIT60459.2023.10440989
  (primary; the peer-reviewed version). Secondary pointer: arXiv:2312.10679. That
  paper spells its source dataset "CLINIC150"; the correct name is **CLINC150**
  (Larson et al., EMNLP-IJCNLP 2019, doi:10.18653/v1/D19-1131), and the
  manuscript uses the correct name.
- **BANKING77:** Casanueva et al., "Efficient Intent Detection with Dual Sentence
  Encoders", NLP4ConvAI 2020, doi:10.18653/v1/2020.nlp4convai-1.5.

`bnintent30-en` vs `bnintent30-bn` is the **primary** comparison (parallel,
controlled). `banking77` is a **generality check** — a different domain and a
much larger label space — and runs last.

### 3.2 Download, pinned
- **BNIntent30:** from `JahidBasher/GAN-BnBERT`, path `dataset/BNIntent30/`.
  `main` was resolved to a **specific commit SHA once**, recorded in
  `configs/data.yaml` and in the manifest; every download uses that SHA, never
  `main`.
- **BANKING77:** the two CSVs from PolyAI's GitHub, downloaded into
  `data/raw/banking77/`, with URLs and sha256 recorded in `configs/data.yaml`.
  The pipeline **does not depend on the `datasets` loading script** (or a pinned
  `datasets==3.6.0`); it reads the CSVs with pandas, which leaves one less thing to
  break.
- BANKING77's license is recorded in the README after verification, not assumed.

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

Duplicate counts are reported under BOTH exact-string and normalized keys so the
difference is visible in the paper.

The steps are applied in this order, at row level, to the union of problems found
in either language (R9):

1. **Step 1.** Drop rows whose `bn_text` is literally `TRANSLATION_FAILED`.
   After this, BNIntent30 totals must be exactly 4,433 = 2,952 train / 595 val /
   886 test, matching the published paper. The code asserts it (it matches).
2. **Step 2.** Drop all rows involved in a label conflict — one normalized key
   carrying more than one distinct intent. EVERY row bearing a conflicted key is
   dropped. Such a row is unlearnable in training and unanswerable in test: any
   prediction is scored wrong.
3. **Step 3.** Resolve cross-split duplicates with priority test > val > train.
   If one normalized key appears in more than one split, the occurrence in the
   EVALUATION split is kept and the training or val occurrence dropped. Rationale
   for the direction, which belongs in the paper: Macro-F1 weights all classes
   equally, so removing test rows destabilises the metric, whereas losing a
   small fraction of the much larger training pool costs almost nothing, and the
   k-shot regimes sample only k per class anyway. The measuring instrument is kept
   intact; the training material is spent instead.
4. **Step 4.** Resolve within-split duplicates. If one normalized key appears
   more than once inside the same split, the lowest uid is kept and the rest
   dropped. Rationale: machine translation collapses distinct English sentences
   onto one Bangla sentence (for example, a request and a shorter paraphrase of
   the same request can receive an identical translation), so Bangla has far more
   within-split duplicates than English (counts in
   `results/tables/data_cleaning.md`). Left in place, a Bangla k-shot train set
   would hold fewer DISTINCT utterances than its English counterpart at the same
   k — then k does not mean the same thing in the two tracks, so the two lines in
   the paper's main figure no longer share an x-axis, and any English-vs-Bangla
   difference would be confounded by effective training-set size.
5. **Step 5.** Assign a stable uid to every surviving row and freeze it. uids
   are deterministic and independent of processing order.
6. **Step 6.** Write `results/tables/data_cleaning.md`: counts removed per step
   per split, per-class before/after, both duplicate counts (exact and
   normalized), and which language triggered each removal.

**STEPS 2, 3 AND 4 APPLY TO BANKING77 AS WELL.** No dataset is exempt.

**ASSERTED AFTER ALL CLEANING:** every class in every track retains at least 20
training rows (the largest k). The three smallest classes per track are reported.
If any class fell below 20, the pipeline stops with an error; k is never silently
reduced.

This cleaning report is a paper contribution, not bookkeeping. Genuine
train/test overlap in a published Bangla benchmark, caused by translation
collapse, deserves its own paragraph and its own before/after table.

### 3.4 Licensing — code public, data not
GAN-BnBERT has no license file and the paper states no data license. Therefore:
- the repo may be public, but **no BNIntent30 text, cleaned or raw, is
  committed** — `data/` is gitignored except the two split manifests
  (`data/splits/manifest.json`, `data/splits_dirty/manifest.json`)
- reproducibility comes from the download script + manifest + sha256 checks (R3)
- a short email to the authors requesting permission to redistribute derived
  splits is drafted in `docs/data_license_request.md`

### 3.5 Validation sets are k-matched

The validation set scales with k: for a run at k, the validation split holds at
most k examples per class, drawn from that track's val portion by the same
sha256(seed, uid) ranking used for train, and NESTED the same way (the k=5 val
set is a subset of the k=10 val set, and so on). k="full" uses the full val
portion.

Model selection — early stopping, epoch choice, anything that touches the
weights or picks a checkpoint — uses ONLY the k-matched validation set.

Rationale: a run that trains on 150 rows but early-stops on the 568 labelled
rows of the full validation portion (`results/tables/data_cleaning.md`, final
BNIntent30 counts) is not a 5-shot result. In a real 5-shot setting those 568
labels do not exist; if they did, they would be training data. Reviewers raise
this, and correctly.

Additionally, each run logs metrics on the FULL val portion under the key
`val_full_diagnostic`, explicitly marked `"used_for_selection": false`. This
answers "how much would a larger validation set have helped?" without
contaminating the headline claim.

The k-matching constraint binds at k=5 and k=10. BNIntent30 holds at most 20
validation rows per class, so at k=20 and k=full the k-matched validation set
equals the full validation portion. At k=20 the validation set is therefore of
comparable size to the training set, which matches established few-shot practice
of using a development set the same size as the training set (Gao, Fisch & Chen,
"Making Pre-trained Language Models Better Few-shot Learners", ACL 2021, §3;
verified, `docs/citation_checklist.md` entry 3).


---

## 4. Feature families

Three families, **identical across all three tracks**:

1. **TF-IDF** — word-level, chi-squared selected to a fixed `max_features`
2. **Character n-gram hashes** — n = 3,4,5, hashed to a fixed dimension
3. **Utterance statistics** — token count, char count, mean token length, digit
   ratio, punctuation ratio, uppercase ratio

The total feature dimension is **identical for every k**, so that `W_p` has
a stable shape across runs. The realised dimension per family is logged in every
run config. The dimensions (TF-IDF columns, hash buckets) were selected ONCE on
k-matched validation data only, with the handcrafted branch alone, by the rule
"smallest configuration within one SD of the best", with the grid extended
downward in any dimension whose chosen value sat on the grid's lower edge, until
the choice was bracketed, and are frozen in `configs/default.yaml`
(`scripts/select_feature_dims.py`, `results/tables/feature_dim_selection.md`,
which also states how flat performance is across the grid).

**One F for all tracks — the BANKING77 check of the frozen dimensions.** The
dimensions were selected on the two 30-class tracks. The same grid, method and
one-SD rule run on `banking77` alone (`scripts/check_feature_dims_banking77.py`,
`results/tables/feature_dim_check_banking77.md`, validation only) place the frozen
configuration just OUTSIDE BANKING77's own one-SD plateau: narrowly (about a fifth
of BANKING77's seed SD below its threshold), not clearly. Under the rule "when
uncertain, take the option that weakens our claim" (§9), one F is kept for every
track: a per-track F would add a selection step in favour of the handcrafted
branch, whereas the frozen F can only disadvantage that branch on BANKING77.
Consequence, which the paper states: on BANKING77 the handcrafted branch runs
slightly below its own validation optimum, so a weak handcrafted result there may
partly reflect this and must not by itself be read as "handcrafted features do not
transfer".

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
evidence for retaining the char n-gram family. The table is reported.

### POS features are DROPPED from all tracks
A Bangla POS tagger exists (`bnlp_toolkit`'s CRF `BengaliPOS`) and roughly works,
but using it would require mapping bnlp's BIS-style tags and the English tagger's
Penn Treebank tags onto a shared UPOS set — otherwise the two languages'
feature vectors are not comparable, which breaks R9 in spirit — plus pinning a
model file that bnlp downloads from a GitHub `master` branch at runtime, which
breaks R3.

That is roughly a day of work for the weakest-contributing family. It is dropped
from **all** tracks, which preserves the control, and it is not reintroduced.

The paper frames this honestly and usefully: *the absence of a reliable Bangla
POS tagger with a tag set comparable to English is itself an instance of the
low-resource problem this paper studies.* It is recorded under limitations and
future work.

---

## 5. Scope

| Dimension | Value |
|---|---|
| Encoder | `xlm-roberta-base`, **frozen**, mean-pooled (§6) |
| Tracks | `bnintent30-en`, `bnintent30-bn`, `banking77` |
| k | 5, 10, 20, full |
| seeds | 13, 42, 87, 100, 2026 |
| heads | baseline, concat, baseline-cap, concat-cap, sagff |
| run conditions | the five heads + `sagff-noise-perepoch` + `sagff-permuted-fixed` |

3 tracks x 4 k x 5 seeds x 7 conditions = **420 runs**, plus 40 leaky-condition
runs = **460**. On cached embeddings the measured time per optimizer step is
1.6-7.2 ms for the small heads (`baseline`, `concat`) and 12.9-26.1 ms for the
large ones (`baseline-cap`, `concat-cap`, `sagff` and the two controls), as
per-cell means over seeds (`results/tables/run_timing.md`). Wall time scales with
training-set size times epochs: a few seconds at k=5, about a minute for
BNIntent30 at k=full, and several minutes for BANKING77 at k=full with the large
heads (`results/tables/run_timing.md`).

**Run order matters.** The grid is resumable and follows this order, so that if
time runs out the core claim is already complete:
1. `bnintent30-bn` and `bnintent30-en` — the primary parallel comparison
2. `banking77` — the generality check

Within each track the conditions run in the order `sagff`, `baseline-cap`,
`concat-cap`, `sagff-noise-perepoch`, `sagff-permuted-fixed`, `baseline`, `concat`
(`src.train.CONDITIONS`), and the leaky-condition set runs after all clean runs.
The paper's own claims come first: `baseline-cap` is half of primary comparison S4
and is never dropped. If time had run out, what would have been dropped is the
comparison to prior-work-style setups (`baseline`, `concat`), not the paper's own
claims.

**Plus the leaky-condition set.** `bnintent30-bn`, heads `sagff` and
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

**Out of scope:** CLINC150, HWU64, MASSIVE, the Banglish/code-mixed subset,
scalar-vs-vector gate ablation, prompt/LLM baselines, hyperparameter search
(beyond the two one-off, validation-only selections already made and frozen:
feature dimensions §4 and learning rate §2), POS features.

**Optional supplementary, not run:** a full-fine-tuning supplementary on free
cloud GPU — `bnintent30-en` and `bnintent30-bn`, `baseline` + `sagff`, all k,
**one seed (13)** = 16 runs, to show the trend survives fine-tuning. It would be
started only after everything above is finished and verified, and only with the
supervisor's agreement.

---

## 6. Encoder caching

`scripts/cache_embeddings.py`:
- loads `xlm-roberta-base` once, `model.eval()`, inside `torch.no_grad()`
- **mean pooling over tokens with the attention mask**, not the `<s>`/CLS token.
  This matters: CLS is only a good sentence representation *after* fine-tuning.
  With a frozen encoder, mean pooling is substantially better. Every run JSON
  records `pooling: "mean"`.
- fixed `max_length`, with the truncation count logged. Implemented:
  `FacebookAI/xlm-roberta-base` pinned to its hub revision in `src/embeddings.py`;
  `max_length` = 128; the cache per track covers every uid of the clean and leaky
  conditions; timings, truncation counts and shapes are in each sidecar JSON
- writes `data/cache/<track>_<encoder>_<pooling>.npy` plus a sidecar JSON with
  the `uid` order, encoder name, revision, pooling, max_length and sha256
- is idempotent: if the cache and sidecar exist and match, it does nothing
- training code loads embeddings **by `uid`**, never by positional index

Measured time on the original CPU: 775 s (about 13 minutes) for all three tracks,
once (the `wall_clock_s` fields of the three sidecars in `data/cache/`, written by
`scripts/cache_embeddings.py`; an independent re-run from a fresh clone took 845 s).

---

## 7. Layout

```
├── README.md               # reproduction commands
├── requirements.txt        # direct dependencies
├── requirements.lock       # exact environment of the runs
├── .gitignore              # data/ except the two split manifests; docs/drafts/
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
│   ├── models.py           # baseline / concat / baseline-cap / concat-cap / sagff heads
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
│   ├── check_feature_dims_banking77.py  # BANKING77 check of the frozen dimensions
│   ├── head_capacity.py    # baseline-cap width + parameter table
│   ├── feature_report.py
│   ├── stopping_diagnostic.py  # best-epoch vs final-epoch gate
│   ├── compare_lr_reproduction.py  # LR selection re-run vs earlier run, exact equality
│   ├── compare_stopping_to_grid.py # stopping-diagnostic runs vs grid runs, exact equality
│   ├── record_anomalies.py # results/runs_anomalies.md (recorded, never corrected)
│   ├── rerun_anomaly.py    # bit-for-bit re-run of an anomalous run's spec
│   ├── smoke_test.py
│   └── run_all.py
├── tests/
│   ├── test_leakage.py     # R1 enforcement
│   ├── test_parallel.py    # R9 enforcement
│   ├── test_cleaning.py    # §3.3 assertions
│   ├── test_models.py      # §2 head spec (vector gate, capacity match)
│   └── test_analysis.py    # §10 statistics against independent reference values
├── results/
│   ├── runs/               # one JSON per run
│   ├── selection/          # JSON logs of the one-off selection runs (not results)
│   ├── smoke/              # smoke-test JSON
│   ├── diagnostics/        # diagnostic runs (not results)
│   ├── tables/
│   └── figures/
└── docs/
    ├── PROTOCOL.md         # this file (originally CLAUDE.md at the repository root)
    ├── code_map.md         # what each source file does
    ├── preregistration_disclosure.md  # manuscript-ready registration and disclosure text
    ├── review_report.md    # pre-handover review
    ├── ai_use_disclosure.md, citation_checklist.md, data_license_request.md, manuscript_notes.md
    └── drafts/             # manuscript drafts (gitignored)
```

---

## 8. The tests

`tests/test_leakage.py` asserts at minimum the following. For each, what it
PROVES and what it does NOT prove — a test is not cited for more than it
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
   training code calls `featurize_run` — train.py does, and its run JSON records
   the fingerprint.

`tests/test_parallel.py` asserts R9 for every (k, seed). PROVES identical uid
sets, order and labels across the two BNIntent30 tracks. Does NOT prove the two
texts of a uid are good translations of each other.

`tests/test_cleaning.py` asserts the §3.3 step-1 count (4,433 =
2,952/595/886), that no `TRANSLATION_FAILED` survives, that no dedup key has two
labels, that no dedup key appears in two splits or twice within a split, and
that every class keeps at least 20 train rows. PROVES the cleaned row sets meet
§3.3. Does NOT prove the official labels are correct.

`tests/test_models.py` asserts the §2 spec: the gate is a `(B, d_model)` vector
in (0, 1); the gate bias comes from the config and the mean gate at
initialisation is ~0.5; feature-free heads ignore `h_feat`; feature heads use
it; `baseline-cap` is within 20% and `concat-cap` within 1% of `sagff`'s
parameter count; logged counts match the modules; the grid order is the run
order fixed in §5; both controls use the `sagff` head with the right
permutation mode; the permutation is deterministic, seed- and split-dependent
and a true permutation; the per-epoch permutation differs between epochs and is
reproducible from (seed, epoch). PROVES the architecture and the control's
construction. Does NOT prove anything about training.

`tests/test_analysis.py` checks the statistics behind §10 (Spearman, Wilcoxon,
rank-biserial correlation, Holm adjustment, direction and status rules, and the
gate statistics of a run) against values computed independently of
`src/analysis.py`. PROVES the reported statistics are computed as specified.
Does NOT prove the analysis was the right one to register.

`scripts/run_all.py` runs the whole test suite first and **refuses to start the
grid if anything fails.**

---

## 9. Methodological conventions

- **When a methodological choice is genuinely uncertain, the option that WEAKENS
  the paper's claim is taken.** This rule guided every methodological choice after
  it was adopted.
- **Design and confound reasoning may come from discussion with the supervisor;
  every NUMBER comes from the code.** Three times in this project a number asserted
  from intuition ("runs take seconds", "loss reaches zero by epoch 3", "~76 rows")
  was wrong and was corrected by measurement. No number is written that has not
  been measured.
- **Smoke test before the grid.** One track, k=5, one seed, one head, with a valid
  JSON log confirmed before the grid was run.
- **Determinism:** `random`, `numpy` and `torch` are seeded;
  `torch.use_deterministic_algorithms(True)` is set where possible; anything that
  cannot be made deterministic is logged.
- **CPU-aware:** thread counts are set explicitly and logged. Head training on
  cached embeddings costs 1.6-26.1 ms per optimizer step depending on the head
  (`results/tables/run_timing.md`); a run whose time per step is far above that is
  recomputing something (e.g. embeddings) and is investigated. Long runs are
  expected only where the step count is large (BANKING77 at k=full).
- **`wall_clock_s` is wall time, not a compute measurement.** It can be inflated
  by machine suspension (sleep), as happened to one grid run. Anomalies are
  RECORDED in `results/runs_anomalies.md` (generated by
  `scripts/record_anomalies.py`), never corrected in the run JSONs; analysis
  code excludes a run only from the table kinds named there.
- **Blockers are reported early and in plain language:** a surprising number, a
  count that does not match §3.3, a test that cannot be satisfied.
- Each working stage is committed with a clear message.
- No dependency is added that a stage does not need, and working code that is not
  in the way is not refactored.

---

## 10. Pre-specified analyses

**Fixed before any grid run.** This section was committed before a single grid
run existed; the paper names that commit (`git log --format=%h -S
"## 10. Pre-specified analyses" --follow -- docs/PROTOCOL.md` prints it: 44bd224). Choosing statistics after
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
- **Amendment (three readouts).** S1, S2 and S3 are reported on three gate
  readouts: the best-epoch displacement (`gate_displacement.mean`, PRIMARY), the
  final-epoch displacement (`gate_displacement_final_epoch.mean`, sensitivity) and
  the convergence-anchored displacement (`gate_displacement_converged.mean`, the
  first epoch with mean training loss below 0.01; sensitivity). If all three agree
  in direction, the paper states that in one sentence and the stopping question is
  closed. Runs whose loss never falls below the threshold have no
  convergence-anchored value; they are excluded from that readout only, with the
  reduced n stated.
- **Why the primary readout is the conservative one.** In the stopping diagnostic
  (`results/tables/stopping_diagnostic.md`, five seeds), the best-epoch gate at k=5
  sits ABOVE the final-epoch gate on every track (best minus final is positive in
  all fifteen k=5 runs), while at k=full the two agree within noise. Under S1's
  predicted direction (a more negative displacement at small k), the best-epoch
  readout therefore yields a SMALLER k-trend than the final-epoch readout would.
  Whatever the stopping point does, it suppresses the effect we predict rather than
  inflating it. The paper says so explicitly.

**Disclosure: the stopping diagnostic.** The analysis plan above was fixed at
commit 44bd224, before any grid run. A 30-run stopping diagnostic (`sagff`, all
three tracks, k = 5 and k = full, all five seeds; `results/diagnostics/stopping/`)
was examined afterwards. These runs are deterministic, so they are identical to the
corresponding grid runs: they are a subset of S1's data, seen after the plan was
fixed. They were not deleted or regenerated. Deleting them would only make the
record false, because regenerating them yields identical numbers. In response to
the stopping behaviour they revealed (a MECHANISM, not the direction of any
result), two sensitivity readouts were added: the per-epoch gate trajectory and
the convergence-anchored readout, in the commit that introduced this amendment
(`git log --format=%h -S "Amendment (three readouts)" --follow -- docs/PROTOCOL.md`). Rule for
any later change: it must be justified by a mechanism, never by the direction of a
result. The direction of the primary k-trend was not examined before the full grid
completed.

**Commitment.** The k-trend in the 30 diagnostic runs is not reported or
interpreted until the full grid is complete; the gate values in those runs are
used only for the stopping-point question they were run for. Manuscript-ready
wording: `docs/preregistration_disclosure.md`.

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

---

## 11. Decisions made after registration

§10 is the registration and is left exactly as committed. This section, placed
directly after it, lists every analysis or design decision taken after the plan was
registered at `44bd224` (2026-10-11 23:53, UTC+6), with its commit and date and
whether any test-set result was visible when it was made. The same text is in
`docs/preregistration_disclosure.md`, worded so that the manuscript can quote it.
No erratum to §10 is needed: none of the sentences corrected in this protocol after
review lies inside §10.

Timeline of test-set results: before registration, one smoke-test run existed
(`bnintent30-bn`, k = 5, seed 13, `sagff`; commit `9b2036f`, 2026-10-11 13:19) and
one unlogged sanity run of the fixed-permutation control (§2). The 30-run stopping
diagnostic (commit `9e44c19`, 2026-10-12 00:53) was the first set of test-set
results after registration. The grid ran from 2026-10-12 01:24 to 13:21 at commit
`8e0a39e`. The registered tests were first computed at commit `e73d392`
(2026-10-13 02:49).

1. **Feature-dimension re-selection** (commit `7e7f8f6`, 2026-10-11 23:58; run
   from the registration commit's tree). After registration, before the grid and
   before the stopping diagnostic, on validation data only (test files never
   opened). The selection grid was extended downward until the choice was
   bracketed, and the frozen dimensions became tfidf 256 / chargram 256
   (`results/tables/feature_dim_selection.md`). No test-set result other than the
   two pre-registration runs above was visible.
2. **Learning-rate re-selection** (commit `fdb9e52`, 2026-10-12 00:20; run from
   `7e7f8f6`). After registration, before the grid and before the stopping
   diagnostic, on validation data only, with the `baseline` head only (test data
   never loaded). The grid was extended to 1e-2 and the chosen value stayed 3e-3
   (`results/tables/lr_selection.md`). No test-set result other than the two
   pre-registration runs above was visible.
3. **Holm family size** (commit `e73d392`, 2026-10-13 02:49). Decided after the grid
   had completed, in the commit that first computed the registered tests, so it is
   treated as decided after results were visible. §10 states a family of "nine
   tests" while enumerating eight. Every reported Holm-adjusted p-value and status
   uses m = 9, the stated and more conservative number; Holm adjustment with m = 8
   is reported separately as exploratory. No test's status differs between the two
   in any of the 24 test-by-readout combinations
   (`results/tables/T2_preregistered_tests.md`).
4. **Wilcoxon direction rule, including "direction ambiguous"** (observed direction
   defined as the sign of the matched-pairs rank-biserial correlation in `e73d392`,
   2026-10-13 02:49; the "direction ambiguous" rule added in `c56a7de`, 2026-10-13
   03:07). Decided after results were seen: §10 defines neither. When the median
   paired difference and the rank-biserial correlation disagree in sign, the
   observed direction is reported as ambiguous, with both signs, and is not
   resolved in either favour. The rule applies to one test, S4. S4 is
   non-significant under every reading: unadjusted p = 0.6063 and Holm-adjusted
   p = 1.0000 (m = 9) on all three readouts, so no test's status depends on the
   rule.

---

## 12. Spanish and Hindi extension (post-registration, registered before any run)

**Status and timing.** This extension was requested by the supervisor after the
main grid, the registered analyses (§10) and the manuscript results were complete.
It is therefore post-registration with respect to §10, and is registered here, in
its own commit (`git log --format=%h -S "## 12. Spanish and Hindi extension"
-- docs/PROTOCOL.md` prints it), on 2026-11-09, BEFORE any translation, embedding
or run of either new track. At that commit no Spanish or Hindi text, embedding
cache, split or run exists in the repository or under `data/`. §10, §11, every
existing track, split, run and result, and every existing table and figure are
unchanged by this extension. Any change to this section after the first
translation is made is recorded as a dated amendment with its justification,
under the same rule as §10: justified by a mechanism, never by the direction of a
result.

**Purpose.** The Bangla gain of `sagff` over `baseline` at small k could reflect
machine translation itself, a general lower-resource-language effect, or neither.
Two further machine-translated tracks of the same rows separate these accounts:
Spanish (well represented in XLM-R) and Hindi.

### Design

- **Tracks.** `bnintent30-es` and `bnintent30-hi`: machine translations (source
  `en`; targets `es`, `hi`) of the English `text` field of BNIntent30, made with
  the Google Cloud Translation API. Recorded for each language: the API version,
  the date of translation, and the SHA-256 of the translation file. Translations
  are cached under `data/` (gitignored) and never committed (§3.4).
- **Rows.** Exactly the existing clean rows and splits of `bnintent30-en` /
  `bnintent30-bn` (same uids). No re-cleaning: the clean row set is not
  recomputed on the new languages, and the existing tracks and results are
  unchanged. As a diagnostic, report per language the deduplication-key
  collisions among these rows on the translated text (label conflicts,
  cross-split, within-split; same key as §3.3). If cross-split collisions exist,
  also report a sensitivity run excluding them.
- **k-shot subsets.** The same uids per (k, seed) as `bnintent30-en` /
  `bnintent30-bn`, in the same order; the existing split manifest is not
  modified. A test verifies the identity of uids per (k, seed, split) across
  en, bn, es and hi before any run, and must pass before the grid may run.
- **Frozen as in the main grid.** Encoder `xlm-roberta-base` at the same pinned
  revision, mean pooling, maximum length 128 (the longest tokenised utterance of
  each new track is recorded; any truncation is reported); feature families and dimensions
  (tfidf 256 + chargram 256 + stats 6, F = 518), fit on each run's training split
  only (R1); learning rate 3e-3; early stopping on the k-matched validation split
  with the same patience and epoch cap; seeds {13, 42, 87, 100, 2026};
  k in {5, 10, 20, full}. No tuning of any kind on the new tracks.
- **Heads.** `baseline` and `sagff`. 2 heads x 4 k x 5 seeds = 40 runs per
  language, 80 in total.
- **Isolation from existing results.** The 80 run logs are written to a separate
  directory, not to `results/runs/`, so that regenerating every existing table
  and figure reproduces them exactly. New results go to new tables only
  (`results/tables/T8_*`). No existing table or figure changes.

### Predictions and decision rule

- **E1 (primary, per language).** Paired test Macro-F1 difference `sagff` minus
  `baseline`, pooled over k in {5, 10} and the five seeds (n = 10 pairs per
  language), two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`, zero
  differences dropped). Holm correction over the 2 languages (a family of its
  own, separate from §10's), alpha = 0.05. Reported as in §10: statistic, p,
  Holm-adjusted p, n (and n after dropping zeros), median paired difference and
  matched-pairs rank-biserial correlation. The observed direction follows the
  rule of §11 item 4; "significantly positive" means Holm-adjusted p < 0.05 with a
  positive observed direction.
- **Interpretation, registered in advance.**
  - es and hi both significantly positive: consistent with a
    machine-translation artefact.
  - hi positive, es not: consistent with a lower-resource-language effect.
  - neither positive: the Bangla gain is not explained by translation or by a
    general lower-resource effect.
  - Any other pattern (for example es positive and hi not) is reported as
    observed, without a registered interpretation.
- **Registered expectations.** es: not significantly positive (XLM-R represents
  Spanish well). hi: no directional expectation.
- **E2 (descriptive).** Spearman rho(k, gate displacement) per language, n = 20
  (4 k x 5 seeds), on the primary readout of §10 (displacement at the epoch
  selected by early stopping). Not part of any test family.
- **E3 (descriptive).** `baseline` test Macro-F1 at k = 5 for es, hi, en and bn
  side by side (en and bn from the existing grid, unchanged).

**Known differences from the Bangla track, stated in advance.** The Bangla text
was produced by the dataset's authors with the Google Translator API and then
manually reviewed (Hasan et al., 2023; `docs/citation_checklist.md` entry 4a);
the Spanish and Hindi translations are new, unreviewed
API output of the same English rows. The new tracks also inherit the Bangla-driven
row removals of §3.3 (R9) rather than being cleaned on their own text. Both points
are reported with the results.

### Amendment 2026-11-09 (before any translation)

Pins the cross-split sensitivity analysis named under "Rows" above as RESCORING,
not retraining. Made before any translation, embedding or run of either new track;
the text above is unchanged.

- For each language and each run, a test row is "colliding" if its dedup key (§3.3
  key, on that language's translated text) equals the key of any row in that run's
  own training subset (k- and seed-specific).
- Sensitivity analysis: the same trained models; test Macro-F1 recomputed on the
  test split with the colliding rows removed; E1 recomputed on these scores with
  the same test and Holm family. No model is retrained.
- Implementation: each run of the extension records its predicted label for every
  test uid, so the rescoring uses the same predictions and needs no re-inference.
- Reported per language and k: number of colliding test rows (mean and range over
  seeds).
- Justification (mechanism, not result): leakage inflates the score at evaluation,
  when the model meets an item it already saw in training; removing those items
  from scoring removes the inflation without changing the training pool, and keeps
  the run count fixed. This follows the practice of evaluating on the
  non-overlapping subset (Lewis et al., 2021; `docs/citation_checklist.md`
  entry 26).
- If no cross-split collisions exist for a language, the sensitivity analysis is
  reported as not applicable.

### Amendment 2026-11-10 (after translation, before any embedding or run)

**The translations were written by Claude, not by the Cloud Translation API.** This
replaces the registered translation method under "Tracks" above ("made with the
Google Cloud Translation API"). It is recorded after the translations were made
and before any embedding, split, run or quality check of either new track. The
text above is unchanged.

- **How the translations were made** (confirmed from the original session
  transcript). Claude (Anthropic, model Opus 5.5), in a Claude Code session in VS
  Code on another PC on 2026-11-10, read only the English `text` field of each
  BNIntent30 per-intent file (150 sentences at a time). It wrote the Hindi and
  Spanish translations itself into per-file text files, one "Hindi<TAB>Spanish"
  line per sentence. A small merge script inserted them into the per-intent files
  as `hi_text` and `es_text` without altering any other field. No translation
  library, model or external API was used. The user's instruction is saved
  verbatim in `docs/translation_prompt.md`.
- **What was checked on arrival** (2026-11-10, in this repository). For all 4,500
  raw uids, the incoming copy's `text`, `bn_text`, `intent` and `split` are
  identical to the repository's `data/raw/bnintent30/` files (0 differences), and
  all 4,096 clean uids have non-empty Spanish and Hindi text. The repository's raw
  files, clean files and splits still match their recorded checksums.
- **Reference files.** The intermediate text files and the merge script stayed on
  that PC and are no longer available. The reference translations are therefore
  the two extracted files below, one `{uid, text}` line per raw uid (4,500 lines
  each, gitignored, never committed), identified by their SHA-256:
  - `data/extension/es.jsonl`:
    `26587bc208e24739b203c1c3ac8f25bfccbf9ea4ad10c6e0fee7b925cc4eb727`
  - `data/extension/hi.jsonl`:
    `c4b272ccaab6e76c902ffecce36285f357ddd0ad108813d686fbb2dfc5feed7a`

  Every later step reads these files and fails if either hash differs.
- **"API version and date" (registered above)** are replaced by the model name
  (Opus 5.5), the date (2026-11-10) and the two hashes above.
- **Consequence.** The three translated tracks no longer share a translation
  process. Bangla was machine-translated with Google Translate and reviewed by its
  authors; Spanish and Hindi were written by an LLM from English only and were not
  reviewed. A difference between bn and es/hi may therefore reflect the
  translation process as well as the language. In the E1 interpretations above,
  "machine-translation artefact" can only be read as "an artefact of translated
  text", not as an artefact of one shared translation system. The decision rule
  itself, E1-E3 and every other design element are unchanged.
- **Disclosure.** Claude's generation of the Spanish and Hindi data is recorded in
  `docs/ai_use_disclosure.md`.

### Amendment 2026-11-12: Bangla translation-method control (registered before any translation)

**Status and timing.** Registered in its own commit on 2026-11-12, AFTER the results
of the main grid and of E1-E3 (tables T8a-T8e) were visible, and BEFORE any
translation, embedding or run of the new track. At this commit no file
`data/extension/bn_cl.jsonl`, split, cache or run of the track exists. The text
above is unchanged. Any later change is a dated note, justified by a mechanism,
never by the direction of a result.

**Purpose.** The existing Bangla text is Google Translate output reviewed by the
dataset's authors; es and hi were written by Claude. A difference between bn and
es/hi therefore confounds language with translation method. This control holds the
language at Bangla and changes the translation method.

**Design.**
- **Track `bnintent30-bn-cl`.** Bangla written by Claude (Anthropic), model
  Fable 5.2 (model ID `claude-fable-5-2`), in a Claude Code session of the Claude
  desktop app on the repository's PC, from the English text only, for the 4,096
  clean uids. Procedure as for es/hi: one batch per intent (that intent's clean
  rows, at most 150 sentences), each batch file holding a row index, the uid and
  the English sentence; the translator writes one Bangla line per row index; a
  merge script joins them by index and uid, and checks that every row has exactly
  one non-empty line containing Bengali script. No translation library, model or
  external API is used.
- **Blindness to the existing Bangla.** The translator reads only uid, English
  text and intent, from batch files exported from `data/clean/bnintent30-en.jsonl`.
  It does not open `bn_text`, `data/clean/bnintent30-bn.jsonl` or any Bangla split
  file. The share of rows whose bn-cl text is identical to the original `bn_text`
  is computed by script, not by reading.
- **Rows, splits, k-subsets.** Exactly the clean rows and splits of
  `bnintent30-en` / `bnintent30-bn` (same uids, order and labels), in its own split
  directory with its own manifest; a test verifies uid identity per
  (k, seed, split) across en, bn, es, hi and bn-cl and must pass (not be skipped)
  before any run.
- **Frozen settings.** As in the main grid and E1: encoder revision, mean pooling,
  maximum length 128 (longest utterance and any truncation reported), feature
  families and dimensions (F = 518), learning rate 3e-3, early stopping, seeds
  {13, 42, 87, 100, 2026}, k in {5, 10, 20, full}. No tuning.
- **Heads and runs.** `baseline` and `sagff`: 2 x 4 x 5 = 40 runs. Logs go to the
  extension run directory (`results/extension/runs/`), never `results/runs/`;
  each run saves its predicted label per test uid.
- **Isolation.** New results go to new tables only (`results/tables/T9_*`) and new
  files under `results/extension/`. T8a-T8e and every other existing table, figure
  and run are unchanged and must regenerate identically.
- **Provenance.** The intermediate batch files (inputs and the translator's
  outputs) are kept under `data/extension/provenance/` (gitignored). The SHA-256
  of `data/extension/bn_cl.jsonl` is recorded in a further dated note after
  translation and before any embedding or run; every later step fails if it
  differs.

**Prediction and decision rule.**
- **E4 (primary, a single test).** Paired test Macro-F1 difference `sagff` minus
  `baseline` on bn-cl, pooled over k in {5, 10} and the five seeds (n = 10),
  two-sided Wilcoxon signed-rank (`scipy.stats.wilcoxon`, zero differences
  dropped), alpha = 0.05, no multiplicity adjustment (one test, a family of its
  own). Reported as E1 is; observed direction by the rule of §11 item 4;
  "significantly positive" means p < 0.05 with a positive observed direction.
- **Interpretation, registered in advance.**
  - Significantly positive: the Bangla gain is not specific to the Google
    translation.
  - Not significantly positive: the gain may depend on the Google-translated
    text.
  - A significantly negative result is reported as observed; for the
    interpretation it counts as not significantly positive.
- **Registered expectation.** None: no directional expectation.

**Descriptive (no test family).**
- `baseline` test Macro-F1 at k = 5 on bn-cl next to bn (en, es, hi for
  reference).
- Spearman rho(k, gate displacement at the selected epoch) on bn-cl, n = 20.
- Test Macro-F1 of both heads per k on bn-cl next to bn, and the E4 statistic
  computed on the original bn runs of the main grid, for reference.
- Collision diagnostics on the bn-cl text and the rescoring sensitivity analysis,
  as for es/hi (amendment 2026-11-09).
- LaBSE similarity to English and intent preservation for bn-cl, as for bn, es
  and hi; no row is excluded on that basis.
- Share of rows where the bn-cl text is identical to the original `bn_text`
  (exact string, and under the §3.3 key).

**Known limits, stated in advance.**
- Different session, PC and model from the es/hi translation (es/hi: Opus 5.5 on
  another PC, 2026-11-10; bn-cl: Fable 5.2 on this PC). bn-cl against es/hi is
  therefore not a same-translator comparison; the control compares bn-cl with the
  original bn.
- The translating session has worked on this repository before and may have been
  shown a small number of original `bn_text` strings during the leakage audit; the
  model may also have seen public BNIntent30 text in training. Blindness is
  procedural (what the translation step reads), and the identical-row share above
  is its only measured check.
- Like es and hi, the bn-cl text is not reviewed by a human translator.
- With n = 10 the test has limited power; a non-significant E4 is not evidence
  that the two Bangla texts behave identically.
