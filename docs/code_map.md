# Code map

A guide to every source file in this repository, for a reader who has never seen
the project. The research protocol that the code implements is
`docs/PROTOCOL.md`; section numbers below (§, R1-R9) refer to it.

## How the pieces fit together

The pipeline runs in one direction, and every stage writes files that the next
stage reads:

```
scripts/download_data.py      raw data            data/raw/            (+ pins in configs/data.yaml)
src/data.py                   cleaning, uids      data/clean/          (+ results/tables/data_cleaning.md)
scripts/make_splits.py        k-shot splits       data/splits/, data/splits_dirty/ (+ committed manifests)
scripts/cache_embeddings.py   frozen encoder      data/cache/          (via src/embeddings.py)
src/train.py                  one run             results/runs/<run_id>.json
   (uses src/features.py, src/models.py, src/embeddings.py)
scripts/run_all.py            the whole grid      460 run files
src/analysis.py               tables, figures     results/tables/T*.md, results/figures/
```

Nothing in `data/` except the two `manifest.json` files is committed, because
BNIntent30 has no licence. The committed manifests list the row identifiers and
sha256 checksums of every split file, so a reader who downloads the raw data and
re-runs the pipeline can prove they rebuilt exactly the same splits.

Three one-off selection scripts (`select_feature_dims.py`, `select_lr.py`,
`head_capacity.py`) were run once before the grid and wrote frozen values into
`configs/default.yaml`; every later step reads those values from the config.

## Library code (`src/`)

**`src/__init__.py`.** Empty. It makes `src` a package so that scripts and tests
can write `from src.data import ...`.

**`src/data.py`.** Loads the two raw datasets, cleans them and assigns every row
a stable identifier (uid). BNIntent30 rows get uids of the form
`bn30.<file>.<row>`, BANKING77 rows `b77.<split>.<row>`, fixed at load time so
they never depend on processing order. It defines `dedup_key`, the single
function that decides whether two utterances count as "the same" (Unicode NFC,
lowercase, punctuation replaced by spaces, tokens sorted); every other file that
compares texts imports it from here. Cleaning (§3.3) drops untranslated
placeholder rows, rows whose key carries more than one intent, cross-split
duplicates (keeping the test copy) and within-split duplicates. For BNIntent30 the
English and Bangla texts are cleaned together, so the two language tracks keep
exactly the same rows (R9). BANKING77 has no official validation split, so one
sixth of each class's training rows is moved to validation, keeping duplicate
groups together. `build_tracks()` returns the cleaned tracks and is called by
`make_splits.py`, `cache_embeddings.py`, `src/analysis.py` (table T5) and the
tests; `build_dirty_tracks()` returns the uncleaned "leaky" condition used as a
contrast. Run as `python -m src.data`, it writes `data/clean/` and the cleaning
report `results/tables/data_cleaning.md`.

**`src/features.py`.** The three handcrafted feature families (§4): word TF-IDF
reduced by a chi-squared selector, hashed character 3- to 5-grams with idf
weighting, and six standardised utterance statistics. The class
`HandcraftedFeatures` enforces rule R1 (fit on the training split only) in code:
it refuses to transform before it is fitted, refuses to be fitted twice, and
refuses a `split` argument other than `"train"`. The output width is fixed
(TF-IDF columns are zero-padded when a small training set has a small
vocabulary) so the model's projection matrix has the same shape for every k.
`featurize_run()` is the only entry point the rest of the code uses: it fits one
fresh instance on the run's training split and transforms every split with that
same object. Its tokenizer is `src.data.norm_key`, so the TF-IDF branch sees text
exactly as the deduplication key does. Used by `src/train.py`,
`scripts/select_feature_dims.py`, `scripts/feature_report.py` and the leakage
tests.

**`src/embeddings.py`.** Sentence embeddings from the frozen, pinned
`xlm-roberta-base` encoder, mean-pooled over tokens with the attention mask.
`encode()` computes them; it is called only by `scripts/cache_embeddings.py`,
which stores one matrix per track in `data/cache/` with a sidecar JSON listing
the uid order and the file's sha256. `load_embeddings()` is what training uses:
it verifies the checksum and returns rows looked up by uid, never by position.
The module docstring records why embedding the test split with a frozen encoder
does not break R1 (rule R1b): no parameter is estimated from the data.

**`src/models.py`.** The five classification heads (§2), all small PyTorch
modules that take a cached embedding and, for three of them, a handcrafted
feature vector: `Baseline` (embedding only), `Concat` (embedding concatenated with
projected features), `BaselineCap` and `ConcatCap` (the same with an extra hidden
layer, sized so their parameter count matches SAGFF), and `SAGFF`, whose
per-dimension sigmoid gate mixes the two branches (gate high means relying on the
encoder). `build_head()` constructs a head from the `heads` section of
`configs/default.yaml`; `cap_hidden_for()` and `concat_cap_hidden_for()` compute
the matching hidden widths and are used once, by `scripts/head_capacity.py`.

**`src/train.py`.** One training run, start to finish, producing exactly one JSON
log (R6). A `RunSpec` names the track, k, seed and condition (the five heads plus
two permutation controls that feed SAGFF deliberately scrambled features).
`run()` loads the split files named in the manifest and checks their sha256,
builds features with `featurize_run`, loads embeddings by uid, trains with AdamW
and early stopping on the k-matched validation split, and records metrics, the
head's parameter count, the git commit and library versions. For SAGFF-type runs
it records gate statistics on the test split at initialisation, at every epoch,
at the first epoch with training loss below 0.01, at the final epoch and at the
selected epoch (R7). A failed run still writes its JSON, and an existing file is
never overwritten. `require_clean_tree()` is used by every script that produces
results, so no result can come from uncommitted code. Called by
`scripts/run_all.py`, `smoke_test.py`, `select_lr.py`, `stopping_diagnostic.py`
and `rerun_anomaly.py`.

**`src/analysis.py`.** Turns the 460 run logs in `results/runs/` into every table
and figure the manuscript uses (R8). It computes the pre-registered tests of §10
(Spearman and Wilcoxon with Holm adjustment, on three gate readouts) and writes
tables T1-T7, the headline, predictions, timing and run-accounting tables, the
six figures F1-F6 and their captions. Table T3 also calls
`scripts/leakage_dose.py` on the leaky split files, and T5 calls
`src.data.build_tracks()`, so those two tables need the raw data and splits on
disk. A run listed in `results/runs_anomalies.md` is excluded only from the table
kinds named there (currently only the timing table). Run as
`python -m src.analysis all`.

## Pipeline scripts (`scripts/`)

**`scripts/download_data.py`.** Downloads BNIntent30 (a zip of one pinned GitHub
commit) and the two BANKING77 CSV files (pinned commit), and checks every file's
sha256 against `configs/data.yaml`. On its first ever run it resolved the
branches to commits and recorded them; it now only reads those pins. It rewrites
`configs/data.yaml` on every run (same content if nothing changed). Feeds
`src/data.py`.

**`scripts/make_splits.py`.** Builds, for every track, k (5, 10, 20, full) and
seed, the files `train.jsonl`, `val.jsonl` (at most k per class, the only split
used for model selection), `test.jsonl` and `val_full.jsonl`. Rows are chosen per
class by ranking sha256(seed, uid), so no random-number library is involved and
smaller k sets are subsets of larger ones. Writes the clean condition to
`data/splits/` and the leaky condition to `data/splits_dirty/`. If a manifest
already exists it only checks the regenerated files against it and fails on any
difference. Uses `src.data`; its functions are also imported by the
reproducibility test in `tests/test_leakage.py`.

**`scripts/verify_manifest.py`.** Recomputes the sha256 of every raw file and
every split file on disk, compares them and the uid order with the two manifests
and `configs/data.yaml`, and exits non-zero on any mismatch (R3).

**`scripts/cache_embeddings.py`.** Collects every uid that any clean or leaky
split uses for a track, embeds them once with `src.embeddings.encode`, and writes
`data/cache/<track>_xlm-roberta-base_mean.npy` plus its sidecar. Skips a track
whose cache already matches.

**`scripts/run_all.py`.** Runs the full grid of 460 runs in the fixed order of
§5 (Bangla and English BNIntent30 first, BANKING77 last, then the leaky
condition). It refuses to start unless the working tree is committed and the
whole test suite passes, and it skips any run whose JSON already exists, so it can
be resumed. Progress lines show status and time only, never metrics; they go to the
screen and are appended to `logs/grid.log`, which the script creates if needed and
which `record_anomalies.py` reads.

**`scripts/smoke_test.py`.** One SAGFF run (Bangla, k=5, seed 13) written to
`results/smoke/`, followed by a check that the JSON contains every field the
protocol requires. Used once before the grid.

## One-off selection scripts (run once, results frozen)

**`scripts/select_feature_dims.py`.** Chose the TF-IDF and character-hash
dimensions on validation data only, using the handcrafted features alone with a
logistic regression, by the rule "smallest configuration within one standard
deviation of the best", extending the grid downward until the choice was
bracketed. Wrote `features` in `configs/default.yaml`,
`results/selection/feature_dims/scores.json` and
`results/tables/feature_dim_selection.md`. Refuses to run again without
`--reselect`.

**`scripts/select_lr.py`.** Chose the learning rate on validation data only, with
the baseline head only, from a grid extended at whichever edge the winner sat.
Each cell is a `src.train` run with `evaluate_test=False`, so test data is never
loaded. Wrote `training.lr` in `configs/default.yaml`, the run logs in
`results/selection/lr/` and `results/tables/lr_selection.md`. Refuses to run again
without `--reselect`.

**`scripts/head_capacity.py`.** Fixes the head configuration: computes the hidden
widths that match the capped heads' parameter counts to SAGFF, counts every
head's parameters from the instantiated modules, and writes the `heads` section of
`configs/default.yaml` and `results/tables/head_capacity.md`. Note that it
rewrites the config every time it runs.

**`scripts/check_feature_dims_banking77.py`.** Repeats the feature-dimension
selection procedure on BANKING77 alone, without changing anything, to report
whether the frozen dimensions also fall within BANKING77's own one-SD plateau.
Reuses functions from `select_feature_dims.py`.

## Diagnostic and reporting scripts

**`scripts/leakage_dose.py`.** For the Bangla track, counts how many test rows
share a deduplication key with the run's training or validation split, in the
leaky condition and (as a control that must be zero) the clean one. Writes
`results/tables/leakage_dose.md`; its `dose()` function is also used by
`src/analysis.py` for table T3.

**`scripts/feature_report.py`.** Describes the realised features for every track
and k over all seeds: dimensions, sparsity, out-of-vocabulary rates, test rows for
which the TF-IDF family is entirely zero, and a check that no two different
deduplication keys produce identical feature vectors. Writes
`results/tables/feature_overview.md` and `results/tables/tfidf_coverage.md`.

**`scripts/stopping_diagnostic.py`.** Thirty SAGFF runs (three tracks, k=5 and
full, five seeds) logged to `results/diagnostics/stopping/`, comparing the gate at
the early-stopping epoch with the gate at the last trained epoch. Reuses existing
logs if present. Writes `results/tables/stopping_diagnostic.md`.

**`scripts/compare_lr_reproduction.py`.** Compares the learning-rate selection
runs made from an uncommitted tree (`results/selection/lr_218bd44_uncommitted/`)
with the re-runs from a committed tree (`results/selection/lr/`) using exact
equality, as evidence that training is deterministic. Writes
`results/tables/lr_reproduction.md`.

**`scripts/compare_stopping_to_grid.py`.** Compares the 30 stopping-diagnostic runs
(made before the per-epoch gate trajectory and the convergence-anchored readout were
added to `src/train.py`) with the corresponding grid runs, using exact equality, as
evidence that adding those readouts did not change training. Writes
`results/tables/stopping_vs_grid.md`.

**`scripts/rerun_anomaly.py`.** Re-runs the specification of one stored run into
`results/diagnostics/anomaly_rerun/` and checks that every training output is
bit-identical, to show that an anomaly (a machine sleep during the run) affected
only wall-clock time.

**`scripts/record_anomalies.py`.** Writes `results/runs_anomalies.md`, which
documents that anomaly from the grid log and the Windows event log and ends with
a machine-readable block that `src/analysis.py` reads to drop the run from the
timing table only. It can only be re-run on the original machine (see the review
report).

## Tests (`tests/`)

**`tests/conftest.py`.** Shared fixtures that load both manifests, every split
file and the rebuilt cleaned tracks. Missing data makes the tests fail rather
than skip, so the suite cannot pass on an empty checkout by accident.

**`tests/test_leakage.py`.** The checks behind R1, R3, R5 and §3.5: the TF-IDF
vocabulary is exactly the training split's token set and every TF-IDF column of every
split is a selected training term present in that row, the feature class refuses misuse, no
deduplication key is shared between or repeated within splits, k-shot and
k-matched validation sizes are right and the test split has every class, smaller
k sets are nested in larger ones, `make_splits.py` reproduces the manifest byte
for byte, and the feature pipeline used at test time is the one object fitted once
on exactly the training texts.

**`tests/test_parallel.py`.** R9: for every k and seed, the English and Bangla
BNIntent30 tracks have the same uids in the same order with the same labels, and
the two text fields are not swapped.

**`tests/test_cleaning.py`.** §3.3: the post-step-1 counts, counted directly from the
raw rows, match the published paper (4,433 rows), no placeholder survives, no key has two labels or appears
twice, every class keeps at least 20 training rows, and uids are deterministic.

**`tests/test_models.py`.** §2: the SAGFF gate is a vector in (0, 1) with its bias
taken from the config, featureless heads ignore features and feature heads use
them, the capped heads are parameter-matched, parameter counts of the modules and
the config equal the counts written out from the §2 definitions, the grid that
`run_all.py` builds follows the §5 order, and the permutation controls are
deterministic true permutations that change with seed, split and (for the
per-epoch control) epoch.

**`tests/test_analysis.py`.** The statistics behind the registered analyses:
Spearman, Wilcoxon with its rank-biserial correlation, Holm adjustment, the
direction and status rules and `registered_tests()` itself, each checked against
values computed independently (scipy called directly, formulas written out, worked
Holm examples, synthetic run logs whose answers are known). It also checks
`src.train.gate_stats` against numpy and that every committed displacement equals its
readout minus the gate mean at initialisation.

## Configuration and data records

**`requirements.txt`, `requirements.lock`.** The direct dependencies, and the exact
environment (every package, including the CPU build of PyTorch) that produced the
runs; the lock file's versions match each run log's `libraries` field.


**`configs/data.yaml`.** Pinned upstream commits, download URLs, raw-file
checksums and dataset citations, written by `download_data.py`.

**`configs/default.yaml`.** The frozen feature dimensions, head configuration and
training hyperparameters shared by every run, each with a record of how and from
which commit it was selected.

**`data/splits/manifest.json`, `data/splits_dirty/manifest.json`.** The uid lists
and sha256 checksums of every split file, written by `make_splits.py` and checked
by `verify_manifest.py`, the tests and `src/train.py`.
