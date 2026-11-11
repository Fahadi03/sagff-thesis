# SAGFF thesis experiments

*When Do Handcrafted Features Help? Scarcity-Aware Feature Fusion for
Low-Resource Intent Classification in English and Bangla.*

- Research protocol and repository rules: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).
  The file was originally named `CLAUDE.md`, the configuration file read by the AI
  coding assistant (Claude Code) that implemented the pipeline; see
  [`docs/ai_use_disclosure.md`](docs/ai_use_disclosure.md).
- What each source file does and how they connect: [`docs/code_map.md`](docs/code_map.md).
- Pre-registration and post-registration decisions:
  [`docs/preregistration_disclosure.md`](docs/preregistration_disclosure.md).

## Environment

The runs were produced with Python 3.13.3 on Windows 11, CPU only.
`requirements.lock` pins the exact environment (every transitive package, and the
CPU build of PyTorch 2.14.1); `requirements.txt` lists the direct dependencies.
The commands below use the Windows layout `.venv/Scripts/python`; on Linux and
macOS use `.venv/bin/python`.

On Windows, clone into a short path (for example `C:\sagff`): installing PyTorch
writes files whose full path exceeds the 260-character limit when the clone sits
deep in the folder tree, unless Windows long-path support is enabled.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock
```

## Reproduce

Each command writes the files named after it. Times are for the original CPU.

```bash
# 1. Data (BNIntent30 text is never committed: it is downloaded and rebuilt here)
.venv/Scripts/python scripts/download_data.py      # data/raw/; pinned commits and checksums in configs/data.yaml
.venv/Scripts/python -m src.data                   # data/clean/, results/tables/data_cleaning.md
.venv/Scripts/python scripts/make_splits.py        # data/splits/ (clean), data/splits_dirty/ (leaky); checked against both manifests
.venv/Scripts/python scripts/verify_manifest.py    # exits non-zero on any sha256 mismatch

# 2. Tests (about 1 minute; they need the data from step 1)
.venv/Scripts/python -m pytest tests/ -v

# 3. Frozen-encoder embeddings (about 13 minutes; downloads xlm-roberta-base, about 1.1 GB)
.venv/Scripts/python scripts/cache_embeddings.py   # data/cache/; idempotent

# 4. The paper's tables and figures, from the committed run logs in results/runs/
.venv/Scripts/python -m src.analysis all           # results/tables/T1-T7, headline, predictions, accounting, timing; results/figures/F1-F6 + captions

# 5. Supporting tables
.venv/Scripts/python scripts/leakage_dose.py       # results/tables/leakage_dose.md
.venv/Scripts/python scripts/feature_report.py     # results/tables/feature_overview.md, tfidf_coverage.md
.venv/Scripts/python scripts/compare_lr_reproduction.py  # results/tables/lr_reproduction.md
.venv/Scripts/python scripts/compare_stopping_to_grid.py # results/tables/stopping_vs_grid.md
.venv/Scripts/python scripts/stopping_diagnostic.py      # results/tables/stopping_diagnostic.md (reuses the committed diagnostic logs)

# 6. The grid (460 runs): test suite first, committed code required
.venv/Scripts/python scripts/run_all.py            # results/runs/; progress also appended to logs/grid.log
```

Steps 1, 4 and 5 reproduce the committed files byte for byte, with the
exceptions listed below. `scripts/check_numeric_equality.py <commit>` compares every
numeric table cell and every figure on disk with their versions at a given commit and
prints the full text diff. Tables T3 and T5 also need the raw data and the leaky
split files from step 1.

`scripts/run_all.py` skips every run whose log already exists, so on a fresh
clone it runs the tests and then skips all 460 runs. To re-run the grid from
scratch, move `results/runs/` aside first (for example to `results/runs_committed/`)
and compare the new logs with the committed ones. The grid took 11.9 h of wall
time on the original machine, of which 8.1 h was a machine sleep during one run
(`results/runs_anomalies.md`); without the sleep the runs took 3.8 h (sum of the
`wall_clock_s` fields minus the recorded sleep).

### Spanish and Hindi extension (docs/PROTOCOL.md §12)

The extension adds two tracks, `bnintent30-es` and `bnintent30-hi`: the clean
BNIntent30 rows and splits with the text translated from English. It is
post-registration; its plan, predictions and amendments are in §12 of the protocol.
It has its own splits, embedding cache and run directory, so none of the commands
above is affected by it.

The translations are not in the repository and cannot be rebuilt by a script: they
were written by Claude from the English text (§12, amendment 2026-11-10;
`docs/translation_prompt.md`). The reference files are `data/extension/es.jsonl` and
`data/extension/hi.jsonl`, one `{"uid", "text"}` line per raw uid, identified by the
SHA-256 hashes recorded in §12. Every script below stops if a hash differs. Without
those two files, only the last command runs: it rebuilds T8a-T8d from the committed
run logs and then stops at T8e, which needs the split files.

```bash
# E1. Splits and collision diagnostics (needs data/extension/es.jsonl and hi.jsonl)
.venv/Scripts/python scripts/extension_prepare.py     # data/extension/splits/ (checked against its manifest), results/extension/collisions.md

# E2. Translation-quality checks (downloads LaBSE, about 1.9 GB; descriptive, no row is excluded)
.venv/Scripts/python scripts/translation_quality.py   # results/extension/translation_quality.md, low_similarity_uids.txt, review_sample_uids.txt

# E3. Tests, including the uid-identity test across en, bn, es and hi
.venv/Scripts/python -m pytest tests/ -v

# E4. Frozen-encoder embeddings for es and hi (about 7 minutes; separate cache)
.venv/Scripts/python scripts/extension_embed.py       # data/extension/cache/, results/extension/embeddings.md

# E5. The 80 runs (about 15 minutes): test suite first, committed code required
.venv/Scripts/python scripts/run_extension.py         # results/extension/runs/; progress also appended to logs/extension.log

# E6. The registered analysis (E1, E2, E3, rescoring), from the committed run logs
.venv/Scripts/python scripts/extension_analysis.py    # results/tables/T8a-T8e
```

`scripts/run_extension.py` skips every run whose log already exists and refuses to
start unless `tests/test_extension.py` passed (a skip does not count). On a clone
without the extension data, `tests/test_extension.py` is skipped and the main test
suite is unaffected.

### Outputs that are not byte-identical on regeneration, and why

- `results/tables/stopping_diagnostic.md` records the commit that generated it,
  so it changes with every new commit; its numbers do not.
- `scripts/smoke_test.py` (one `sagff` run, used once before the grid) overwrites
  `results/smoke/`. The committed smoke log was made at `4dab8fe`, before the final
  learning rate and feature dimensions were selected; it is not a result.
- `scripts/head_capacity.py` and `scripts/download_data.py` rewrite
  `configs/default.yaml` (heads section) and `configs/data.yaml`; with unchanged
  inputs the bytes are identical, and they write nothing when nothing changed.

### One-off steps that are frozen and are not re-run

- `scripts/select_feature_dims.py` (feature dimensions) and `scripts/select_lr.py`
  (learning rate, baseline head only) were each run once on validation data; their
  results are frozen in `configs/default.yaml` and they refuse to run again without
  `--reselect`. `scripts/check_feature_dims_banking77.py` repeats the feature
  selection procedure on BANKING77 without selecting anything. Their reports
  (`feature_dim_selection.md`, `lr_selection.md`, `feature_dim_check_banking77.md`)
  record the commit they ran from.
- `scripts/rerun_anomaly.py` re-ran one anomalous run to show it is bit-identical.
- `scripts/record_anomalies.py` wrote `results/runs_anomalies.md` from
  `logs/grid.log` and the Windows System event log of the original machine. It
  cannot be re-run elsewhere; `src/analysis.py` reads only the committed file.

Each split directory `<track>/k<k>_seed<seed>/` holds `train.jsonl`, `val.jsonl`
(k-matched; the only split used for model selection), `test.jsonl`,
`val_full.jsonl` (diagnostic only) and `meta.json`.

## Data and licenses

| Dataset | Source (pinned) | License |
|---|---|---|
| BNIntent30 (Hasan et al., ICCIT 2023, doi:10.1109/ICCIT60459.2023.10440989; also arXiv:2312.10679) | `JahidBasher/GAN-BnBERT` @ `213bbcf` (`dataset/BNIntent30/`) | **None stated** (no license file; paper states no data license). No BNIntent30 rows are committed in the current tree; only uids and checksums in the two manifests. The commit history contains three example utterances quoted in the protocol file: a one-word Bangla utterance (commit `52af2fc` only) and two short English utterances used to illustrate translation collapse (commits `34c29f9` to `d964a5a`; removed in `7d78a50`). The English side of BNIntent30 derives from CLINC150. BNIntent30's English text is excerpted from CLINC150 (Larson et al., EMNLP-IJCNLP 2019, doi:10.18653/v1/D19-1131), which its official repository (`clinc/oos-eval`, `LICENSE`) releases under the Creative Commons Attribution 3.0 Unported license (https://creativecommons.org/licenses/by/3.0/), permitting redistribution with attribution; that license does not cover BNIntent30's Bangla translations. Permission request draft: `docs/data_license_request.md`. |
| BANKING77 (Casanueva et al., NLP4ConvAI 2020, doi:10.18653/v1/2020.nlp4convai-1.5) | `PolyAI-LDN/task-specific-datasets` @ `57ec275` (`banking_data/*.csv`) | CC-BY-4.0, per the repository's LICENSE file (verified via the GitHub license API). BANKING77 text is also never committed here; it is downloaded by `scripts/download_data.py`. |

## Provenance practice

- Every selection, diagnostic and grid run refuses to start from uncommitted code
  and records the commit it ran from; files the experiments themselves write
  (`results/`, `logs/`) do not count as uncommitted code.
- A correction to a frozen artefact is made in its own commit, touching only what
  is corrected, so the history shows exactly what changed and what did not. One
  exception is recorded below (`4579f64`).
- Pre-registration: the registered analyses S1-S5 (their tests, statistics,
  predicted directions and correction method) are unchanged since `44bd224`, the
  commit that fixed them before any grid run. The surrounding text of §10 was
  edited in three commits, none of which altered any analysis:
  - `80b01ae` (2026-10-12): corrected the git command that prints the registration commit.
  - `32b865c` (2026-10-12): added the disclosed amendment (two sensitivity readouts
    after the stopping diagnostic), the paragraph on why the primary readout is the
    conservative one, the disclosure and the commitment.
  - `4579f64` (2026-10-26): rewrote both git lookup commands for the file's new name
    (`--follow -- docs/PROTOCOL.md`), as part of the commit that moved the file
    rather than in a commit of its own.
  Decisions made after registration are listed in `docs/preregistration_disclosure.md`
  and in §11 of the protocol.

## Labels in commit messages

The commit history cannot be rewritten, and its messages use the working labels of
the project. They map to the protocol as follows: *Stage 1*, data download and
cleaning; *Stage 1B*, the normalized deduplication key, cleaning of all tracks,
k-matched validation and the leaky splits; *Stage 2*, feature families and the
leakage dose; *Stages 3 and 5*, heads, controls and the embedding cache;
*Task K1/K2*, the bracketed re-selection of feature dimensions and learning rate;
*Task L*, the learning-rate reproduction check; *Task M*, the stopping diagnostic;
*Tasks N, O*, the per-epoch gate trajectory and the convergence-anchored readout;
*Task Q*, the BANKING77 feature-dimension check; *Stage 7, 7b, 7c*, the reporting
layer (tables, figures, descriptive comparisons).
