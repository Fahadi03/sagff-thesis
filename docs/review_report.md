# Pre-handover review report

Review of the repository at commit `dd08f59` (branch `main`, clean tree), carried out
as an external examiner would before the work goes to the thesis supervisor.
Phase 1 only: nothing in the repository was changed except the addition of this
file and `docs/code_map.md`. Nothing was committed.

## How the review was done

- Read `docs/PROTOCOL.md`, `README.md`, every file in `src/`, `scripts/`, `tests/`,
  `configs/` and `docs/`, and the generated tables.
- Cloned the repository into a fresh temporary directory
  (`%LOCALAPPDATA%\Temp\sr`), created a new virtual environment from
  `requirements.txt` with the system Python 3.13.3, and followed `README.md` line by
  line. The embedding step used an empty Hugging Face cache (`HF_HOME` pointed at a
  new folder), so the encoder was downloaded as a stranger would download it.
- In the clone: ran the full test suite, regenerated every table and figure that can
  be regenerated, compared each output byte for byte with the committed file
  (`git show HEAD:<path>`), and re-ran six grid runs from scratch to compare them with
  the committed run logs.

## Summary

The core result chain reproduces exactly. Following the README in a fresh clone
rebuilt the same raw data, the same 320 split files (manifest verified) and the same
frozen-encoder embeddings (sha256 equal to the value recorded in all 460 run logs).
All 53 tests pass. Every table and figure produced by `python -m src.analysis all`
and by the report scripts is byte-identical to the committed file, PDFs and PNGs
included, and six grid runs re-run from scratch reproduce their committed logs bit
for bit (only wall time, commit hash and log path differ).

The problems are in the documentation and around the edges of the pipeline. The
README never gives the command that produces the paper's tables and figures, and
its last command fails on a fresh clone. Several sentences in the protocol contradict
the generated tables, which is exactly what rule R8 forbids. Two BNIntent30
utterances are quoted in the protocol although the repository states that none is
committed. The README makes a claim about §10 being byte-identical that one
`git diff` disproves. Large parts of the protocol and the code comments still read
as instructions to an AI assistant, with task letters from the working conversation.

## Reproduction results (checks 1 and 6)

| README step | Result in the fresh clone |
|---|---|
| `python -m venv .venv`; `pip install -r requirements.txt` | OK, but see S3 and S4: fails at a long clone path; 15 unpinned transitive packages resolved to newer versions than the original `.venv` |
| `scripts/download_data.py` | OK; all checksums match; `configs/data.yaml` rewritten with identical bytes |
| `python -m src.data` | OK; `data_cleaning.md` byte-identical |
| `scripts/make_splits.py` | OK; 60 clean and 20 leaky splits match both manifests |
| `scripts/verify_manifest.py` | OK; 320 split files in 80 entries match |
| `scripts/leakage_dose.py` | OK; byte-identical |
| `scripts/feature_report.py` | OK; `feature_overview.md`, `tfidf_coverage.md` byte-identical |
| `scripts/head_capacity.py` | OK; `head_capacity.md` byte-identical; rewrites `configs/default.yaml` (identical bytes) |
| `scripts/cache_embeddings.py` | OK, 845 s total; all three cache sha256 values equal the `encoder.cache_sha256` recorded in every run log |
| `python -m pytest tests/ -v` | **53 passed** in 53 s |
| `scripts/smoke_test.py` | Runs and validates, but overwrites the committed, stale `results/smoke/` log (S5) |
| `scripts/stopping_diagnostic.py` | Runs; table differs from the committed one only in the embedded "table generated at commit" hash (S5) |
| `nohup ... scripts/run_all.py > logs/grid.log 2>&1 &` | **Fails as written**: `logs/` does not exist in a clone (M2). With `logs/` created: tests pass, all 460 runs skipped because their logs exist |
| (not in README) `python -m src.analysis all` | OK; T1-T7, headline, predictions, run accounting, run timing, captions and F1-F6 (PDF and PNG) all **byte-identical** |
| (not in README) `scripts/compare_lr_reproduction.py` | OK; byte-identical |
| (not in README) `scripts/check_feature_dims_banking77.py` | Runs; every number identical; only the "Run from commit" line differs |
| (not in README) `scripts/select_feature_dims.py --reselect` | Reproduces the frozen choice (256/256) and every score; only the commit hash and date differ in the table, `scores.json` and `configs/default.yaml` |
| (not in README) `scripts/record_anomalies.py` | **Cannot run** outside the original machine: needs `logs/grid.log` (gitignored) and the Windows event log (S9) |
| Six grid runs re-run from scratch | Bit-identical to the committed logs: `bnintent30-bn k5 s13 sagff`, `bnintent30-en k10 s87 sagff-noise-perepoch`, `banking77 k5 s42 concat-cap`, `bnintent30-bn k20 s100 baseline (leaky)`, `banking77 k10 s2026 sagff-permuted-fixed`, `bnintent30-en kfull s42 concat` |

## Must fix

**M1. README omits the command that produces the paper's tables and figures.**
`README.md:9-26`. `python -m src.analysis all` (T1-T7, F1-F6, captions, headline,
accounting) appears nowhere in the README, nor do `compare_lr_reproduction.py`,
`check_feature_dims_banking77.py`, `rerun_anomaly.py` or `record_anomalies.py`. A
stranger following the README can rebuild the data but not the results. The README
should also say that T3 and T5 need the raw data and split files on disk.

**M2. The README's grid command fails on a fresh clone.** `README.md:25`.
`> logs/grid.log` fails with "No such file or directory" because `logs/` is gitignored
and absent. The line also mixes `nohup` and `&` (POSIX) with the Windows-only
`.venv/Scripts/python`, and on a clone it is a no-op because all 460 run logs are
committed. The README should say so and explain how to reproduce runs from scratch
(for example, point `RunSpec.log_dir` elsewhere or move `results/runs/` aside).

**M3. Protocol sentences that contradict the generated tables (R8).**
- `docs/PROTOCOL.md:479-480`: "early-stops on 577 labelled rows". The cleaned
  BNIntent30 validation portion has **568** rows (`results/tables/data_cleaning.md:36`).
- `docs/PROTOCOL.md:578-579` and `:772-776`: "roughly 14-22 ms per optimizer step".
  `results/tables/run_timing.md` shows 1.6-7.2 ms for `baseline`/`concat` and
  12.9-26.1 ms for the large heads. The number is wrong for small heads and
  out of range for large ones.
- `docs/PROTOCOL.md:644`: "Expect roughly 20 minutes on CPU". The sidecars record
  775 s on the original machine (845 s in the clone); the README says about 13 min.

**M4. BNIntent30 text is quoted in the repository, against its own licensing claim.**
`docs/PROTOCOL.md:437`. The step-4 rationale quotes two English utterances (a
request and a shorter paraphrase of it) as the example of translation collapse; both
are exact `text` values of BNIntent30 rows (checked against the raw files).
`README.md:40` and `.gitignore:1-2` state that no BNIntent30 text is committed. The
English side derives from CLINC150, so the licensing risk is probably small, but the
stated rule is absolute. Replace them with a description. Separately, commit
`52af2fc` (`CLAUDE.md`, line 215) quotes a one-word Bangla utterance that is an exact
`bn_text` value. It cannot be removed without rewriting history, which is excluded,
so the README claim should be worded to match ("no dataset rows; one single-word
example in the history of the protocol file"). Everything else is clean: no Bangla
text in any tracked file or historical blob apart from that word, the Unicode range
bounds in `src/data.py:552` and `tests/test_parallel.py:8`, and a common Bangla word
used as a tokenizer example in `src/features.py:32`, which is not a dataset row. No run log, table or
selection file contains any utterance field.

**M5. The README's statement that §10 is byte-identical is false as written.**
`README.md:48-51` says that after `80b01ae` "the pre-specified analyses remain
byte-identical to `44bd224`". The text of §10 was changed in `80b01ae` (the command),
`32b865c` (the disclosed amendment) and `4579f64` (both lookup commands rewritten
to `--follow -- docs/PROTOCOL.md`). The `4579f64` change was bundled into the file-move
commit, which breaks the repository's own rule that a correction to a frozen artefact
gets its own commit (`README.md:48-49`). The statistics, predictions, family and
rules did not change; the README should say that, list the three text edits, and
not claim byte identity. (§10 itself is not to be touched.) Both lookup commands in
§10 do print the right commits (`44bd224`, `32b865c`).

**M6. Analysis decisions made after registration are not disclosed.**
`docs/preregistration_disclosure.md` covers only the stopping diagnostic. Not covered:
- the feature dimensions (`7e7f8f6`) and the learning rate (`fdb9e52`) were
  re-selected after the analysis plan was fixed at `44bd224`, although before any grid
  run and on validation data only;
- the rule that Wilcoxon "observed direction" is the sign of the rank-biserial
  correlation, and the "direction ambiguous" rule that now applies to S4
  (`src/analysis.py:182-201`), were introduced in `c56a7de`, after the grid had
  finished. §10 defines neither. S4 is non-significant either way, so no status
  changes, but an examiner will ask when the rule was written.
All three should be stated in the disclosure text.

**M7. Assistant-directed text in the two documents the supervisor reads first.**
- `README.md:5-7`: "Claude Code does not load it automatically under this name:
  point any future session at it explicitly."
- `README.md:9`: "(Stage 1)".
- `docs/PROTOCOL.md:3-7`: the same loading note.
- `docs/PROTOCOL.md:12-14`: "If anything in the repo still reflects v1, fix it to match this file."
- `docs/PROTOCOL.md:20-23`: "If a rule blocks you, STOP and report it in plain language. Do not work around it ..."
- `docs/PROTOCOL.md:37-38`: "State it that way in the paper."
- `docs/PROTOCOL.md:61-62, 75-76, 134, 136-137, 300, 373-374, 454, 549, 566, 613, 621-622`: commands to the implementer ("Do not widen ...", "Write this distinction as a comment ...", "Do not omit it", "Keep it as is", "STOP and report", "do not add them back", "do not build, do not suggest building", "Ask before starting this").
- `docs/PROTOCOL.md:238`: "If your gate tensor is `(B, 1)` you have built the wrong model."
- `docs/PROTOCOL.md:492-496`: "VERIFIED ... Do not re-check; do not carry this note into the manuscript."
- `docs/PROTOCOL.md:520`: "(Task Q)"; `:682`: "(Task M)"; `:291, 296, 307-308, 883`: control labels "I-A"/"I-B" that are never defined as names.
- `docs/PROTOCOL.md:748`: "the grid order is the supervisor's ruling"; `:763`: "Design and confound reasoning may come from the supervisor".
- `docs/PROTOCOL.md:759-787` (§9): written as working instructions ("Commit after each working stage", "Report blockers early", "Stop and find out why").
These can all be rewritten in neutral protocol voice without changing any rule; §10
stays as it is.

**M8. `docs/drafts/` does not exist, so check 4 could not be completed for the drafts.**
No `docs/drafts/` directory is in the repository or the working folder.
`docs/citation_checklist.md:16-23` records the same thing and lists ten citations
(P1-P10) as pending. Every number in the drafts still needs to be traced to a
generated table before handover.

## Should fix

**S1. The code that produces every confirmatory statistic has no tests.**
`src/analysis.py:145-281` (`spearman`, `wilcoxon` with its rank-biserial, `holm`,
`observed`, `status`, `registered_tests`) and `src/train.py:163-168, 324-344`
(`gate_stats`, the three displacement fields) are untested. A handful of tests on
small fixed inputs (a known Holm example, a hand-computed rank-biserial, a
displacement sign) would close this without changing any output.

**S2. Tests that cannot fail, or cannot fail for the reason they name.**
- `tests/test_leakage.py:38-41`: a TF-IDF column index `>= tfidf_selected` is
  impossible by construction, since `SelectKBest` outputs exactly that many columns;
  `:42-43` (the vocabulary cannot grow at transform time in scikit-learn) is the same.
  The protocol says honestly what the test does not prove, but as written it cannot fail.
- `tests/test_cleaning.py:16-21`: `clean_bnintent30` already raises on a mismatch
  (`src/data.py:249-253`), so the test's own assertion is unreachable.
- `tests/test_models.py:84-88`: compares `CONDITIONS` with a copy of the same literal;
  a change detector, not a check on the order the grid actually used.
- `tests/test_models.py:75-81`: the config's parameter counts were written by
  `scripts/head_capacity.py` from the same modules, so the test checks the config
  against its own source.

**S3. Environment assumptions in the README.**
- `README.md:12-25`: Windows paths (`.venv/Scripts/python`) throughout; the Linux and
  macOS variant appears once, as a comment.
- The Python version is not stated (all runs used 3.13.3).
- Only top-level packages are pinned (`requirements.txt`). The fresh install resolved
  15 transitive packages to versions other than the original `.venv`, including
  `pillow` 12.3.0 → 13.0.0, `python-dateutil` 2.9.0 → 3.0.0, `filelock` 3.32 → 4.2 and
  `networkx` 3.6 → 3.7. Outputs were still identical, but a `pip freeze` lock file of
  the environment that produced the runs would remove the doubt.
- `requirements.txt:9`: the CPU-only PyTorch index is mentioned only in a comment.
- The full install fails when the clone sits at a long path on Windows: pip could not
  write a PyTorch header because the path exceeded 260 characters. Recommend a short
  clone path or Windows long-path support in the README.

**S4. README steps rewrite tracked "frozen" files.**
`scripts/download_data.py:135` rewrites `configs/data.yaml` and
`scripts/head_capacity.py:84` rewrites `configs/default.yaml` (heads section) every
time they run. Today the bytes come out identical, but a frozen config that a
reporting step rewrites is fragile: any library change in YAML dumping would dirty the
tree, and `run_all.py` would then refuse to start. The README lists `head_capacity.py`
as a routine step (`README.md:20`).

**S5. Committed outputs that cannot be byte-identical after regeneration.**
- `results/tables/stopping_diagnostic.md:5`, `results/tables/feature_dim_check_banking77.md:5`,
  `results/selection/feature_dims_banking77/scores.json` (the `git.commit` field), and
  the two selection tables embed the commit that generated them, so they change with
  every new commit. The committed `stopping_diagnostic.md` already names `4579f64`, not HEAD.
- `results/smoke/bnintent30-bn__k5__seed13__sagff.json` dates from `4dab8fe`, before the
  final learning rate and feature dimensions were selected, and is overwritten by the
  README's smoke step. It is not a result, but an examiner who opens it sees a
  configuration that no reported run used.
Decide whether these are allowed to differ, and say so in the README.

**S6. Conversational markers in code, comments, docstrings and generated text.**
- Task letters: `scripts/stopping_diagnostic.py:1` (Task M),
  `scripts/check_feature_dims_banking77.py:1, 81` (Task Q), `scripts/compare_lr_reproduction.py:1` (Task L),
  `scripts/select_lr.py:10` and `scripts/select_feature_dims.py:19` (Task K),
  `scripts/select_feature_dims.py:160` (K3), `scripts/leakage_dose.py:1` (Stage 1b, Task A),
  `src/data.py:426` (Task B), `:448` (Task C), `src/analysis.py:1` (Stage 7), `:73` (Task T), `:485` (Task U).
- Single-letter tags: `src/train.py:293` (N), `:297, 338` (O), `:311, 331` (M), `:326` (G2);
  `src/models.py:71` and `scripts/head_capacity.py:26` (G1); `tests/test_models.py:37` (G1), `:104` (G3), `:115` (I-A).
- "Stage" keys: `src/data.py:89, 91, 493, 537-538` ("the Stage 1 key", "the Stage 1b key"); `requirements.txt:1, 5, 9, 11, 17`.
- Rulings: `scripts/run_all.py:1` ("the ruled order"); `tests/test_models.py:84` (`test_grid_order_is_the_supervisor_ruling`).
- Imperatives to a reader in error messages: `src/data.py:253, 326`, `tests/test_cleaning.py:20, 62`,
  `scripts/download_data.py:72` ("STOP and report", "Do not continue; report it").
- `docs/citation_checklist.md:16-21`: "were reported as added ... no file of those names
  was found on Desktop, Downloads or Documents" (conversation history and this machine's folders).
- `docs/manuscript_notes.md:3-4` is fine; `docs/data_license_request.md:3-4` addresses
  the reader ("send from your university account"), which is acceptable for a template.

**Flag: some of these strings are written into committed tables.** `src/data.py:493,
537-538` produce `results/tables/data_cleaning.md:8, 184` ("Stage 1 key", "Stage 1b key");
`scripts/check_feature_dims_banking77.py:81` produces the title of
`feature_dim_check_banking77.md` ("Task Q"); `scripts/select_feature_dims.py:160`
produces "Plateau (K3)" in `feature_dim_selection.md:15`. Changing these strings
changes no number, but it changes those tables' bytes, which conflicts with the
byte-identity requirement for Phase 2. Your decision: change the strings and
regenerate (text-only diff), or leave them.

**S7. Hard-coded values that belong in `configs/` or duplicate a config value.**
- `src/embeddings.py:31-33`: `MAX_LENGTH = 128`, `BATCH_SIZE = 64`, `N_THREADS = 6` (the last duplicates `training.threads`); the comment on line 31 hand-types measured token counts.
- `scripts/head_capacity.py:25-27`: `TOLERANCE`, `GATE_BIAS_INIT = 0.0`, `DROPOUT = 0.1`. The protocol says the gate bias "lives in the config, not in the code" (`docs/PROTOCOL.md:283-284`), but the script is its source and writes it into the config. `TOLERANCE` is duplicated in `tests/test_models.py:14`.
- `scripts/stopping_diagnostic.py:29`: `LOSS_ZERO = 0.01` duplicates `training.converged_loss_threshold`; `:78` hard-codes "< 0.01" again in the header.
- `src/analysis.py:443`: "training loss < 0.01" hard-coded in the T6 text; `:600` hard-codes the seed list "(13, 42, 87, 100, 2026)"; `:642, 660, 735` say "five seeds" instead of using `SEEDS`.
- `scripts/check_feature_dims_banking77.py:53`: `n_classes = 77` instead of reading the manifest.
- `src/train.py:62` `GATE_BINS = 20`, repeated as `20` in `scripts/smoke_test.py:32` and `np.linspace(0, 1, 21)` in `src/analysis.py:665`.
- `scripts/record_anomalies.py:24, 60`: run id and the date `"2026-10-12"` hard-coded.
- `src/models.py:29` `D_MODEL = 768` duplicates `heads.d_model`.
- The fixed design lists (`TRACKS`, `KS`, `SEEDS`) are redefined in nine files: `src/analysis.py:55-59`, `scripts/run_all.py:29-31`, `make_splits.py:48-49`, `select_lr.py:41-43`, `select_feature_dims.py:53-55`, `check_feature_dims_banking77.py:37-38`, `stopping_diagnostic.py:24-26`, `feature_report.py:27-33`, `leakage_dose.py:23`, plus `tests/test_leakage.py:13`. R4 fixes them; one definition would make that visible.

**S8. Duplicated logic.**
- sha256 helpers: `src/train.py:96`, `src/embeddings.py:41`, `scripts/verify_manifest.py:30`, `scripts/download_data.py:59`.
- Optimizer-step count: `src/analysis.py:121-123` and `scripts/record_anomalies.py:52-55`.
- Split-file loaders: `tests/conftest.py:31`, `scripts/feature_report.py:36`, `scripts/select_feature_dims.py:62`, `scripts/leakage_dose.py:28`, `src/train.py:130`.
- Mean (min-max) formatters: `scripts/stopping_diagnostic.py:32`, `feature_report.py:75`, `leakage_dose.py:40`, and the inline versions in `select_lr.py:115` and `select_feature_dims.py:170`.
- Bangla-script test: `src/data.py:552` and `tests/test_parallel.py:7-8`.
- `src/train.py:310` recomputes the condition already held in `track_gate` (`:263`).

**S9. Evidence that lives outside the committed pipeline.**
- `scripts/record_anomalies.py:28-36, 66` reads `logs/grid.log` (gitignored) and the
  Windows System event log, so `results/runs_anomalies.md` (which `src/analysis.py`
  uses to exclude a run from the timing table) can be regenerated on no other machine.
  Commit the relevant excerpt of the evidence or state this in the README.
- `docs/PROTOCOL.md:301-303`: "A sanity run of the fixed-permutation design showed the
  gate moving toward memorizable noise." No log of that run is committed. It supports a
  stated limitation, so it should be traceable or described as unlogged.
- `docs/manuscript_notes.md:14-16`: "The nine feature-dimension configurations evaluated
  in both feature selections also agree to the printed precision" cites
  `feature_dim_selection.md`, which does not contain the earlier selection; those runs
  (from uncommitted trees, `configs/default.yaml:55-56`) are not in the repository.
- `docs/manuscript_notes.md:16-18`: "Logging the per-epoch gate trajectory was likewise
  verified not to perturb training." The evidence is in the repository (I checked: all
  30 logs in `results/diagnostics/stopping/` are identical to the corresponding grid runs
  in curve, metrics and every gate statistic), but no committed script or table
  performs that comparison.

**S10. Long functions.** `src/train.py:198-374` (`_run`, 177 lines), `src/analysis.py:596-744`
(`figures`, 149 lines), `scripts/select_feature_dims.py:89-203` (`main`, 115), `src/data.py:478-560`
(`write_report`, 83), `src/analysis.py:207-281` (`registered_tests`, 75; its four test
blocks repeat the same triple loop), `src/analysis.py:748-790` (`run_all_analyses` also builds the
run-accounting table inline). Splitting them needs care not to change any output.

**S11. Missing docstrings on public functions.** 84 public module-level functions
have none (37 in `src/`, 47 in `scripts/`). The
library modules matter most: `src/analysis.py` (19, including `t1`-`t6`, `figures`,
`run_all_analyses`, `spearman`, `status`), `src/data.py` (8, including `step2_conflicts`,
`load_bnintent30_raw`, `write_report`), `src/train.py` (`gate_stats`, `macro_f1_acc`,
`library_versions`), `src/features.py` (`utterance_stats`, `word_tokens`), `src/embeddings.py`
(`cache_paths`, `texts_sha256`), `src/models.py` (`count_params`). Every script's `main`
lacks one.

**S12. Protocol layout section out of date.** `docs/PROTOCOL.md:648-701`: the
`models.py` comment lists four heads (no `concat-cap`); the `.gitignore` comment omits
the leaky manifest; `check_feature_dims_banking77.py` and `compare_lr_reproduction.py`
are missing; `docs/` lists only `PROTOCOL.md`.

**S13. Personal information in git history.** Every commit carries the author email
`abdullahfahad9277@gmail.com`. That is normal, but it becomes public if the repository
is published, and it cannot be changed without rewriting history. Decide before
publishing. No credentials, tokens or keys were found in any tracked file.

## Optional

- **O1.** `scripts/smoke_test.py:10`: `import json` is unused.
- **O2.** `src/analysis.py:57`: `K_ORDER = KS` is a redundant alias; `:373` modifies `sys.path` inside `t3` to import a script (move `dose` into `src/`).
- **O3.** Three different head orders: `src/models.py:28` (`HEAD_ORDER`, also `heads.order` in `configs/default.yaml:58-63`), `src/analysis.py:60` (`HEADS`) and `src/train.py:59` (`CONDITIONS`). Each is deliberate, but the reason is not written down.
- **O4.** `src/analysis.py:392`: variable named `l` (easily misread as `1`); `f1` is both a function in `analysis.py` and a variable name in the tests.
- **O5.** `scripts/run_all.py:51`: "§" comes out as mojibake in `logs/grid.log` on Windows (the console encoding is cp1252); seen in the clone.
- **O6.** `results/diagnostics/anomaly_rerun_<run_id>.json` (the comparison summary) and `results/diagnostics/anomaly_rerun/<run_id>.json` (the re-run's log) differ by one character in their paths; clearer names would help a reader.
- **O7.** Commit messages carry the same task and stage labels ("Task M", "Tasks N, O", "Stage 1B"). History cannot be rewritten, so a short glossary in the README would let a reader decode them.
- **O8.** `docs/PROTOCOL.md:820-828` with T2: under the convergence-anchored readout, S1 is non-significant for both BNIntent30 tracks (Holm p 0.127 and 0.295) although the direction agrees. §10's rule "if all three agree in direction ... the stopping question is closed" refers to direction only; manuscript prose should not suggest that significance also agrees.
- **O9.** `scripts/cache_embeddings.py` prints a long "UNEXPECTED keys" warning from `transformers` (the unused masked-LM head). It is harmless, but a stranger will not know that; one line in the README would help.

## Hygiene check (check 5)

- Tracked files: 993, 28.7 MB in total; the largest is `data/splits/manifest.json` (5.8 MB). No `.venv`, caches, raw data, embeddings, logs or scratch files are tracked; `.gitignore` covers them.
- No BNIntent30 text in any tracked file, run log, table or selection file, apart from M4.
- No credentials or tokens. Personal information: the author email in git history (S13); `results/runs_anomalies.md` records the times this machine slept on 2026-10-12, which is harmless.

## Fixes that would change committed outputs (to decide before Phase 2)

1. Rewording the conversational strings that are written into tables (S6, flagged
   paragraph): text-only changes to `data_cleaning.md`, `feature_dim_check_banking77.md`
   and `feature_dim_selection.md`.
2. Regenerating the stale smoke log (S5): changes `results/smoke/...json`.
3. Any change that makes the commit-stamped tables (S5) commit-independent changes
   those tables once.

Nothing else on this list touches a computed number or a generated file.

---

## Phase 2 outcome (2026-10-28, branch `pre-handover-fixes`)

**Must fix: all eight done.** M1, M2: README gives `python -m src.analysis all` and
every supporting generator; `scripts/run_all.py` creates `logs/` and writes
`logs/grid.log` itself. M3: the three protocol sentences now give the generated values
and their sources (none was inside §10, so no erratum was needed). M4: the two quoted
utterances are replaced by a description; the README states which example
utterances remain in the history (a one-word Bangla utterance in `52af2fc`; the two
English utterances in `34c29f9` to `dd08f59`, found by a full-history scan). M5: the README lists the three
§10 text edits and states that S1-S5 are unchanged. M6: four post-registration
decisions are disclosed in `docs/preregistration_disclosure.md` and in a new §11 of the
protocol. M7: README and protocol §0-§9 rewritten in protocol voice; §10 is
byte-identical. M8: drafts traced (`docs/drafts/number_trace.md`, not committed) and
citations verified (`docs/citation_checklist.md`).

**Should fix, done.** S1 (`tests/test_analysis.py`); S2 (the four tests rewritten and
each confirmed to fail on a deliberate mutation); S3 (`requirements.lock`, Python
version, long-path note); S4 (`download_data.py` and `head_capacity.py` no longer
rewrite an unchanged config); S5 documented in the README; S6 (labels removed from code
and from `data_cleaning.md`); S7 in part (converged-loss threshold, seed list, bin
count and class count now read from their sources); S9 in part
(`scripts/compare_stopping_to_grid.py` and `results/tables/stopping_vs_grid.md` now
carry the trajectory-perturbation evidence; the untraceable feature-selection claim is
withdrawn; the sanity run is described as unlogged); S11 for `src/` (37 docstrings);
S12 (layout section).

**Labels left unchanged in the three frozen selection reports** (they can only be
regenerated by re-running a frozen selection, so neither the reports nor the strings
that generate them were changed): "(Task Q)" in the title of
`results/tables/feature_dim_check_banking77.md` and "Plateau (K3)" in
`results/tables/feature_dim_selection.md`. `results/tables/lr_selection.md` contains no
such label.

**Deferred (each would need a refactor or touches evidence; none blocks handover):**
- S7 remainder: `MAX_LENGTH`, `BATCH_SIZE`, `N_THREADS` in `src/embeddings.py` not
  moved to the config (they enter the cache check); `TRACKS`/`KS`/`SEEDS` still defined
  per script; `scripts/record_anomalies.py` keeps its fixed run id and date (it records
  one historical event); the tolerance literal in `tests/test_models.py` is kept on
  purpose as an independent statement of the §2 spec.
- S8 (duplicated helpers) and S10 (long functions): not refactored.
- S9 remainder: the `logs/grid.log` and event-log evidence behind
  `results/runs_anomalies.md` is still not committed; the README says so.
- S5: the stale smoke log is documented, not regenerated (it is a committed run log).
- S11 for `scripts/`: most `main` functions still lack docstrings.
- S13: the author email in history is the author's decision.
- Optional items: only O1, O2 and O5 were done (O5 because `run_all.py` now writes its
  log in UTF-8).
