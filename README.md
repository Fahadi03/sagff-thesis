# SAGFF thesis experiments

*When Do Handcrafted Features Help? Scarcity-Aware Feature Fusion for
Low-Resource Intent Classification in English and Bangla.* Research protocol and
repository rules: [`docs/PROTOCOL.md`](docs/PROTOCOL.md) (formerly `CLAUDE.md` at the
repository root). Claude Code does not load it automatically under this name: point
any future session at it explicitly. AI-use disclosure: `docs/ai_use_disclosure.md`.

## Reproduce the data (Stage 1)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python scripts/download_data.py      # pinned commits, checksums in configs/data.yaml
.venv/Scripts/python -m src.data                   # cleaning -> data/clean/, results/tables/data_cleaning.md
.venv/Scripts/python scripts/make_splits.py        # data/splits/ (clean) and data/splits_dirty/ (leaky condition)
.venv/Scripts/python scripts/verify_manifest.py    # exits non-zero on any sha256 mismatch
.venv/Scripts/python scripts/leakage_dose.py       # results/tables/leakage_dose.md
.venv/Scripts/python scripts/feature_report.py     # results/tables/feature_overview.md
.venv/Scripts/python scripts/head_capacity.py      # results/tables/head_capacity.md
.venv/Scripts/python scripts/cache_embeddings.py   # frozen XLM-R, once (~13 min on CPU); idempotent
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python scripts/smoke_test.py         # one sagff run -> results/smoke/
.venv/Scripts/python scripts/stopping_diagnostic.py  # best vs final epoch gate -> results/tables/
nohup .venv/Scripts/python scripts/run_all.py > logs/grid.log 2>&1 &   # the grid: tests first, clean tree required
```

`scripts/select_feature_dims.py` was run once (validation only) and its result is
frozen in `configs/default.yaml`; it refuses to re-run without `--reselect`.
The same holds for `scripts/select_lr.py` (learning rate, baseline head only).

Each split directory `<track>/k<k>_seed<seed>/` holds `train.jsonl`, `val.jsonl`
(k-matched; the only split used for model selection), `test.jsonl`,
`val_full.jsonl` (diagnostic only) and `meta.json`.

## Data and licenses

| Dataset | Source (pinned) | License |
|---|---|---|
| BNIntent30 | `JahidBasher/GAN-BnBERT` @ `213bbcf` (`dataset/BNIntent30/`) | **None stated** (no license file; paper states no data license). No BNIntent30 text is committed here; only uids and checksums in the two manifests. Permission request draft: `docs/data_license_request.md`. |
| BANKING77 | `PolyAI-LDN/task-specific-datasets` @ `57ec275` (`banking_data/*.csv`) | CC-BY-4.0, per the repository's LICENSE file (verified via the GitHub license API). |

## Provenance practice

- Every selection, diagnostic and grid run refuses to start from uncommitted code
  and records the commit it ran from; files the experiments themselves write
  (`results/`, `logs/`) do not count as uncommitted code.
- A correction to a frozen artefact is made in its OWN commit, touching only what
  is corrected, so the history shows exactly what changed and what did not.
  Example: `80b01ae` fixed the git command quoted in §10 of the protocol (then `CLAUDE.md`) and nothing else,
  so the pre-specified analyses remain byte-identical to `44bd224`.
- Pre-registration: `44bd224` (analysis plan); `32b865c` (two sensitivity readouts
  added after a disclosed stopping diagnostic). See `docs/preregistration_disclosure.md`.
