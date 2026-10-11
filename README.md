# SAGFF thesis experiments

*When Do Handcrafted Features Help? Scarcity-Aware Feature Fusion for
Low-Resource Intent Classification in English and Bangla.* Rules: see `CLAUDE.md`.

## Reproduce the data (Stage 1)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python scripts/download_data.py      # pinned commits, checksums in configs/data.yaml
.venv/Scripts/python -m src.data                   # cleaning -> data/clean/, results/tables/data_cleaning.md
.venv/Scripts/python scripts/make_splits.py        # data/splits/ (clean) and data/splits_dirty/ (leaky condition)
.venv/Scripts/python scripts/verify_manifest.py    # exits non-zero on any sha256 mismatch
.venv/Scripts/python -m pytest tests/ -v
```

Each split directory `<track>/k<k>_seed<seed>/` holds `train.jsonl`, `val.jsonl`
(k-matched; the only split used for model selection), `test.jsonl`,
`val_full.jsonl` (diagnostic only) and `meta.json`.

## Data and licenses

| Dataset | Source (pinned) | License |
|---|---|---|
| BNIntent30 | `JahidBasher/GAN-BnBERT` @ `213bbcf` (`dataset/BNIntent30/`) | **None stated** (no license file; paper states no data license). No BNIntent30 text is committed here; only uids and checksums in the two manifests. Permission request draft: `docs/data_license_request.md`. |
| BANKING77 | `PolyAI-LDN/task-specific-datasets` @ `57ec275` (`banking_data/*.csv`) | CC-BY-4.0, per the repository's LICENSE file (verified via the GitHub license API). |
