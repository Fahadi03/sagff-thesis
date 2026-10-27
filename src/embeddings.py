"""Frozen-encoder sentence embeddings: compute once, cache, load by uid (docs/PROTOCOL.md §6).

R1b - WHY ENCODING VAL AND TEST HERE IS NOT LEAKAGE.
The encoder is FROZEN: its weights come from pretraining and are never updated,
and no parameter of any kind is estimated from this dataset at this step.
Embedding every utterance (train, val and test) once is therefore a fixed,
data-independent function applied row by row - exactly like tokenisation.
Contrast the handcrafted features (src/features.py): TF-IDF vocabulary, chi2
selection, idf weights and the stats scaler ARE estimated from data, so they
are fit per run on that run's train split only (R1). The paper states this
distinction in its methodology section.

Pooling: MEAN over tokens using the attention mask (special tokens <s>, </s>
included, padding excluded) - not the <s>/CLS vector, which is only a good
sentence representation after fine-tuning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
ENCODER = "FacebookAI/xlm-roberta-base"
ENCODER_SHORT = "xlm-roberta-base"
REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"  # resolved once from the HF hub, pinned
POOLING = "mean"
MAX_LENGTH = 128   # measured longest utterance: 101 tokens (banking77); 42 bn, 32 en
BATCH_SIZE = 64
N_THREADS = 6      # set explicitly and logged (§9)


def cache_paths(track: str) -> tuple[Path, Path]:
    """(matrix, sidecar) paths of a track's embedding cache."""
    stem = f"{track}_{ENCODER_SHORT}_{POOLING}"
    return CACHE / f"{stem}.npy", CACHE / f"{stem}.json"


def sha256_file(p: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def texts_sha256(uids: list[str], texts: list[str]) -> str:
    """Fingerprint of the (uid, text) pairs a cache was built from."""
    return hashlib.sha256("\x1e".join(f"{u}\x1f{t}" for u, t in zip(uids, texts)).encode("utf-8")).hexdigest()


def encode(texts: list[str]) -> tuple[np.ndarray, int, dict]:
    """Mean-pooled frozen XLM-R embeddings, (n, 768) float32; also returns the
    truncation count and environment info."""
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(N_THREADS)
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION)
    model = AutoModel.from_pretrained(ENCODER, revision=REVISION)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    full_lengths = [len(ids) for ids in tok(texts)["input_ids"]]
    truncated = sum(n > MAX_LENGTH for n in full_lengths)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = tok(texts[i:i + BATCH_SIZE], padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
            hidden = model(**batch).last_hidden_state                # (b, t, 768)
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy().astype(np.float32))
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "threads": torch.get_num_threads(), "max_tokens_seen": max(full_lengths)}
    return np.concatenate(out), truncated, env


def load_embeddings(track: str, uids: list[str]) -> np.ndarray:
    """Rows of the cached matrix for `uids`, in that order - looked up BY UID,
    never by position. Verifies the cache file against its sidecar sha256."""
    npy, sidecar = cache_paths(track)
    if not npy.exists() or not sidecar.exists():
        raise FileNotFoundError(f"No embedding cache for {track}. Run: python scripts/cache_embeddings.py")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if sha256_file(npy) != meta["sha256"]:
        raise RuntimeError(f"{npy.name} does not match the sha256 in its sidecar; rebuild the cache.")
    index = {u: i for i, u in enumerate(meta["uids"])}
    missing = [u for u in uids if u not in index]
    if missing:
        raise KeyError(f"{len(missing)} uids not in the {track} cache, e.g. {missing[:3]}")
    return np.load(npy)[[index[u] for u in uids]]
