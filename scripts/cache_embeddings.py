"""Build the frozen-encoder embedding caches, one per track (CLAUDE.md §6, R1b).

See src/embeddings.py for why encoding val and test with a frozen encoder is
NOT a violation of R1: no parameter is estimated from this dataset here.

Each cache covers every uid referenced by any manifest for that track (clean
and, for bnintent30-bn, the dirty condition), so every run can look its rows
up by uid. Writes data/cache/<track>_xlm-roberta-base_mean.npy plus a sidecar
JSON (uid order, encoder, revision, pooling, max_length, truncation count,
sha256). Idempotent: if the cache and sidecar exist and match, nothing is done.

Usage:  python scripts/cache_embeddings.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import TRACKS, build_dirty_tracks, build_tracks  # noqa: E402
from src.embeddings import (ENCODER, MAX_LENGTH, POOLING, REVISION, cache_paths,  # noqa: E402
                            encode, sha256_file, texts_sha256)


def rows_per_track() -> dict:
    """{track: (uids, texts)} - the union of clean and dirty rows, sorted by uid."""
    tracks, _ = build_tracks()
    rows = {t: {r["uid"]: r["text"] for r in tracks[t]["rows"]} for t in TRACKS}
    for t, d in build_dirty_tracks().items():
        for r in d["rows"]:
            if rows[t].setdefault(r["uid"], r["text"]) != r["text"]:
                raise ValueError(f"{t} uid {r['uid']} has different text in clean and dirty builds")
    return {t: (sorted(m), [m[u] for u in sorted(m)]) for t, m in rows.items()}


def up_to_date(track: str, uids: list[str], texts: list[str]) -> bool:
    npy, sidecar = cache_paths(track)
    if not (npy.exists() and sidecar.exists()):
        return False
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    return (meta.get("encoder") == ENCODER and meta.get("revision") == REVISION
            and meta.get("pooling") == POOLING and meta.get("max_length") == MAX_LENGTH
            and meta.get("uids") == uids and meta.get("texts_sha256") == texts_sha256(uids, texts)
            and meta.get("sha256") == sha256_file(npy))


def main() -> None:
    for track, (uids, texts) in rows_per_track().items():
        npy, sidecar = cache_paths(track)
        if up_to_date(track, uids, texts):
            print(f"{track}: cache up to date ({npy.name}), nothing to do")
            continue
        t0 = time.time()
        X, truncated, env = encode(texts)
        wall = time.time() - t0
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, X)
        meta = {
            "track": track, "encoder": ENCODER, "revision": REVISION, "pooling": POOLING,
            "max_length": MAX_LENGTH, "n": len(uids), "dim": int(X.shape[1]),
            "truncated": truncated, "texts_sha256": texts_sha256(uids, texts),
            "sha256": sha256_file(npy), "wall_clock_s": round(wall, 1), **env,
            "frozen": True, "fit_on_dataset": False,
            "note": "R1b: frozen encoder, no parameter estimated from this dataset",
            "uids": uids,
        }
        sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8", newline="\n")
        print(f"{track}: shape {X.shape}, truncated {truncated}, max tokens {env['max_tokens_seen']}, "
              f"{wall:.1f}s, threads {env['threads']}")


if __name__ == "__main__":
    main()
