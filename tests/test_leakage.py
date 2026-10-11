"""R1/R3/R5/§3.5 enforcement (CLAUDE.md §8). Must pass before any grid run.

Every text comparison uses src.data.dedup_key, the single deduplication key (§3.3)."""
import hashlib
from collections import Counter, defaultdict
from itertools import combinations

import pytest

from conftest import DIRTY_DIR, FILES, SPLITS_DIR, check
from src.data import TRACKS, dedup_key

KS_ORDER = ("5", "10", "20", "full")


# 1 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("track", TRACKS)
def test_vectorizer_indices_within_train_vocabulary(track, split_files):
    """A pipeline fit on train, applied to val/test, never produces a TF-IDF column outside
    the train-fit (selected) vocabulary; padding columns stay zero; the vocabulary does
    not grow at transform time; every split has the fixed total dimension."""
    from src.features import featurize_run
    problems = []
    for (t, k, seed), files in split_files.items():
        if t != track:
            continue
        pipeline = None
        try:
            pipeline, X = featurize_run(files)
        except Exception as e:  # report, don't hide
            problems.append(f"k={k} seed={seed}: featurize_run raised {e!r}")
            continue
        vocab_before = len(pipeline._tfidf.vocabulary_)
        n_sel = pipeline.tfidf_selected
        for name, M in X.items():
            if M.shape[1] != pipeline.total_dim:
                problems.append(f"k={k} seed={seed} {name}: dim {M.shape[1]} != {pipeline.total_dim}")
            cols = M[:, :pipeline.tfidf_dim].tocoo().col
            if cols.size and cols.max() >= n_sel:
                problems.append(f"k={k} seed={seed} {name}: TF-IDF column {cols.max()} >= "
                                f"train-fit vocabulary size {n_sel}")
        if len(pipeline._tfidf.vocabulary_) != vocab_before:
            problems.append(f"k={k} seed={seed}: vocabulary grew during transform")
    check(problems, f"[{track}] feature indices outside the train-fit vocabulary")


def test_pipeline_refuses_misuse():
    """R1 structurally: no transform before fit, no fit on a non-train split, no refit."""
    from src.features import (FeaturesAlreadyFittedError, FeaturesNotFittedError,
                              HandcraftedFeatures)
    with pytest.raises(FeaturesNotFittedError):
        HandcraftedFeatures().transform(["x"])
    for bad in ("test", "val", "val_full", "train+val"):
        with pytest.raises(ValueError, match="TRAIN split only"):
            HandcraftedFeatures().fit(["a b"], [0], split=bad)
    p = HandcraftedFeatures().fit(["a b", "c d"], [0, 1], split="train")
    with pytest.raises(FeaturesAlreadyFittedError):
        p.fit(["e f", "g h"], [0, 1], split="train")


# 2 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("track", TRACKS)
def test_splits_disjoint_by_normalized_key(track, split_files):
    """train, val, test and val_full share zero normalized keys across splits, every (k, seed).
    (val is a subset of val_full, so that one pair is exempt.)"""
    problems = []
    for (t, k, seed), files in split_files.items():
        if t != track:
            continue
        keys = {f: {dedup_key(r["text"]): r["uid"] for r in rows} for f, rows in files.items()}
        for a, b in combinations(FILES, 2):
            if {a, b} == {"val", "val_full"}:
                continue
            for key in sorted(keys[a].keys() & keys[b].keys()):
                problems.append(f"k={k} seed={seed}: {a} uid {keys[a][key]} and {b} uid "
                                f"{keys[b][key]} share the dedup key {key!r}")
    check(problems, f"[{track}] dedup-key overlap between splits")


# 2b ------------------------------------------------------------------------ #
@pytest.mark.parametrize("track", TRACKS)
def test_no_normalized_key_twice_within_a_split(track, split_files):
    """§3.3 step 4: inside any split file, every normalized key occurs at most once."""
    problems = []
    for (t, k, seed), files in split_files.items():
        if t != track:
            continue
        for f, rows in files.items():
            seen = defaultdict(list)
            for r in rows:
                seen[dedup_key(r["text"])].append(r["uid"])
            problems += [f"k={k} seed={seed} {f}: key {key!r} on uids {uids}"
                         for key, uids in seen.items() if len(uids) > 1]
    check(problems, f"[{track}] duplicate dedup keys inside a split")


# 3 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("track", TRACKS)
def test_kshot_counts_and_full_way_test(track, split_files, manifest):
    """train and val (k-matched, §3.5) hold exactly min(k, available) per class, so never
    more than k; test holds ALL classes of the track (R5)."""
    n_classes = len(manifest["label_names"][track])
    problems = []
    full_train = {tuple(sorted(r["uid"] for r in f["train"]))
                  for (t, k, s), f in split_files.items() if t == track and k == "full"}
    if len(full_train) != 1:
        problems.append("k=full train differs in content between seeds")
    for (t, k, seed), files in split_files.items():
        if t != track:
            continue
        available = {"train": Counter(r["label_id"] for r in split_files[(t, "full", seed)]["train"]),
                     "val": Counter(r["label_id"] for r in files["val_full"])}
        if k != "full":
            for f in ("train", "val"):
                per_class = Counter(r["label_id"] for r in files[f])
                for lid in range(n_classes):
                    want = min(int(k), available[f][lid])
                    if per_class[lid] != want:
                        problems.append(f"k={k} seed={seed} {f}: class {lid} has "
                                        f"{per_class[lid]} rows (expected {want}, max {k})")
        elif [r["uid"] for r in files["val"]] != [r["uid"] for r in files["val_full"]]:
            problems.append(f"k=full seed={seed}: val is not the full val portion")
        missing = set(range(n_classes)) - {r["label_id"] for r in files["test"]}
        if missing:
            problems.append(f"k={k} seed={seed}: test is missing classes {sorted(missing)}")
    check(problems, f"[{track}] k-shot / k-matched val / full-way violations")


# 3b ------------------------------------------------------------------------ #
@pytest.mark.parametrize("condition", ["clean", "dirty"])
def test_kshot_sets_are_nested(condition, split_files, dirty_files):
    """R3 requirement: for each (track, seed), train and val at k=5 ⊆ 10 ⊆ 20 ⊆ full."""
    files = split_files if condition == "clean" else dirty_files
    problems = []
    for track, seed in sorted({(t, s) for (t, k, s) in files}):
        for f in ("train", "val"):
            sets = [{r["uid"] for r in files[(track, k, seed)][f]} for k in KS_ORDER]
            for (ka, a), (kb, b) in zip(zip(KS_ORDER, sets), zip(KS_ORDER[1:], sets[1:])):
                if not a <= b:
                    problems.append(f"{track} seed={seed} {f}: k={ka} is not a subset of k={kb} "
                                    f"({len(a - b)} uids missing)")
    check(problems, f"[{condition}] k-sets not nested")


# 4 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("condition", ["clean", "dirty"])
def test_make_splits_reproduces_manifest(condition, tmp_path, manifest, dirty_manifest):
    """Re-running make_splits gives byte-identical files matching the manifest sha256."""
    from scripts.make_splits import build_condition_tracks, compare, generate
    committed = manifest if condition == "clean" else dirty_manifest
    on_disk_base = SPLITS_DIR if condition == "clean" else DIRTY_DIR
    regenerated = generate(tmp_path, build_condition_tracks(condition), condition)
    problems = compare(regenerated, committed)
    for e in regenerated["splits"]:
        for f in FILES:
            new = (tmp_path / e["path"] / f"{f}.jsonl").read_bytes()
            if new != (on_disk_base / e["path"] / f"{f}.jsonl").read_bytes():
                problems.append(f"{e['path']}/{f}.jsonl: regenerated bytes differ from disk")
            if hashlib.sha256(new).hexdigest() != e["sha256"][f]:
                problems.append(f"{e['path']}/{f}.jsonl: sha256 does not match its own record")
    check(problems, f"[{condition}] make_splits.py is not reproducible")


# 5 ------------------------------------------------------------------------- #
def test_test_time_pipeline_is_train_fitted_object(split_files, monkeypatch):
    """Identity, not a refit: within one run, fit is called exactly once and every split,
    test included, is transformed by that same fitted object, whose fitted state the
    transforms do not change."""
    from src.features import HandcraftedFeatures, _fingerprint, featurize_run
    calls = []
    original_fit = HandcraftedFeatures.fit

    def counting_fit(self, *a, **kw):
        calls.append(id(self))
        return original_fit(self, *a, **kw)

    monkeypatch.setattr(HandcraftedFeatures, "fit", counting_fit)
    problems = []
    for key in [(t, k, 13) for t in TRACKS for k in KS_ORDER]:
        calls.clear()
        pipeline, _ = featurize_run(split_files[key])
        logged = {name: v["transformed_by"] for name, v in pipeline.transform_log.items()}
        if calls != [id(pipeline)]:
            problems.append(f"{key}: fit called {len(calls)} time(s) (expected once, on the "
                            "returned object)")
        if pipeline.n_fits != 1:
            problems.append(f"{key}: n_fits={pipeline.n_fits}")
        if set(logged) != set(split_files[key]) or set(logged.values()) != {id(pipeline)}:
            problems.append(f"{key}: a split was transformed by a different object: {logged}")
        snapshot = (tuple(pipeline.selected_terms), pipeline._scaler.mean_.tobytes(),
                    pipeline._char_idf.idf_.tobytes())
        pipeline.transform([r["text"] for r in split_files[key]["test"]])
        if snapshot != (tuple(pipeline.selected_terms), pipeline._scaler.mean_.tobytes(),
                        pipeline._char_idf.idf_.tobytes()):
            problems.append(f"{key}: fitted state changed when transforming test")
        if pipeline.train_fingerprint != _fingerprint([r["text"] for r in split_files[key]["train"]]):
            problems.append(f"{key}: pipeline was not fit on exactly the train split")
    check(problems, "test-time feature pipeline is not the train-fitted object")
