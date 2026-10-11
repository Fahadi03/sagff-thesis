"""R1/R3/R5/§3.5 enforcement (CLAUDE.md §8). Must pass before any grid run.

Every text comparison uses src.data.norm_key, the single comparison key (§3.3)."""
import hashlib
from collections import Counter, defaultdict
from itertools import combinations

import pytest

from conftest import DIRTY_DIR, FILES, SPLITS_DIR, check
from src.data import TRACKS, norm_key

KS_ORDER = ("5", "10", "20", "full")


# 1 ------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Stage 2: needs src/features.py. Will assert that a vectorizer fit on "
                         "train yields no feature index >= the train-fit vocabulary size on test.")
def test_vectorizer_indices_within_train_vocabulary():
    pass


# 2 ------------------------------------------------------------------------- #
@pytest.mark.parametrize("track", TRACKS)
def test_splits_disjoint_by_normalized_key(track, split_files):
    """train, val, test and val_full share zero normalized keys across splits, every (k, seed).
    (val is a subset of val_full, so that one pair is exempt.)"""
    problems = []
    for (t, k, seed), files in split_files.items():
        if t != track:
            continue
        keys = {f: {norm_key(r["text"]): r["uid"] for r in rows} for f, rows in files.items()}
        for a, b in combinations(FILES, 2):
            if {a, b} == {"val", "val_full"}:
                continue
            for key in sorted(keys[a].keys() & keys[b].keys()):
                problems.append(f"k={k} seed={seed}: {a} uid {keys[a][key]} and {b} uid "
                                f"{keys[b][key]} share the normalized key {key!r}")
    check(problems, f"[{track}] normalized-key overlap between splits")


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
                seen[norm_key(r["text"])].append(r["uid"])
            problems += [f"k={k} seed={seed} {f}: key {key!r} on uids {uids}"
                         for key, uids in seen.items() if len(uids) > 1]
    check(problems, f"[{track}] duplicate normalized keys inside a split")


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
@pytest.mark.skip(reason="Stage 2: needs src/features.py and src/train.py. Will assert the fitted "
                         "feature pipeline used at test time IS the train-time object (identity "
                         "check, not a refit).")
def test_test_time_pipeline_is_train_fitted_object():
    pass
