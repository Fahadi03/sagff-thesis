"""One complete run -> exactly one JSON log (CLAUDE.md R6, R7).

    load split (by manifest; sha256 verified) -> fit handcrafted features on TRAIN
    only (featurize_run) -> load frozen embeddings BY UID -> build head ->
    gate statistics at initialisation (test split) -> train with early stopping
    on the k-matched VALIDATION split -> gate statistics at the FINAL epoch ->
    restore best epoch -> evaluate on test -> gate statistics at the best epoch ->
    one JSON, even on failure.

Gate readouts for sagff-head runs (all on the TEST split, none used for selection):
  gate_init_*       at initialisation
  gate_*            at the epoch selected by early stopping   (primary, R7)
  gate_final_*      at the final trained epoch                (sensitivity)
  gate_converged_*  at the first epoch whose mean training loss falls below
                    training.converged_loss_threshold         (sensitivity; deterministic,
                                                               free of validation noise)
  curve[*].gate_mean / gate_std   at every epoch (the trajectory; scalars only)

Per-epoch control, validation quirk (also a manuscript footnote): during training
the k-matched validation split of `sagff-noise-perepoch` is scored under that
epoch's own permutation, which is what early stopping sees; the validation metric
logged after training is recomputed under the fixed evaluation permutation, so it
need not equal the value that drove selection. Both are noise by construction.

Run conditions, in grid order (§5, CONDITIONS): sagff, baseline-cap, concat-cap,
sagff-noise-perepoch, sagff-permuted-fixed, baseline, concat. The two controls
use the sagff head with real, train-fitted features whose utterance -> vector
mapping is permuted (every marginal property of the branch is kept):
  sagff-noise-perepoch   re-permuted every epoch from (seed, epoch): carries no
                         stable content, cannot be memorized. Tests whether the
                         gate can reject a worthless branch.
  sagff-permuted-fixed   one fixed permutation per split: a memorizable,
                         non-generalizing identifier. Tests whether the gate is
                         fooled by exploitability. Its failure is a result.

Hyperparameters come from configs/default.yaml and are identical for every head
and regime (R2). The learning rate was selected once, on validation, with the
baseline head only (scripts/select_lr.py).
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
SPLIT_BASES = {False: ROOT / "data" / "splits", True: ROOT / "data" / "splits_dirty"}
CONDITIONS = ("sagff", "baseline-cap", "concat-cap", "sagff-noise-perepoch",
              "sagff-permuted-fixed", "baseline", "concat")
PERMUTATION_MODE = {"sagff-noise-perepoch": "per-epoch", "sagff-permuted-fixed": "fixed"}
GATE_BINS = 20


@dataclass
class RunSpec:
    track: str
    k: str
    seed: int
    condition: str
    dirty: bool = False
    lr: float | None = None              # None -> configs/default.yaml training.lr
    evaluate_test: bool = True           # False for selection runs: test is never loaded
    log_dir: Path = field(default_factory=lambda: ROOT / "results" / "runs")
    tag: str = ""                        # extra run_id suffix (selection runs)

    @property
    def head(self) -> str:
        return "sagff" if self.condition in PERMUTATION_MODE else self.condition

    @property
    def permutation_mode(self) -> str | None:
        return PERMUTATION_MODE.get(self.condition)

    @property
    def permuted(self) -> bool:
        return self.permutation_mode is not None

    @property
    def run_id(self) -> str:
        rid = f"{self.track}__k{self.k}__seed{self.seed}__{self.condition}"
        return rid + ("__dirty" if self.dirty else "") + (f"__{self.tag}" if self.tag else "")


# --------------------------------------------------------------------------- #
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_state() -> dict:
    """Commit and cleanliness of the CODE. Files the experiments themselves write
    (results/, logs/) do not count as dirt, otherwise every run after the first
    would be recorded as uncommitted."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    changed = [line[3:] for line in git("status", "--porcelain").splitlines()]
    dirt = [path for path in changed if not path.startswith(("results/", "logs/"))]
    return {"commit": git("rev-parse", "HEAD").strip(), "working_tree_dirty": bool(dirt),
            "dirty_paths": dirt[:20]}


def require_clean_tree(what: str) -> dict:
    """Refuse to run a selection or grid from uncommitted code (every number must
    trace to a commit). Returns the git state to record as provenance."""
    state = git_state()
    if state["working_tree_dirty"]:
        raise SystemExit(f"{what}: the working tree has uncommitted changes {state['dirty_paths']}. "
                         "Commit first, so every number traces to a commit.")
    return state


def library_versions() -> dict:
    import scipy
    import sklearn
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__}


def load_split(spec: RunSpec) -> tuple[dict, dict, list[str]]:
    """Read the split files listed in the manifest and verify their sha256.
    Test (and its sha) is only touched when spec.evaluate_test is True."""
    base = SPLIT_BASES[spec.dirty]
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["splits"]
                 if e["track"] == spec.track and e["k"] == spec.k and e["seed"] == spec.seed)
    names = ["train", "val", "val_full"] + (["test"] if spec.evaluate_test else [])
    splits, sha = {}, {}
    for n in names:
        raw = (base / entry["path"] / f"{n}.jsonl").read_bytes()
        if _sha256(raw) != entry["sha256"][n]:
            raise RuntimeError(f"{entry['path']}/{n}.jsonl does not match the manifest sha256")
        splits[n] = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        sha[n] = entry["sha256"][n]
    return splits, sha, manifest["label_names"][spec.track]


def permutation(seed: int, split: str, uids: list[str], epoch: int | None = None) -> np.ndarray:
    """RNG-free, deterministic permutation of rows, keyed on (seed, split, uid) and,
    for the per-epoch control, the epoch."""
    tag = split if epoch is None else f"{split}\x1fepoch{epoch}"
    return np.array(sorted(range(len(uids)),
                           key=lambda i: _sha256(f"{seed}\x1fpermute\x1f{tag}\x1f{uids[i]}".encode())))


def macro_f1_acc(y_true, y_pred, n_classes) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)),
                                       zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred))}


def gate_stats(gates: np.ndarray, prefix: str) -> dict:
    hist, _ = np.histogram(gates, bins=GATE_BINS, range=(0.0, 1.0))
    return {f"{prefix}mean": round(float(gates.mean()), 6),
            f"{prefix}std": round(float(gates.std()), 6),
            f"{prefix}mean_per_dim": [round(float(v), 6) for v in gates.mean(axis=0)],
            f"{prefix}hist": [int(v) for v in hist]}


# --------------------------------------------------------------------------- #
def run(spec: RunSpec) -> dict:
    """Execute one run and write exactly one JSON. Refuses to overwrite."""
    out = Path(spec.log_dir) / f"{spec.run_id}.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; run files are never overwritten (R6)")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log_dir = Path(spec.log_dir).resolve()
    log = {"run_id": spec.run_id, "status": "running",
           "spec": {**asdict(spec), "head": spec.head,
                    "log_dir": (log_dir.relative_to(ROOT) if log_dir.is_relative_to(ROOT)
                                else log_dir).as_posix()},
           "dirty": spec.dirty, "permuted": spec.permuted,
           "permutation_mode": spec.permutation_mode, "git": git_state()}
    try:
        log.update(_run(spec))
        log["status"] = "ok"
    except Exception as e:  # R6: a failed run still writes its JSON
        log["status"] = "failed"
        log["error"] = repr(e)
        log["traceback"] = traceback.format_exc()
    log["wall_clock_s"] = round(time.time() - t0, 2)
    out.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return log


def _run(spec: RunSpec) -> dict:
    import torch
    from torch import nn

    from src.embeddings import ENCODER, POOLING, REVISION, cache_paths, load_embeddings
    from src.features import featurize_run
    from src.models import build_head, count_params

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tr, heads_cfg = cfg["training"], cfg["heads"]
    lr = float(spec.lr if spec.lr is not None else tr["lr"])
    if spec.condition not in CONDITIONS:
        raise ValueError(f"unknown condition {spec.condition!r}")

    # determinism (§9)
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(tr["threads"]))

    splits, split_sha, label_names = load_split(spec)
    n_classes = len(label_names)

    # handcrafted features: fit on TRAIN only, same object transforms every split (R1)
    pipeline, X = featurize_run(splits)
    feat_report = pipeline.report()
    uids = {n: [r["uid"] for r in rows] for n, rows in splits.items()}
    perm_info = None
    if spec.permutation_mode is not None:
        # Both controls: the evaluation-time mapping (val/val_full/test, and train
        # for the fixed control) is one fixed permutation per split.
        perm_info = {"mode": spec.permutation_mode}
        for name in splits:
            p = permutation(spec.seed, name, uids[name])
            X[name] = X[name][p]
            perm_info[name] = {"n": len(p), "fixed_points": int((p == np.arange(len(p))).sum())}

    # frozen embeddings, looked up BY UID
    E = {n: load_embeddings(spec.track, [r["uid"] for r in rows]) for n, rows in splits.items()}
    sidecar = json.loads(cache_paths(spec.track)[1].read_text(encoding="utf-8"))
    y = {n: np.array([r["label_id"] for r in rows]) for n, rows in splits.items()}
    T = {n: (torch.from_numpy(E[n]), torch.from_numpy(X[n].toarray().astype(np.float32)),
             torch.from_numpy(y[n])) for n in splits}

    def epoch_features(name: str, epoch: int):
        """Features seen during training epoch `epoch`. Only the per-epoch control
        re-permutes (from seed, epoch, split); otherwise the split's fixed tensor."""
        f = T[name][1]
        if spec.permutation_mode != "per-epoch":
            return f
        return f[torch.from_numpy(permutation(spec.seed, name, uids[name], epoch=epoch))]

    torch.manual_seed(spec.seed)  # head init depends only on the seed
    model = build_head(spec.head, n_classes, heads_cfg, float(heads_cfg["dropout"]))
    n_params = count_params(model)

    def predict(name, features=None):
        model.eval()
        e, f, _ = T[name]
        with torch.no_grad():
            logits, aux = model(e, f if features is None else features)
        return logits.argmax(-1).numpy(), aux.get("gate")

    gate_log = {}
    track_gate = spec.head == "sagff" and spec.evaluate_test
    if track_gate:
        _, g0 = predict("test")
        gate_log.update(gate_stats(g0.numpy(), "gate_init_"))
    converged_threshold = float(tr["converged_loss_threshold"])
    converged_epoch = None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tr["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(spec.seed)
    e_tr, _, y_tr = T["train"]
    best = {"val_macro_f1": -1.0, "epoch": -1, "state": None}
    curve, patience_left = [], int(tr["patience"])
    for epoch in range(1, int(tr["max_epochs"]) + 1):
        model.train()
        f_tr = epoch_features("train", epoch)
        order = torch.randperm(len(y_tr), generator=gen)
        total = 0.0
        for i in range(0, len(order), int(tr["batch_size"])):
            idx = order[i:i + int(tr["batch_size"])]
            opt.zero_grad()
            loss = loss_fn(model(e_tr[idx], f_tr[idx])[0], y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
            opt.step()
            total += float(loss.detach()) * len(idx)
        val = macro_f1_acc(y["val"], predict("val", epoch_features("val", epoch))[0], n_classes)["macro_f1"]
        train_loss = total / len(y_tr)
        point = {"epoch": epoch, "train_loss": round(train_loss, 6), "val_macro_f1": round(val, 6)}
        if track_gate:
            # N: gate trajectory (scalars only); never used for selection
            g_ep = predict("test")[1].numpy()
            point["gate_mean"] = round(float(g_ep.mean()), 6)
            point["gate_std"] = round(float(g_ep.std()), 6)
            # O: convergence-anchored readout, first epoch with loss below the threshold
            if converged_epoch is None and train_loss < converged_threshold:
                converged_epoch = epoch
                gate_log.update(gate_stats(g_ep, "gate_converged_"))
        curve.append(point)
        if val > best["val_macro_f1"]:           # strict improvement; ties keep the earlier epoch
            best = {"val_macro_f1": val, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            patience_left = int(tr["patience"])
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    final_epoch = len(curve)
    if spec.head == "sagff" and spec.evaluate_test:
        # M: gate at the FINAL trained epoch, before restoring the selected one
        _, g_final = predict("test")
        gate_log.update(gate_stats(g_final.numpy(), "gate_final_"))
    model.load_state_dict(best["state"])

    metrics = {"val": {**macro_f1_acc(y["val"], predict("val")[0], n_classes),
                       "used_for_selection": True, "n": int(len(y["val"]))},
               "val_full_diagnostic": {**macro_f1_acc(y["val_full"], predict("val_full")[0], n_classes),
                                       "used_for_selection": False, "n": int(len(y["val_full"]))}}
    if spec.evaluate_test:
        pred, g1 = predict("test")
        metrics["test"] = {**macro_f1_acc(y["test"], pred, n_classes), "n": int(len(y["test"]))}
        if spec.head == "sagff":
            gate_log.update(gate_stats(g1.numpy(), "gate_"))
            init_dim = np.array(gate_log["gate_init_mean_per_dim"])
            gate_log["gate_displacement"] = {   # derived: what the paper analyses (G2)
                "mean": round(gate_log["gate_mean"] - gate_log["gate_init_mean"], 6),
                "mean_abs_per_dim": round(float(np.mean(np.abs(
                    np.array(gate_log["gate_mean_per_dim"]) - init_dim))), 6),
            }
            gate_log["gate_displacement_final_epoch"] = {   # M: stopping-point sensitivity
                "epoch": final_epoch,
                "mean": round(gate_log["gate_final_mean"] - gate_log["gate_init_mean"], 6),
                "mean_abs_per_dim": round(float(np.mean(np.abs(
                    np.array(gate_log["gate_final_mean_per_dim"]) - init_dim))), 6),
                "best_minus_final_gate_mean": round(gate_log["gate_mean"] - gate_log["gate_final_mean"], 6),
            }
            # O: convergence-anchored sensitivity readout (None if loss never fell below the threshold)
            gate_log["gate_displacement_converged"] = None if converged_epoch is None else {
                "epoch": converged_epoch, "loss_threshold": converged_threshold,
                "mean": round(gate_log["gate_converged_mean"] - gate_log["gate_init_mean"], 6),
                "mean_abs_per_dim": round(float(np.mean(np.abs(
                    np.array(gate_log["gate_converged_mean_per_dim"]) - init_dim))), 6),
            }

    return {
        "config": {"track": spec.track, "k": spec.k, "seed": spec.seed, "condition": spec.condition,
                   "head": spec.head, "dirty": spec.dirty, "permuted": spec.permuted,
                   "permutation_mode": spec.permutation_mode,
                   "training": {**tr, "lr": lr}, "heads": {kk: heads_cfg[kk] for kk in
                   ("d_model", "feat_dim", "dropout", "gate_bias_init", "baseline_cap_hidden",
                    "concat_cap_hidden")},
                   "features": {kk: cfg["features"][kk] for kk in ("tfidf_dim", "chargram_dim", "stats_dim")}},
        "libraries": library_versions(),
        "determinism": {"torch_deterministic_algorithms": True, "threads": torch.get_num_threads(),
                        "seeded": ["random", "numpy", "torch", "shuffle generator"]},
        "encoder": {"name": ENCODER, "revision": REVISION, "pooling": POOLING,
                    "max_length": sidecar["max_length"], "cache_sha256": sidecar["sha256"]},
        "splits": {"sha256": split_sha, "sizes": {n: len(r) for n, r in splits.items()},
                   "label_names": label_names},
        "features": {"dims": feat_report["dims"], "tfidf_vocab_size": feat_report["tfidf_vocab_size"],
                     "tfidf_selected": feat_report["tfidf_selected"],
                     "tfidf_padded_columns": feat_report["tfidf_padded_columns"],
                     "degenerate_stats": feat_report["degenerate_stats"],
                     "transform_log": feat_report["transform_log"],
                     "all_zero_tfidf_test_rows": feat_report["transform_log"].get("test", {}).get("tfidf_empty_rows"),
                     "fit": feat_report["fit"], "used_by_head": bool(model.uses_features),
                     "permutation": perm_info},
        "head": {"name": spec.head, "param_count": n_params},
        "training_log": {"best_epoch": best["epoch"], "epochs_run": len(curve),
                         "stopped_early": len(curve) < int(tr["max_epochs"]), "curve": curve},
        "metrics": metrics,
        "gate": gate_log or None,
    }
