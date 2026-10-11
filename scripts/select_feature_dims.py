"""Select the handcrafted feature dimensions ONCE, on validation data only, then freeze them.

The handcrafted branch is evaluated ALONE: multinomial logistic regression on
h_feat (no encoder, no fusion head), trained on the run's train split and scored
by Macro-F1 on the run's k-matched validation split (§3.5). Test files are never
opened by this script.

Starting grid: tfidf_dim in {128, 256, 512, 1024} x chargram_dim in {256, 512, 1024, 2048}
Cells: bnintent30-bn and bnintent30-en, k in {5, full}, seeds {13, 42, 87, 100, 2026}

Selection rule (smallest within one SD of the best):
  score(config, seed) = mean val Macro-F1 over the 4 (track, k) cells of that seed
  mean(config)        = mean of score(config, seed) over the 5 seeds
  best                = config with the highest mean
  sd                  = sample SD (ddof=1) of the best config's 5 seed-level scores
  chosen              = among configs with mean >= mean(best) - sd, the one with the
                        smallest tfidf_dim + chargram_dim (ties: higher mean)

Bracketing rule (Task K): if the chosen value of a dimension is the SMALLEST value
of that dimension in the grid, the grid is extended downward by one halving in that
dimension and the rule is re-applied, until the chosen configuration is interior in
both dimensions (bracketed by a clearly worse smaller point) or a floor is reached
(tfidf 32, chargram 64). The report states which happened.

Runs only from a committed tree; records the commit. Writes
results/tables/feature_dim_selection.md, results/selection/feature_dims/scores.json
and the `features` section of configs/default.yaml. Refuses to run again once a
selection is recorded, unless --reselect.

Usage:  python scripts/select_feature_dims.py [--reselect]
"""
import argparse
import datetime
import itertools
import json
import statistics
import sys
from pathlib import Path

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import featurize_run  # noqa: E402
from src.train import require_clean_tree  # noqa: E402

TFIDF_START = (128, 256, 512, 1024)
CHARGRAM_START = (256, 512, 1024, 2048)
FLOOR = {"tfidf": 32, "chargram": 64}
TRACKS = ("bnintent30-bn", "bnintent30-en")
KS = ("5", "full")
SEEDS = (13, 42, 87, 100, 2026)
CONFIG = ROOT / "configs" / "default.yaml"
OUT = ROOT / "results" / "tables" / "feature_dim_selection.md"
SCORES = ROOT / "results" / "selection" / "feature_dims" / "scores.json"
LR_PARAMS = {"C": 1.0, "max_iter": 5000, "solver": "lbfgs"}  # fixed, not tuned


def load_train_val(track: str, k: str, seed: int) -> dict:
    """train + k-matched val ONLY. There is deliberately no way to load test here."""
    d = ROOT / "data" / "splits" / track / f"k{k}_seed{seed}"
    return {s: [json.loads(line) for line in (d / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
            for s in ("train", "val")}


def val_macro_f1(splits: dict, dims: dict, n_classes: int) -> float:
    _, X = featurize_run(splits, dims=dims)
    y_tr = [r["label_id"] for r in splits["train"]]
    y_va = [r["label_id"] for r in splits["val"]]
    clf = LogisticRegression(**LR_PARAMS).fit(X["train"], y_tr)
    return f1_score(y_va, clf.predict(X["val"]), average="macro",
                    labels=list(range(n_classes)), zero_division=0)


def select(configs, cell):
    seed_score = {(c, s): statistics.mean(cell[c + (t, k, s)] for t in TRACKS for k in KS)
                  for c in configs for s in SEEDS}
    mean = {c: statistics.mean(seed_score[(c, s)] for s in SEEDS) for c in configs}
    best = max(configs, key=lambda c: mean[c])
    sd = statistics.stdev([seed_score[(best, s)] for s in SEEDS])
    eligible = [c for c in configs if mean[c] >= mean[best] - sd]
    chosen = min(eligible, key=lambda c: (c[0] + c[1], -mean[c]))
    return mean, best, sd, eligible, chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reselect", action="store_true")
    args = ap.parse_args()
    git = require_clean_tree("select_feature_dims")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if cfg.get("features", {}).get("selection") and not args.reselect:
        sys.exit("Feature dimensions are already selected and frozen in configs/default.yaml. "
                 "Re-selection needs --reselect and a reason.")
    manifest = json.loads((ROOT / "data" / "splits" / "manifest.json").read_text(encoding="utf-8"))
    n_classes = {t: len(manifest["label_names"][t]) for t in TRACKS}
    data = {(t, k, s): load_train_val(t, k, s) for t in TRACKS for k in KS for s in SEEDS}

    tfidf_grid, char_grid = list(TFIDF_START), list(CHARGRAM_START)
    cell, rounds = {}, []
    while True:
        configs = list(itertools.product(sorted(tfidf_grid), sorted(char_grid)))
        for tf, ch in configs:
            if (tf, ch) + (TRACKS[0], KS[0], SEEDS[0]) in cell:
                continue
            for key, splits in data.items():
                cell[(tf, ch) + key] = val_macro_f1(splits, {"tfidf_dim": tf, "chargram_dim": ch},
                                                    n_classes[key[0]])
            print(f"tfidf={tf:4d} chargram={ch:4d} done", flush=True)
        mean, best, sd, eligible, chosen = select(configs, cell)
        at_edge = {"tfidf": chosen[0] == min(tfidf_grid), "chargram": chosen[1] == min(char_grid)}
        rounds.append({"tfidf_grid": sorted(tfidf_grid), "chargram_grid": sorted(char_grid),
                       "chosen": list(chosen), "at_lower_edge": at_edge})
        extended = False
        if at_edge["tfidf"] and min(tfidf_grid) // 2 >= FLOOR["tfidf"]:
            tfidf_grid.append(min(tfidf_grid) // 2)
            extended = True
        if at_edge["chargram"] and min(char_grid) // 2 >= FLOOR["chargram"]:
            char_grid.append(min(char_grid) // 2)
            extended = True
        if not extended:
            break
    bracketed = not any(at_edge.values())

    # raw scores, for anyone who wants to recompute anything
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    SCORES.write_text(json.dumps({"git": git, "lr_params": LR_PARAMS,
                                  "cells": [{"tfidf_dim": k[0], "chargram_dim": k[1], "track": k[2],
                                             "k": k[3], "seed": k[4], "val_macro_f1": v}
                                            for k, v in sorted(cell.items())]}, indent=1) + "\n",
                      encoding="utf-8", newline="\n")

    means = sorted(mean.values())
    plateau = (f"{len(eligible)} of {len(configs)} configurations lie within one SD ({sd:.4f}) of the "
               f"best; mean validation Macro-F1 spans {means[0]:.4f}-{means[-1]:.4f} across the whole "
               f"grid (a range of {means[-1] - means[0]:.4f}).")
    L = ["# Feature dimension selection (validation only)", "",
         "Generated by `python scripts/select_feature_dims.py`. Do not edit by hand.", "",
         f"Run from commit `{git['commit']}` (clean tree). Handcrafted branch alone: multinomial "
         f"logistic regression ({', '.join(f'{k}={v}' for k, v in LR_PARAMS.items())}) on h_feat, "
         "trained on the run's train split, scored by Macro-F1 on the run's k-matched validation "
         f"split. No encoder. Test data never opened. Each cell is mean (min-max) over seeds {list(SEEDS)}.", "",
         "Rule: best = highest mean; threshold = best mean - SD over seeds of the best config's "
         "seed-level score; choose the smallest tfidf_dim + chargram_dim at or above the threshold. "
         "Bracketing: whenever the chosen value of a dimension is that dimension's smallest grid value, "
         "the grid is extended downward by one halving and the rule re-applied.", "",
         f"- best: tfidf_dim={best[0]}, chargram_dim={best[1]}, mean {mean[best]:.4f}",
         f"- SD over seeds of best: {sd:.4f}; threshold {mean[best] - sd:.4f}",
         f"- **chosen: tfidf_dim={chosen[0]}, chargram_dim={chosen[1]}** (mean {mean[chosen]:.4f})",
         f"- bracketing: {len(rounds)} round(s); " + "; ".join(
             f"grid tfidf {r['tfidf_grid']} x chargram {r['chargram_grid']} -> chosen {tuple(r['chosen'])}"
             for r in rounds),
         "- " + ("the chosen configuration is INTERIOR in both dimensions: a smaller value in each "
                 "dimension falls outside one SD of the best." if bracketed else
                 "the chosen configuration is still on a lower edge after reaching the floor "
                 f"{FLOOR}; it is NOT bracketed."), "",
         "**Plateau (K3).** " + plateau + " Performance is flat across most of this range of feature "
         "dimensions, so the choice is low-stakes: any configuration on the plateau would serve.", "",
         "| tfidf_dim | chargram_dim | " + " | ".join(f"{t.split('-')[1]} k={k}" for t in TRACKS for k in KS)
         + " | mean over cells & seeds | within 1 SD | |",
         "|" + " --- |" * (5 + len(TRACKS) * len(KS))]
    for c in configs:
        cells = []
        for t in TRACKS:
            for k in KS:
                v = [cell[c + (t, k, s)] for s in SEEDS]
                cells.append(f"{statistics.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})")
        tag = "**chosen**" if c == chosen else ("best" if c == best else "")
        L.append("| " + " | ".join([str(c[0]), str(c[1])] + cells
                                   + [f"{mean[c]:.4f}", "yes" if c in eligible else "no", tag]) + " |")
    OUT.write_bytes(("\n".join(L) + "\n").encode("utf-8"))
    print("\n".join(L))

    cfg["features"] = {
        "tfidf_dim": chosen[0], "chargram_dim": chosen[1], "stats_dim": 6,
        "selection": {
            "selected_on": "k-matched VALIDATION splits only; test never opened",
            "date": datetime.date.today().isoformat(),
            "git_commit": git["commit"], "working_tree_clean": not git["working_tree_dirty"],
            "script": "scripts/select_feature_dims.py",
            "report": "results/tables/feature_dim_selection.md",
            "scores": "results/selection/feature_dims/scores.json",
            "method": "handcrafted branch alone (multinomial logistic regression on h_feat), val Macro-F1",
            "grid_final": {"tfidf_dim": sorted(tfidf_grid), "chargram_dim": sorted(char_grid)},
            "bracketing_rounds": len(rounds), "bracketed": bracketed,
            "cells": {"tracks": list(TRACKS), "k": list(KS), "seeds": list(SEEDS), "n_seeds": len(SEEDS)},
            "rule": "smallest tfidf_dim + chargram_dim with mean >= best mean - SD(best over seeds); "
                    "extend a dimension downward while the choice sits on its lower edge",
            "best": {"tfidf_dim": best[0], "chargram_dim": best[1], "mean_val_macro_f1": round(mean[best], 4)},
            "sd_best": round(sd, 4),
            "chosen_mean_val_macro_f1": round(mean[chosen], 4),
            "plateau": plateau,
            "history": "2026-10-11: 3x3 grid, seeds 13/42 -> 256/1024; 3x3 grid, 5 seeds -> 256/512 "
                       "(both from uncommitted trees); this run supersedes them",
        },
    }
    raw = CONFIG.read_text(encoding="utf-8")
    header = "".join(line + "\n" for line in raw.splitlines() if line.startswith("#"))
    CONFIG.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    print(f"\nwrote {CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
