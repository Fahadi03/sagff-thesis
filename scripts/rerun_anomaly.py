"""Re-run an anomalous run's spec into a scratch directory and compare it, bit for
bit, with the stored run (results/runs/). Training is deterministic, so every
training-produced field must be identical; if it is, the anomaly (e.g. a machine
sleep inside the run) affected nothing but wall time.

Compared with exact equality: the full per-epoch curve (loss, val Macro-F1, gate
mean/std), best and final epochs, all metrics, all gate statistics, the features
report. Writes results/diagnostics/anomaly_rerun_<run_id>.json; the re-run's own
log goes to results/diagnostics/anomaly_rerun/ (never results/runs/).

Usage:  python scripts/rerun_anomaly.py banking77 full 2026 sagff-noise-perepoch
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import RunSpec, require_clean_tree, run  # noqa: E402

OUT_DIR = ROOT / "results" / "diagnostics"


def main() -> None:
    track, k, seed, condition = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    git = require_clean_tree("rerun_anomaly")
    spec = RunSpec(track=track, k=k, seed=seed, condition=condition,
                   log_dir=OUT_DIR / "anomaly_rerun")
    stored = json.loads((ROOT / "results" / "runs" / f"{spec.run_id}.json").read_text(encoding="utf-8"))
    path = Path(spec.log_dir) / f"{spec.run_id}.json"
    new = json.loads(path.read_text(encoding="utf-8")) if path.exists() else run(spec)
    checks = {
        "curve (loss, val Macro-F1, per-epoch gate mean/std), every epoch":
            new["training_log"]["curve"] == stored["training_log"]["curve"],
        "best epoch and epochs run":
            (new["training_log"]["best_epoch"], new["training_log"]["epochs_run"])
            == (stored["training_log"]["best_epoch"], stored["training_log"]["epochs_run"]),
        "metrics (val, val_full_diagnostic, test)": new["metrics"] == stored["metrics"],
        "gate statistics (init, best, final, converged; per-dim, histograms, displacements)":
            new["gate"] == stored["gate"],
        "features report and permutation info": new["features"] == stored["features"],
    }
    result = {"run_id": spec.run_id, "stored_commit": stored["git"]["commit"],
              "rerun_git": git, "rerun_wall_clock_s": new["wall_clock_s"],
              "stored_wall_clock_s": stored["wall_clock_s"], "checks": checks,
              "all_identical": all(checks.values())}
    out = OUT_DIR / f"anomaly_rerun_{spec.run_id}.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    for name, ok in checks.items():
        print(("IDENTICAL  " if ok else "DIFFERENT  ") + name)
    print(f"all identical: {result['all_identical']}; wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
