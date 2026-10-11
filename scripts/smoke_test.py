"""Smoke test (CLAUDE.md §9): one track, k=5, one seed, one head, then validate
the JSON log. bnintent30-bn, k=5, seed 13, sagff.

The log goes to results/smoke/ (never results/runs/, so the real grid run of the
same cell is not blocked). A previous smoke log is deleted first: it is a test
artifact, not a result.

Usage:  python scripts/smoke_test.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import RunSpec, run  # noqa: E402

SMOKE_DIR = ROOT / "results" / "smoke"
REQUIRED = ["run_id", "status", "git", "config", "libraries", "determinism", "encoder", "splits",
            "features", "head", "training_log", "metrics", "gate", "wall_clock_s"]


def validate(log: dict) -> list[str]:
    p = [f"missing key {k}" for k in REQUIRED if k not in log]
    if log.get("status") != "ok":
        return p + [f"status={log.get('status')}: {log.get('error')}"]
    g, n_test, d = log["gate"], log["metrics"]["test"]["n"], log["config"]["heads"]["d_model"]
    for prefix in ("gate_", "gate_init_"):
        if len(g[f"{prefix}mean_per_dim"]) != d:
            p.append(f"{prefix}mean_per_dim has length {len(g[f'{prefix}mean_per_dim'])}, expected {d}")
        if len(g[f"{prefix}hist"]) != 20 or sum(g[f"{prefix}hist"]) != n_test * d:
            p.append(f"{prefix}hist must have 20 bins summing to n_test*d_model")
    if "gate_displacement" not in g:
        p.append("missing derived gate_displacement")
    if log["metrics"]["val_full_diagnostic"]["used_for_selection"] is not False:
        p.append("val_full_diagnostic must be marked used_for_selection=false")
    if not log["head"]["param_count"]:
        p.append("missing head param_count")
    if log["features"]["all_zero_tfidf_test_rows"] is None:
        p.append("missing all_zero_tfidf_test_rows")
    return p


def main() -> None:
    spec = RunSpec(track="bnintent30-bn", k="5", seed=13, condition="sagff", log_dir=SMOKE_DIR)
    (SMOKE_DIR / f"{spec.run_id}.json").unlink(missing_ok=True)
    log = run(spec)
    problems = validate(log)
    print(f"smoke run {spec.run_id}: status={log['status']}, wall {log['wall_clock_s']}s")
    if problems:
        print("INVALID:", *problems, sep="\n  - ")
        sys.exit(1)
    print(f"valid JSON log: {SMOKE_DIR.relative_to(ROOT) / (spec.run_id + '.json')}")


if __name__ == "__main__":
    main()
