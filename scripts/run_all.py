"""Run the full grid (docs/PROTOCOL.md §5), resumably, in the fixed run order.

Before anything runs:
  1. the whole test suite must pass (pytest tests/), otherwise refuse (§8);
  2. the code must be committed (every number traces to a commit).

Order: tracks bnintent30-bn, bnintent30-en, banking77; within each track the
conditions in src.train.CONDITIONS (sagff, baseline-cap, concat-cap,
sagff-noise-perepoch, sagff-permuted-fixed, baseline, concat); within each
condition k = 5, 10, 20, full and seeds 13, 42, 87, 100, 2026. Then the leaky
condition: bnintent30-bn, sagff then baseline, all k, all seeds (dirty=true).

Resumable: a run whose JSON already exists is skipped. A FAILED run is never
re-run (§10, failed-run rule); it is counted and reported.

Progress goes to stdout and is appended to logs/grid.log (created if missing; the
directory is gitignored). scripts/record_anomalies.py reads that file, so its line
format ("YYYY-MM-DD HH:MM:SS [i/n] <run_id>: <status> ...") is kept stable.

Usage:  python scripts/run_all.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import CONDITIONS, RunSpec, require_clean_tree, run  # noqa: E402

TRACKS = ("bnintent30-bn", "bnintent30-en", "banking77")
KS = ("5", "10", "20", "full")
SEEDS = (13, 42, 87, 100, 2026)
DIRTY = ("bnintent30-bn", ("sagff", "baseline"))
RUN_DIR = ROOT / "results" / "runs"
GRID_LOG = ROOT / "logs" / "grid.log"


def grid() -> list[RunSpec]:
    """Every run of the grid, in the order of docs/PROTOCOL.md §5."""
    specs = [RunSpec(track=t, k=k, seed=s, condition=c)
             for t in TRACKS for c in CONDITIONS for k in KS for s in SEEDS]
    specs += [RunSpec(track=DIRTY[0], k=k, seed=s, condition=c, dirty=True)
              for c in DIRTY[1] for k in KS for s in SEEDS]
    return specs


def log(msg: str) -> None:
    """Print a timestamped progress line and append it to logs/grid.log."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    GRID_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GRID_LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def main() -> None:
    """Refuse on uncommitted code or a failing test suite; otherwise run every missing run."""
    git = require_clean_tree("run_all")
    log(f"grid at commit {git['commit']} (clean tree)")
    log("running the test suite first (docs/PROTOCOL.md §8)...")
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ROOT,
                           capture_output=True, text=True)
    log("pytest: " + (tests.stdout.strip().splitlines() or ["(no output)"])[-1])
    if tests.returncode != 0:
        log("pytest output (tail):\n" + tests.stdout[-4000:] + "\n" + tests.stderr[-2000:])
        log("REFUSING to start the grid: the test suite did not pass.")
        sys.exit(1)

    specs = grid()
    total, t0 = len(specs), time.time()
    counts = {"ok": 0, "failed": 0, "skipped_ok": 0, "skipped_failed": 0}
    log(f"grid: {total} runs ({len(TRACKS)} tracks x {len(CONDITIONS)} conditions x {len(KS)} k x "
        f"{len(SEEDS)} seeds + {len(DIRTY[1]) * len(KS) * len(SEEDS)} leaky-condition runs)")
    for i, spec in enumerate(specs, 1):
        path = RUN_DIR / f"{spec.run_id}.json"
        if path.exists():
            status = json.loads(path.read_text(encoding="utf-8"))["status"]
            counts["skipped_ok" if status == "ok" else "skipped_failed"] += 1
            log(f"[{i}/{total}] {spec.run_id}: exists ({status}), skipped")
            continue
        result = run(spec)
        counts["ok" if result["status"] == "ok" else "failed"] += 1
        # Progress is BLIND: status and timing only, no metrics or gate values, so
        # nothing about the results is on view while the grid is running.
        log(f"[{i}/{total}] {spec.run_id}: {result['status']}"
            + ("" if result["status"] == "ok" else f" ERROR {result.get('error')}")
            + f" ({result['wall_clock_s']:.1f}s; elapsed {(time.time() - t0) / 60:.1f} min)")
    log(f"grid finished in {(time.time() - t0) / 3600:.2f} h: {counts}")


if __name__ == "__main__":
    main()
