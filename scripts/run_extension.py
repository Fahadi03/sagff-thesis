"""§12 extension, step 3: the 80 runs (bnintent30-es and bnintent30-hi), resumably.

Before anything runs:
  1. the code must be committed (every number traces to a commit);
  2. the whole test suite must pass, AND tests/test_extension.py (the uid-identity
     test) must have PASSED, not been skipped (§12: "must pass before the grid may run");
  3. the extension embedding caches must exist.

Runs: tracks es, hi; heads sagff, baseline (src.train.CONDITIONS order); k = 5, 10, 20,
full; seeds 13, 42, 87, 100, 2026: 2 x 2 x 4 x 5 = 80. Every setting is the main grid's
(configs/default.yaml, unchanged). The opt-in hooks of src/train.py point the runs at the
extension's own splits and cache, write the logs to results/extension/runs/ (never
results/runs/), and save the predicted label of every test uid (§12 rescoring).

Resumable: an existing run JSON is skipped; a failed run is never re-run. Progress
(status and timing only, blind to results) goes to stdout and logs/extension.log.

Usage:  python scripts/run_extension.py
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import cache_paths  # noqa: E402
from src.extension import CACHE_DIR, HEADS, KS, LANGS, RUN_DIR, SEEDS, SPLIT_BASE, TRACK  # noqa: E402
from src.train import RunSpec, require_clean_tree, run  # noqa: E402

LOG = ROOT / "logs" / "extension.log"


def grid() -> list[RunSpec]:
    return [RunSpec(track=TRACK[lang], k=k, seed=s, condition=h, log_dir=RUN_DIR,
                    split_base=SPLIT_BASE, cache_dir=CACHE_DIR, save_predictions=True)
            for lang in LANGS for h in HEADS for k in KS for s in SEEDS]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def pytest(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "-rs"], cwd=ROOT,
                          capture_output=True, text=True)


def main() -> None:
    git = require_clean_tree("run_extension")
    log(f"extension at commit {git['commit']} (clean tree)")
    full = pytest("tests/")
    log("pytest tests/: " + (full.stdout.strip().splitlines() or ["(no output)"])[-1])
    ext = pytest("tests/test_extension.py")
    summary = (ext.stdout.strip().splitlines() or [""])[-1]
    log("pytest tests/test_extension.py: " + summary)
    passed = re.search(r"(\d+) passed", summary)
    if full.returncode or ext.returncode or "skipped" in summary or not passed:
        log("REFUSING to start: the test suite did not pass, or the uid-identity test was skipped.")
        sys.exit(1)
    for lang in LANGS:
        if not cache_paths(TRACK[lang], CACHE_DIR)[0].exists():
            log(f"REFUSING to start: no embedding cache for {TRACK[lang]}; run scripts/extension_embed.py")
            sys.exit(1)

    specs = grid()
    total, t0 = len(specs), time.time()
    counts = {"ok": 0, "failed": 0, "skipped_ok": 0, "skipped_failed": 0}
    log(f"extension: {total} runs ({len(LANGS)} languages x {len(HEADS)} heads x {len(KS)} k x {len(SEEDS)} seeds)")
    for i, spec in enumerate(specs, 1):
        path = RUN_DIR / f"{spec.run_id}.json"
        if path.exists():
            status = json.loads(path.read_text(encoding="utf-8"))["status"]
            counts["skipped_ok" if status == "ok" else "skipped_failed"] += 1
            log(f"[{i}/{total}] {spec.run_id}: exists ({status}), skipped")
            continue
        result = run(spec)
        counts["ok" if result["status"] == "ok" else "failed"] += 1
        log(f"[{i}/{total}] {spec.run_id}: {result['status']}"
            + ("" if result["status"] == "ok" else f" ERROR {result.get('error')}")
            + f" ({result['wall_clock_s']:.1f}s; elapsed {(time.time() - t0) / 60:.1f} min)")
    log(f"extension finished in {(time.time() - t0) / 3600:.2f} h: {counts}")


if __name__ == "__main__":
    main()
