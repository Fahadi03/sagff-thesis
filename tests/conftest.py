"""Shared fixtures. Tests FAIL (not skip) when data is missing: a silently
skipped leakage test is worse than a red one."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
MANIFEST = SPLITS_DIR / "manifest.json"


def explain(title: str, problems: list, limit: int = 15) -> str:
    """Readable failure message: a title, a count, and the first few problems."""
    lines = [f"{title}: {len(problems)} problem(s)"]
    lines += [f"  - {p}" for p in problems[:limit]]
    if len(problems) > limit:
        lines.append(f"  ... and {len(problems) - limit} more")
    return "\n".join(lines)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def check(problems: list, title: str) -> None:
    """Fail with a clean, readable message (no traceback/fixture dump) if problems exist."""
    if problems:
        pytest.fail(explain(title, problems), pytrace=False)


@pytest.fixture(scope="session")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.fail(f"{MANIFEST.relative_to(ROOT)} not found. Run: python scripts/download_data.py "
                    "&& python scripts/make_splits.py", pytrace=False)
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def split_files(manifest) -> dict:
    """{(track, k, seed): {"train"/"val"/"test": rows}} read from disk."""
    out = {}
    for e in manifest["splits"]:
        d = SPLITS_DIR / e["path"]
        if not d.exists():
            pytest.fail(f"Split directory {d.relative_to(ROOT)} is missing. "
                        "Run: python scripts/make_splits.py", pytrace=False)
        out[(e["track"], e["k"], e["seed"])] = {s: read_jsonl(d / f"{s}.jsonl")
                                                 for s in ("train", "val", "test")}
    return out


@pytest.fixture(scope="session")
def built():
    """(tracks, report) rebuilt from data/raw by src.data."""
    from src.data import build_tracks
    try:
        return build_tracks()
    except FileNotFoundError as e:
        pytest.fail(f"{e}", pytrace=False)
