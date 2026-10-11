"""Shared fixtures. Tests FAIL (not skip) when data is missing: a silently
skipped leakage test is worse than a red one."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
DIRTY_DIR = ROOT / "data" / "splits_dirty"
MANIFEST = SPLITS_DIR / "manifest.json"
DIRTY_MANIFEST = DIRTY_DIR / "manifest.json"
FILES = ("train", "val", "test", "val_full")


def explain(title: str, problems: list, limit: int = 15) -> str:
    """Readable failure message: a title, a count, and the first few problems."""
    lines = [f"{title}: {len(problems)} problem(s)"]
    lines += [f"  - {p}" for p in problems[:limit]]
    if len(problems) > limit:
        lines.append(f"  ... and {len(problems) - limit} more")
    return "\n".join(lines)


def check(problems: list, title: str) -> None:
    """Fail with a clean, readable message (no traceback/fixture dump) if problems exist."""
    if problems:
        pytest.fail(explain(title, problems), pytrace=False)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        pytest.fail(f"{path.relative_to(ROOT)} not found. Run: python scripts/download_data.py "
                    "&& python scripts/make_splits.py", pytrace=False)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_files(manifest: dict, base: Path) -> dict:
    """{(track, k, seed): {file name: rows}} read from disk."""
    out = {}
    for e in manifest["splits"]:
        d = base / e["path"]
        if not d.exists():
            pytest.fail(f"Split directory {d.relative_to(ROOT)} is missing. "
                        "Run: python scripts/make_splits.py", pytrace=False)
        out[(e["track"], e["k"], e["seed"])] = {f: read_jsonl(d / f"{f}.jsonl") for f in FILES}
    return out


@pytest.fixture(scope="session")
def manifest() -> dict:
    return _load_manifest(MANIFEST)


@pytest.fixture(scope="session")
def dirty_manifest() -> dict:
    return _load_manifest(DIRTY_MANIFEST)


@pytest.fixture(scope="session")
def split_files(manifest) -> dict:
    return _load_files(manifest, SPLITS_DIR)


@pytest.fixture(scope="session")
def dirty_files(dirty_manifest) -> dict:
    return _load_files(dirty_manifest, DIRTY_DIR)


@pytest.fixture(scope="session")
def built():
    """(tracks, report) rebuilt from data/raw by src.data."""
    from src.data import build_tracks
    try:
        return build_tracks()
    except FileNotFoundError as e:
        pytest.fail(f"{e}", pytrace=False)
