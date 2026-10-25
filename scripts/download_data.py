"""Download the raw data, pinned to exact upstream commits (docs/PROTOCOL.md §3.2).

First run:  resolves GAN-BnBERT `main` (and the PolyAI repo `master`) to commit
            SHAs ONCE, downloads those commits, and records SHAs, URLs and the
            sha256 of every raw file in configs/data.yaml.
Later runs: read the pinned SHAs from configs/data.yaml, never resolve a branch
            again, and fail loudly if any downloaded file's sha256 differs from
            the recorded one.

Usage:  python scripts/download_data.py
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data.yaml"
RAW = ROOT / "data" / "raw"

BN30_REPO = "JahidBasher/GAN-BnBERT"
BN30_BRANCH = "main"
BN30_PATH = "dataset/BNIntent30/"

B77_REPO = "PolyAI-LDN/task-specific-datasets"

# Dataset citations, verified against the publisher/archive record
# (docs/citation_checklist.md). Primary = peer-reviewed version.
BN30_CITATION = {
    "primary": "M. Hasan, M. J. I. Basher, M. T. R. Shawon, \"Bengali Intent Classification with "
               "Generative Adversarial BERT\", 2023 26th International Conference on Computer and "
               "Information Technology (ICCIT), IEEE, pp. 1-6, 2023",
    "doi": "10.1109/ICCIT60459.2023.10440989",
    "secondary": "arXiv:2312.10679",
    "note": "the paper spells its source dataset 'CLINIC150'; the dataset's name is CLINC150 "
            "(Larson et al., EMNLP-IJCNLP 2019, doi:10.18653/v1/D19-1131)",
}
B77_CITATION = {
    "primary": "I. Casanueva, T. Temcinas, D. Gerz, M. Henderson, I. Vulic, \"Efficient Intent "
               "Detection with Dual Sentence Encoders\", Proceedings of the 2nd Workshop on NLP for "
               "Conversational AI, ACL, pp. 38-45, 2020",
    "doi": "10.18653/v1/2020.nlp4convai-1.5",
}
B77_BRANCH = "master"
B77_FILES = {"train": "banking_data/train.csv", "test": "banking_data/test.csv"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sagff-thesis-download"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve_commit(repo: str, branch: str) -> str:
    data = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{branch}"))
    return data["sha"]


def check_or_record(recorded: dict, name: str, digest: str) -> None:
    """Record a file's sha256 the first time; afterwards it must match."""
    if name in recorded and recorded[name] != digest:
        sys.exit(f"ERROR: sha256 mismatch for {name}\n  recorded: {recorded[name]}\n"
                 f"  got:      {digest}\nThe upstream file changed. Do not continue; report it.")
    recorded[name] = digest


def download_bnintent30(cfg: dict) -> None:
    sec = cfg.setdefault("bnintent30", {"repo": BN30_REPO, "path": BN30_PATH})
    sec["citation"] = BN30_CITATION
    if "commit" not in sec:
        sec["commit"] = resolve_commit(BN30_REPO, BN30_BRANCH)
        sec["resolved_from_branch"] = BN30_BRANCH
        print(f"Resolved {BN30_REPO}@{BN30_BRANCH} -> {sec['commit']} (recorded; will not resolve again)")
    sha = sec["commit"]
    print(f"BNIntent30: {BN30_REPO} @ {sha}")
    sec["archive_url"] = f"https://codeload.github.com/{BN30_REPO}/zip/{sha}"

    out = RAW / "bnintent30"
    out.mkdir(parents=True, exist_ok=True)
    recorded = sec.setdefault("files_sha256", {})
    z = zipfile.ZipFile(io.BytesIO(fetch(sec["archive_url"])))
    members = [m for m in z.namelist()
               if m.split("/", 1)[-1].startswith(BN30_PATH) and m.endswith(".json")]
    if len(members) != 30:
        sys.exit(f"ERROR: expected 30 BNIntent30 JSON files at {sha}, found {len(members)}.")
    for m in sorted(members):
        name = Path(m).name  # basename only: never trust archive paths
        content = z.read(m)
        check_or_record(recorded, name, sha256_bytes(content))
        (out / name).write_bytes(content)
    print(f"  wrote {len(members)} files to {out.relative_to(ROOT)}")


def download_banking77(cfg: dict) -> None:
    sec = cfg.setdefault("banking77", {"repo": B77_REPO})
    sec["citation"] = B77_CITATION
    if "commit" not in sec:
        sec["commit"] = resolve_commit(B77_REPO, B77_BRANCH)
        sec["resolved_from_branch"] = B77_BRANCH
        print(f"Resolved {B77_REPO}@{B77_BRANCH} -> {sec['commit']} (recorded; will not resolve again)")
    sha = sec["commit"]
    print(f"BANKING77: {B77_REPO} @ {sha}")

    out = RAW / "banking77"
    out.mkdir(parents=True, exist_ok=True)
    sec.setdefault("license", "CC-BY-4.0 (LICENSE file of PolyAI-LDN/task-specific-datasets)")
    urls = sec.setdefault("urls", {})
    recorded = sec.setdefault("files_sha256", {})
    for split, path in B77_FILES.items():
        urls[split] = f"https://raw.githubusercontent.com/{B77_REPO}/{sha}/{path}"
        content = fetch(urls[split])
        name = f"{split}.csv"
        check_or_record(recorded, name, sha256_bytes(content))
        (out / name).write_bytes(content)
        print(f"  wrote {out.relative_to(ROOT) / name} ({len(content):,} bytes)")


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    cfg = cfg or {}
    download_bnintent30(cfg)
    download_banking77(cfg)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Written by scripts/download_data.py. Pinned upstream commits and raw-file\n"
              "# checksums (docs/PROTOCOL.md §3.2). Do not edit by hand.\n")
    CONFIG.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                      encoding="utf-8", newline="\n")
    print(f"Recorded pins and checksums in {CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
