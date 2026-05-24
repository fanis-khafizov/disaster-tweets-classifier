"""Data download helpers.

Two open sources are combined to comfortably exceed the 10 MB requirement of
the course while still being downloadable without authentication:

* **Kaggle "nlp-getting-started"** -- ~1.4 MB, primary task data with the same
  schema as the public test set;
* **HumAID** (Alam et al., 2021, https://crisisnlp.qcri.org/humaid_dataset) --
  ~16 MB after extraction, ~77K crisis-related tweets with humanitarian
  category labels which are mapped to the binary target.

The primary mechanism is ``dvc pull``; if no DVC remote is reachable,
``download_data`` falls back to direct HTTP downloads.
"""

from __future__ import annotations

import logging
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from disaster_tweets_classifier.constants import DATA_DIR

logger = logging.getLogger(__name__)

KAGGLE_URLS: dict[str, str] = {
    "train.csv": "https://raw.githubusercontent.com/Krishnarohith10/nlp-getting-started/master/train.csv",
    "test.csv": "https://raw.githubusercontent.com/Krishnarohith10/nlp-getting-started/master/test.csv",
}
HUMAID_ZIP_URL: str = "https://crisisnlp.qcri.org/data/humaid/humaid_data_all.zip"
HUMAID_SUBDIR: str = "humaid"
HUMAID_FILES: tuple[str, ...] = ("train.jsonl", "dev.jsonl", "test.jsonl")


def _run_dvc_pull(targets: list[str] | None = None) -> bool:
    """Run ``dvc pull``. Returns ``True`` on success."""
    cmd = ["dvc", "pull"]
    if targets:
        cmd.extend(targets)
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("dvc pull failed: %s", exc)
        return False


def _download_kaggle(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, url in KAGGLE_URLS.items():
        out = target_dir / name
        if out.exists():
            logger.info("Already present: %s", out)
            continue
        logger.info("Downloading %s -> %s", url, out)
        urlretrieve(url, out)  # noqa: S310 -- trusted public mirror


def _download_humaid(target_dir: Path) -> None:
    humaid_dir = target_dir / HUMAID_SUBDIR
    humaid_dir.mkdir(parents=True, exist_ok=True)
    if all((humaid_dir / name).exists() for name in HUMAID_FILES):
        logger.info("HumAID already present at %s", humaid_dir)
        return

    zip_path = humaid_dir / "humaid_data_all.zip"
    if not zip_path.exists():
        logger.info("Downloading HumAID from %s", HUMAID_ZIP_URL)
        urlretrieve(HUMAID_ZIP_URL, zip_path)  # noqa: S310 -- trusted public mirror

    logger.info("Extracting %s", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.startswith("__MACOSX") or member.endswith("/"):
                continue
            archive.extract(member, humaid_dir)
    zip_path.unlink(missing_ok=True)


def download_data(
    use_dvc: bool = True,
    target_dir: Path | None = None,
    include_humaid: bool = True,
) -> Path:
    """Ensure raw data is materialized locally.

    Parameters
    ----------
    use_dvc:
        When ``True``, try ``dvc pull`` first; fall back to HTTP only on failure.
    target_dir:
        Directory to materialize raw files into. Defaults to ``data/raw``.
    include_humaid:
        Whether to fetch the HumAID extension dataset.

    Returns
    -------
    Path
        The directory containing ``train.csv``, ``test.csv`` and ``humaid/*``.
    """
    raw_dir = target_dir if target_dir is not None else DATA_DIR / "raw"
    if use_dvc:
        _run_dvc_pull(["data/raw"])
    # Always supplement what's missing via HTTP (idempotent: skips existing files).
    _download_kaggle(raw_dir)
    if include_humaid:
        _download_humaid(raw_dir)
    return raw_dir
