"""Project-wide constants that are not expected to change.

Tunable hyperparameters live in Hydra configs (``configs/``), not here.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

TARGET_COLUMN: str = "target"
TEXT_COLUMN: str = "text"
ID_COLUMN: str = "id"

LABEL_NAMES: tuple[str, ...] = ("non_disaster", "disaster")
NUM_CLASSES: int = 2
