"""Plot training curves from MLflow metric history."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from disaster_tweets_classifier.constants import PLOTS_DIR

logger = logging.getLogger(__name__)


def _safe_history(client: MlflowClient, run_id: str, metric: str) -> list:
    try:
        return client.get_metric_history(run_id, metric)
    except Exception as exc:
        logger.warning("Failed to fetch metric %s: %s", metric, exc)
        return []


def save_training_curves(
    tracking_uri: str,
    run_id: str,
    metrics: tuple[str, ...] = ("train/loss", "val/loss", "val/f1"),
    output_dir: Path = PLOTS_DIR,
) -> list[Path]:
    """Pull metric histories from MLflow and save matplotlib plots to disk."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for metric in metrics:
        history = _safe_history(client, run_id, metric)
        if not history:
            continue
        steps = [point.step for point in history]
        values = [point.value for point in history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values, marker="o", linewidth=1.5)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = output_dir / f"{metric.replace('/', '_')}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(out_path)
    return saved
