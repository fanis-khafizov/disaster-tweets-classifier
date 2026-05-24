"""Unified CLI entry point: ``dtc <command>``.

Hydra is used through the Compose API so that ``fire`` can expose discrete
subcommands while still benefiting from hierarchical config groups.
"""

from __future__ import annotations

import logging
from typing import Any

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from disaster_tweets_classifier.constants import CONFIGS_DIR
from disaster_tweets_classifier.data.download import download_data as _download_data
from disaster_tweets_classifier.inference.export_onnx import export_to_onnx
from disaster_tweets_classifier.inference.predict import run_inference
from disaster_tweets_classifier.training.train import run_training


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_config(overrides: tuple[str, ...] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        return compose(config_name="config", overrides=list(overrides or ()))


def train(*overrides: str) -> str:
    """Run the full Lightning training pipeline."""
    cfg = _load_config(overrides)
    run_id = run_training(cfg)
    print(f"Training finished. MLflow run id: {run_id}")
    return run_id


def infer(*overrides: str) -> str:
    """Run batch inference on a CSV file."""
    cfg = _load_config(overrides)
    output_path = run_inference(cfg)
    print(f"Predictions written to: {output_path}")
    return str(output_path)


def export_onnx(*overrides: str) -> str:
    """Export the trained model checkpoint to ONNX."""
    cfg = _load_config(overrides)
    onnx_path = export_to_onnx(cfg)
    print(f"ONNX model saved to: {onnx_path}")
    return str(onnx_path)


def download_data(*overrides: str) -> str:
    """Materialize raw data via DVC (or a public fallback)."""
    cfg = _load_config(overrides)
    target = _download_data(
        use_dvc=cfg.data.use_dvc,
        target_dir=None,
        include_humaid=cfg.data.use_humaid,
    )
    print(f"Data is available at: {target}")
    return str(target)


def cli() -> Any:
    """Fire entry point."""
    return fire.Fire(
        {
            "train": train,
            "infer": infer,
            "export-onnx": export_onnx,
            "download-data": download_data,
        }
    )


def main() -> None:
    _configure_logging()
    cli()


if __name__ == "__main__":
    main()
