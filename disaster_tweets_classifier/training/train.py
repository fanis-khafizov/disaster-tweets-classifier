"""End-to-end training entry point: data download -> Lightning fit -> plots."""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from disaster_tweets_classifier.constants import PROJECT_ROOT
from disaster_tweets_classifier.data.datamodule import DisasterTweetsDataModule
from disaster_tweets_classifier.data.download import download_data
from disaster_tweets_classifier.models.deberta_classifier import DisasterTweetsClassifier
from disaster_tweets_classifier.utils.git_utils import get_git_commit_id
from disaster_tweets_classifier.utils.plotting import save_training_curves

logger = logging.getLogger(__name__)


def _build_datamodule(cfg: DictConfig) -> DisasterTweetsDataModule:
    return DisasterTweetsDataModule(
        data_dir=Path(cfg.data.data_dir),
        tokenizer_name=cfg.data.tokenizer_name,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        seed=cfg.seed,
        use_humaid=cfg.data.use_humaid,
        humaid_max_samples=cfg.data.humaid_max_samples,
    )


def _build_model(cfg: DictConfig, total_steps: int) -> DisasterTweetsClassifier:
    return DisasterTweetsClassifier(
        model_name=cfg.model.model_name,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_ratio=cfg.model.warmup_ratio,
        total_steps=total_steps,
    )


def _build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    checkpoint_dir = Path(cfg.training.checkpoint.dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            mode=cfg.training.early_stopping.mode,
            patience=cfg.training.early_stopping.patience,
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=cfg.training.checkpoint.filename,
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            save_top_k=cfg.training.checkpoint.save_top_k,
            auto_insert_metric_name=cfg.training.checkpoint.auto_insert_metric_name,
            save_last=True,
        ),
    ]


def _build_logger(cfg: DictConfig, commit_id: str) -> MLFlowLogger:
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        run_name=cfg.logging.run_name,
        tags={"git_commit": commit_id},
    )
    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return mlf_logger


def run_training(cfg: DictConfig) -> str:
    """Run the full training pipeline. Returns the MLflow run id."""
    pl.seed_everything(cfg.seed, workers=True)

    download_data(
        use_dvc=cfg.data.use_dvc,
        target_dir=Path(cfg.data.data_dir),
        include_humaid=cfg.data.use_humaid,
    )

    datamodule = _build_datamodule(cfg)
    datamodule.setup()

    steps_per_epoch = max(1, len(datamodule.train_dataloader()))
    total_steps = steps_per_epoch * cfg.training.max_epochs
    model = _build_model(cfg, total_steps=total_steps)

    commit_id = get_git_commit_id(PROJECT_ROOT)
    mlf_logger = _build_logger(cfg, commit_id)

    trainer_kwargs: dict = {
        "max_epochs": cfg.training.max_epochs,
        "accelerator": cfg.training.accelerator,
        "devices": cfg.training.devices,
        "precision": cfg.training.precision,
        "gradient_clip_val": cfg.training.gradient_clip_val,
        "log_every_n_steps": cfg.training.log_every_n_steps,
        "deterministic": cfg.training.deterministic,
        "callbacks": _build_callbacks(cfg),
        "logger": mlf_logger,
    }
    if cfg.training.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = cfg.training.limit_train_batches
    if cfg.training.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = cfg.training.limit_val_batches
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, datamodule=datamodule)

    if cfg.logging.save_plots:
        saved = save_training_curves(
            tracking_uri=cfg.logging.tracking_uri,
            run_id=mlf_logger.run_id,
            metrics=tuple(cfg.logging.plot_metrics),
        )
        logger.info("Saved %d training plots", len(saved))

    return mlf_logger.run_id
