"""Batch inference on a CSV file using a trained Lightning checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from disaster_tweets_classifier.constants import ID_COLUMN, TEXT_COLUMN
from disaster_tweets_classifier.data.datamodule import TweetDataset
from disaster_tweets_classifier.data.download import download_data
from disaster_tweets_classifier.data.preprocessing import clean_text_for_transformer
from disaster_tweets_classifier.models.deberta_classifier import DisasterTweetsClassifier

logger = logging.getLogger(__name__)


def _load_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if TEXT_COLUMN not in frame.columns:
        raise ValueError(
            f"Input file must contain a '{TEXT_COLUMN}' column, got {list(frame.columns)}"
        )
    if ID_COLUMN not in frame.columns:
        frame = frame.assign(**{ID_COLUMN: range(len(frame))})
    return frame


def run_inference(cfg: DictConfig) -> Path:
    """Predict labels for the CSV referenced by ``cfg.infer.input_path``."""
    if cfg.data.use_dvc:
        download_data(
            use_dvc=True,
            target_dir=Path(cfg.data.data_dir),
            include_humaid=False,
        )

    input_path = Path(cfg.infer.input_path)
    frame = _load_input(input_path)
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").map(clean_text_for_transformer)

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    dataset = TweetDataset(
        texts=frame[TEXT_COLUMN].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
    )
    loader = DataLoader(dataset, batch_size=cfg.infer.batch_size, shuffle=False)

    checkpoint_path = Path(cfg.infer.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = DisasterTweetsClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    probs: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(
                    device
                ),
            )
            probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().tolist())

    threshold = float(cfg.infer.threshold)
    predictions = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].tolist(),
            "probability": probs,
            "target": [int(prob >= threshold) for prob in probs],
        }
    )

    output_path = Path(cfg.infer.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)
    return output_path
