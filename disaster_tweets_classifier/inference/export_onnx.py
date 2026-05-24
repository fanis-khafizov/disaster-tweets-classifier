"""Export a trained Lightning checkpoint to ONNX."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from disaster_tweets_classifier.models.deberta_classifier import DisasterTweetsClassifier

logger = logging.getLogger(__name__)

_OPSET_VERSION = 17


def export_to_onnx(cfg: DictConfig) -> Path:
    """Trace the model with dummy inputs and write an ONNX graph."""
    checkpoint_path = Path(cfg.infer.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = DisasterTweetsClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    max_length = int(cfg.data.max_length)
    dummy_input_ids = torch.zeros((1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long)
    dummy_token_type_ids = torch.zeros((1, max_length), dtype=torch.long)

    onnx_path = Path(cfg.infer.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        onnx_path.as_posix(),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=_OPSET_VERSION,
    )
    logger.info("ONNX model saved to %s", onnx_path)
    return onnx_path
