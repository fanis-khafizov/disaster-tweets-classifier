"""DeBERTa V3 fine-tuning LightningModule."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from disaster_tweets_classifier.constants import NUM_CLASSES


class DisasterTweetsClassifier(pl.LightningModule):
    """Wraps a HuggingFace sequence-classification model with metrics and optimization."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
        )

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_auroc = BinaryAUROC()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch["labels"]
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, logits = self._shared_step(batch)
        preds = logits.argmax(dim=-1)
        self.train_f1.update(preds, batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, logits = self._shared_step(batch)
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]

        self.val_f1.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_auroc.update(probs, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/recall", self.val_recall, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_epoch=True)

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.hparams.total_steps:
            return {"optimizer": optimizer}

        warmup_steps = int(self.hparams.warmup_ratio * self.hparams.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
