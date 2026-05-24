"""Lightning DataModule for the disaster tweets dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from disaster_tweets_classifier.constants import (
    DATA_DIR,
    ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
)
from disaster_tweets_classifier.data.preprocessing import clean_text_for_transformer

HUMAID_POSITIVE_LABELS: frozenset[str] = frozenset(
    {
        "caution_and_advice",
        "displaced_people_and_evacuations",
        "infrastructure_and_utility_damage",
        "injured_or_dead_people",
        "missing_or_found_people",
        "requests_or_urgent_needs",
        "rescue_volunteering_or_donation_effort",
        "sympathy_and_support",
    }
)
HUMAID_NEGATIVE_LABELS: frozenset[str] = frozenset({"not_humanitarian", "other_relevant_information"})


def _load_humaid(humaid_dir: Path, max_samples: int | None, seed: int) -> pd.DataFrame:
    """Load HumAID JSONL splits, map to binary target, optionally subsample."""
    frames: list[pd.DataFrame] = []
    for name in ("train.jsonl", "dev.jsonl"):
        path = humaid_dir / name
        if not path.exists():
            continue
        frames.append(pd.read_json(path, lines=True))
    if not frames:
        return pd.DataFrame(columns=[TEXT_COLUMN, TARGET_COLUMN])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["class_label"].isin(HUMAID_POSITIVE_LABELS | HUMAID_NEGATIVE_LABELS)]
    combined[TARGET_COLUMN] = combined["class_label"].isin(HUMAID_POSITIVE_LABELS).astype(int)
    combined = combined.rename(columns={"tweet_text": TEXT_COLUMN})[[TEXT_COLUMN, TARGET_COLUMN]]
    combined = combined.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
    if max_samples is not None and len(combined) > max_samples:
        combined = combined.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return combined


class TweetDataset(Dataset):
    """Lightweight tweet dataset returning pre-tokenized tensors."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class DisasterTweetsDataModule(pl.LightningDataModule):
    """Loads, cleans, and splits the disaster tweets corpus."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR / "raw",
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_length: int = 128,
        batch_size: int = 16,
        num_workers: int = 2,
        val_size: float = 0.15,
        seed: int = 42,
        use_humaid: bool = True,
        humaid_max_samples: int | None = 10_000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.tokenizer: AutoTokenizer | None = None
        self.train_dataset: TweetDataset | None = None
        self.val_dataset: TweetDataset | None = None
        self.test_dataset: TweetDataset | None = None

    def prepare_data(self) -> None:
        # No-op: data is materialized externally via DVC / download_data.
        pass

    def setup(self, stage: str | None = None) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(
                f"Expected {train_path} to exist. Run `dtc download-data` first."
            )

        kaggle_frame = pd.read_csv(train_path)[[TEXT_COLUMN, TARGET_COLUMN]]
        sources: list[pd.DataFrame] = [kaggle_frame]

        if self.hparams.use_humaid:
            humaid_frame = _load_humaid(
                humaid_dir=self.data_dir / "humaid",
                max_samples=self.hparams.humaid_max_samples,
                seed=self.hparams.seed,
            )
            if not humaid_frame.empty:
                sources.append(humaid_frame)

        frame = pd.concat(sources, ignore_index=True)
        frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").map(clean_text_for_transformer)

        train_frame, val_frame = train_test_split(
            frame,
            test_size=self.hparams.val_size,
            random_state=self.hparams.seed,
            stratify=frame[TARGET_COLUMN],
        )
        self.train_dataset = self._build_dataset(train_frame, with_labels=True)
        self.val_dataset = self._build_dataset(val_frame, with_labels=True)

        test_path = self.data_dir / "test.csv"
        if test_path.exists():
            test_frame = pd.read_csv(test_path)
            test_frame[TEXT_COLUMN] = (
                test_frame[TEXT_COLUMN].fillna("").map(clean_text_for_transformer)
            )
            self.test_dataset = self._build_dataset(test_frame, with_labels=False)

    def _build_dataset(self, frame: pd.DataFrame, *, with_labels: bool) -> TweetDataset:
        labels = frame[TARGET_COLUMN].astype(int).tolist() if with_labels else None
        return TweetDataset(
            texts=frame[TEXT_COLUMN].tolist(),
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("No test dataset is loaded.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def get_ids(self) -> list[int]:
        test_path = self.data_dir / "test.csv"
        if not test_path.exists():
            return []
        return pd.read_csv(test_path, usecols=[ID_COLUMN])[ID_COLUMN].tolist()
