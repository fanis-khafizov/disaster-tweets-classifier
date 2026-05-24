"""Unit tests for HumAID binary mapping inside the DataModule."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from disaster_tweets_classifier.constants import TARGET_COLUMN, TEXT_COLUMN
from disaster_tweets_classifier.data.datamodule import (
    HUMAID_NEGATIVE_LABELS,
    HUMAID_POSITIVE_LABELS,
    _load_humaid,
)


def _write_humaid(tmp_path: Path) -> Path:
    humaid_dir = tmp_path / "humaid"
    humaid_dir.mkdir()
    rows = [
        {"tweet_text": "people trapped under rubble", "class_label": "injured_or_dead_people"},
        {"tweet_text": "donate to relief efforts now", "class_label": "rescue_volunteering_or_donation_effort"},
        {"tweet_text": "random twitter chatter", "class_label": "not_humanitarian"},
        {"tweet_text": "weather report", "class_label": "other_relevant_information"},
        {"tweet_text": "should be skipped", "class_label": "unknown_label"},
    ]
    (humaid_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    return humaid_dir


def test_humaid_labels_are_partitioned() -> None:
    assert HUMAID_POSITIVE_LABELS.isdisjoint(HUMAID_NEGATIVE_LABELS)


def test_load_humaid_applies_binary_mapping(tmp_path: Path) -> None:
    humaid_dir = _write_humaid(tmp_path)
    frame = _load_humaid(humaid_dir=humaid_dir, max_samples=None, seed=42)

    assert isinstance(frame, pd.DataFrame)
    assert set(frame.columns) == {TEXT_COLUMN, TARGET_COLUMN}
    assert len(frame) == 4  # unknown_label row is filtered out

    targets_by_text = dict(zip(frame[TEXT_COLUMN], frame[TARGET_COLUMN], strict=True))
    assert targets_by_text["people trapped under rubble"] == 1
    assert targets_by_text["random twitter chatter"] == 0


def test_load_humaid_subsamples(tmp_path: Path) -> None:
    humaid_dir = _write_humaid(tmp_path)
    frame = _load_humaid(humaid_dir=humaid_dir, max_samples=2, seed=0)
    assert len(frame) == 2


def test_load_humaid_returns_empty_when_missing(tmp_path: Path) -> None:
    frame = _load_humaid(humaid_dir=tmp_path / "missing", max_samples=None, seed=42)
    assert frame.empty
    assert set(frame.columns) == {TEXT_COLUMN, TARGET_COLUMN}
