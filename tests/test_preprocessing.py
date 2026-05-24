"""Unit tests for text preprocessing utilities."""

from __future__ import annotations

import pytest

from disaster_tweets_classifier.data.preprocessing import (
    clean_text_for_baseline,
    clean_text_for_transformer,
)


@pytest.mark.parametrize(
    ("raw", "expected_substring"),
    [
        ("Hello https://t.co/abc world", "Hello world"),
        ("<b>bold</b> text", "bold text"),
        ("Soooooo loooong", "Soo loong"),
    ],
)
def test_transformer_cleaning_removes_noise(raw: str, expected_substring: str) -> None:
    cleaned = clean_text_for_transformer(raw)
    assert expected_substring in cleaned


def test_transformer_cleaning_handles_non_string() -> None:
    assert clean_text_for_transformer(None) == ""  # type: ignore[arg-type]
    assert clean_text_for_transformer(123) == ""  # type: ignore[arg-type]


def test_transformer_cleaning_demojizes_emoji() -> None:
    cleaned = clean_text_for_transformer("fire 🔥 everywhere")
    assert "fire" in cleaned.lower()
    assert "everywhere" in cleaned.lower()


def test_baseline_cleaning_lowercases_and_strips_specials() -> None:
    cleaned = clean_text_for_baseline("Hello @user! Check #fire 123")
    assert cleaned == cleaned.lower()
    assert "@" not in cleaned
    assert "123" not in cleaned


def test_baseline_cleaning_keeps_alphabetic_words() -> None:
    cleaned = clean_text_for_baseline("Forest fire near La Ronge")
    for word in ("forest", "fire", "near", "la", "ronge"):
        assert word in cleaned
