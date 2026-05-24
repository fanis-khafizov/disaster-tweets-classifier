"""Triton Inference Server test client.

Sends a small batch of tweets to the ``disaster_tweets_onnx`` model over HTTP,
runs softmax + threshold on the client side, and prints predictions.

Run a local Triton server with the ONNX backend (see README), then::

    uv run python triton/client.py --texts="['earthquake hit the city','love this party']"
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from disaster_tweets_classifier.data.preprocessing import clean_text_for_transformer

DEFAULT_TEXTS: tuple[str, ...] = (
    "Forest fire near La Ronge Sask. Canada",
    "I love this party, it's lit!",
    "13,000 people receive #wildfires evacuation orders in California",
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict(
    texts: list[str] | None = None,
    server_url: str = "localhost:8000",
    model_name: str = "disaster_tweets_onnx",
    tokenizer_name: str = "microsoft/deberta-v3-base",
    max_length: int = 128,
    threshold: float = 0.5,
) -> list[dict]:
    """Query Triton and print predictions for each input text."""
    if texts is None:
        texts = list(DEFAULT_TEXTS)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    cleaned = [clean_text_for_transformer(text) for text in texts]
    encoded = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

    client = httpclient.InferenceServerClient(url=server_url)
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        httpclient.InferInput("token_type_ids", token_type_ids.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    inputs[2].set_data_from_numpy(token_type_ids)

    outputs = [httpclient.InferRequestedOutput("logits")]
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    logits = response.as_numpy("logits")

    probs = _softmax(logits)[:, 1]
    results = [
        {
            "text": original,
            "probability": float(prob),
            "target": int(prob >= threshold),
        }
        for original, prob in zip(texts, probs, strict=True)
    ]
    for row in results:
        print(f"target={row['target']} | p={row['probability']:.4f} | {row['text']}")
    return results


def main() -> None:
    fire.Fire(predict)


if __name__ == "__main__":
    main()
