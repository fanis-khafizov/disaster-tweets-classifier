"""Public inference entry point.

Usage::

    python infer.py infer.input_path=path/to/file.csv infer.output_path=preds.csv
"""

from __future__ import annotations

import sys

from disaster_tweets_classifier.commands import infer


def main() -> None:
    infer(*sys.argv[1:])


if __name__ == "__main__":
    main()
