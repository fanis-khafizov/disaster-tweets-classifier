"""Helpers for capturing the current git revision in experiment metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import git

logger = logging.getLogger(__name__)


def get_git_commit_id(repo_path: str | Path = ".") -> str:
    """Return the short commit id, or ``"unknown"`` if not in a git repo."""
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
        return repo.head.commit.hexsha[:8]
    except (git.InvalidGitRepositoryError, git.NoSuchPathError) as exc:
        logger.warning("Could not determine git commit: %s", exc)
        return "unknown"
