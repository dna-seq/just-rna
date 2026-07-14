"""Explicit environment configuration boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from platformdirs import user_cache_path

CACHE_ENV_VAR = "JUST_RNA_CACHE_DIR"


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved process settings.

    Loading ``.env`` is intentionally explicit and never happens on package import.
    """

    cache_directory: Path


def load_settings(dotenv_path: Path | None = None) -> Settings:
    """Load ``.env`` and return an immutable settings snapshot."""

    load_dotenv(dotenv_path=dotenv_path, override=False)
    configured_cache = os.getenv(CACHE_ENV_VAR)
    cache_directory = (
        Path(configured_cache).expanduser()
        if configured_cache
        else user_cache_path("just-rna", appauthor=False) / "assets"
    )
    return Settings(cache_directory=cache_directory)

