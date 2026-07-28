"""
One source of truth for where things live on disk.

WHY THIS EXISTS. Every module used to compute the project root by counting
directory levels — `Path(__file__).resolve().parents[2]`. That is correct
exactly until someone moves a folder, at which point every path silently points
somewhere wrong: collectors write to a store nobody reads, credential lookups
miss, and nothing raises. Moving the code into `backend/` broke six such call
sites at once.

This resolves the root by SEARCHING UPWARD FOR A MARKER instead, so the depth
of any given module stops mattering. Override with HORMUZ_ROOT when the layout
is unusual — notably inside the container, where the repo markers are excluded
by .dockerignore.

Layout this assumes:

    <root>/
      backend/        code (this package)
      data/           collected data (gitignored)
      secrets/        credentials (gitignored)
      ports/          legacy reference data
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Files that only ever exist at the project root. `backend/` is included so a
# stray CLAUDE.md elsewhere cannot fool the search.
_MARKERS = ("CLAUDE.md", "docker-compose.yml")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """
    Project root, resolved by marker search rather than by counting parents.

    Order: HORMUZ_ROOT env var -> upward marker search -> structural fallback.
    """
    env = os.environ.get("HORMUZ_ROOT", "").strip()
    if env:
        return Path(env).resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / m).exists() for m in _MARKERS) and (parent / "backend").is_dir():
            return parent

    # Fallback for environments where the markers are absent (the Docker image
    # excludes *.md and compose files): backend/common/paths.py -> common ->
    # backend -> root.
    return here.parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("HORMUZ_DATA_DIR", "").strip() or repo_root() / "data")


def raw_dir() -> Path:
    return data_dir() / "raw"


def derived_dir() -> Path:
    return data_dir() / "derived"


def reference_dir() -> Path:
    return data_dir() / "reference"


def secrets_dir() -> Path:
    return repo_root() / "secrets"


def ports_dir() -> Path:
    """Legacy reference data (uae_ports.*, aisstream key). Not yet migrated."""
    return repo_root() / "ports"
