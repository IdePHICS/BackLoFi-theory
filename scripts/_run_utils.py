"""Shared single-run helpers for the experiment script cores.

Every ``scripts/_*_core.py`` runner follows the same one-config-one-file
contract: skip if a results JSON already exists, otherwise compute and write
it.  These two helpers own that file-level cache/save boilerplate so the cores
don't each copy-paste it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_cached_or_none(out_path: Path | None) -> dict[str, Any] | None:
    """Return the cached results dict at *out_path*, or ``None`` to (re)run.

    Logs a cache hit; on a corrupt/unreadable cache file logs a warning, deletes
    it, and returns ``None`` so the caller recomputes.  ``None`` ``out_path``
    (no caching requested) also returns ``None``.
    """
    if out_path is None or not out_path.exists():
        return None
    log.info("Cache hit: %s", out_path)
    try:
        with open(out_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        log.warning("Corrupt cache file, re-running: %s", out_path)
        out_path.unlink()
        return None


def save_results(out_path: Path | None, results: dict[str, Any]) -> None:
    """Write *results* as indented JSON to *out_path* (creating parent dirs).

    No-op when *out_path* is ``None`` (caching not requested).
    """
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a cancel mid-write must not leave a partial file that a later
    # resume treats as a valid cache.  Write a pid-unique temp then os.replace
    # (atomic rename on the same filesystem).
    tmp = out_path.with_name(f".{out_path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, out_path)
    log.info("Results saved to %s", out_path)
