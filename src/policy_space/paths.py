"""Explicit paths for optional model repositories and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def external_source_root(
    cfg: dict[str, Any], *, env_name: str, cfg_key: str, project: str
) -> Path:
    """Resolve one model source checkout without assuming a parent repository."""

    raw = os.environ.get(env_name) or cfg.get(cfg_key)
    if not raw:
        raise RuntimeError(
            f"{project} source is not configured; set {env_name} or model.cfg.{cfg_key}"
        )
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{project} source directory does not exist: {path}")
    return path


def resolve_data_path(value: str, *, base: Path | None = None) -> Path:
    """Resolve a local data path against an explicitly configured base."""

    path = Path(value).expanduser()
    if path.is_absolute() or base is None:
        return path.resolve()
    return (base / path).resolve()


__all__ = ["external_source_root", "resolve_data_path"]
