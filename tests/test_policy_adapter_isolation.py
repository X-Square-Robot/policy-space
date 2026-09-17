"""Policy adapters must remain usable without importing ManaEnv."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_policy_adapters_do_not_import_manaenv() -> None:
    offenders: list[str] = []
    for path in (ROOT / "policies").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module_names: list[str]
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module_names = [node.module or ""]
            else:
                continue
            if any(
                name == "manaenv" or name.startswith("manaenv.")
                for name in module_names
            ):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_model_source_does_not_assume_a_parent_manaenv_3rd_directory() -> None:
    offenders = []
    for path in (ROOT / "policies").rglob("model.py"):
        source = path.read_text(encoding="utf-8")
        if '"3rd" /' in source or "'3rd' /" in source:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
