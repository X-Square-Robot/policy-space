"""Loader and reference checks for versioned public contract YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml


def _documents(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict) or not isinstance(document.get("id"), str):
            raise ValueError(f"{path}: contract must be an object with a string id")
        identity = document["id"]
        if identity in result:
            raise ValueError(f"Duplicate contract id: {identity}")
        result[identity] = document
    return result


@dataclass(frozen=True)
class ContractRegistry:
    protocol: dict[str, Any]
    observations: dict[str, dict[str, Any]]
    actions: dict[str, dict[str, Any]]
    embodiments: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, root: str | Path) -> "ContractRegistry":
        root_path = Path(root)
        protocol_path = root_path / "protocol" / "v2.yaml"
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
        if protocol.get("protocol_version") != 2:
            raise ValueError(f"{protocol_path}: expected protocol_version 2")
        return cls(
            protocol=protocol,
            observations=_documents(root_path / "observations"),
            actions=_documents(root_path / "actions"),
            embodiments=_documents(root_path / "embodiments"),
        )

    @classmethod
    def builtin(cls) -> "ContractRegistry":
        configured = os.environ.get("POLICY_SPACE_CONTRACTS_DIR")
        if configured:
            return cls.load(Path(configured).expanduser())
        resource = files("policy_space_protocol.contracts").joinpath("schemas")
        with as_file(resource) as schema_root:
            return cls.load(schema_root)

    def validate_references(self) -> None:
        for identity, profile in self.embodiments.items():
            for schema in profile.get("supported_observation_schemas", []):
                if schema not in self.observations:
                    raise ValueError(f"{identity}: unknown observation schema {schema!r}")
            for schema in profile.get("supported_action_schemas", []):
                if schema not in self.actions:
                    raise ValueError(f"{identity}: unknown action schema {schema!r}")


__all__ = ["ContractRegistry"]
