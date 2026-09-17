"""Guards for the lightweight Policy Space protocol distribution."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "packages" / "protocol"


def test_protocol_distribution_has_an_independent_package_boundary() -> None:
    pyproject_path = PROTOCOL_ROOT / "pyproject.toml"
    assert pyproject_path.is_file()
    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert document["project"]["name"] == "policy-space-protocol"
    assert document["project"]["dependencies"] == ["PyYAML>=6.0"]
    assert (PROTOCOL_ROOT / "src" / "policy_space_protocol" / "__init__.py").is_file()


def test_full_distribution_bundles_the_lightweight_protocol_package() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "packages/protocol/src" in document["tool"]["setuptools"]["packages"]["find"]["where"]
    package_data = document["tool"]["setuptools"]["package-data"]
    assert package_data["policy_space_protocol.contracts"] == ["schemas/*/*.yaml"]
    assert "policy_space.contracts" not in package_data
    assert not (ROOT / "src" / "policy_space" / "contracts" / "schemas").exists()
