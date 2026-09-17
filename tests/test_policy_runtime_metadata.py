from __future__ import annotations

import pytest

from policy_space.base import PolicyModel
from policy_space.server import load_policy_model


def test_load_policy_model_merges_runtime_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(
        PolicyModel,
        "runtime_metadata",
        lambda _self: {"capabilities": {"test_runtime_capability": {"enabled": True}}},
    )

    _model, metadata = load_policy_model("example_policy")

    assert metadata["policy_name"] == "example_policy"
    assert metadata["capabilities"] == {"test_runtime_capability": {"enabled": True}}


def test_load_policy_model_rejects_runtime_override_of_static_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        PolicyModel, "runtime_metadata", lambda _self: {"policy_name": "spoofed"}
    )

    with pytest.raises(ValueError, match="cannot override static metadata"):
        load_policy_model("example_policy")
