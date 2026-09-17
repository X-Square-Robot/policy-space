"""Versioned schema registry tests."""

from __future__ import annotations

from importlib.resources import files

from policy_space.contracts.registry import ContractRegistry


def test_shipped_contract_registry_has_resolvable_references() -> None:
    registry = ContractRegistry.builtin()

    assert registry.protocol["protocol_version"] == 2
    assert "bimanual_rgb_joint@1" in registry.observations
    assert "bimanual_joint_position@1" in registry.actions
    assert registry.embodiments["ex001_6r@1"]["supported_observation_schemas"]
    registry.validate_references()


def test_schema_owns_wire_shape_while_embodiment_only_references_schema() -> None:
    registry = ContractRegistry.builtin()

    action = registry.actions["bimanual_joint_position@1"]
    embodiment = registry.embodiments["ex001_6r@1"]
    assert action["action_dim"] == 14
    assert "action_dim" not in embodiment
    assert "joint_dim" not in embodiment


def test_builtin_contracts_are_packaged_with_contract_loader() -> None:
    schema_root = files("policy_space_protocol.contracts").joinpath("schemas")
    assert schema_root.joinpath("protocol", "v2.yaml").is_file()
    registry = ContractRegistry.builtin()
    assert registry.protocol["protocol_version"] == 2
    registry.validate_references()
