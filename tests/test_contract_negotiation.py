"""Public metadata and schema-pair negotiation contracts."""

from __future__ import annotations

import pytest

from policy_space.contracts import (
    SchemaPair,
    schema_pairs_from_metadata,
    select_schema_pair,
)
from policy_space.protocol import validate_server_metadata


def _metadata(**extra):
    return {
        "server": "policy_space",
        "version": 2,
        "protocol": "policy_space",
        "protocol_version": "2",
        "server_instance_id": "server-a",
        "session_mode": "exclusive",
        "supported_schema_pairs": [
            {
                "observation_schema": "bimanual_rgb_joint@1",
                "action_schema": "bimanual_joint_position@1",
                "embodiment_constraints": [],
            }
        ],
        **extra,
    }


def test_metadata_accepts_unknown_optional_and_namespaced_extensions() -> None:
    metadata = _metadata(
        display_name="Example", **{"vendor.example/feature": {"enabled": True}}
    )

    assert validate_server_metadata(metadata) == metadata


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", "1"),
        ("server_instance_id", ""),
        ("session_mode", "parallel_unsafe"),
    ],
)
def test_metadata_rejects_invalid_required_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        validate_server_metadata(_metadata(**{field: value}))


def test_schema_pair_selection_does_not_require_per_embodiment_tuples() -> None:
    policy_pairs = [SchemaPair("bimanual_rgb_joint@1", "bimanual_joint_position@1")]
    bridge_pairs = [SchemaPair("bimanual_rgb_joint@1", "bimanual_joint_position@1")]

    selected = select_schema_pair(policy_pairs, bridge_pairs, embodiment="arx_r5@1")

    assert selected == policy_pairs[0]


def test_schema_pair_selection_enforces_optional_embodiment_constraint() -> None:
    policy_pairs = [
        SchemaPair(
            "bimanual_rgb_joint@1",
            "bimanual_joint_position@1",
            embodiment_constraints=("ex001_6r@1",),
        )
    ]

    with pytest.raises(ValueError, match="embodiment"):
        select_schema_pair(policy_pairs, policy_pairs, embodiment="arx_r5@1")


def test_schema_pair_selection_rejects_ambiguity() -> None:
    policy_pairs = [
        SchemaPair("bimanual_rgb_joint@1", "bimanual_joint_position@1"),
        SchemaPair("bimanual_rgb_ee@1", "bimanual_ee_absolute@1"),
    ]

    with pytest.raises(ValueError, match="multiple"):
        select_schema_pair(policy_pairs, policy_pairs, embodiment="ex001_6r@1")


def test_schema_pair_parser_requires_nonempty_unique_pairs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        schema_pairs_from_metadata({})
    with pytest.raises(ValueError, match="duplicates"):
        schema_pairs_from_metadata(
            {
                "supported_schema_pairs": [
                    {
                        "observation_schema": "obs@1",
                        "action_schema": "act@1",
                    },
                    {
                        "observation_schema": "obs@1",
                        "action_schema": "act@1",
                    },
                ]
            }
        )


def test_bridge_embodiment_constraint_is_enforced() -> None:
    policy_pairs = [SchemaPair("obs@1", "act@1")]
    bridge_pairs = [SchemaPair("obs@1", "act@1", ("franka@1",))]

    with pytest.raises(ValueError, match="No compatible"):
        select_schema_pair(policy_pairs, bridge_pairs, embodiment="ex001_6r@1")
