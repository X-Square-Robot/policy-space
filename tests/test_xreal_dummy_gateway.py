"""Contracts for the test-only XReal PolicySpace dummy gateway."""

from __future__ import annotations

import math

import pytest

from tests import xreal_dummy_gateway


def test_dummy_metadata_matches_locked_xreal_contract() -> None:
    assert xreal_dummy_gateway.METADATA == {
        "server": "policy_space",
        "version": 2,
        "policy_name": "dreamzero_x2robot",
        "protocol": "policy_space",
        "protocol_version": "2",
        "server_instance_id": "xreal-dummy-gateway",
        "session_mode": "exclusive",
        "embodiment": "ArtiXon Arm-6A",
        "action_dim": 14,
        "action_mode": "delta_ee",
        "action_horizon": 24,
    }


def test_dummy_infer_actions_returns_finite_24_by_14_chunk() -> None:
    response = xreal_dummy_gateway.dispatch("infer_actions", {})
    actions = response["actions"]

    assert isinstance(actions, list)
    assert len(actions) == 24
    assert all(len(row) == 14 for row in actions)
    assert all(
        math.isfinite(value) and value == 0.0 for row in actions for value in row
    )


@pytest.mark.parametrize(
    "endpoint", ["initialize_episode", "ingest_observation", "finalize_episode"]
)
def test_dummy_accepts_lifecycle_endpoints(endpoint: str) -> None:
    assert xreal_dummy_gateway.dispatch(endpoint, {}) == {"ok": True}


def test_dummy_rejects_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="Unknown endpoint"):
        xreal_dummy_gateway.dispatch("unexpected", {})
