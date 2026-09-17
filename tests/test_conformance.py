"""Action-output contract validation tests."""

from __future__ import annotations

import numpy as np
import pytest

from policy_space.conformance import (
    make_observation_fixture,
    normalize_action_output,
    validate_observation_input,
)
from policy_space.contracts.registry import ContractRegistry

CONTRACT = {"response_key": "actions", "action_dim": 14, "dtype": "float32"}


def test_action_output_is_normalized_to_finite_float32() -> None:
    response = normalize_action_output(np.zeros((2, 14), dtype=np.float64), CONTRACT)

    assert response["actions"].shape == (2, 14)
    assert response["actions"].dtype == np.float32


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((0, 14), dtype=np.float32),
        np.zeros((2, 13), dtype=np.float32),
        np.full((2, 14), np.nan, dtype=np.float32),
        np.full((2, 14), 1e300, dtype=np.float64),
        np.zeros((2, 14), dtype=np.complex64),
        np.zeros((2, 14), dtype=np.bool_),
    ],
)
def test_action_output_rejects_invalid_shape_or_values(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_action_output(value, CONTRACT)


def test_single_action_is_normalized_to_one_step_sequence() -> None:
    response = normalize_action_output(np.zeros(14, dtype=np.float32), CONTRACT)

    assert response["actions"].shape == (1, 14)


def test_action_output_horizon_must_equal_static_metadata() -> None:
    with pytest.raises(ValueError, match="action_horizon 24; got 23"):
        normalize_action_output(
            np.zeros((23, 14), dtype=np.float32),
            CONTRACT,
            expected_horizon=24,
        )

    response = normalize_action_output(
        np.zeros((24, 14), dtype=np.float32),
        CONTRACT,
        expected_horizon=24,
    )
    assert response["actions"].shape == (24, 14)


def test_semantic_wire_validates_envelope_metadata() -> None:
    contract = {
        "response_key": "action_chunk",
        "action_dim": 20,
        "wire_schema": "wire@1",
        "base_representation": "vx_vy_yaw_rate",
    }

    with pytest.raises(ValueError, match="wire_schema"):
        normalize_action_output(
            {"action_chunk": np.zeros((1, 20), dtype=np.float32)},
            contract,
        )


def test_public_observation_fixture_matches_contract() -> None:
    contract = ContractRegistry.builtin().observations["bimanual_rgb_ee@1"]
    observation = make_observation_fixture(contract)

    validate_observation_input(observation, contract)
    assert sorted(observation["state"]) == ["follow1_pos", "follow2_pos"]


def test_public_observation_rejects_missing_camera_role() -> None:
    contract = ContractRegistry.builtin().observations["bimanual_rgb_ee@1"]
    observation = make_observation_fixture(contract)
    del observation["images"]["camera_right"]

    with pytest.raises(ValueError, match="camera_right"):
        validate_observation_input(observation, contract)
