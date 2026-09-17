"""Contract validation shared by the CLI and service runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def validate_observation_input(
    observation: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Validate one mapped observation against its selected public schema."""

    if not isinstance(observation, Mapping):
        raise ValueError("observation must be an object")
    fields = contract.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("observation contract requires a fields object")

    image_contract = fields.get("images")
    if isinstance(image_contract, Mapping):
        images = observation.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("observation requires an images object")
        for role in image_contract.get("roles") or []:
            if role not in images:
                raise ValueError(f"observation is missing image role {role!r}")
            image = np.asarray(images[role])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"image {role!r} must have uint8 shape [height,width,3]; "
                    f"got dtype={image.dtype}, shape={image.shape}"
                )

    state_contract = fields.get("state")
    if isinstance(state_contract, Mapping):
        state = observation.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("observation requires a state object")
        if state_contract.get("type") == "named_float_vectors":
            for name, value in state.items():
                _validate_state_vector(name, value, expected_shape=None)
        else:
            for name, specification in state_contract.items():
                if name not in state:
                    raise ValueError(f"observation is missing state field {name!r}")
                expected_shape = (
                    tuple(specification.get("shape") or [])
                    if isinstance(specification, Mapping)
                    else None
                )
                _validate_state_vector(name, state[name], expected_shape)


def _validate_state_vector(
    name: str, value: Any, expected_shape: tuple[int, ...] | None
) -> None:
    array = np.asarray(value)
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"state field {name!r} must have shape {expected_shape}; got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"state field {name!r} must be floating point; got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"state field {name!r} contains NaN or infinity")


def make_observation_fixture(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Create a minimal deterministic observation for local conformance checks."""

    fields = contract.get("fields") or {}
    image_contract = fields.get("images") or {}
    images = {
        role: np.zeros((2, 2, 3), dtype=np.uint8)
        for role in image_contract.get("roles") or []
    }
    state_contract = fields.get("state") or {}
    if state_contract.get("type") == "named_float_vectors":
        state: dict[str, np.ndarray] = {}
    else:
        state = {
            name: np.zeros(tuple(specification.get("shape") or []), dtype=np.float32)
            for name, specification in state_contract.items()
        }
    return {
        "episode_id": "policy-space-check",
        "step": 0,
        "instruction": "conformance check",
        "images": images,
        "state": state,
    }


def normalize_action_output(
    output: Any,
    contract: Mapping[str, Any],
    *,
    expected_horizon: int | None = None,
) -> dict[str, Any]:
    """Return a validated wire response for one declared action contract."""

    response_key = str(contract.get("response_key") or "actions")
    if isinstance(output, Mapping):
        response = dict(output)
        if response_key not in response:
            raise ValueError(f"policy output requires {response_key!r}")
        value = response[response_key]
    else:
        if response_key != "actions":
            raise ValueError(
                f"action contract requires a {response_key!r} response envelope"
            )
        response = {}
        value = output

    array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    expected_dim = int(contract["action_dim"])
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != expected_dim:
        raise ValueError(
            f"{response_key} must have shape [T,{expected_dim}] with T >= 1; "
            f"got {array.shape}"
        )
    if expected_horizon is not None:
        if isinstance(expected_horizon, bool) or not isinstance(
            expected_horizon, (int, np.integer)
        ):
            raise ValueError(
                "metadata action_horizon must be a positive integer; "
                f"got {expected_horizon!r}"
            )
        expected_horizon = int(expected_horizon)
        if expected_horizon < 1:
            raise ValueError(
                "metadata action_horizon must be a positive integer; "
                f"got {expected_horizon!r}"
            )
        if array.shape[0] != expected_horizon:
            raise ValueError(
                f"{response_key} horizon must equal metadata action_horizon "
                f"{expected_horizon}; got {array.shape[0]}"
            )
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{response_key} must be numeric; got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{response_key} contains NaN or infinity")

    if response_key != "actions":
        wire_checks = {
            "wire_schema": "wire_schema",
            "base_representation": "base_representation",
            "rotation": "ee_rotation_representation",
            "pose_reference": "ee_reference",
        }
        for contract_key, response_field in wire_checks.items():
            expected = contract.get(contract_key)
            if expected is not None and response.get(response_field) != expected:
                raise ValueError(
                    f"policy output {response_field} must be {expected!r}; "
                    f"got {response.get(response_field)!r}"
                )

    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{response_key} exceeds the finite float32 range")
    response[response_key] = normalized
    return response


__all__ = [
    "make_observation_fixture",
    "normalize_action_output",
    "validate_observation_input",
]
