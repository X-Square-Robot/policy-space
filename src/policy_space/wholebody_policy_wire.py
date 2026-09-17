"""Gateway-local packing for the Quanta X1 whole-body policy wire.

Native model services retain their own named response fields. This module owns
only the shared, model-neutral ``[T, 20]`` envelope returned by PolicySpace.
Simulator conversion remains exclusively in ManaEnv.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

WIRE_SCHEMA = "manaenv_ex001_wholebody_policy_wire_20d_v1"
ACTION_DIM = 20
BASE_REPRESENTATION = "vx_vy_yaw_rate"
EE_ROTATION_REPRESENTATION = "euler_xyz"
EE_REFERENCE = "init_ee_pose"


def pack_named_response(
    response: Mapping[str, Any],
    *,
    source_name: str,
    base_field: str,
) -> dict[str, Any]:
    """Validate native whole-body fields and pack the public policy-wire envelope."""
    if not isinstance(response, Mapping):
        raise TypeError(
            f"native {source_name} response must be a mapping, got {type(response).__name__}"
        )
    if "error" in response:
        raise RuntimeError(
            f"native {source_name} returned error: {response['error']!r}"
        )

    required = {
        "follow1_pos": 7,
        "follow2_pos": 7,
        base_field: 3,
        "lift": 1,
        "head_pos": 2,
    }
    missing = [name for name in required if name not in response]
    if missing:
        raise ValueError(
            f"native {source_name} response missing whole-body fields: {', '.join(missing)}"
        )

    fields = {
        name: _as_chunk(
            response[name], source_name=source_name, field=name, width=width
        )
        for name, width in required.items()
    }
    horizon = fields["follow1_pos"].shape[0]
    mismatched = {
        name: list(value.shape)
        for name, value in fields.items()
        if value.shape[0] != horizon
    }
    if mismatched:
        raise ValueError(
            f"native {source_name} response has inconsistent action horizon {horizon}: {mismatched}"
        )

    chunk = np.empty((horizon, ACTION_DIM), dtype=np.float32)
    chunk[:, 0:7] = fields["follow1_pos"]
    chunk[:, 7:14] = fields["follow2_pos"]
    chunk[:, 14:17] = fields[base_field]
    chunk[:, 17:18] = fields["lift"]
    chunk[:, 18:20] = fields["head_pos"]
    return {
        "wire_schema": WIRE_SCHEMA,
        "action_dim": ACTION_DIM,
        "base_representation": BASE_REPRESENTATION,
        "ee_rotation_representation": EE_ROTATION_REPRESENTATION,
        "ee_reference": EE_REFERENCE,
        "action_chunk": chunk,
    }


def _as_chunk(value: Any, *, source_name: str, field: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != width:
        raise ValueError(
            f"native {source_name} {field} must have shape [T,{width}], got {array.shape!r}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"native {source_name} {field} contains NaN/Inf")
    return np.ascontiguousarray(array, dtype=np.float32)
