"""Pure protocol tests for the WallX Quanta X1 whole-body gateway."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODEL_PATH = (
    Path(__file__).parents[1] / "policies" / "wallx" / "wallx_quanta_x1_whole_body" / "model.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "wallx_quanta_x1_model_test", _MODEL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
pack_wallx_named_response = _MODULE.pack_wallx_named_response


def _native_response(horizon: int = 2) -> dict[str, np.ndarray]:
    return {
        "follow1_pos": np.arange(horizon * 7, dtype=np.float32).reshape(horizon, 7),
        "follow2_pos": (100 + np.arange(horizon * 7, dtype=np.float32)).reshape(
            horizon, 7
        ),
        "velocity_decomposed_odom": (
            200 + np.arange(horizon * 3, dtype=np.float32)
        ).reshape(horizon, 3),
        "lift": (300 + np.arange(horizon, dtype=np.float32)).reshape(horizon, 1),
        "head_pos": (400 + np.arange(horizon * 2, dtype=np.float32)).reshape(
            horizon, 2
        ),
    }


def test_wallx_named_response_packs_exact_policy_wire_layout():
    source = _native_response()
    packed = pack_wallx_named_response(source)

    assert packed["wire_schema"] == "manaenv_ex001_wholebody_policy_wire_20d_v1"
    assert packed["action_dim"] == 20
    assert packed["base_representation"] == "vx_vy_yaw_rate"
    assert packed["ee_rotation_representation"] == "euler_xyz"
    assert packed["ee_reference"] == "init_ee_pose"
    np.testing.assert_array_equal(packed["action_chunk"][:, 0:7], source["follow1_pos"])
    np.testing.assert_array_equal(
        packed["action_chunk"][:, 7:14], source["follow2_pos"]
    )
    np.testing.assert_array_equal(
        packed["action_chunk"][:, 14:17], source["velocity_decomposed_odom"]
    )
    np.testing.assert_array_equal(packed["action_chunk"][:, 17:18], source["lift"])
    np.testing.assert_array_equal(packed["action_chunk"][:, 18:20], source["head_pos"])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda response: response.pop("head_pos"), "missing whole-body fields"),
        (
            lambda response: response.__setitem__(
                "lift", np.zeros((1, 1), dtype=np.float32)
            ),
            "inconsistent action horizon",
        ),
        (
            lambda response: response.__setitem__(
                "follow1_pos", np.zeros((2, 6), dtype=np.float32)
            ),
            "shape",
        ),
    ],
)
def test_wallx_named_response_fails_closed(mutator, message):
    response = _native_response()
    mutator(response)
    with pytest.raises(ValueError, match=message):
        pack_wallx_named_response(response)
