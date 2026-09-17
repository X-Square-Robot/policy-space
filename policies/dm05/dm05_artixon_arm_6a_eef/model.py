"""DM0.5 X2Real EE adapter for PolicySpace v2.

The OpenDM checkpoint speaks ``[xyz, rot6d_row, gripper] * 2`` (20-D), while
the public ArtiXon Arm-6A PolicySpace wire speaks ``[xyz, euler_xyz, gripper] * 2``
(14-D).  This adapter keeps that conversion at the model boundary.
"""

from __future__ import annotations

import base64
import binascii
import http.client
import json
import os
import socket
import struct
import zlib
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from scipy.spatial.transform import Rotation

from policy_space.base import PolicyModel

MODEL_ACTION_DIM = 20
MODEL_ACTION_HORIZON = 50
WIRE_ACTION_DIM = 14
# x2real's registered ArtiXon Arm-6A wire contract accepts 32 rows. DM0.5 still
# predicts 50 rows internally; truncation is confined to this adapter.
WIRE_ACTION_HORIZON = 32
ROBOT_TYPE = "x2robot_dreamzero_dual_arm"
CAMERA_KEYS = ("camera_front", "camera_left", "camera_right")
STATE_KEYS = ("follow1_pos", "follow2_pos")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _encode_png(image: Any, camera: str) -> str:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{camera} must be HWC RGB, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    array = np.ascontiguousarray(array)
    h, w = array.shape[:2]
    scanlines = b"".join(b"\0" + array[row].tobytes() for row in range(h))
    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = _PNG_SIGNATURE + _png_chunk(b"IHDR", header)
    png += _png_chunk(b"IDAT", zlib.compress(scanlines)) + _png_chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def _euler_to_rot6d(euler: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler, dtype=np.float32)
    if euler.shape[-1] != 3:
        raise ValueError(f"Euler rotation must end in 3, got {euler.shape}")
    leading = euler.shape[:-1]
    matrices = Rotation.from_euler("xyz", euler.reshape(-1, 3), degrees=False).as_matrix()
    # Training conversion calls this rot6d_row: first two matrix rows, row-major.
    return matrices[:, :2, :].reshape(*leading, 6).astype(np.float32, copy=False)


def _rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must end in 6, got {rot6d.shape}")
    leading = rot6d.shape[:-1]
    rows = rot6d.reshape(-1, 2, 3)
    # Project noisy predictions to SO(3), preserving the row-major convention.
    matrices = np.stack((rows[:, 0], rows[:, 1], np.cross(rows[:, 0], rows[:, 1])), axis=1)
    left, _, right_t = np.linalg.svd(matrices)
    normalized = left @ right_t
    reflected = np.linalg.det(normalized) < 0
    if np.any(reflected):
        left = left.copy()
        left[reflected, :, -1] *= -1
        normalized[reflected] = left[reflected] @ right_t[reflected]
    return Rotation.from_matrix(normalized).as_euler("xyz", degrees=False).reshape(*leading, 3).astype(np.float32)


def _wire_to_model(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (WIRE_ACTION_DIM,):
        raise ValueError(f"wire state must be shape (14,), got {state.shape}")
    arms = []
    for start in (0, 7):
        arm = state[start : start + 7]
        arms.append(np.concatenate((arm[:3], _euler_to_rot6d(arm[3:6]), arm[6:7])))
    return np.concatenate(arms).astype(np.float32, copy=False)


def _model_to_wire(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != MODEL_ACTION_DIM:
        raise ValueError(f"OpenDM EE actions must be [T,20], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("OpenDM EE actions contain NaN/Inf")
    arms = []
    for start in (0, 10):
        arm = actions[:, start : start + 10]
        arms.append(np.concatenate((arm[:, :3], _rot6d_to_euler(arm[:, 3:9]), arm[:, 9:10]), axis=1))
    return np.concatenate(arms, axis=1).astype(np.float32, copy=False)


class _HTTPClient:
    def __init__(self, endpoint: str, timeout: float) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path != "/v1/infer":
            raise ValueError("backend_url must be an http(s) /v1/infer endpoint")
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self._parsed = parsed

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        cls = http.client.HTTPSConnection if self._parsed.scheme == "https" else http.client.HTTPConnection
        conn = cls(self._parsed.hostname, self._parsed.port, timeout=self.timeout)
        try:
            conn.request("POST", self._parsed.path, body=body, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if not 200 <= response.status < 300:
                raise RuntimeError(f"OpenDM HTTP {response.status}: {raw[:512]!r}")
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise RuntimeError("OpenDM response must be a JSON object")
            return decoded
        except (OSError, socket.timeout, http.client.HTTPException) as exc:
            raise RuntimeError(f"OpenDM request failed: {exc}") from exc
        finally:
            conn.close()


class Model(PolicyModel):
    """PolicySpace v2 model adapter; OpenDM remains the native HTTP backend."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        endpoint = os.environ.get("DM05_X2REAL_EE_BACKEND_URL", cfg.get("backend_url", "http://127.0.0.1:8002/v1/infer"))
        timeout = float(os.environ.get("DM05_X2REAL_EE_TIMEOUT_SECONDS", cfg.get("timeout_seconds", 180)))
        self._client = _HTTPClient(str(endpoint), timeout)
        self._checkpoint_id = str(
            os.environ.get("DM05_X2REAL_EE_CHECKPOINT_ID")
            or cfg.get("checkpoint_id")
            or ""
        )
        self._latest_obs: dict[str, Any] | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        del episode_info
        self._latest_obs = None

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict) or not isinstance(obs.get("images"), dict) or not isinstance(obs.get("state"), dict):
            raise TypeError("DM0.5 EE observation requires images and state mappings")
        for camera in CAMERA_KEYS:
            if camera not in obs["images"]:
                raise KeyError(f"missing camera {camera!r}")
        state = []
        for key in STATE_KEYS:
            value = np.asarray(obs["state"].get(key), dtype=np.float32).reshape(-1)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"state[{key!r}] must be finite shape (7,), got {value.shape}")
            state.append(value)
        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("DM0.5 EE requires a non-empty instruction")
        self._latest_obs = {"images": obs["images"], "state": state, "instruction": instruction}

    def infer_actions(self) -> dict[str, Any]:
        if self._latest_obs is None:
            raise RuntimeError("No DM0.5 EE observation buffered")
        state14 = np.concatenate(self._latest_obs["state"]).astype(np.float32, copy=False)
        payload = {
            "observation": {
                "prompt": self._latest_obs["instruction"],
                "state": _wire_to_model(state14).tolist(),
                "images": {str(i): _encode_png(self._latest_obs["images"][key], key) for i, key in enumerate(CAMERA_KEYS, 1)},
                "robot_type": ROBOT_TYPE,
            }
        }
        response = self._client.post(payload)
        wire = _model_to_wire(response.get("actions"))
        if wire.shape[0] < WIRE_ACTION_HORIZON:
            raise ValueError(f"OpenDM returned only {wire.shape[0]} actions, expected at least {WIRE_ACTION_HORIZON}")
        return {"actions": wire[:WIRE_ACTION_HORIZON]}

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "dm05_checkpoint_id": self._checkpoint_id,
            "dm05_model_action_shape": [MODEL_ACTION_HORIZON, MODEL_ACTION_DIM],
            "dm05_wire_action_shape": [WIRE_ACTION_HORIZON, WIRE_ACTION_DIM],
            "dm05_model_rotation": "rot6d_row",
            "dm05_wire_rotation": "euler_xyz",
            "dm05_action_semantics": "absolute_init_ee_pose_relative",
            "dm05_backend": self._client.endpoint,
        }

    def finalize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        self._latest_obs = None

    def close(self) -> None:
        self._latest_obs = None
