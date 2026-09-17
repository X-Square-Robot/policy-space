"""WallX Quanta X1 native-response gateway for the whole-body policy wire.

The native WallX service owns model loading and returns its established
serialized Quanta X1 fields. This PolicySpace model forwards one observation to
that service, validates its named response, and packs the public semantic
action vector. It does *not* perform simulator conversions: ManaEnv owns
home-EE composition, wheel conversion, and gripper scaling.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
from websockets.sync.client import connect

from policy_space.base import PolicyModel
from policy_space.wholebody_policy_wire import pack_named_response


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value
    return cfg.get(cfg_key, default)


def _as_chunk(value: Any, *, field: str, width: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != width:
        raise ValueError(f"native WallX {field} must have shape [T,{width}], got {arr.shape!r}")
    if not np.isfinite(arr).all():
        raise ValueError(f"native WallX {field} contains NaN/Inf")
    return np.ascontiguousarray(arr, dtype=np.float32)


def pack_wallx_named_response(response: dict[str, Any]) -> dict[str, Any]:
    """Pack native serialized Quanta X1 action fields into the public 20-D wire.

    Native WallX serializes its internal 6-D rotations to Euler XYZ in
    ``follow*_pos``. This gateway must not convert rotations again. Values stay
    reset-home-relative; the benchmark reconstructs absolute IK targets.
    """
    return pack_named_response(response, source_name="WallX", base_field="velocity_decomposed_odom")


def _jpeg_base64(image: Any, *, quality: int) -> str:
    import cv2

    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape!r}")
    if array.dtype != np.uint8:
        if (float(np.max(array)) if array.size else 0.0) <= 1.0:
            array = array * 255.0
        array = array.clip(0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("failed to JPEG-encode PolicySpace image")
    return base64.b64encode(encoded).decode("ascii")


class _NativeWallXClient:
    """Persistent flat-protocol client for a loopback WallX service."""

    def __init__(self, endpoint: str, *, timeout_seconds: float) -> None:
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._ws = None

    def _connect(self) -> None:
        self.close()
        self._ws = connect(self._endpoint, open_timeout=self._timeout_seconds, max_size=None, compression=None)
        metadata = self._ws.recv(timeout=self._timeout_seconds)
        if not isinstance(metadata, bytes):
            raise RuntimeError("native WallX metadata must be msgpack bytes")
        if not isinstance(msgpack.unpackb(metadata, object_hook=msgpack_numpy.decode), dict):
            raise RuntimeError("native WallX metadata must be a mapping")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            self._connect()
        assert self._ws is not None
        try:
            self._ws.send(msgpack.packb(payload, default=msgpack_numpy.encode))
            raw = self._ws.recv(timeout=self._timeout_seconds)
            if isinstance(raw, str):
                raise RuntimeError(f"native WallX server error: {raw}")
            response = msgpack.unpackb(raw, object_hook=msgpack_numpy.decode)
            if not isinstance(response, dict):
                raise RuntimeError(f"native WallX response must be a mapping, got {type(response).__name__}")
            return response
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None


class Model(PolicyModel):
    """PolicySpace facade around a native WallX Quanta X1 inference service."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        nested = cfg.get("model", {}).get("cfg", {}) if isinstance(cfg.get("model"), dict) else {}
        knobs = dict(cfg)
        knobs.update(nested)
        address = str(_env_or(knobs, "WALLX_NATIVE_ADDRESS", "native_address", "127.0.0.1"))
        port = int(_env_or(knobs, "WALLX_NATIVE_PORT", "native_port", 8002))
        endpoint = str(_env_or(knobs, "WALLX_NATIVE_ENDPOINT", "native_endpoint", f"ws://{address}:{port}"))
        self._client = _NativeWallXClient(endpoint, timeout_seconds=float(knobs.get("timeout_seconds", 180.0)))
        self._instruction = str(_env_or(knobs, "WALLX_INSTRUCTION", "instruction", ""))
        self._jpeg_quality = int(knobs.get("jpeg_quality", 95))
        self._latest_obs: dict[str, Any] | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        self._latest_obs = None
        if episode_info and isinstance(episode_info.get("instruction"), str) and episode_info["instruction"].strip():
            self._instruction = episode_info["instruction"]

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = obs

    def infer_actions(self) -> dict[str, Any]:
        if self._latest_obs is None:
            raise RuntimeError("No observation buffered. Call ingest_observation first.")
        started = time.monotonic()
        native = self._client.infer(self._to_native_request(self._latest_obs))
        semantic = pack_wallx_named_response(native)
        semantic["server_timing"] = {
            "native_infer_ms": round((time.monotonic() - started) * 1000.0, 3),
            **(native.get("server_timing") if isinstance(native.get("server_timing"), dict) else {}),
        }
        return semantic

    def close(self) -> None:
        self._client.close()

    def _to_native_request(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs.get("images") or {}

        def image_for(*names: str) -> Any:
            for name in names:
                if name in images and images[name] is not None:
                    return images[name]
            raise ValueError(f"missing native WallX camera; accepted aliases={names!r}, present={sorted(images)!r}")

        views = {
            "camera_front": _jpeg_base64(
                image_for("camera_front", "head_cam", "face_view"), quality=self._jpeg_quality
            ),
            "camera_left": _jpeg_base64(
                image_for("camera_left", "left_wrist_cam", "left_wrist_view"), quality=self._jpeg_quality
            ),
            "camera_right": _jpeg_base64(
                image_for("camera_right", "right_wrist_cam", "right_wrist_view"), quality=self._jpeg_quality
            ),
        }
        state = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in (obs.get("state") or {}).items()
            if value is not None
        }
        required_state = ("follow1_pos", "follow2_pos", "velocity_decomposed_odom", "lift", "head_pos")
        missing = [name for name in required_state if name not in state]
        if missing:
            raise ValueError(f"missing native WallX Quanta X1 state fields: {', '.join(missing)}")
        instruction = obs.get("instruction")
        return {
            "views": views,
            "state": state,
            "instruction": instruction if isinstance(instruction, str) and instruction.strip() else self._instruction,
        }
