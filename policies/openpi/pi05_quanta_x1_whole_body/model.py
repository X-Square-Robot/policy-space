"""Pi0.5 Quanta X1 native-response adapter for the whole-body policy wire.

The native OpenPI service owns model loading and inference. This deployment
adapter only turns its named Quanta X1 response into the model-neutral action wire
consumed by ManaEnv. Simulator-specific conversions deliberately remain in
``PolicySpaceImpl`` and the ManaEnv whole-body policy-wire contract.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import msgpack
import numpy as np
from websockets.sync.client import connect

from policy_space.base import PolicyModel
from policy_space.wholebody_policy_wire import pack_named_response
from policy_space.transport import send_websocket_bounded, unpack_ndarray


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value
    return cfg.get(cfg_key, default)


def _pack_openpi_array(obj: Any) -> Any:
    """Encode arrays with the OpenPI ``msgpack_numpy`` envelope, not pickle."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported native Pi0.5 dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def pack_native_pi05_request(payload: dict[str, Any]) -> bytes:
    """Pack one native OpenPI request as msgpack bytes."""
    if not isinstance(payload, dict):
        raise TypeError(f"native Pi0.5 request must be a mapping, got {type(payload).__name__}")
    packed = msgpack.packb(payload, default=_pack_openpi_array)
    if not isinstance(packed, bytes):
        raise TypeError("native Pi0.5 request packing must return bytes")
    return packed


def unpack_native_pi05_message(raw: Any, *, kind: str) -> dict[str, Any]:
    """Decode one OpenPI websocket frame.

    The native server sends msgpack bytes. A text frame is the exception
    traceback path from ``WebsocketPolicyServer``, not JSON metadata.
    """
    if isinstance(raw, str):
        raise RuntimeError(f"native Pi0.5 {kind} error: {raw}")
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise RuntimeError(f"native Pi0.5 {kind} must be msgpack bytes, got {type(raw).__name__}")
    message = msgpack.unpackb(bytes(raw), object_hook=unpack_ndarray)
    if not isinstance(message, dict):
        raise RuntimeError(f"native Pi0.5 {kind} must be a mapping, got {type(message).__name__}")
    return message


def pack_pi05_named_response(response: dict[str, Any]) -> dict[str, Any]:
    """Pack named native Pi0.5 Quanta X1 actions into the public 20-D wire."""
    return pack_named_response(response, source_name="Pi0.5", base_field="velocity_decomposed")


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


class _NativePi05Client:
    """Persistent client for the native Quanta X1 Pi0.5 WebSocket server."""

    def __init__(self, endpoint: str, *, timeout_seconds: float) -> None:
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._ws = None

    def _connect(self) -> None:
        self.close()
        self._ws = connect(self._endpoint, open_timeout=self._timeout_seconds, max_size=None, compression=None)
        unpack_native_pi05_message(self._ws.recv(timeout=self._timeout_seconds), kind="metadata")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            self._connect()
        assert self._ws is not None
        try:
            send_websocket_bounded(
                self._ws,
                pack_native_pi05_request(payload),
                timeout_seconds=self._timeout_seconds,
                operation="native Pi0.5",
            )
            return unpack_native_pi05_message(self._ws.recv(timeout=self._timeout_seconds), kind="response")
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
    """PolicySpace facade around a native Quanta X1 Pi0.5 inference service."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        nested = cfg.get("model", {}).get("cfg", {}) if isinstance(cfg.get("model"), dict) else {}
        knobs = dict(cfg)
        knobs.update(nested)
        address = str(_env_or(knobs, "PI05_NATIVE_ADDRESS", "native_address", "127.0.0.1"))
        port = int(_env_or(knobs, "PI05_NATIVE_PORT", "native_port", 8002))
        endpoint = str(_env_or(knobs, "PI05_NATIVE_ENDPOINT", "native_endpoint", f"ws://{address}:{port}"))
        self._client = _NativePi05Client(endpoint, timeout_seconds=float(knobs.get("timeout_seconds", 180.0)))
        self._instruction = str(_env_or(knobs, "PI05_INSTRUCTION", "instruction", ""))
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
        semantic = pack_pi05_named_response(native)
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
            raise ValueError(f"missing native Pi0.5 camera; accepted aliases={names!r}, present={sorted(images)!r}")

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
            name: np.asarray(value, dtype=np.float32).reshape(-1).tolist()
            for name, value in (obs.get("state") or {}).items()
            if value is not None
        }
        required_state = ("follow1_pos", "follow2_pos", "velocity_decomposed_odom", "lift", "head_pos")
        missing = [name for name in required_state if name not in state]
        if missing:
            raise ValueError(f"missing native Pi0.5 Quanta X1 state fields: {', '.join(missing)}")
        instruction = obs.get("instruction")
        return {
            "views": views,
            "state": state,
            "instruction": instruction if isinstance(instruction, str) and instruction.strip() else self._instruction,
        }
