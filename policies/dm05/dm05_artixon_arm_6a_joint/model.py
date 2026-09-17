"""Thin PolicySpace gateway for the DM0.5 X2Real HTTP inference service."""

from __future__ import annotations

import base64
import binascii
import http.client
import json
import os
import socket
import struct
import threading
import time
import zlib
from typing import Any
from urllib.parse import SplitResult, urlsplit

import numpy as np

from policy_space.base import PolicyModel

CAMERA_KEYS = ("camera_front", "camera_left", "camera_right")
STATE_KEYS = ("follow1_pos", "follow2_pos")
ROBOT_TYPE = "x2robot_dreamzero_dual_arm_joint"
RAW_ACTION_SHAPE = (50, 14)
PACKED_ACTION_SHAPE = (50, 26)
_MAX_RESPONSE_BYTES = 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value
    return cfg.get(cfg_key, default)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload))


def _validate_rgb_image(image: Any, *, camera: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"DM0.5 camera {camera!r} must be a numpy.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise TypeError(f"DM0.5 camera {camera!r} must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"DM0.5 camera {camera!r} must be non-empty HWC RGB, got {image.shape!r}")
    return np.ascontiguousarray(image)


def encode_rgb_png_base64(image: np.ndarray, *, camera: str) -> str:
    """Encode one HWC RGB uint8 image without resizing or cropping it."""
    rgb = _validate_rgb_image(image, camera=camera)
    height, width = rgb.shape[:2]
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = _PNG_SIGNATURE + _png_chunk(b"IHDR", header)
    png += _png_chunk(b"IDAT", zlib.compress(scanlines))
    png += _png_chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def pack_x2real_action(raw_actions: Any) -> np.ndarray:
    """Validate DM0.5's absolute 50x14 chunk and pack ArtiXon Arm-6A's 26D layout."""
    try:
        physical = np.asarray(raw_actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("OpenDM actions must be a numeric array") from exc
    if physical.shape != RAW_ACTION_SHAPE:
        raise ValueError(f"OpenDM actions must have shape {RAW_ACTION_SHAPE}, got {physical.shape!r}")
    if not np.isfinite(physical).all():
        raise ValueError("OpenDM actions contain NaN/Inf")

    packed = np.zeros(PACKED_ACTION_SHAPE, dtype=np.float32)
    packed[:, : RAW_ACTION_SHAPE[1]] = physical
    return packed


class _OpenDMHTTPClient:
    """Single-request JSON client with separate connect and read timeouts."""

    def __init__(self, endpoint: str, *, connect_timeout_seconds: float, read_timeout_seconds: float) -> None:
        self.endpoint = endpoint
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.read_timeout_seconds = float(read_timeout_seconds)
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("OpenDM connect/read timeouts must be greater than zero")

        parsed = urlsplit(endpoint)
        self._validate_endpoint(parsed)
        self._parsed = parsed
        self._request_target = parsed.path
        self._lock = threading.Lock()
        self._active_connection: http.client.HTTPConnection | None = None

    @staticmethod
    def _validate_endpoint(parsed: SplitResult) -> None:
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("OpenDM backend URL must use http:// or https://")
        if not parsed.hostname:
            raise ValueError("OpenDM backend URL must include a hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("OpenDM backend URL must not contain credentials")
        if parsed.query or parsed.fragment or parsed.path != "/v1/infer":
            raise ValueError("OpenDM backend URL must be the complete /v1/infer endpoint without query or fragment")

    def _new_connection(self) -> http.client.HTTPConnection:
        connection_cls = http.client.HTTPSConnection if self._parsed.scheme == "https" else http.client.HTTPConnection
        return connection_cls(
            self._parsed.hostname,
            self._parsed.port,
            timeout=self.connect_timeout_seconds,
        )

    def post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        connection = self._new_connection()
        with self._lock:
            self._active_connection = connection
        try:
            try:
                connection.connect()
            except (OSError, socket.timeout, http.client.HTTPException) as exc:
                raise ConnectionError(f"OpenDM connect failed: {exc}") from exc

            if connection.sock is None:
                raise ConnectionError("OpenDM connect failed: connection has no socket")
            connection.sock.settimeout(self.read_timeout_seconds)
            try:
                connection.request(
                    "POST",
                    self._request_target,
                    body=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
                response = connection.getresponse()
                response_body = response.read(_MAX_RESPONSE_BYTES + 1)
            except (OSError, socket.timeout, http.client.HTTPException) as exc:
                raise TimeoutError(f"OpenDM POST/read failed: {exc}") from exc

            if len(response_body) > _MAX_RESPONSE_BYTES:
                raise RuntimeError(f"OpenDM response exceeds {_MAX_RESPONSE_BYTES} bytes")
            if not 200 <= response.status < 300:
                detail = response_body[:512].decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenDM HTTP {response.status}: {detail}")
            try:
                decoded = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("OpenDM response is not valid UTF-8 JSON") from exc
            if not isinstance(decoded, dict):
                raise RuntimeError(f"OpenDM response must be a JSON object, got {type(decoded).__name__}")
            return decoded
        finally:
            connection.close()
            with self._lock:
                if self._active_connection is connection:
                    self._active_connection = None

    def close(self) -> None:
        with self._lock:
            connection, self._active_connection = self._active_connection, None
        if connection is not None:
            connection.close()


class Model(PolicyModel):
    """PolicySpace protocol adapter; OpenDM model loading stays on the GPU server."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        backend_url = str(_env_or(cfg, "DM05_X2REAL_BACKEND_URL", "backend_url", "http://127.0.0.1:8002/v1/infer"))
        connect_timeout = float(_env_or(cfg, "DM05_X2REAL_CONNECT_TIMEOUT_SECONDS", "connect_timeout_seconds", 30.0))
        read_timeout = float(_env_or(cfg, "DM05_X2REAL_READ_TIMEOUT_SECONDS", "read_timeout_seconds", 30.0))
        self._checkpoint_id = str(
            _env_or(
                cfg,
                "DM05_X2REAL_CHECKPOINT_ID",
                "checkpoint_id",
                "",
            )
        )
        seed_value = _env_or(cfg, "DM05_X2REAL_SAMPLING_SEED", "sampling_seed", None)
        self._sampling_seed = None if seed_value in (None, "") else int(seed_value)
        self._client = _OpenDMHTTPClient(
            backend_url,
            connect_timeout_seconds=connect_timeout,
            read_timeout_seconds=read_timeout,
        )
        self._latest_obs: dict[str, Any] | None = None
        self._last_error: str | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        del episode_info
        self._latest_obs = None
        self._last_error = None

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = self._validate_observation(obs)
        self._last_error = None

    def infer_actions(self) -> dict[str, Any]:
        if self._latest_obs is None:
            raise RuntimeError("No DM0.5 observation buffered. Call ingest_observation first.")
        started = time.monotonic()
        try:
            response = self._client.post_json(self._build_request(self._latest_obs))
            if "actions" not in response:
                raise RuntimeError("OpenDM response is missing 'actions'")
            packed = pack_x2real_action(response["actions"])
            backend_metadata = response.get("metadata")
            if backend_metadata is not None and not isinstance(backend_metadata, dict):
                raise RuntimeError("OpenDM response metadata must be a JSON object when present")
            backend_latency = (backend_metadata or {}).get("latency_ms")
            if backend_latency is not None:
                try:
                    backend_latency = float(backend_latency)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("OpenDM metadata.latency_ms must be numeric") from exc
                if not np.isfinite(backend_latency):
                    raise RuntimeError("OpenDM metadata.latency_ms must be finite")
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

        return {
            "actions": packed,
            "dm05_timing": {
                "backend_latency_ms": backend_latency,
                "gateway_roundtrip_ms": round((time.monotonic() - started) * 1000.0, 3),
            },
            "dm05_shapes": {"raw": list(RAW_ACTION_SHAPE), "packed": list(PACKED_ACTION_SHAPE)},
        }

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "dm05_backend": {
                "url": self._client.endpoint,
                "connect_timeout_seconds": self._client.connect_timeout_seconds,
                "read_timeout_seconds": self._client.read_timeout_seconds,
                "automatic_retries": 0,
            },
            "dm05_checkpoint_id": self._checkpoint_id,
            "dm05_input_contract": {
                "camera_slots": {"1": "head", "2": "left_wrist", "3": "right_wrist"},
                "state_shape": [14],
                "image_preprocessing": "none_gateway_png_only",
            },
            "dm05_output_contract": {"raw_shape": [50, 14], "packed_shape": [50, 26], "dtype": "float32"},
            "dm05_robot_type": ROBOT_TYPE,
            "dm05_action_semantics": "absolute_joint",
            "dm05_sampling_seed": self._sampling_seed,
        }

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _validate_observation(obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs, dict):
            raise TypeError(f"DM0.5 observation must be a mapping, got {type(obs).__name__}")
        images = obs.get("images")
        state = obs.get("state")
        if not isinstance(images, dict):
            raise TypeError("DM0.5 observation.images must be a mapping")
        if not isinstance(state, dict):
            raise TypeError("DM0.5 observation.state must be a mapping")

        validated_images: dict[str, np.ndarray] = {}
        for camera in CAMERA_KEYS:
            if camera not in images:
                raise KeyError(f"DM0.5 observation is missing camera {camera!r}")
            image = images[camera]
            validated_images[camera] = _validate_rgb_image(image, camera=camera).copy()

        validated_state: dict[str, np.ndarray] = {}
        for key in STATE_KEYS:
            if key not in state:
                raise KeyError(f"DM0.5 observation is missing state field {key!r}")
            try:
                value = np.asarray(state[key], dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"DM0.5 state {key!r} must contain numeric values") from exc
            if value.shape != (7,):
                raise ValueError(f"DM0.5 state {key!r} must have shape (7,), got {value.shape!r}")
            if not np.isfinite(value).all():
                raise ValueError(f"DM0.5 state {key!r} contains NaN/Inf")
            validated_state[key] = value.copy()

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("DM0.5 observation requires the current episode instruction")
        return {"images": validated_images, "state": validated_state, "instruction": instruction}

    def _build_request(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = {
            str(index): encode_rgb_png_base64(obs["images"][camera], camera=camera)
            for index, camera in enumerate(CAMERA_KEYS, start=1)
        }
        state = np.concatenate([obs["state"][key] for key in STATE_KEYS]).astype(np.float32, copy=False)
        request: dict[str, Any] = {
            "observation": {
                "prompt": obs["instruction"],
                "state": [float(value) for value in state],
                "images": images,
                "robot_type": ROBOT_TYPE,
            }
        }
        if self._sampling_seed is not None:
            request["sampling"] = {"seed": self._sampling_seed}
        return request
