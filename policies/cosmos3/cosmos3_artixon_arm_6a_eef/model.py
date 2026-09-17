"""PolicyModel implementation for Cosmos 3 X2Robot.

Migrates the old SDK ``Policy`` interface to the new ``PolicyModel`` base class.
Speaks a versioned 14-D bimanual end-effector msgpack WebSocket protocol.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from policy_space.base import PolicyModel

# --- Protocol constants ---
PROTOCOL_NAME = "cosmos3-x2robot"
PROTOCOL_VERSION = 1
ACTION_SPACE = "x2_bimanual_ee_init_relative"
WIRE_POSE_REFERENCE = "init_ee_pose"
ROTATION_FORMAT = "euler_xyz"
ACTION_DIM = 14
STATE_SPACE = "x2_bimanual_ee_init_relative"
TRANSLATION_RELATIVE_FRAME = "robot_root"
ROTATION_COMPOSITION = "relative_left_multiply"

_DEFAULT_CAMERA_MAP = {
    "camera_front": "observation/head_image",
    "camera_left": "observation/left_wrist_image",
    "camera_right": "observation/right_wrist_image",
}

_STATE_FIELDS = ("follow1_pos", "follow2_pos")
_STATE_WIRE_KEYS = {
    "follow1_pos": "observation/left_eef",
    "follow2_pos": "observation/right_eef",
}


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip() != "":
        return raw
    if cfg.get(cfg_key) is not None:
        return cfg[cfg_key]
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Inline msgpack WebSocket client
# ---------------------------------------------------------------------------


class _MsgpackWebsocketClient:
    """Minimal msgpack WebSocket client with ndarray codec."""

    def __init__(self, host: str, port: int, timeout_seconds: float) -> None:
        import msgpack
        from websockets.sync.client import connect

        self._msgpack = msgpack
        self._timeout_seconds = float(timeout_seconds)
        self._ws = connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=self._timeout_seconds,
            ping_interval=None,
            proxy=None,
        )
        metadata_raw = self._ws.recv(timeout=self._timeout_seconds)
        if isinstance(metadata_raw, str):
            raise RuntimeError(f"server returned text during metadata handshake: {metadata_raw}")
        self._server_metadata = self._unpack(metadata_raw)

    @staticmethod
    def _pack_array(value: Any) -> Any:
        if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported ndarray dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        if isinstance(value, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": value.item(),
                b"dtype": value.dtype.str,
            }
        return value

    @staticmethod
    def _unpack_array(value: dict) -> Any:
        if b"__ndarray__" in value:
            return np.ndarray(
                buffer=value[b"data"],
                dtype=np.dtype(value[b"dtype"]),
                shape=value[b"shape"],
            )
        if b"__npgeneric__" in value:
            return np.dtype(value[b"dtype"]).type(value[b"data"])
        return value

    def _pack(self, value: Any) -> bytes:
        return self._msgpack.packb(value, default=self._pack_array)

    def _unpack(self, value: bytes) -> Any:
        return self._msgpack.unpackb(value, object_hook=self._unpack_array)

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._pack(obs))
        response = self._ws.recv(timeout=self._timeout_seconds)
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpack(response)

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self._ws.close()


# ---------------------------------------------------------------------------
# PolicyModel implementation
# ---------------------------------------------------------------------------


class Model(PolicyModel):
    """Cosmos 3 X2Robot policy using the PolicyModel interface."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)

        self.server_address = str(_env_or(cfg, "COSMOS3_X2ROBOT_SERVER_ADDRESS", "server_address", "127.0.0.1"))
        self.server_port = int(_env_or(cfg, "COSMOS3_X2ROBOT_SERVER_PORT", "server_port", 8002))
        timeout_ms = int(cfg.get("timeout_ms", 120000) or 120000)
        self.timeout_seconds = float(
            _env_or(cfg, "COSMOS3_X2ROBOT_TIMEOUT_SECONDS", "timeout_seconds", timeout_ms / 1000.0)
        )
        self.instruction = str(_env_or(cfg, "COSMOS3_X2ROBOT_INSTRUCTION", "instruction", ""))
        self.require_instruction = _as_bool(cfg.get("require_instruction", True))

        self.camera_map: dict[str, str] = dict(cfg.get("camera_map") or _DEFAULT_CAMERA_MAP)
        self.model_fps = float(cfg.get("model_fps", 15.0))
        self.control_fps = float(cfg.get("control_fps", 30.0))
        ratio = self.control_fps / self.model_fps
        self.action_repeat = int(round(ratio))
        if self.model_fps <= 0 or self.control_fps <= 0 or self.action_repeat < 1:
            raise ValueError("model_fps and control_fps must be positive")
        if not np.isclose(ratio, self.action_repeat, rtol=0.0, atol=1e-6):
            raise ValueError(f"Integer FPS ratio required: control_fps/model_fps={ratio:.6f}")

        self.expected_model_chunk_size = int(cfg.get("model_chunk_size", 32))
        self.require_server_metadata = _as_bool(cfg.get("require_server_metadata", True))

        self._client: _MsgpackWebsocketClient | None = None
        self._obs: dict[str, Any] | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        if episode_info:
            instruction = episode_info.get("instruction")
            if isinstance(instruction, str) and instruction.strip():
                self.instruction = instruction
        self._obs = None
        client = self._ensure_client()
        try:
            client.reset()
        except Exception:
            pass

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._obs = obs

    def infer_actions(self) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("ingest_observation() must be called before infer_actions()")
        client = self._ensure_client()
        request, current_wire_state = self._build_request(self._obs)
        try:
            result = client.infer(request)
        except Exception as exc:
            raise RuntimeError(
                f"Cosmos3 X2Robot infer failed " f"(ws://{self.server_address}:{self.server_port}): {exc}"
            ) from exc

        model_actions = self._parse_result(result)
        control_actions = np.repeat(model_actions, self.action_repeat, axis=0)
        actions = np.concatenate([current_wire_state[None, :], control_actions], axis=0)
        return np.ascontiguousarray(actions, dtype=np.float32)

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> _MsgpackWebsocketClient:
        if self._client is not None:
            return self._client
        try:
            self._client = _MsgpackWebsocketClient(
                host=self.server_address,
                port=self.server_port,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Cosmos3 X2Robot at " f"ws://{self.server_address}:{self.server_port}: {exc}"
            ) from exc
        metadata = self._client.get_server_metadata()
        try:
            self._validate_server_metadata(dict(metadata) if isinstance(metadata, dict) else {})
        except Exception:
            self.close()
            raise
        return self._client

    def _validate_server_metadata(self, metadata: dict[str, Any]) -> None:
        if not metadata and not self.require_server_metadata:
            return
        expected = {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "action_space": ACTION_SPACE,
            "action_dim": ACTION_DIM,
            "wire_pose_reference": WIRE_POSE_REFERENCE,
            "rotation": ROTATION_FORMAT,
            "translation_relative_frame": TRANSLATION_RELATIVE_FRAME,
            "rotation_composition": ROTATION_COMPOSITION,
            "state_space": STATE_SPACE,
        }
        mismatches = {k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v}
        if mismatches:
            raise RuntimeError(f"Incompatible server metadata. Mismatches: {mismatches}")
        chunk_size = int(metadata.get("model_chunk_size", -1))
        if chunk_size != self.expected_model_chunk_size:
            raise RuntimeError(f"Server model_chunk_size={chunk_size}, expected {self.expected_model_chunk_size}")
        server_fps = float(metadata.get("model_fps", -1.0))
        if not np.isclose(server_fps, self.model_fps, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"Server model_fps={server_fps}, expected {self.model_fps}")

    def _build_request(self, obs: dict[str, Any]) -> tuple:
        images = obs.get("images") or {}
        request: dict[str, Any] = {}
        missing = [k for k in self.camera_map if images.get(k) is None]
        if missing:
            raise ValueError(f"Missing required cameras: {missing}")
        for sdk_name, wire_key in self.camera_map.items():
            request[wire_key] = self._prep_frame(images[sdk_name])

        state = obs.get("state") or {}
        parts = []
        for field in _STATE_FIELDS:
            vec = self._state_vec(state, field)
            request[_STATE_WIRE_KEYS[field]] = vec
            parts.append(vec)
        current_wire_state = np.concatenate(parts).astype(np.float32, copy=False)

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = self.instruction
        if self.require_instruction and (not isinstance(instruction, str) or not instruction.strip()):
            raise ValueError(
                "Cosmos3 X2Robot requires a non-empty task instruction. "
                "Pass via obs or set COSMOS3_X2ROBOT_INSTRUCTION."
            )
        request["prompt"] = str(instruction or "")
        return request, current_wire_state

    def _parse_result(self, result: Any) -> np.ndarray:
        if not isinstance(result, dict) or "action" not in result:
            raise RuntimeError("Response missing 'action' key")
        actions = np.asarray(result["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        expected_shape = (self.expected_model_chunk_size, ACTION_DIM)
        if actions.shape != expected_shape:
            raise RuntimeError(f"Action shape {actions.shape} != expected {expected_shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("Server returned NaN/Inf actions")
        return np.ascontiguousarray(actions)

    @staticmethod
    def _prep_frame(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Image must be HWC RGB, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _state_vec(state: dict, key: str) -> np.ndarray:
        value = state.get(key)
        if value is None:
            raise ValueError(f"Missing required state field: {key!r}")
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape != (7,):
            raise ValueError(f"state[{key!r}] must be shape (7,), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"state[{key!r}] contains NaN/Inf")
        return np.ascontiguousarray(arr)
