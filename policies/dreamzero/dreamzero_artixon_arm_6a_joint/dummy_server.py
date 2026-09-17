"""Protocol-only DreamZero X2Robot server returning a hold-position chunk.

This server does not load DreamZero.  It exists to validate the metadata,
endpoint, history, state/action, and ManaEnv transport contracts safely before
an X2Robot-post-trained checkpoint is available.
"""

from __future__ import annotations

import argparse
from typing import Any

import msgpack
import numpy as np
from policy import (ACTION_DIM, ACTION_SPACE, MODEL_CHUNK_SIZE, PROTOCOL_NAME,
                    PROTOCOL_VERSION, ROTATION_COMPOSITION, ROTATION_FORMAT,
                    STATE_SPACE, TRANSLATION_RELATIVE_FRAME,
                    WIRE_POSE_REFERENCE)
from websockets.sync.server import serve

METADATA = {
    "protocol": PROTOCOL_NAME,
    "protocol_version": PROTOCOL_VERSION,
    "action_space": ACTION_SPACE,
    "action_dim": ACTION_DIM,
    "wire_pose_reference": WIRE_POSE_REFERENCE,
    "rotation": ROTATION_FORMAT,
    "translation_relative_frame": TRANSLATION_RELATIVE_FRAME,
    "rotation_composition": ROTATION_COMPOSITION,
    "model_chunk_size": MODEL_CHUNK_SIZE,
    "model_fps": 30,
    "control_fps": 30,
    "history_owner": "client",
    "needs_session_id": True,
    "state_space": STATE_SPACE,
    "backend": "dummy-hold-position",
}


def _pack_array(value: Any) -> Any:
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
    raise TypeError(f"Cannot encode {type(value).__name__}")


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


def _pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_array)


def _unpack(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_unpack_array)


def _validate_image(request: dict[str, Any], key: str, request_index: int) -> None:
    if key not in request:
        raise ValueError(f"missing {key}")
    image = np.asarray(request[key])
    expected_ndim = 3 if request_index == 0 else 4
    if image.ndim != expected_ndim or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"{key} must be {'HWC' if request_index == 0 else 'THWC'} uint8, "
            f"got shape={image.shape} dtype={image.dtype}"
        )
    if request_index > 0 and image.shape[0] != 4:
        raise ValueError(f"{key} history must contain 4 frames, got {image.shape[0]}")


def _handler(connection: Any) -> None:
    connection.send(_pack(METADATA))
    active_session = None
    expected_request_index = 0
    for raw in connection:
        try:
            if isinstance(raw, str):
                raise ValueError("binary msgpack frames are required")
            request = _unpack(raw)
            if not isinstance(request, dict):
                raise ValueError("request must be a dict")
            endpoint = request.get("endpoint")
            if endpoint == "reset":
                session_id = request.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("reset requires a non-empty session_id")
                active_session = session_id
                expected_request_index = 0
                connection.send(_pack({"ok": True, "session_id": active_session}))
                continue
            if endpoint != "infer":
                raise ValueError(f"unknown endpoint {endpoint!r}")
            if request.get("session_id") != active_session or active_session is None:
                raise ValueError("infer session_id does not match the last reset")
            request_index = int(request.get("request_index", -1))
            if request_index != expected_request_index:
                raise ValueError(
                    f"request_index={request_index}, expected {expected_request_index}"
                )
            for key in (
                "observation/head_image",
                "observation/left_wrist_image",
                "observation/right_wrist_image",
            ):
                _validate_image(request, key, request_index)
            for key in ("observation/left_eef", "observation/right_eef"):
                absolute = np.asarray(request.get(key), dtype=np.float32).reshape(-1)
                if absolute.shape != (7,) or not np.isfinite(absolute).all():
                    raise ValueError(
                        f"absolute EE state {key} must be finite shape (7,)"
                    )
            left_echo = np.asarray(
                request.get("observation/left_eef_init_relative"), dtype=np.float32
            ).reshape(-1)
            right_echo = np.asarray(
                request.get("observation/right_eef_init_relative"), dtype=np.float32
            ).reshape(-1)
            if left_echo.shape != (7,) or right_echo.shape != (7,):
                raise ValueError("init-relative EE echo state must be 7+7")
            wire_state = np.concatenate([left_echo, right_echo])
            if not np.isfinite(wire_state).all():
                raise ValueError("wire state contains NaN/Inf")
            action = np.repeat(wire_state[None, :], MODEL_CHUNK_SIZE, axis=0)
            connection.send(
                _pack(
                    {
                        "action": np.ascontiguousarray(action, dtype=np.float32),
                        "request_index": request_index,
                        "backend": "dummy-hold-position",
                    }
                )
            )
            expected_request_index += 1
        except Exception as exc:
            connection.send(f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print(f"DreamZero X2Robot dummy server listening on ws://{args.host}:{args.port}")
    with serve(
        _handler, args.host, args.port, compression=None, max_size=None
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
