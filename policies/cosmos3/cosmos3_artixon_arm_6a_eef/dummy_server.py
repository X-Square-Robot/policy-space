#!/usr/bin/env python3
"""Zero-GPU X2Robot protocol server returning a safe hold-position chunk."""

from __future__ import annotations

import argparse
from typing import Any

import msgpack
import numpy as np
from policy import (
    ACTION_DIM,
    ACTION_SPACE,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ROTATION_COMPOSITION,
    ROTATION_FORMAT,
    STATE_SPACE,
    TRANSLATION_RELATIVE_FRAME,
    WIRE_POSE_REFERENCE,
)
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

_IMAGE_KEYS = (
    "observation/head_image",
    "observation/left_wrist_image",
    "observation/right_wrist_image",
)


def server_metadata(chunk_size: int, model_fps: float) -> dict:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "embodiment": "ArtiXon Arm-6A",
        "action_space": ACTION_SPACE,
        "action_dim": ACTION_DIM,
        "model_action_dim": ACTION_DIM,
        "model_rotation": "euler_xyz",
        "model_chunk_size": int(chunk_size),
        "model_fps": float(model_fps),
        "wire_pose_reference": WIRE_POSE_REFERENCE,
        "rotation": ROTATION_FORMAT,
        "translation_relative_frame": TRANSLATION_RELATIVE_FRAME,
        "rotation_composition": ROTATION_COMPOSITION,
        "state_space": STATE_SPACE,
        "backend": "hold-dummy",
    }


def validate_request(obs: dict) -> np.ndarray:
    if not isinstance(obs.get("prompt"), str):
        raise ValueError("'prompt' must be a string")
    missing = [key for key in _IMAGE_KEYS if key not in obs]
    if missing:
        raise ValueError(f"missing images: {missing}")
    for key in _IMAGE_KEYS:
        image = np.asarray(obs[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key} must be HWC uint8 RGB, got {image.shape}/{image.dtype}")

    arms = []
    for key in ("observation/left_eef", "observation/right_eef"):
        value = np.asarray(obs.get(key), dtype=np.float32).reshape(-1)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite shape (7,), got {value.shape}")
        arms.append(value)
    return np.concatenate(arms).astype(np.float32)


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


def make_handler(chunk_size: int, model_fps: float):
    metadata = server_metadata(chunk_size, model_fps)

    def handler(websocket) -> None:
        websocket.send(_pack(metadata))
        print(f"[dummy-cosmos3-x2robot] client connected; metadata={metadata}")
        try:
            for message in websocket:
                obs = _unpack(message)
                try:
                    current = validate_request(obs)
                    action = np.repeat(current[None, :], chunk_size, axis=0)
                    websocket.send(_pack({"action": action, "server_timing": {"dummy": 0.0}}))
                    print("[dummy-cosmos3-x2robot] hold response " f"prompt={obs['prompt']!r} action={action.shape}")
                except (TypeError, ValueError) as exc:
                    websocket.send(f"dummy-cosmos3-x2robot validation error: {exc}")
        except ConnectionClosed:
            print("[dummy-cosmos3-x2robot] client disconnected")

    return handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-chunk-size", type=int, default=32)
    parser.add_argument("--model-fps", type=float, default=15.0)
    args = parser.parse_args()
    if args.model_chunk_size < 1 or args.model_fps <= 0:
        parser.error("model chunk size and FPS must be positive")

    print(
        f"[dummy-cosmos3-x2robot] serving ws://{args.host}:{args.port} "
        f"chunk={args.model_chunk_size} dim={ACTION_DIM} fps={args.model_fps}"
    )
    with serve(
        make_handler(args.model_chunk_size, args.model_fps),
        args.host,
        args.port,
        max_size=None,
        compression=None,
    ) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
