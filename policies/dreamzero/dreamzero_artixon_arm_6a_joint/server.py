#!/usr/bin/env python3
"""Strict DreamZero server for an X2Robot-post-trained EE14 checkpoint.

The checkpoint-specific modality mapping lives in a required JSON contract.
There are no guessed action keys, zero-filled missing outputs, or DROID/YAM
fallbacks.  This wrapper keeps the upstream DreamZero checkout unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from policy import (ACTION_DIM, ACTION_SPACE, MODEL_CHUNK_SIZE, PROTOCOL_NAME,
                    PROTOCOL_VERSION, ROTATION_COMPOSITION, ROTATION_FORMAT,
                    STATE_SPACE, TRANSLATION_RELATIVE_FRAME,
                    WIRE_POSE_REFERENCE)
from websockets.sync.server import serve

_CAMERA_WIRE_KEYS = (
    "observation/head_image",
    "observation/left_wrist_image",
    "observation/right_wrist_image",
)
_STATE_WIRE_KEYS = ("observation/left_eef", "observation/right_eef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coverage(fields: list[dict[str, Any]], slice_key: str) -> list[int]:
    covered: list[int] = []
    for field in fields:
        bounds = field.get(slice_key)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{slice_key} must be [start,end], got {bounds!r}")
        start, end = (int(bounds[0]), int(bounds[1]))
        if start < 0 or end <= start or end > ACTION_DIM:
            raise ValueError(f"invalid {slice_key}={bounds!r}")
        covered.extend(range(start, end))
    return covered


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "action_space": ACTION_SPACE,
        "action_dim": ACTION_DIM,
        "wire_pose_reference": WIRE_POSE_REFERENCE,
        "rotation": ROTATION_FORMAT,
        "translation_relative_frame": TRANSLATION_RELATIVE_FRAME,
        "rotation_composition": ROTATION_COMPOSITION,
        "model_chunk_size": MODEL_CHUNK_SIZE,
        "history_owner": "client",
        "relative_action_precomputed": True,
        "dreamzero_relative_action": False,
        "state_space": STATE_SPACE,
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible X2Robot checkpoint contract: {mismatches}")

    required_text = (
        "checkpoint_id",
        "embodiment_tag",
        "language_key",
        "training_config_sha256",
        "normalization_stats_sha256",
    )
    for key in required_text:
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ValueError(f"contract.{key} must be a non-empty string")
        if contract[key].startswith("REPLACE_"):
            raise ValueError(f"contract.{key} still contains the example placeholder")

    image_size = contract.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or min(int(image_size[0]), int(image_size[1])) <= 0
    ):
        raise ValueError("contract.image_size must be positive [height,width]")
    videos = contract.get("video_keys")
    if not isinstance(videos, dict) or set(videos) != set(_CAMERA_WIRE_KEYS):
        raise ValueError(f"contract.video_keys must map exactly {_CAMERA_WIRE_KEYS}")
    if not all(isinstance(value, str) and value for value in videos.values()):
        raise ValueError("contract.video_keys model values must be non-empty strings")

    state_fields = contract.get("state_fields")
    action_fields = contract.get("action_fields")
    if not isinstance(state_fields, list) or not state_fields:
        raise ValueError("contract.state_fields must be a non-empty list")
    if not isinstance(action_fields, list) or not action_fields:
        raise ValueError("contract.action_fields must be a non-empty list")
    for field in state_fields:
        if field.get("wire_key") not in _STATE_WIRE_KEYS:
            raise ValueError(f"invalid state wire_key: {field.get('wire_key')!r}")
        if not isinstance(field.get("model_key"), str) or not field["model_key"]:
            raise ValueError("every state field needs model_key")
    for field in action_fields:
        if not isinstance(field.get("model_key"), str) or not field["model_key"]:
            raise ValueError("every action field needs model_key")

    state_covered = _coverage(state_fields, "wire_slice")
    # State slices are per-arm 7-D, so each arm must be covered exactly once.
    for wire_key in _STATE_WIRE_KEYS:
        arm_fields = [field for field in state_fields if field["wire_key"] == wire_key]
        if sorted(_coverage(arm_fields, "wire_slice")) != list(range(7)):
            raise ValueError(
                f"state_fields for {wire_key} must cover slots 0..6 exactly once"
            )
    if len(state_covered) != ACTION_DIM:
        raise ValueError("state_fields must describe exactly 14 physical channels")

    action_covered = _coverage(action_fields, "wire_slice")
    if (
        sorted(action_covered) != list(range(ACTION_DIM))
        or len(action_covered) != ACTION_DIM
    ):
        raise ValueError("action_fields must cover wire slots 0..13 exactly once")
    if float(contract.get("model_fps", -1)) <= 0:
        raise ValueError("contract.model_fps must be positive")
    return contract


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


class DreamZeroX2RobotService:
    def __init__(
        self,
        model_path: Path,
        contract_path: Path,
        dreamzero_root: Path,
        tokenizer_path: Path | None,
        training_config_path: Path,
        normalization_stats_path: Path,
    ) -> None:
        self.contract_path = contract_path.resolve()
        self.contract = load_contract(self.contract_path)
        self.model_path = model_path.resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {self.model_path}")
        if not dreamzero_root.is_dir():
            raise FileNotFoundError(f"DreamZero root does not exist: {dreamzero_root}")
        # GrootSimPolicy does not consume arbitrary CLI-side config/stat files;
        # it loads these two artifacts from the checkpoint itself. Verify the
        # supplied audit copies AND the embedded files against the same frozen
        # digest so the contract cannot describe different preprocessing from
        # what inference actually instantiates.
        for label, artifact_path, embedded_path, digest_key in (
            (
                "training config",
                training_config_path,
                self.model_path / "experiment_cfg" / "conf.yaml",
                "training_config_sha256",
            ),
            (
                "normalization metadata",
                normalization_stats_path,
                self.model_path / "experiment_cfg" / "metadata.json",
                "normalization_stats_sha256",
            ),
        ):
            artifact_path = artifact_path.resolve()
            if not artifact_path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {artifact_path}")
            if not embedded_path.is_file():
                raise FileNotFoundError(
                    f"checkpoint is missing the {label} actually loaded by GrootSimPolicy: "
                    f"{embedded_path}"
                )
            actual = _sha256(artifact_path)
            embedded = _sha256(embedded_path)
            expected = self.contract[digest_key]
            if actual != expected or embedded != expected:
                raise ValueError(
                    f"{label} SHA256 mismatch: contract={expected}, supplied={actual}, "
                    f"checkpoint_embedded={embedded}, supplied_path={artifact_path}, "
                    f"embedded_path={embedded_path}"
                )
        if str(dreamzero_root) not in sys.path:
            sys.path.insert(0, str(dreamzero_root))

        import torch
        import torch.distributed as dist
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from torch.distributed.device_mesh import init_device_mesh

        if not torch.cuda.is_available():
            raise RuntimeError("DreamZero X2Robot real inference requires CUDA")
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29501")
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)
        device_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("ip",))
        embodiment = EmbodimentTag(self.contract["embodiment_tag"])
        self.policy = GrootSimPolicy(
            embodiment_tag=embodiment,
            model_path=str(self.model_path),
            tokenizer_path_override=(
                str(tokenizer_path.resolve()) if tokenizer_path else None
            ),
            device="cuda",
            device_mesh=device_mesh,
        )
        self._torch = torch
        self._Batch = __import__("tianshou.data", fromlist=["Batch"]).Batch
        self._active_session: str | None = None
        self._expected_request_index = 0
        self._prompt: str | None = None
        self._infer_lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "action_space": ACTION_SPACE,
            "action_dim": ACTION_DIM,
            "wire_pose_reference": WIRE_POSE_REFERENCE,
            "rotation": ROTATION_FORMAT,
            "translation_relative_frame": TRANSLATION_RELATIVE_FRAME,
            "rotation_composition": ROTATION_COMPOSITION,
            "model_chunk_size": MODEL_CHUNK_SIZE,
            "model_fps": float(self.contract["model_fps"]),
            "history_owner": "client",
            "needs_session_id": True,
            "checkpoint_id": self.contract["checkpoint_id"],
            "checkpoint_path": str(self.model_path),
            "checkpoint_contract_sha256": _sha256(self.contract_path),
            "training_config_sha256": self.contract["training_config_sha256"],
            "normalization_stats_sha256": self.contract["normalization_stats_sha256"],
            "backend": "dreamzero",
            "state_space": STATE_SPACE,
        }

    def reset(self, session_id: str) -> None:
        self._active_session = session_id
        self._expected_request_index = 0
        self._prompt = None
        head = self.policy.trained_model.action_head
        if not hasattr(head, "current_start_frame"):
            raise RuntimeError("checkpoint action head has no DreamZero causal cursor")
        head.current_start_frame = 0
        if hasattr(head, "language"):
            head.language = None

    @staticmethod
    def _resize_video(value: Any, target_h: int, target_w: int, key: str) -> np.ndarray:
        import cv2

        video = np.asarray(value)
        if video.dtype != np.uint8 or video.ndim not in (3, 4) or video.shape[-1] != 3:
            raise ValueError(
                f"{key} must be HWC/THWC uint8 RGB, got {video.shape}/{video.dtype}"
            )
        frames = video[None, ...] if video.ndim == 3 else video
        resized = np.stack(
            [
                (
                    frame
                    if frame.shape[:2] == (target_h, target_w)
                    else cv2.resize(
                        frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )
                )
                for frame in frames
            ],
            axis=0,
        )
        return np.ascontiguousarray(resized)

    def _convert_obs(self, request: dict[str, Any]) -> dict[str, Any]:
        request_index = int(request.get("request_index", -1))
        if (
            request.get("session_id") != self._active_session
            or self._active_session is None
        ):
            raise ValueError("session_id does not match the last reset")
        if request_index != self._expected_request_index:
            raise ValueError(
                f"request_index={request_index}, expected={self._expected_request_index}"
            )
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self._prompt is not None and prompt != self._prompt:
            raise ValueError("prompt changed inside a session; reset the episode first")
        self._prompt = prompt

        target_h, target_w = (int(v) for v in self.contract["image_size"])
        converted: dict[str, Any] = {}
        expected_frames = 1 if request_index == 0 else 4
        for wire_key, model_key in self.contract["video_keys"].items():
            if wire_key not in request:
                raise ValueError(f"missing {wire_key}")
            video = self._resize_video(request[wire_key], target_h, target_w, wire_key)
            if video.shape[0] != expected_frames:
                raise ValueError(
                    f"request {request_index} requires {expected_frames} frame(s) for {wire_key}, "
                    f"got {video.shape[0]}"
                )
            converted[model_key] = video

        for field in self.contract["state_fields"]:
            wire_key = field["wire_key"]
            state = np.asarray(request.get(wire_key), dtype=np.float32).reshape(-1)
            if state.shape != (7,) or not np.isfinite(state).all():
                raise ValueError(
                    f"{wire_key} must be finite shape (7,), got {state.shape}"
                )
            start, end = (int(v) for v in field["wire_slice"])
            converted[field["model_key"]] = (
                state[start:end].reshape(1, -1).astype(np.float64)
            )
        for key in (
            "observation/left_eef_init_relative",
            "observation/right_eef_init_relative",
        ):
            echo = np.asarray(request.get(key), dtype=np.float32).reshape(-1)
            if echo.shape != (7,) or not np.isfinite(echo).all():
                raise ValueError(f"{key} must be finite shape (7,), got {echo.shape}")
        converted[self.contract["language_key"]] = prompt
        return converted

    @staticmethod
    def _action_value(action: Any, key: str) -> Any:
        try:
            if key in action:
                return action[key]
        except Exception:
            pass
        try:
            return getattr(action, key)
        except (AttributeError, KeyError):
            raise KeyError(
                f"checkpoint output is missing required action key {key!r}"
            ) from None

    def _convert_action(self, action: Any) -> np.ndarray:
        wire = np.empty((MODEL_CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
        for field in self.contract["action_fields"]:
            start, end = (int(v) for v in field["wire_slice"])
            width = end - start
            value = self._action_value(action, field["model_key"])
            if self._torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            while value.ndim > 2 and value.shape[0] == 1:
                value = value[0]
            if value.ndim == 1 and width == 1:
                value = value[:, None]
            expected = (MODEL_CHUNK_SIZE, width)
            if value.shape != expected:
                raise RuntimeError(
                    f"action {field['model_key']!r} must have shape {expected}, got {value.shape}"
                )
            wire[:, start:end] = value
        if not np.isfinite(wire).all():
            raise RuntimeError("checkpoint returned NaN/Inf actions")
        return np.ascontiguousarray(wire)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        converted = self._convert_obs(request)
        with self._infer_lock, self._torch.inference_mode():
            result, _video = self.policy.lazy_joint_forward_causal(
                self._Batch(obs=converted)
            )
        action = self._convert_action(result.act)
        request_index = self._expected_request_index
        self._expected_request_index += 1
        return {"action": action, "request_index": request_index}


def make_handler(service: DreamZeroX2RobotService):
    # DreamZero causal state is process-global, so only one active client may
    # own it at a time. ManaEnv itself is single-client for this integration.
    connection_lock = threading.Lock()

    def handler(connection: Any) -> None:
        connection.send(_pack(service.metadata))
        if not connection_lock.acquire(blocking=False):
            connection.send("DreamZero X2Robot server already has an active client")
            connection.close()
            return
        try:
            for raw in connection:
                try:
                    if isinstance(raw, str):
                        raise ValueError("binary msgpack frames are required")
                    request = _unpack(raw)
                    endpoint = request.get("endpoint")
                    if endpoint == "reset":
                        session_id = request.get("session_id")
                        if not isinstance(session_id, str) or not session_id:
                            raise ValueError("reset requires a non-empty session_id")
                        service.reset(session_id)
                        connection.send(_pack({"ok": True, "session_id": session_id}))
                    elif endpoint == "infer":
                        connection.send(_pack(service.infer(request)))
                    else:
                        raise ValueError(f"unknown endpoint {endpoint!r}")
                except Exception as exc:
                    connection.send(f"{type(exc).__name__}: {exc}")
        finally:
            connection_lock.release()

    return handler


def main() -> None:
    default_root = Path(__file__).resolve().parents[2] / "3rd" / "dreamzero"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-contract", type=Path, required=True)
    parser.add_argument("--dreamzero-root", type=Path, default=default_root)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--training-config-path", type=Path, required=True)
    parser.add_argument("--normalization-stats-path", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    service = DreamZeroX2RobotService(
        args.model_path,
        args.checkpoint_contract,
        args.dreamzero_root,
        args.tokenizer_path,
        args.training_config_path,
        args.normalization_stats_path,
    )
    print(f"DreamZero X2Robot server listening on ws://{args.host}:{args.port}")
    print(f"metadata={service.metadata}")
    with serve(
        make_handler(service), args.host, args.port, compression=None, max_size=None
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
