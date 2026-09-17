"""Cosmos3-Nano-Policy-DROID model loaded by Policy Space.

Loads the Cosmos3 action policy model directly in-process via the
cosmos_framework RobolabPolicyService, without a separate GPU server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from policy_space.base import PolicyModel
from policy_space.paths import external_source_root

_DEFAULT_CAMERA_MAP = {
    "wrist": "observation/wrist_image_left",
    "exterior_1": "observation/exterior_image_1_left",
    "exterior_2": "observation/exterior_image_2_left",
}

_STATE_SPEC = (
    ("observation/joint_position", "joint_position", 7),
    ("observation/gripper_position", "gripper_position", 1),
)


def _ensure_cosmos_imports(cosmos_root: Path) -> None:
    cosmos_str = str(cosmos_root)
    if cosmos_str not in sys.path:
        sys.path.insert(0, cosmos_str)


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip():
        return raw
    if cfg.get(cfg_key) is not None:
        return cfg[cfg_key]
    return default


class Model(PolicyModel):
    """Cosmos3 DROID model that loads RobolabPolicyService directly."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self._cosmos_root = external_source_root(
            cfg,
            env_name="COSMOS_ROOT",
            cfg_key="cosmos_root",
            project="Cosmos",
        )
        self._instruction = str(_env_or(cfg, "COSMOS3_INSTRUCTION", "instruction", ""))
        self._camera_map: dict[str, str] = dict(
            cfg.get("camera_map") or _DEFAULT_CAMERA_MAP
        )
        self._composed_image_camera: str | None = (
            cfg.get("composed_image_camera") or None
        )

        checkpoint_path = str(
            _env_or(
                cfg,
                "COSMOS3_CHECKPOINT_PATH",
                "checkpoint_path",
                "nvidia/Cosmos3-Nano-Policy-DROID",
            )
        )
        self._checkpoint_path = str(self._resolve_path(checkpoint_path))

        _ensure_cosmos_imports(self._cosmos_root)
        from cosmos_framework.inference.common.init import init_script

        init_script()

        from cosmos_framework.scripts.action_policy_server_robolab import (
            RobolabPolicyService,
            RobolabServerArgs,
        )

        server_args = RobolabServerArgs(
            checkpoint_path=self._checkpoint_path,
            port=0,
            host="127.0.0.1",
            domain_name=str(cfg.get("domain_name", "droid_lerobot")),
            decode_video=False,
            seed=int(cfg.get("seed", 42)),
            deterministic_seed=bool(cfg.get("deterministic_seed", False)),
            guidance=float(cfg.get("guidance", 1.0)),
            num_steps=int(cfg.get("num_steps", 10)),
            shift=float(cfg.get("shift", 1.0)),
            conditioning_fps=float(cfg.get("conditioning_fps", 15.0)),
            resolution=cfg.get("resolution") or None,
            action_chunk_size=int(cfg.get("action_chunk_size", 16)),
            action_dim=int(cfg.get("action_dim", 8)),
            image_height=int(cfg.get("image_height", 180)),
            image_width=int(cfg.get("image_width", 320)),
            action_space=cfg.get("action_space", "joint_pos"),
            use_state=bool(cfg.get("use_state", True)),
            history_length=int(cfg.get("history_length", 1)),
        )

        print(f"[Cosmos3 DROID] Loading model: {self._checkpoint_path}")
        self._service = RobolabPolicyService(server_args)
        print("[Cosmos3 DROID] Model loaded successfully")

        self._latest_obs: dict[str, Any] | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        if episode_info:
            instr = episode_info.get("instruction")
            if isinstance(instr, str) and instr.strip():
                self._instruction = instr
        self._latest_obs = None

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = obs

    def infer_actions(self) -> np.ndarray:
        if self._latest_obs is None:
            raise RuntimeError(
                "No observation buffered. Call ingest_observation first."
            )
        request = self._build_request(self._latest_obs)
        result = self._service.infer(request)
        return self._parse_actions(result)

    def close(self) -> None:
        self._latest_obs = None

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        if "/" not in p and not path.exists():
            return path
        resolved = (self._cosmos_root / path).resolve()
        return resolved if resolved.exists() else path

    def _build_request(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs.get("images") or {}
        request: dict[str, Any] = {}

        if (
            self._composed_image_camera
            and images.get(self._composed_image_camera) is not None
        ):
            request["observation/image"] = _prep_frame(
                images[self._composed_image_camera]
            )
        else:
            for sdk_name, wire_key in self._camera_map.items():
                img = images.get(sdk_name)
                if img is not None:
                    request[wire_key] = _prep_frame(img)

        state = obs.get("state") or {}
        for wire_key, src_key, dim in _STATE_SPEC:
            request[wire_key] = _state_vec(state, src_key, dim)

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = self._instruction or "complete the task"
        request["prompt"] = instruction
        return request

    @staticmethod
    def _parse_actions(result: Any) -> np.ndarray:
        if not isinstance(result, dict) or "action" not in result:
            raise RuntimeError(
                f"Cosmos3 response missing 'action' key: {type(result).__name__}"
            )
        actions = np.asarray(result["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        return actions


def _prep_frame(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _state_vec(state: dict, key: str, dim: int) -> np.ndarray:
    val = state.get(key)
    if val is None:
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(val, dtype=np.float32).reshape(-1)
    if arr.shape[0] < dim:
        arr = np.concatenate([arr, np.zeros(dim - arr.shape[0], dtype=np.float32)])
    return arr[:dim]
