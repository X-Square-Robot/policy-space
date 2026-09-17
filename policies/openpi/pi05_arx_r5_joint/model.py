"""OpenPI Pi0.5 ARX R5 bimanual joint PolicySpace adapter."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from policy_space.base import PolicyModel


ACTION_DIM = 14
ACTION_HORIZON = 32
STATE_FIELDS = ("follow1_pos", "follow2_pos")
CAMERA_FIELDS = {
    "camera_front": "observation/image",
    "camera_left": "observation/left_wrist_image",
    "camera_right": "observation/right_wrist_image",
}


def _finite_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(array)


def _rgb_image(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty HWC RGB, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def build_joint_observation(obs: dict[str, Any], *, instruction: str) -> dict[str, Any]:
    """Map the PolicySpace bimanual joint envelope to OpenPI X2RealInputs."""
    if not isinstance(obs, dict):
        raise TypeError(f"observation must be a mapping, got {type(obs).__name__}")
    images = obs.get("images")
    state = obs.get("state")
    if not isinstance(images, dict):
        raise TypeError("observation.images must be a mapping")
    if not isinstance(state, dict):
        raise TypeError("observation.state must be a mapping")

    result: dict[str, Any] = {}
    for camera, model_key in CAMERA_FIELDS.items():
        if camera not in images:
            raise ValueError(f"missing camera {camera!r}")
        result[model_key] = _rgb_image(images[camera], name=f"images[{camera!r}]")

    result["observation/state"] = np.concatenate(
        [_finite_vector(state[field], name=f"state[{field!r}]", size=7) for field in STATE_FIELDS]
    )
    current_instruction = obs.get("instruction")
    result["prompt"] = (
        current_instruction.strip()
        if isinstance(current_instruction, str) and current_instruction.strip()
        else instruction.strip()
    )
    if not result["prompt"]:
        raise ValueError("joint Pi0.5 inference requires a non-empty instruction")
    return result


def validate_joint_actions(result: dict[str, Any]) -> np.ndarray:
    """Validate OpenPI's unnormalized absolute 14D joint action output."""
    if not isinstance(result, dict) or "actions" not in result:
        raise RuntimeError("OpenPI response is missing the 'actions' key")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(f"OpenPI joint actions must have shape (T, {ACTION_DIM}), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("OpenPI joint actions contain NaN/Inf")
    return np.ascontiguousarray(actions)


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value
    return cfg.get(cfg_key, default)


def _resolve_checkpoint_path(raw_path: str, *, openpi_root: Path) -> str:
    value = raw_path.strip()
    if not value:
        raise ValueError("Pi0.5 joint checkpoint_path is required")
    if "://" in value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(openpi_root / path)


def _create_openpi_policy(train_config_name: str, checkpoint_path: str, device: str, openpi_root: Path) -> Any:
    root = str(openpi_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as exc:
        raise RuntimeError("OpenPI is unavailable; set OPENPI_ROOT to the OpenPI source checkout") from exc
    try:
        config = training_config.get_config(train_config_name)
    except Exception as exc:
        raise ValueError(f"Unknown OpenPI train_config {train_config_name!r}") from exc
    if getattr(config.model, "action_horizon", ACTION_HORIZON) != ACTION_HORIZON:
        raise ValueError(f"Pi0.5 joint checkpoint requires action_horizon={ACTION_HORIZON}")
    policy = policy_config.create_trained_policy(config, checkpoint_path, pytorch_device=device)
    return policy


class Model(PolicyModel):
    """Load the ARX R5 Pi0.5 JAX checkpoint and emit 14D joint actions."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        openpi_root = Path(str(_env_or(cfg, "OPENPI_ROOT", "openpi_root", ""))).expanduser()
        if not str(openpi_root):
            raise ValueError("Pi0.5 joint openpi_root is required")
        self.train_config = str(_env_or(cfg, "OPENPI_TRAIN_CONFIG", "train_config", "pi05_arx_r5_joint"))
        checkpoint = str(_env_or(cfg, "OPENPI_CHECKPOINT_PATH", "checkpoint_path", ""))
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint, openpi_root=openpi_root)
        self.device = str(_env_or(cfg, "OPENPI_DEVICE", "device", "cuda:0"))
        self.instruction = str(cfg.get("instruction", ""))
        self.action_horizon = int(cfg.get("action_horizon", ACTION_HORIZON))
        if self.action_horizon <= 0 or self.action_horizon > ACTION_HORIZON:
            raise ValueError(f"action_horizon must be in [1, {ACTION_HORIZON}]")
        print(f"[OpenPI Pi0.5 ARX R5 joint] Loading checkpoint: {self.checkpoint_path}")
        self._policy = _create_openpi_policy(self.train_config, self.checkpoint_path, self.device, openpi_root)
        print("[OpenPI Pi0.5 ARX R5 joint] Model loaded successfully")
        self._latest_obs: dict[str, Any] | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        if isinstance(episode_info, dict) and isinstance(episode_info.get("instruction"), str):
            if episode_info["instruction"].strip():
                self.instruction = episode_info["instruction"]
        self._latest_obs = None

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = obs

    def infer_actions(self) -> dict[str, Any]:
        if self._latest_obs is None:
            raise RuntimeError("No observation buffered. Call ingest_observation first.")
        started = time.monotonic()
        result = self._policy.infer(build_joint_observation(self._latest_obs, instruction=self.instruction))
        actions = validate_joint_actions(result)[: self.action_horizon]
        return {
            "actions": actions,
            "openpi_timing": {"infer_ms": round((time.monotonic() - started) * 1000.0, 3)},
        }

    def finalize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        self._latest_obs = None

    def close(self) -> None:
        self._latest_obs = None
