"""OpenPI JAX pi0.5 ARX R5 policy, preserving the 14D training wire."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from policy_space.base import PolicyModel

ACTION_DIM = 14
ACTION_HORIZON = 32
MODEL_ACTION_DIM = 32
ASSET_ID = "pi05_arx_ee"
STATE_FIELDS = ("follow1_pos", "follow2_pos")
CAMERA_FIELDS = {
    "camera_front": "observation.images.faceImg",
    "camera_left": "observation.images.leftImg",
    "camera_right": "observation.images.rightImg",
}
MODEL_CAMERAS = {
    "observation.images.faceImg": "base_0_rgb",
    "observation.images.leftImg": "left_wrist_0_rgb",
    "observation.images.rightImg": "right_wrist_0_rgb",
}


def _finite_vector(value: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite ({size},) vector, got {array.shape}")
    return array.copy()


def _rgb_image(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) < 1:
        raise ValueError(f"{name} must be non-empty HWC RGB, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 RGB, got {array.dtype}")
    return np.ascontiguousarray(array).copy()


def build_arx_observation(obs: dict[str, Any], *, instruction: str = "") -> dict[str, Any]:
    """Preserve XYZ/RPY poses and normalized grippers; do not add state to actions."""
    if not isinstance(obs, dict):
        raise TypeError("ARX observation must be a mapping")
    images, state = obs.get("images"), obs.get("state")
    if not isinstance(images, dict) or not isinstance(state, dict):
        raise TypeError("ARX observation requires images and state mappings")
    result = {}
    for camera, model_key in CAMERA_FIELDS.items():
        if camera not in images:
            raise ValueError(f"missing camera {camera!r}")
        result[model_key] = _rgb_image(images[camera], camera)
    arms = []
    for field in STATE_FIELDS:
        if field not in state:
            raise ValueError(f"missing state field {field!r}")
        arm = _finite_vector(state[field], field, 7)
        if not -1e-6 <= arm[-1] <= 1.0 + 1e-6:
            raise ValueError(f"{field} gripper must be normalized to [0,1]")
        arms.append(arm)
    result["observation.state"] = np.concatenate(arms)
    prompt = obs.get("instruction")
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    result["prompt"] = prompt or instruction.strip()
    if not result["prompt"]:
        raise ValueError("ARX pi0.5 requires a non-empty instruction")
    return result


class ArxInputs:
    """Inference equivalent of the training Ex0016REEInputs(use_right_wrist=True)."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = {
            "state": _finite_vector(data["observation.state"], "observation.state", ACTION_DIM),
            "image": {name: _rgb_image(data[key], key) for key, name in MODEL_CAMERAS.items()},
            "image_mask": {name: np.True_ for name in MODEL_CAMERAS.values()},
        }
        if "action" in data:
            result["actions"] = np.asarray(data["action"], dtype=np.float32)[..., :ACTION_DIM]
        for key in ("instruction", "task", "prompt"):
            if key in data:
                result["prompt"] = data[key]
                break
        return result


class ArxOutputs:
    """Drop only OpenPI's padded action channels after checkpoint unnormalization."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
            raise ValueError(f"OpenPI padded actions must be [T,D>=14], got {actions.shape}")
        return {"actions": actions[:, :ACTION_DIM]}


def validate_arx_actions(result: Any) -> np.ndarray:
    if not isinstance(result, dict) or "actions" not in result:
        raise RuntimeError("OpenPI response is missing actions")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise RuntimeError(f"ARX actions must be (32, 14), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("ARX actions contain NaN/Inf")
    actions = actions.copy()
    # Bound normalized gripper predictions only; XYZ/RPY stay in trained units.
    actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0.0, 1.0)
    return np.ascontiguousarray(actions)


def _validate_checkpoint(path: Path) -> None:
    required = [
        path / "_CHECKPOINT_METADATA",
        path / "params" / "_METADATA",
        path / "params" / "manifest.ocdbt",
        path / "assets" / ASSET_ID / "norm_stats.json",
    ]
    for item in required:
        if not item.is_file():
            raise FileNotFoundError(f"ARX checkpoint file missing: {item}")
    metadata = json.loads(required[0].read_text())
    if not metadata.get("commit_timestamp_nsecs"):
        raise ValueError("ARX checkpoint has no completed commit timestamp")
    stats = json.loads(required[-1].read_text())["norm_stats"]
    for field in ("state", "actions"):
        for key in ("mean", "std", "q01", "q99"):
            _finite_vector(stats[field][key], f"norm_stats.{field}.{key}", ACTION_DIM)


def build_inference_config(checkpoint: Path):
    """Build the training-equivalent inference pipeline without a training sidecar import."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config

    def data_transforms(_model_config):
        return transforms.Group(inputs=[ArxInputs()], outputs=[ArxOutputs()])

    return config.TrainConfig(
        name=ASSET_ID,
        exp_name="inference",
        model=Pi0Config(pi05=True, action_dim=MODEL_ACTION_DIM, action_horizon=ACTION_HORIZON),
        data=config.SimpleDataConfig(
            repo_id="jade/arx_ee_v21",
            assets=config.AssetsConfig(
                assets_dir=str(checkpoint / "assets"), asset_id=ASSET_ID
            ),
            data_transforms=data_transforms,
        ),
        ema_decay=None,
        wandb_enabled=False,
        policy_metadata={"robot_type": "arx", "valid_action_dim": ACTION_DIM, "fps": 20},
    )


def _create_openpi_policy(openpi_root: Path, checkpoint: Path):
    for directory in (openpi_root, openpi_root / "src", openpi_root / "packages/openpi-client/src"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from openpi.policies import policy_config

    config = build_inference_config(checkpoint)
    # PI0.5 uses quantile normalization, state tokenization and padding to 32.
    # create_trained_policy always reads normalization assets from this checkpoint.
    return policy_config.create_trained_policy(config, checkpoint)


def _required_path(cfg: dict[str, Any], env: str, key: str) -> Path:
    raw = str(os.environ.get(env) or cfg.get(key) or "").strip()
    if not raw:
        raise ValueError(f"{key} is required; configure it or set {env}")
    return Path(raw).expanduser().resolve()


class Model(PolicyModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        openpi_root = _required_path(cfg, "OPENPI_ROOT", "openpi_root")
        if not (openpi_root / "src/openpi").is_dir():
            raise ValueError(f"openpi_root has no src/openpi package: {openpi_root}")
        self.checkpoint_path = _required_path(cfg, "OPENPI_CHECKPOINT_PATH", "checkpoint_path")
        _validate_checkpoint(self.checkpoint_path)
        configured_horizon = cfg.get("action_horizon", ACTION_HORIZON)
        if isinstance(configured_horizon, bool) or configured_horizon != ACTION_HORIZON:
            raise ValueError("ARX checkpoint requires action_horizon=32")
        self._default_instruction = str(cfg.get("instruction") or "").strip()
        self._instruction = self._default_instruction
        self._latest_obs: dict[str, Any] | None = None
        self._policy = _create_openpi_policy(openpi_root, self.checkpoint_path)

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        self._latest_obs = None
        instruction = (episode_info or {}).get("instruction")
        self._instruction = (
            instruction.strip()
            if isinstance(instruction, str) and instruction.strip()
            else self._default_instruction
        )

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = None
        self._latest_obs = build_arx_observation(obs, instruction=self._instruction)

    def infer_actions(self) -> dict[str, Any]:
        if self._latest_obs is None:
            raise RuntimeError("No ARX observation buffered; call ingest_observation first")
        return {"actions": validate_arx_actions(self._policy.infer(self._latest_obs))}

    def finalize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        self._latest_obs = None
        self._instruction = self._default_instruction

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "openpi_checkpoint": str(self.checkpoint_path),
            "openpi_backend": "jax",
            "openpi_asset_id": ASSET_ID,
            "openpi_model_action_shape": [ACTION_HORIZON, MODEL_ACTION_DIM],
            "openpi_wire_action_shape": [ACTION_HORIZON, ACTION_DIM],
            "openpi_normalization": "checkpoint_quantiles",
        }

    def close(self) -> None:
        self._latest_obs = None
        self._policy = None
