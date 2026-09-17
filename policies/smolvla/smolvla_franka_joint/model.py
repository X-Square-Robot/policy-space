"""SmolVLA Franka adapter for PolicySpace protocol v2."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from policy_space import PolicyModel


CAMERA_MAP = {
    "wrist": "observation.images.camera1",
    "exterior_1": "observation.images.camera2",
    "exterior_2": "observation.images.camera3",
}


def _value(cfg: dict[str, Any], env_name: str, key: str, default: Any) -> Any:
    value = os.environ.get(env_name)
    if value is not None and str(value).strip():
        return value
    return cfg.get(key, default)


def _required_path(cfg: dict[str, Any], env_name: str, key: str) -> Path:
    value = _value(cfg, env_name, key, "")
    if not value:
        raise RuntimeError(f"Set {env_name} or model.cfg.{key}")
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{env_name} path does not exist: {path}")
    return path


def _image(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"SmolVLA camera {name!r} must be numpy.ndarray")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"SmolVLA camera {name!r} must be uint8 HWC RGB, got {value.shape}/{value.dtype}")
    return np.ascontiguousarray(value)


def _interpolated_horizon(horizon: int, multiplier: float) -> int:
    if horizon < 1 or not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("SmolVLA interpolation requires a positive horizon and finite multiplier >= 1")
    scaled = horizon * multiplier
    output_horizon = round(scaled)
    if not np.isclose(scaled, output_horizon, rtol=0.0, atol=1e-9):
        raise ValueError("SmolVLA horizon times interpolation_multiplier must be an integer")
    return output_horizon


def _interpolate_joint_actions(actions: np.ndarray, multiplier: float) -> np.ndarray:
    """Resample 7 absolute joints linearly and hold each gripper command until its next native tick."""
    rows = actions.shape[0]
    output_horizon = _interpolated_horizon(rows, multiplier)
    if multiplier == 1.0:
        return np.ascontiguousarray(actions)
    # Eight values here are joints+gripper, never an XYZ/quaternion pose.
    # Uniform output ticks preserve the requested time scale; the terminal
    # value is held when the last output tick falls beyond the last keyframe.
    source_ticks = np.arange(rows, dtype=np.float64)
    output_ticks = np.minimum(np.arange(output_horizon, dtype=np.float64) / multiplier, rows - 1)
    result = np.empty((output_horizon, actions.shape[1]), dtype=np.float32)
    for joint in range(actions.shape[1] - 1):
        result[:, joint] = np.interp(output_ticks, source_ticks, actions[:, joint])
    result[:, -1] = actions[np.floor(output_ticks).astype(np.int64), -1]
    return np.ascontiguousarray(result)


class Model(PolicyModel):
    """Run a trained LeRobot SmolVLA policy behind the Franka v2 wire."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        project = _required_path(cfg, "SMOLVLA_PROJECT", "project_path")
        checkpoint = _required_path(cfg, "SMOLVLA_CHECKPOINT_PATH", "checkpoint_path")
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"SmolVLA checkpoint config missing: {checkpoint / 'config.json'}")

        source_root = project / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy
        from lerobot.policies.utils import prepare_observation_for_inference

        self._prepare_observation = prepare_observation_for_inference
        self._device = torch.device(str(_value(cfg, "SMOLVLA_DEVICE", "device", "cuda")))
        self._policy = SmolVLAPolicy.from_pretrained(str(checkpoint), local_files_only=True)
        self._policy.eval()
        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self._device)}},
        )

        self._checkpoint = checkpoint
        self._camera_map = dict(cfg.get("camera_map") or CAMERA_MAP)
        self._instruction = str(_value(cfg, "SMOLVLA_INSTRUCTION", "instruction", ""))
        self._require_instruction = bool(cfg.get("require_instruction", True))
        self._action_dim = int(cfg.get("action_dim", 8))
        self._native_action_horizon = int(cfg.get("action_horizon", 50))
        self._interpolation_multiplier = float(cfg.get("interpolation_multiplier", 1.0))
        self._action_horizon = _interpolated_horizon(
            self._native_action_horizon, self._interpolation_multiplier
        )
        self._latest: dict[str, Any] | None = None

        configured_dim = int(self._policy.config.output_features["action"].shape[0])
        configured_horizon = int(self._policy.config.n_action_steps)
        if configured_dim != self._action_dim:
            raise ValueError(f"SmolVLA checkpoint action dim is {configured_dim}, expected {self._action_dim}")
        if configured_horizon != self._native_action_horizon:
            raise ValueError(
                f"SmolVLA checkpoint action horizon is {configured_horizon}, "
                f"but model.cfg.action_horizon requests {self._native_action_horizon}"
            )

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        instruction = (episode_info or {}).get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            self._instruction = instruction.strip()
        self._latest = None

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict):
            raise TypeError("SmolVLA observation must be a mapping")
        images = obs.get("images")
        state = obs.get("state")
        if not isinstance(images, dict) or not isinstance(state, dict):
            raise TypeError("SmolVLA observation requires images and state mappings")

        mapped_images = {}
        for wire_name, model_name in self._camera_map.items():
            if wire_name not in images:
                raise KeyError(f"SmolVLA observation is missing camera {wire_name!r}")
            mapped_images[model_name] = _image(images[wire_name], wire_name).copy()

        state_raw = state.get("observation.state", state.get("state"))
        if state_raw is None:
            joints = state.get("joint_position", state.get("observation/joint_position"))
            gripper = state.get("gripper_position", state.get("observation/gripper_position"))
            if joints is None or gripper is None:
                raise KeyError("SmolVLA observation state needs observation.state or joint_position/gripper_position")
            state_raw = np.concatenate(
                [np.asarray(joints, dtype=np.float32).reshape(-1), np.asarray(gripper, dtype=np.float32).reshape(-1)]
            )
        state_value = np.array(state_raw, dtype=np.float32, copy=True)
        if state_value.shape != (self._action_dim,):
            raise ValueError(f"SmolVLA observation.state must have shape ({self._action_dim},), got {state_value.shape}")
        if not np.isfinite(state_value).all():
            raise ValueError("SmolVLA observation.state contains NaN/Inf")

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = self._instruction
        if self._require_instruction and not instruction:
            raise ValueError("SmolVLA observation requires an instruction")

        # Franka wire state is 0=open, 1=closed; the checkpoint was trained
        # on normalized finger opening (0=closed, 1=open). Convert before
        # the checkpoint state normalizer, including packed-state inputs.
        state_value[-1] = 1.0 - np.clip(state_value[-1], 0.0, 1.0)
        mapped_images["observation.state"] = np.ascontiguousarray(state_value)
        self._latest = {"observation": mapped_images, "instruction": instruction or "complete the task"}

    @torch.inference_mode()
    def infer_actions(self) -> np.ndarray:
        if self._latest is None:
            raise RuntimeError("No SmolVLA observation buffered")
        observation = dict(self._latest["observation"])
        frame = self._prepare_observation(
            observation,
            self._device,
            task=self._latest["instruction"],
        )
        batch = self._preprocess(frame)
        actions = self._policy.predict_action_chunk(batch)
        actions = self._postprocess(actions)
        array = actions.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        expected = (self._native_action_horizon, self._action_dim)
        if array.shape != expected:
            raise RuntimeError(f"SmolVLA output must have shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError("SmolVLA output contains NaN/Inf")
        # Postprocess has restored physical model units (0=closed, 1=open).
        # Franka's wire/ZeroToOne action is 0=open, 1=closed. Keep the seven
        # arm joints untouched and bound flow-matching gripper overshoot.
        array = array.copy()
        array[:, -1] = 1.0 - np.clip(array[:, -1], 0.0, 1.0)
        return _interpolate_joint_actions(array, self._interpolation_multiplier)

    def finalize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        del episode_info
        self._latest = None

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "smolvla_backend": "lerobot_in_process",
            "smolvla_checkpoint": str(self._checkpoint),
            "smolvla_action_shape": [self._action_horizon, self._action_dim],
            "smolvla_native_action_shape": [self._native_action_horizon, self._action_dim],
            "smolvla_interpolation_multiplier": self._interpolation_multiplier,
            "smolvla_interpolation": {"joints": "linear", "gripper": "previous"},
            "smolvla_device": str(self._device),
            "smolvla_gripper_adapter": "franka_closed01_to_model_open01_v1",
            "smolvla_model_gripper_convention": "0_closed_1_open",
            "smolvla_wire_gripper_convention": "0_open_1_closed",
        }

    def close(self) -> None:
        self._latest = None
        del self._policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["Model"]
