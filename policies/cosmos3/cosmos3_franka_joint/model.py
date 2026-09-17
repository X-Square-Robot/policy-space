"""In-process Cosmos3 Franka adapter for Policy Space protocol v2."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from policy_space.base import PolicyModel
from policy_space.paths import external_source_root

CAMERA_MAP = {
    "wrist": "observation/wrist_image_left",
    "exterior_1": "observation/exterior_image_1_left",
    "exterior_2": "observation/exterior_image_2_left",
}
STATE_SPEC = (
    ("observation/joint_position", "joint_position", 7),
    ("observation/gripper_position", "gripper_position", 1),
)


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip():
        return raw
    if cfg.get(cfg_key) not in (None, ""):
        return cfg[cfg_key]
    return default


def _required_path(
    cfg: dict[str, Any], env_name: str, cfg_key: str, *, directory: bool = True
) -> Path:
    raw = _env_or(cfg, env_name, cfg_key, "")
    if not raw:
        raise RuntimeError(f"Set {env_name} or model.cfg.{cfg_key}")
    path = Path(str(raw)).expanduser().resolve()
    exists = path.is_dir() if directory else path.exists()
    if not exists:
        raise FileNotFoundError(f"{env_name} path does not exist: {path}")
    return path


def _validate_rgb(image: Any, *, camera: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Cosmos3 camera {camera!r} must be a numpy.ndarray, "
            f"got {type(image).__name__}"
        )
    if image.dtype != np.uint8:
        raise TypeError(
            f"Cosmos3 camera {camera!r} must have dtype uint8, got {image.dtype}"
        )
    if (
        image.ndim != 3
        or image.shape[-1] != 3
        or image.shape[0] < 1
        or image.shape[1] < 1
    ):
        raise ValueError(
            f"Cosmos3 camera {camera!r} must be non-empty HWC RGB, "
            f"got {image.shape!r}"
        )
    return np.ascontiguousarray(image)


def _validate_state(state: dict[str, Any], key: str, dim: int) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Cosmos3 observation is missing state field {key!r}")
    try:
        value = np.asarray(state[key], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cosmos3 state {key!r} must be numeric") from exc
    if value.shape != (dim,):
        raise ValueError(
            f"Cosmos3 state {key!r} must have shape {(dim,)}, got {value.shape!r}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Cosmos3 state {key!r} contains NaN/Inf")
    return np.ascontiguousarray(value)


class Model(PolicyModel):
    """Load the Franka checkpoint once and expose an exact H24 x 8 contract."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        framework = external_source_root(
            cfg,
            env_name="COSMOS3_FRAMEWORK",
            cfg_key="framework_path",
            project="Cosmos3 framework",
        )
        checkpoint = _required_path(cfg, "COSMOS3_CHECKPOINT_PATH", "checkpoint_path")
        tokenizer = _required_path(
            cfg, "COSMOS3_QWEN_TOKENIZER_PATH", "qwen_tokenizer_path"
        )
        output_dir = _required_path(cfg, "COSMOS3_OUTPUT_DIR", "output_dir")

        self._checkpoint_path = str(checkpoint)
        self._camera_map = dict(cfg.get("camera_map") or CAMERA_MAP)
        self._instruction = str(_env_or(cfg, "COSMOS3_INSTRUCTION", "instruction", ""))
        self._require_instruction = bool(cfg.get("require_instruction", True))
        self._model_action_horizon = int(cfg.get("model_action_horizon", 32))
        self._action_horizon = int(cfg.get("action_horizon", 24))
        if self._model_action_horizon != 32 or self._action_horizon != 24:
            raise ValueError(
                "The verified Franka adapter requires model H32 and wire H24"
            )

        framework_text = str(framework)
        if framework_text not in sys.path:
            sys.path.insert(0, framework_text)
        # The framework resolves its VAE assets relative to this checkout.
        os.chdir(framework)

        from cosmos_framework.inference.common.init import init_script
        from cosmos_framework.scripts.action_policy_server_robolab import (
            RobolabPolicyService,
            RobolabServerArgs,
        )

        init_script()
        args = RobolabServerArgs(
            checkpoint_path=self._checkpoint_path,
            allow_dcp_checkpoint=True,
            experiment="action_policy_droid_nano",
            experiment_overrides=[
                "model.config.compile.enabled=False",
                "trainer.callbacks.compile_tokenizer.enabled=False",
                "model.config.ema.enabled=False",
                "model.config.vlm_config.tokenizer.pretrained_model_name="
                f"{tokenizer}",
            ],
            host="127.0.0.1",
            port=0,
            output_dir=output_dir,
            seed=0,
            deterministic_seed=False,
            guidance=3.0,
            num_steps=4,
            shift=5.0,
            resolution="480",
            conditioning_fps=20.0,
            action_chunk_size=self._model_action_horizon,
            action_dim=8,
            image_height=540,
            image_width=640,
            action_space="joint_pos",
            use_state=True,
            history_length=1,
            flip_gripper=False,
            format_prompt_as_json=True,
        )
        self._service = RobolabPolicyService(args)
        self._latest_obs: dict[str, Any] | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        instruction = (episode_info or {}).get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            self._instruction = instruction.strip()
        self._latest_obs = None

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict):
            raise TypeError("Cosmos3 observation must be a mapping")
        images = obs.get("images")
        state = obs.get("state")
        if not isinstance(images, dict) or not isinstance(state, dict):
            raise TypeError("Cosmos3 observation requires images and state mappings")

        validated_images: dict[str, np.ndarray] = {}
        for camera in self._camera_map:
            if camera not in images:
                raise KeyError(f"Cosmos3 observation is missing camera {camera!r}")
            validated_images[camera] = _validate_rgb(
                images[camera], camera=camera
            ).copy()
        validated_state = {
            key: _validate_state(state, key, dim).copy()
            for _wire_key, key, dim in STATE_SPEC
        }
        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = self._instruction
        if self._require_instruction and not instruction:
            raise ValueError("Cosmos3 observation requires the current instruction")
        self._latest_obs = {
            "images": validated_images,
            "state": validated_state,
            "instruction": instruction or "complete the task",
        }

    def infer_actions(self) -> np.ndarray:
        if self._latest_obs is None:
            raise RuntimeError(
                "No Cosmos3 observation buffered. Call ingest_observation first."
            )
        result = self._service.infer(self._build_request(self._latest_obs))
        return self._parse_actions(result)

    def finalize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        del episode_info
        self._latest_obs = None

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "cosmos3_backend": "in_process",
            "cosmos3_checkpoint": self._checkpoint_path,
            "cosmos3_model_output_shape": [self._model_action_horizon, 8],
            "cosmos3_wire_output_shape": [self._action_horizon, 8],
        }

    def close(self) -> None:
        self._latest_obs = None

    def _build_request(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = {
            wire_key: obs["images"][camera]
            for camera, wire_key in self._camera_map.items()
        }
        for wire_key, key, _dim in STATE_SPEC:
            request[wire_key] = obs["state"][key]
        request["prompt"] = obs["instruction"]
        return request

    def _parse_actions(self, result: Any) -> np.ndarray:
        if not isinstance(result, dict) or "action" not in result:
            raise RuntimeError("Cosmos3 response is missing 'action'")
        try:
            actions = np.asarray(result["action"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Cosmos3 action must be numeric") from exc
        expected = (self._model_action_horizon, 8)
        if actions.shape != expected:
            raise RuntimeError(
                f"Cosmos3 model output must be {expected}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("Cosmos3 action contains NaN/Inf")
        return np.ascontiguousarray(actions[: self._action_horizon])
