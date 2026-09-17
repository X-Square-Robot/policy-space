"""OpenPI PI0.5 ArtiXon Arm-6A end-effector-pose PolicySpace model.

The model is loaded directly in the PolicySpace process, matching the
DreamZero integration. OpenPI remains isolated in its own environment; this
module adds the explicitly configured OpenPI source root to ``sys.path`` only
when the model is constructed.

ManaEnv wire state/actions use 14-D dual-arm episode-init-relative EE poses:
``[x, y, z, roll, pitch, yaw, gripper] * 2``. The X2Real OpenPI checkpoint
uses the equivalent 20-D representation with configurable rotation-6D layout:
``[x, y, z, rot6d(6), gripper] * 2``.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from policy_space.base import PolicyModel
from policy_space.paths import external_source_root, resolve_data_path

WIRE_ACTION_DIM = 14
MODEL_ACTION_DIM = 20
DEFAULT_ACTION_HORIZON = 32
DEFAULT_TRAIN_CONFIG = "pi05_x2real_lerobotv2_finetune"
COMPATIBLE_TRAIN_CONFIG = "pi05_ex001_6r_joint_finetune"
PYTORCH_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
DEFAULT_CAMERA_MAP = {
    "camera_front": "cam_high",
    "camera_left": "cam_left_wrist",
    "camera_right": "cam_right_wrist",
}
DEFAULT_STATE_FIELDS = ("follow1_pos", "follow2_pos")
DEFAULT_ROTATION_LAYOUT = "row"
ROTATION_LAYOUTS = {"row", "column"}


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value
    if cfg.get(cfg_key) is not None:
        return cfg[cfg_key]
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_pytorch_compile_mode(cfg: dict[str, Any]) -> str | None:
    value = os.environ.get("OPENPI_PYTORCH_COMPILE_MODE")
    if value is None:
        value = cfg.get("pytorch_compile_mode")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "off", "false", "0"}:
        return None
    if normalized not in PYTORCH_COMPILE_MODES:
        raise ValueError(
            "OpenPI PI0.5 ArtiXon Arm-6A pytorch_compile_mode must be one of "
            f"{sorted(PYTORCH_COMPILE_MODES)} or disabled, got {value!r}"
        )
    return normalized


def _resolve_rotation_layout(cfg: dict[str, Any]) -> str:
    value = _env_or(
        cfg,
        "OPENPI_ROTATION_LAYOUT",
        "rotation_layout",
        DEFAULT_ROTATION_LAYOUT,
    )
    normalized = str(value).strip().lower()
    if normalized not in ROTATION_LAYOUTS:
        raise ValueError(
            "OpenPI PI0.5 ArtiXon Arm-6A rotation_layout must be one of "
            f"{sorted(ROTATION_LAYOUTS)}, got {value!r}"
        )
    return normalized


def _resolve_checkpoint_path(raw_path: str, *, source_root: Path) -> str:
    path_value = raw_path.strip()
    if not path_value:
        raise ValueError(
            "OpenPI PI0.5 ArtiXon Arm-6A checkpoint is not configured. "
            "Set OPENPI_CHECKPOINT_PATH or model.cfg.checkpoint_path."
        )
    if "://" in path_value:
        return path_value
    return str(resolve_data_path(path_value, base=source_root))


def _get_openpi_train_config(training_config: Any, train_config_name: str) -> Any:
    """Resolve the X2Real config with an inference-equivalent source fallback."""
    try:
        return training_config.get_config(train_config_name)
    except Exception as exc:
        if train_config_name != DEFAULT_TRAIN_CONFIG:
            raise ValueError(
                f"Unknown OpenPI train_config {train_config_name!r}"
            ) from exc

    # The X2Real training config currently lives in the training worktree. The
    # source-manifest checkout has an inference-equivalent config with the same
    # PI0.5 model, camera repack, ALOHA transforms, and adapt_to_pi=false.
    try:
        return training_config.get_config(COMPATIBLE_TRAIN_CONFIG)
    except Exception as fallback_exc:
        raise ValueError(
            f"OpenPI provides neither {DEFAULT_TRAIN_CONFIG!r} nor its "
            f"inference-compatible fallback {COMPATIBLE_TRAIN_CONFIG!r}"
        ) from fallback_exc


def _validate_openpi_train_config(config: Any) -> None:
    """Reject configs whose inference transforms cannot match the X2 checkpoint."""
    model = getattr(config, "model", None)
    data = getattr(config, "data", None)
    mismatches = {}
    if getattr(model, "action_horizon", None) != DEFAULT_ACTION_HORIZON:
        mismatches["model.action_horizon"] = getattr(model, "action_horizon", None)
    if getattr(data, "adapt_to_pi", None) is not False:
        mismatches["data.adapt_to_pi"] = getattr(data, "adapt_to_pi", None)
    if getattr(data, "use_delta_joint_actions", None) is not True:
        mismatches["data.use_delta_joint_actions"] = getattr(
            data, "use_delta_joint_actions", None
        )
    if mismatches:
        raise ValueError(f"Incompatible OpenPI PI0.5 ArtiXon Arm-6A train_config: {mismatches}")


class _X2EEOutputs:
    """Keep the 20 real X2 EE channels instead of ALOHA's fixed first 14."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < MODEL_ACTION_DIM:
            raise RuntimeError(
                f"OpenPI model actions must have shape (T, >= {MODEL_ACTION_DIM}), got {actions.shape}"
            )
        return {"actions": actions[:, :MODEL_ACTION_DIM]}


def _restore_x2_output_transform(
    policy: Any, aloha_output_type: type, compose: Any
) -> None:
    """Replace exactly one incompatible ALOHA14 output transform, fail closed."""
    composite = getattr(policy, "_output_transform", None)
    output_transforms = getattr(composite, "transforms", None)
    if output_transforms is None:
        raise RuntimeError(
            "Unsupported OpenPI Policy: output transform sequence is unavailable"
        )

    replacements = 0
    corrected = []
    for transform in output_transforms:
        if isinstance(transform, aloha_output_type):
            replacements += 1
            corrected.append(_X2EEOutputs())
        else:
            corrected.append(transform)
    if replacements != 1:
        raise RuntimeError(
            "OpenPI PI0.5 ArtiXon Arm-6A requires exactly one AlohaOutputs transform; "
            f"found {replacements}. Refusing an unverified output contract."
        )
    policy._output_transform = compose(tuple(corrected))


def _validate_x2_normalization(policy: Any, normalize_type: type) -> None:
    """Require one quantile normalizer carrying 20-D X2 state/action stats."""
    composite = getattr(policy, "_input_transform", None)
    input_transforms = getattr(composite, "transforms", None)
    if input_transforms is None:
        raise RuntimeError(
            "Unsupported OpenPI Policy: input transform sequence is unavailable"
        )
    normalizers = [
        transform
        for transform in input_transforms
        if isinstance(transform, normalize_type)
    ]
    if len(normalizers) != 1:
        raise RuntimeError(
            "OpenPI PI0.5 ArtiXon Arm-6A requires exactly one Normalize transform; "
            f"found {len(normalizers)}. Refusing an unverified normalization contract."
        )

    normalizer = normalizers[0]
    if getattr(normalizer, "use_quantiles", None) is not True:
        raise RuntimeError("OpenPI PI0.5 ArtiXon Arm-6A requires the PI0.5 quantile normalizer")
    norm_stats = getattr(normalizer, "norm_stats", None)
    if not isinstance(norm_stats, dict):
        raise RuntimeError(
            "OpenPI PI0.5 ArtiXon Arm-6A checkpoint does not provide normalization stats"
        )
    for key in ("state", "actions"):
        stats = norm_stats.get(key)
        if stats is None:
            raise RuntimeError(
                f"OpenPI PI0.5 ArtiXon Arm-6A checkpoint normalization is missing {key!r}"
            )
        for field in ("mean", "std", "q01", "q99"):
            values = getattr(stats, field, None)
            shape = None if values is None else np.asarray(values).shape
            if shape != (MODEL_ACTION_DIM,):
                raise RuntimeError(
                    f"OpenPI PI0.5 ArtiXon Arm-6A normalization {key}.{field} must have shape "
                    f"({MODEL_ACTION_DIM},), got {shape}"
                )


def _create_openpi_policy(
    train_config_name: str,
    checkpoint_path: str,
    device: str,
    pytorch_compile_mode: str | None,
    openpi_dir: Path,
) -> Any:
    openpi_path = str(openpi_dir)
    if openpi_path not in sys.path:
        sys.path.insert(0, openpi_path)

    try:
        from openpi.policies import policy_config
        from openpi.policies.aloha_policy import AlohaOutputs
        from openpi.training import config as training_config
        from openpi.transforms import Normalize, compose
    except ImportError as exc:
        raise RuntimeError(
            "OpenPI is unavailable. Set OPENPI_ROOT to its source checkout and run Policy Space "
            "with the Python interpreter from OpenPI's uv environment."
        ) from exc

    config = _get_openpi_train_config(training_config, train_config_name)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model, pytorch_compile_mode=pytorch_compile_mode
        ),
    )
    _validate_openpi_train_config(config)
    resolved_name = str(getattr(config, "name", train_config_name))
    if resolved_name != train_config_name:
        print(
            f"[OpenPI PI0.5 ArtiXon Arm-6A] Train config {train_config_name!r} is not present; "
            f"using inference-equivalent {resolved_name!r}"
        )

    policy = policy_config.create_trained_policy(
        config,
        checkpoint_path,
        pytorch_device=device,
    )
    _validate_x2_normalization(policy, Normalize)
    _restore_x2_output_transform(policy, AlohaOutputs, compose)
    return policy


def _euler_xyz_to_rot6d(
    euler_xyz: np.ndarray, *, layout: str = DEFAULT_ROTATION_LAYOUT
) -> np.ndarray:
    """Convert XYZ Euler radians to row- or column-based rotation-6D."""
    euler = np.asarray(euler_xyz, dtype=np.float32)
    if euler.ndim < 1 or euler.shape[-1] != 3:
        raise ValueError(f"Euler rotation must have shape (..., 3), got {euler.shape}")
    leading_shape = euler.shape[:-1]
    matrices = Rotation.from_euler(
        "xyz", euler.reshape(-1, 3), degrees=False
    ).as_matrix()
    if layout == "row":
        rot6d = matrices[:, :2, :].reshape(-1, 6)
    elif layout == "column":
        rot6d = matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    else:
        raise ValueError(f"Unsupported rotation-6D layout: {layout!r}")
    return rot6d.reshape(*leading_shape, 6).astype(np.float32, copy=False)


def _rot6d_to_euler_xyz(
    rot6d: np.ndarray, *, layout: str = DEFAULT_ROTATION_LAYOUT
) -> np.ndarray:
    """Decode row- or column-based rotation-6D after projecting onto SO(3)."""
    rotation = np.asarray(rot6d, dtype=np.float32)
    if rotation.ndim < 1 or rotation.shape[-1] != 6:
        raise ValueError(f"rotation-6D must have shape (..., 6), got {rotation.shape}")
    leading_shape = rotation.shape[:-1]
    flat = rotation.reshape(-1, 6)
    axis0, axis1 = flat[:, :3], flat[:, 3:]
    if layout == "row":
        matrices = np.stack(
            (axis0, axis1, np.cross(axis0, axis1, axis=-1)), axis=-2
        )
    elif layout == "column":
        matrices = np.stack(
            (axis0, axis1, np.cross(axis0, axis1, axis=-1)), axis=-1
        )
    else:
        raise ValueError(f"Unsupported rotation-6D layout: {layout!r}")

    left, _singular_values, right_t = np.linalg.svd(matrices)
    normalized = left @ right_t
    reflection_mask = np.linalg.det(normalized) < 0
    if np.any(reflection_mask):
        left = left.copy()
        left[reflection_mask, :, -1] *= -1
        normalized[reflection_mask] = left[reflection_mask] @ right_t[reflection_mask]

    euler = Rotation.from_matrix(normalized).as_euler("xyz", degrees=False)
    return euler.reshape(*leading_shape, 3).astype(np.float32, copy=False)


def _wire_to_model_pose(
    wire_pose: np.ndarray, *, rotation_layout: str = DEFAULT_ROTATION_LAYOUT
) -> np.ndarray:
    wire = np.asarray(wire_pose, dtype=np.float32)
    if wire.ndim < 1 or wire.shape[-1] != WIRE_ACTION_DIM:
        raise ValueError(
            f"EE wire pose must have shape (..., {WIRE_ACTION_DIM}), got {wire.shape}"
        )
    arms = []
    for start in (0, 7):
        arm = wire[..., start : start + 7]
        arms.append(
            np.concatenate(
                [
                    arm[..., :3],
                    _euler_xyz_to_rot6d(
                        arm[..., 3:6], layout=rotation_layout
                    ),
                    arm[..., 6:7],
                ],
                axis=-1,
            )
        )
    return np.concatenate(arms, axis=-1).astype(np.float32, copy=False)


def _model_to_wire_pose(
    model_pose: np.ndarray, *, rotation_layout: str = DEFAULT_ROTATION_LAYOUT
) -> np.ndarray:
    model = np.asarray(model_pose, dtype=np.float32)
    if model.ndim < 1 or model.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(
            f"OpenPI EE pose must have shape (..., {MODEL_ACTION_DIM}), got {model.shape}"
        )
    arms = []
    for start in (0, 10):
        arm = model[..., start : start + 10]
        arms.append(
            np.concatenate(
                [
                    arm[..., :3],
                    _rot6d_to_euler_xyz(
                        arm[..., 3:9], layout=rotation_layout
                    ),
                    arm[..., 9:10],
                ],
                axis=-1,
            )
        )
    return np.concatenate(arms, axis=-1).astype(np.float32, copy=False)


class Model(PolicyModel):
    """OpenPI PI0.5 policy for ArtiXon Arm-6A dual-arm EE-pose checkpoints."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)

        self.openpi_root = external_source_root(
            cfg,
            env_name="OPENPI_ROOT",
            cfg_key="openpi_root",
            project="OpenPI",
        )

        self.train_config = str(
            _env_or(
                cfg,
                "OPENPI_TRAIN_CONFIG",
                "train_config",
                DEFAULT_TRAIN_CONFIG,
            )
        ).strip()
        if not self.train_config:
            raise ValueError("OpenPI PI0.5 ArtiXon Arm-6A train_config must not be empty")

        checkpoint_path = str(
            _env_or(cfg, "OPENPI_CHECKPOINT_PATH", "checkpoint_path", "")
        )
        self.checkpoint_path = _resolve_checkpoint_path(
            checkpoint_path, source_root=self.openpi_root
        )
        self.device = str(_env_or(cfg, "OPENPI_DEVICE", "device", "cuda:0"))
        self.pytorch_compile_mode = _resolve_pytorch_compile_mode(cfg)
        self.rotation_layout = _resolve_rotation_layout(cfg)

        self.action_horizon = int(cfg.get("action_horizon", DEFAULT_ACTION_HORIZON))
        if self.action_horizon <= 0:
            raise ValueError("OpenPI PI0.5 ArtiXon Arm-6A action_horizon must be positive")

        self.camera_map = dict(cfg.get("camera_map") or DEFAULT_CAMERA_MAP)
        self.state_fields = tuple(cfg.get("state_fields") or DEFAULT_STATE_FIELDS)
        if len(self.state_fields) != 2:
            raise ValueError("OpenPI PI0.5 ArtiXon Arm-6A requires exactly two 7-D EE state fields")

        self.instruction = str(cfg.get("instruction", ""))
        self.require_instruction = _as_bool(cfg.get("require_instruction", True))

        print(f"[OpenPI PI0.5 ArtiXon Arm-6A] Loading checkpoint: {self.checkpoint_path}")
        print(
            f"[OpenPI PI0.5 ArtiXon Arm-6A] Train config: {self.train_config}, device: {self.device}"
        )
        print(
            f"[OpenPI PI0.5 ArtiXon Arm-6A] PyTorch compile mode: {self.pytorch_compile_mode or 'disabled'}"
        )
        print(
            f"[OpenPI PI0.5 ArtiXon Arm-6A] Rotation-6D layout: {self.rotation_layout}"
        )
        self._policy = _create_openpi_policy(
            self.train_config,
            self.checkpoint_path,
            self.device,
            self.pytorch_compile_mode,
            self.openpi_root,
        )
        print("[OpenPI PI0.5 ArtiXon Arm-6A] Model loaded successfully")

        self._latest_obs: dict[str, Any] | None = None

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        if episode_info:
            instruction = episode_info.get("instruction")
            if isinstance(instruction, str) and instruction.strip():
                self.instruction = instruction
        self._latest_obs = None

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = obs

    def infer_actions(self) -> np.ndarray:
        if self._latest_obs is None:
            raise RuntimeError(
                "No observation buffered. Call ingest_observation first."
            )

        policy_obs, current_wire_state = self._build_policy_observation(
            self._latest_obs
        )
        result = self._policy.infer(policy_obs)
        predicted_actions = self._parse_actions(result)

        # PolicySpaceImpl's EE transform skips row 0 as the current-state echo.
        # Prepending it preserves every predicted action row and matches the
        # existing DreamZero/Cosmos3 X2Robot wire contract.
        actions = np.concatenate(
            [current_wire_state[None, :], predicted_actions], axis=0
        )
        return np.ascontiguousarray(actions, dtype=np.float32)

    def close(self) -> None:
        # The model remains resident for the lifetime of the service process.
        self._latest_obs = None

    def _build_policy_observation(
        self, obs: dict[str, Any]
    ) -> tuple[dict[str, Any], np.ndarray]:
        images = obs.get("images") or {}
        missing_cameras = [name for name in self.camera_map if images.get(name) is None]
        if missing_cameras:
            raise ValueError(f"Missing required cameras: {missing_cameras}")

        policy_images = {
            model_name: self._prepare_image(images[policy_space_name])
            for policy_space_name, model_name in self.camera_map.items()
        }

        state = obs.get("state") or {}
        state_parts = [self._state_vector(state, field) for field in self.state_fields]
        wire_state = np.concatenate(state_parts).astype(np.float32, copy=False)

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = self.instruction
        if self.require_instruction and (
            not isinstance(instruction, str) or not instruction.strip()
        ):
            raise ValueError(
                "OpenPI PI0.5 ArtiXon Arm-6A requires a non-empty task instruction. "
                "Pass it in the observation or configure model.cfg.instruction."
            )

        return {
            "images": policy_images,
            "state": _wire_to_model_pose(
                wire_state, rotation_layout=self.rotation_layout
            ),
            "prompt": str(instruction or ""),
        }, wire_state

    def _parse_actions(self, result: Any) -> np.ndarray:
        if not isinstance(result, dict) or "actions" not in result:
            raise RuntimeError("OpenPI response is missing the 'actions' key")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if (
            actions.ndim != 2
            or actions.shape[0] <= 0
            or actions.shape[1] != MODEL_ACTION_DIM
        ):
            raise RuntimeError(
                f"OpenPI X2 actions must have shape (T, {MODEL_ACTION_DIM}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("OpenPI returned NaN/Inf actions")
        wire_actions = _model_to_wire_pose(
            actions[: self.action_horizon], rotation_layout=self.rotation_layout
        )
        return np.ascontiguousarray(wire_actions)

    @staticmethod
    def _prepare_image(image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Image must be HWC RGB, got shape {array.shape}")
        if array.dtype != np.uint8:
            if (
                np.issubdtype(array.dtype, np.floating)
                and array.size
                and float(array.max()) <= 1.0
            ):
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))

    @staticmethod
    def _state_vector(state: dict[str, Any], field: str) -> np.ndarray:
        value = state.get(field)
        if value is None:
            raise ValueError(f"Missing required state field: {field!r}")
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (7,):
            raise ValueError(f"state[{field!r}] must be shape (7,), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"state[{field!r}] contains NaN/Inf")
        return np.ascontiguousarray(array)
