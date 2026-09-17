"""DreamZero X2Robot PolicySpace model for the dual-arm joint checkpoint.

The 6K full-finetune checkpoint predicts two relative 7-D joint vectors
(``6 arm joints + gripper``).  This adapter owns the causal history, converts
those predictions back to absolute simulator targets, and pads the physical
14-D command to ArtiXon Arm-6A's 26-D joint replay layout.  The trailing 12 channels
are record-only master-EE slots and therefore remain zero.
"""

from __future__ import annotations

import collections
import concurrent.futures
import datetime
import functools
import json
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np

from policy_space.base import PolicyModel
from policy_space.paths import external_source_root, resolve_data_path

PHYSICAL_ACTION_DIM = 14
MODEL_CHUNK_SIZE = 24
SIM_ACTION_DIM = 26
ARM_ACTION_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)

DEFAULT_CAMERA_MAP = {
    "camera_front": "video.face",
    "camera_left": "video.left_wrist",
    "camera_right": "video.right_wrist",
}
# Training video delta_indices are [0, 3, ..., 24].  At a refill the deque
# contains the preceding 24 executed control frames plus the current frame.
DEFAULT_RELATIVE_OFFSETS = [-24, -21, -18, -15, -12, -9, -6, -3, 0]

STATE_MODEL_KEYS = {
    "follow1_pos": "state.left_arm_joint_position",
    "follow2_pos": "state.right_arm_joint_position",
}
ACTION_MODEL_KEYS = (
    "action.left_arm_joint_position",
    "action.right_arm_joint_position",
)

_T = TypeVar("_T")


def _ensure_dreamzero_imports(dreamzero_root: Path) -> None:
    dreamzero_str = str(dreamzero_root)
    if dreamzero_str not in sys.path:
        sys.path.insert(0, dreamzero_str)


def _ensure_dist_initialized() -> None:
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        rendezvous_file = Path(f"/tmp/dreamzero_dist_ps_{os.getpid()}")
        if rendezvous_file.exists():
            rendezvous_file.unlink()
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_file}",
            world_size=1,
            rank=0,
        )


def _disable_cudagraphs() -> None:
    """Keep torch.compile while disabling graphs for 1-frame/9-frame inputs."""
    if os.environ.get("DREAMZERO_CUDAGRAPHS", "0") == "1":
        return

    import torch

    if getattr(torch.compile, "_dreamzero_no_cudagraphs", False):
        return
    original = torch.compile

    @functools.wraps(original)
    def compile_without_cudagraphs(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("mode") == "reduce-overhead":
            kwargs.pop("mode")
        return original(*args, **kwargs)

    compile_without_cudagraphs._dreamzero_no_cudagraphs = True  # type: ignore[attr-defined]
    torch.compile = compile_without_cudagraphs  # type: ignore[assignment]
    print("[DreamZero X2Robot] CUDA Graphs disabled (shape-varying causal input)")


def _configure_torch_dynamo() -> None:
    try:
        import torch._dynamo.config as dynamo_cfg
    except Exception:
        return
    for key, default in (
        ("cache_size_limit", 1000),
        ("recompile_limit", 800),
        ("accumulated_cache_size_limit", 1000),
        ("accumulated_recompile_limit", 2000),
    ):
        value = int(os.environ.get(f"DREAMZERO_DYNAMO_{key.upper()}", str(default)))
        if hasattr(dynamo_cfg, key):
            setattr(dynamo_cfg, key, value)


def _as_action_array(value: Any, torch_module: Any, *, key: str) -> np.ndarray:
    if torch_module.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.size % 7:
        raise RuntimeError(
            f"DreamZero output {key!r} cannot be reshaped to 7-D joints: {arr.shape}"
        )
    arr = arr.reshape(-1, 7)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"DreamZero output {key!r} contains NaN/Inf")
    return arr


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {raw!r}"
    )


def _env_or_cfg(
    cfg: dict[str, Any], *, env_name: str, cfg_key: str, default: Any = None
) -> Any:
    """Resolve an explicit launcher override before the checked-in default."""
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value != "":
        return env_value
    return cfg.get(cfg_key, default)


def _env_int_or_cfg(
    cfg: dict[str, Any], *, env_name: str, cfg_key: str, default: Any = None
) -> int:
    """Resolve an integer launcher override before the checked-in default."""
    return int(_env_or_cfg(cfg, env_name=env_name, cfg_key=cfg_key, default=default))


def _env_csv_ints_or_cfg(
    cfg: dict[str, Any], *, env_name: str, cfg_key: str, default: Any = None
) -> list[int]:
    """Resolve a comma-separated integer-list override before the YAML list."""
    value = _env_or_cfg(cfg, env_name=env_name, cfg_key=cfg_key, default=default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = list(value or [])
    if not items:
        raise ValueError(f"{env_name} must contain at least one integer")
    return [int(item) for item in items]


def _resample_rows_linear(values: np.ndarray, target_steps: int) -> np.ndarray:
    """Resample a joint trajectory along time without rotation semantics."""
    source_steps = int(values.shape[0])
    if source_steps == target_steps:
        return np.asarray(values, dtype=np.float32).copy()
    source = np.linspace(0.0, 1.0, source_steps)
    target = np.linspace(0.0, 1.0, target_steps)
    return np.stack(
        [
            np.interp(target, source, values[:, dim])
            for dim in range(int(values.shape[1]))
        ],
        axis=-1,
    ).astype(np.float32)


def _savgol_smooth_rows(
    values: np.ndarray, *, window_length: int, polyorder: int
) -> np.ndarray:
    """Apply a dependency-free zero-derivative Savitzky-Golay filter."""
    half_window = window_length // 2
    offsets = np.arange(-half_window, half_window + 1, dtype=np.float64)
    vandermonde = np.vander(offsets, polyorder + 1, increasing=True)
    coefficients = np.linalg.pinv(vandermonde)[0]
    padded = np.pad(values, ((half_window, half_window), (0, 0)), mode="edge")
    return np.stack(
        [
            np.convolve(padded[:, dim], coefficients[::-1], mode="valid")
            for dim in range(int(values.shape[1]))
        ],
        axis=-1,
    ).astype(np.float32)


class Model(PolicyModel):
    """DreamZero X2Robot joint inference with one CUDA owner thread."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self._dreamzero_root = external_source_root(
            cfg,
            env_name="DREAMZERO_ROOT",
            cfg_key="dreamzero_root",
            project="DreamZero",
        )
        self._camera_map: dict[str, str] = dict(
            cfg.get("camera_map") or DEFAULT_CAMERA_MAP
        )
        self._relative_offsets = _env_csv_ints_or_cfg(
            cfg,
            env_name="DREAMZERO_RELATIVE_OFFSETS",
            cfg_key="relative_offsets",
            default=DEFAULT_RELATIVE_OFFSETS,
        )
        if (
            self._relative_offsets != sorted(set(self._relative_offsets))
            or self._relative_offsets[-1] != 0
        ):
            raise ValueError(
                "DreamZero X2Robot relative_offsets must be unique, chronological, and end at the current frame 0"
            )
        self._history_window = int(
            cfg.get("history_window", abs(min(self._relative_offsets)) + 1)
        )
        if self._history_window < 1:
            raise ValueError("DreamZero X2Robot history_window must be >= 1")
        if self._relative_offsets[0] < -self._history_window + 1:
            raise ValueError(
                "DreamZero X2Robot relative_offsets exceed the configured history_window"
            )
        resolution = cfg.get("image_resolution") or (176, 320)
        self._image_resolution = (int(resolution[0]), int(resolution[1]))
        self._instruction = str(cfg.get("instruction", ""))
        self._require_instruction = bool(cfg.get("require_instruction", True))
        self._action_horizon = int(cfg.get("action_horizon", MODEL_CHUNK_SIZE))
        self._output_action_dim = int(cfg.get("output_action_dim", SIM_ACTION_DIM))
        # GrootSimPolicy.unapply() already reverses the training-time relative-action
        # transform using the latest state before returning ``result.act``.  Adding the
        # state again here doubles every non-zero joint target and quickly drives the
        # simulator into its joint limits.  Keep the compatibility switch only for an
        # inference backend that explicitly returns raw relative deltas.
        self._relative_actions = bool(cfg.get("relative_actions", False))
        self._inference_method = str(
            cfg.get("inference_method", "lazy_joint_forward_causal")
        )
        self._replan_context_mode = (
            str(
                _env_or_cfg(
                    cfg,
                    env_name="DREAMZERO_REPLAN_CONTEXT_MODE",
                    cfg_key="replan_context_mode",
                    default="causal",
                )
            )
            .strip()
            .lower()
        )
        if self._replan_context_mode not in {"causal", "single"}:
            raise ValueError(
                "DreamZero X2Robot replan_context_mode must be 'causal' or 'single', "
                f"got {self._replan_context_mode!r}"
            )
        self._execute_horizon = _env_int_or_cfg(
            cfg,
            env_name="DREAMZERO_EXECUTE_HORIZON",
            cfg_key="execute_horizon",
            default=self._action_horizon,
        )
        if not 1 <= self._execute_horizon <= self._action_horizon:
            raise ValueError(
                f"DreamZero X2Robot execute_horizon must be in [1, {self._action_horizon}], "
                f"got {self._execute_horizon}"
            )
        # DreamZero always predicts video latents jointly with actions. This
        # debug switch controls whether those otherwise-discarded latents are
        # copied to CPU and decoded into per-view MP4 files at episode reset.
        # It does not remove the video branch from the trained model.
        self._save_video_pred = _env_flag(
            "DREAMZERO_SAVE_VIDEO_PRED", bool(cfg.get("save_video_pred", False))
        )
        video_output_dir = cfg.get("video_output_dir") or os.environ.get(
            "DREAMZERO_VIDEO_OUTPUT_DIR", "/tmp/dreamzero_x2robot_video_pred"
        )
        self._video_output_dir = self._resolve_path(str(video_output_dir))
        self._video_fps = float(cfg.get("video_fps", 15.0))
        if self._video_fps <= 0:
            raise ValueError("DreamZero X2Robot video_fps must be > 0")
        debounce_cfg = cfg.get("gripper_debounce") or {}
        self._gripper_debounce_enabled = bool(debounce_cfg.get("enabled", False))
        self._gripper_filter_mode = str(debounce_cfg.get("mode", "binary_slew"))
        self._gripper_close_threshold = float(debounce_cfg.get("close_threshold", 0.15))
        self._gripper_open_threshold = float(debounce_cfg.get("open_threshold", 0.35))
        self._gripper_min_stable_steps = int(debounce_cfg.get("min_stable_steps", 3))
        self._gripper_closed_target = float(debounce_cfg.get("closed_target", 0.0))
        self._gripper_open_target = float(debounce_cfg.get("open_target", 1.0))
        self._gripper_max_delta_per_step = float(
            debounce_cfg.get("max_delta_per_step", 1.0)
        )
        self._gripper_deadband = float(debounce_cfg.get("deadband", 0.02))
        self._gripper_ema_alpha = float(debounce_cfg.get("ema_alpha", 0.65))
        self._gripper_max_close_delta_per_step = float(
            debounce_cfg.get("max_close_delta_per_step", 0.20)
        )
        self._gripper_max_open_delta_per_step = float(
            debounce_cfg.get("max_open_delta_per_step", 0.10)
        )
        self._gripper_release_threshold = float(
            debounce_cfg.get("release_threshold", 0.75)
        )
        self._gripper_release_stable_steps = int(
            debounce_cfg.get("release_stable_steps", 4)
        )
        if self._gripper_filter_mode not in {
            "binary_slew",
            "continuous_slew",
            "continuous_hysteresis",
        }:
            raise ValueError(
                "DreamZero X2Robot gripper debounce mode must be binary_slew, continuous_slew, "
                "or continuous_hysteresis, "
                f"got {self._gripper_filter_mode!r}"
            )
        if self._gripper_close_threshold >= self._gripper_open_threshold:
            raise ValueError(
                "DreamZero X2Robot gripper close_threshold must be below open_threshold"
            )
        if self._gripper_min_stable_steps < 1:
            raise ValueError("DreamZero X2Robot gripper min_stable_steps must be >= 1")
        if not np.isfinite(
            [self._gripper_closed_target, self._gripper_open_target]
        ).all():
            raise ValueError("DreamZero X2Robot gripper hold targets must be finite")
        if self._gripper_closed_target >= self._gripper_open_target:
            raise ValueError(
                "DreamZero X2Robot gripper closed_target must be below open_target"
            )
        if (
            not np.isfinite(self._gripper_max_delta_per_step)
            or self._gripper_max_delta_per_step <= 0
        ):
            raise ValueError(
                "DreamZero X2Robot gripper max_delta_per_step must be finite and > 0"
            )
        if not np.isfinite(self._gripper_deadband) or self._gripper_deadband < 0:
            raise ValueError(
                "DreamZero X2Robot gripper deadband must be finite and >= 0"
            )
        if (
            not np.isfinite(self._gripper_ema_alpha)
            or not 0 < self._gripper_ema_alpha <= 1
        ):
            raise ValueError(
                "DreamZero X2Robot gripper ema_alpha must be finite and in (0, 1]"
            )
        for name, value in (
            ("max_close_delta_per_step", self._gripper_max_close_delta_per_step),
            ("max_open_delta_per_step", self._gripper_max_open_delta_per_step),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"DreamZero X2Robot gripper {name} must be finite and > 0"
                )
        if (
            not self._gripper_open_threshold
            < self._gripper_release_threshold
            <= self._gripper_open_target
        ):
            raise ValueError(
                "DreamZero X2Robot gripper release_threshold must be above open_threshold "
                "and no greater than open_target"
            )
        if self._gripper_release_stable_steps < 1:
            raise ValueError(
                "DreamZero X2Robot gripper release_stable_steps must be >= 1"
            )

        smoothing_cfg = cfg.get("action_smoothing") or {}
        self._action_smoothing_enabled = bool(smoothing_cfg.get("enabled", False))
        self._action_interpolation_multiplier = int(
            smoothing_cfg.get("interpolation_multiplier", 2)
        )
        self._action_smoothing_window = int(smoothing_cfg.get("window_length", 7))
        self._action_smoothing_polyorder = int(smoothing_cfg.get("polyorder", 2))
        self._action_boundary_blend_steps = _env_int_or_cfg(
            smoothing_cfg,
            env_name="DREAMZERO_ACTION_BOUNDARY_BLEND_STEPS",
            cfg_key="boundary_blend_steps",
            default=6,
        )
        if self._action_interpolation_multiplier < 1:
            raise ValueError(
                "DreamZero X2Robot action interpolation_multiplier must be >= 1"
            )
        if self._action_smoothing_window < 3 or self._action_smoothing_window % 2 == 0:
            raise ValueError(
                "DreamZero X2Robot action smoothing window_length must be an odd integer >= 3"
            )
        if not 0 <= self._action_smoothing_polyorder < self._action_smoothing_window:
            raise ValueError(
                "DreamZero X2Robot action smoothing polyorder must be in [0, window_length)"
            )
        if self._action_boundary_blend_steps < 2:
            raise ValueError(
                "DreamZero X2Robot action boundary_blend_steps must be >= 2"
            )

        checkpoint_path = _env_or_cfg(
            cfg,
            env_name="DREAMZERO_CHECKPOINT_PATH",
            cfg_key="checkpoint_path",
            default="",
        )
        if not str(checkpoint_path).strip():
            raise RuntimeError(
                "DreamZero checkpoint is not configured; set "
                "DREAMZERO_CHECKPOINT_PATH or model.cfg.checkpoint_path"
            )
        self._checkpoint_path = str(self._resolve_path(str(checkpoint_path)))
        tokenizer_path = _env_or_cfg(
            cfg,
            env_name="DREAMZERO_TOKENIZER_PATH",
            cfg_key="tokenizer_path",
        )
        self._tokenizer_path = (
            str(self._resolve_path(str(tokenizer_path))) if tokenizer_path else None
        )

        device = str(cfg.get("device", "cuda:0"))
        self._device = device
        skip_img_transform = bool(cfg.get("skip_img_transform", False))
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dreamzero-x2robot",
        )
        self._run(lambda: self._load_owned(device, skip_img_transform))

    def _load_owned(self, device: str, skip_img_transform: bool) -> None:
        _ensure_dreamzero_imports(self._dreamzero_root)
        _configure_torch_dynamo()
        _ensure_dist_initialized()
        # The scheduler applies @torch.compile at import time.
        _disable_cudagraphs()

        import torch
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from tianshou.data import Batch

        self._torch = torch
        self._Batch = Batch
        embodiment = str(
            self.cfg.get("embodiment_tag", "x2robot_dreamzero_dual_arm_joint")
        )
        print(f"[DreamZero X2Robot] Loading checkpoint: {self._checkpoint_path}")
        print(f"[DreamZero X2Robot] Embodiment: {embodiment}, device: {device}")
        self._policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(embodiment),
            model_path=self._checkpoint_path,
            device=device,
            tokenizer_path_override=self._tokenizer_path,
            skip_img_transform=skip_img_transform,
        )

        modality_configs = self._policy.modality_configs
        actual_video = list(modality_configs.video.modality_keys)
        actual_state = list(modality_configs.state.modality_keys)
        actual_action = list(modality_configs.action.modality_keys)
        expected_video = list(self._camera_map.values())
        expected_state = list(STATE_MODEL_KEYS.values())
        if actual_video != expected_video:
            raise RuntimeError(
                f"DreamZero video modalities mismatch: expected={expected_video}, actual={actual_video}"
            )
        if actual_state != expected_state:
            raise RuntimeError(
                f"DreamZero state modalities mismatch: expected={expected_state}, actual={actual_state}"
            )
        if actual_action != list(ACTION_MODEL_KEYS):
            raise RuntimeError(
                f"DreamZero action modalities mismatch: expected={list(ACTION_MODEL_KEYS)}, actual={actual_action}"
            )
        self._language_key = list(modality_configs.language.modality_keys)[0]
        print(
            "[DreamZero X2Robot] Model loaded successfully "
            f"(resolution={self._image_resolution}, history={len(self._relative_offsets)} frames, "
            f"context={self._replan_context_mode}, execute_horizon={self._execute_horizon}, "
            f"smoothing={'on' if self._action_smoothing_enabled else 'off'}, "
            f"gripper_filter={self._gripper_filter_mode if self._gripper_debounce_enabled else 'off'}, "
            f"video_export={'on' if self._save_video_pred else 'off'})"
        )
        self._session_id: str | None = None
        self._frame_history: dict[str, collections.deque[np.ndarray]] = {}
        self._request_index = 0
        self._latest_obs: dict[str, Any] | None = None
        self._observed_since_predict = False
        self._reset_gripper_debounce_state()
        self._last_arm_command: np.ndarray | None = None
        self._video_pred_latents: list[Any] = []
        self._video_episode_index = 0

    def _run(self, fn: Callable[[], _T]) -> _T:
        # websockets.sync gives each connection a different thread, while both
        # torch.compile state and DreamZero's causal KV cache are thread-owned.
        return self._executor.submit(fn).result()

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        self._run(lambda: self._initialize_episode_owned(episode_info))

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._run(lambda: self._ingest_observation_owned(obs))

    def infer_actions(self) -> np.ndarray:
        return self._run(self._infer_actions_owned)

    def runtime_metadata(self) -> dict[str, Any]:
        """Expose launcher-selected execution hints through the handshake."""
        return {"recommended_execute_horizon": int(self._execute_horizon)}

    def close(self) -> None:
        # Process cleanup remains on the model-owned worker thread.
        self._run(self._close_owned)

    def _initialize_episode_owned(
        self, episode_info: dict[str, Any] | None = None
    ) -> None:
        self._flush_prediction_video_owned()
        if episode_info:
            instruction = episode_info.get("instruction")
            if isinstance(instruction, str) and instruction.strip():
                self._instruction = instruction
        self._reset_policy_state_owned()
        self._session_id = str(uuid.uuid4())
        self._frame_history = {}
        self._request_index = 0
        self._latest_obs = None
        self._observed_since_predict = False
        self._reset_gripper_debounce_state()
        self._last_arm_command = None

    def _reset_policy_state_owned(self) -> None:
        head = getattr(
            getattr(self._policy, "trained_model", None), "action_head", None
        )
        if head is None:
            return
        if hasattr(head, "current_start_frame"):
            head.current_start_frame = 0
        if hasattr(head, "language"):
            head.language = None

    def _reset_replan_cursor_owned(self) -> None:
        """Reset only causal time while retaining the episode language cache."""
        head = getattr(
            getattr(self._policy, "trained_model", None), "action_head", None
        )
        if head is not None and hasattr(head, "current_start_frame"):
            head.current_start_frame = 0

    def _ingest_observation_owned(self, obs: dict[str, Any]) -> None:
        self._append_frames(obs)
        self._latest_obs = obs
        self._observed_since_predict = True

    def _infer_actions_owned(self) -> np.ndarray:
        if self._latest_obs is None:
            raise RuntimeError(
                "No observation buffered. Call ingest_observation first."
            )
        if self._session_id is None:
            self._initialize_episode_owned()
        if not self._observed_since_predict:
            self._append_frames(self._latest_obs)

        # A causal call advances the model's KV/RoPE cursor by one complete
        # 24-step block. Receding-horizon execution advances the simulator by
        # fewer steps, so retaining that cursor would put the next prediction
        # at the wrong logical time. ``single`` deliberately rebuilds a fresh
        # anchor from the latest real observation at every replan.
        if self._replan_context_mode == "single":
            self._reset_replan_cursor_owned()

        request = self._build_request(self._latest_obs)
        batch = self._Batch(obs=request)
        with self._torch.inference_mode():
            if self._inference_method == "lazy_joint_forward_causal":
                result, video_pred = self._policy.lazy_joint_forward_causal(batch)
            elif self._inference_method == "lazy_joint_forward_causal_gt_cond":
                result, video_pred = self._policy.lazy_joint_forward_causal_gt_cond(
                    batch
                )
            else:
                result = self._policy.forward(batch)
                video_pred = None

        self._accumulate_prediction_video_owned(video_pred)

        left = _as_action_array(
            result.act[ACTION_MODEL_KEYS[0]], self._torch, key=ACTION_MODEL_KEYS[0]
        )
        right = _as_action_array(
            result.act[ACTION_MODEL_KEYS[1]], self._torch, key=ACTION_MODEL_KEYS[1]
        )
        if left.shape != right.shape:
            raise RuntimeError(
                f"DreamZero arm horizons differ: left={left.shape}, right={right.shape}"
            )
        current = self._current_joint_state(self._latest_obs)
        physical = np.concatenate([left, right], axis=-1)
        if self._relative_actions:
            physical = physical + current[None, :]
        physical = physical[: self._action_horizon]
        if physical.shape != (self._action_horizon, PHYSICAL_ACTION_DIM):
            raise RuntimeError(
                f"DreamZero returned {physical.shape}; expected ({self._action_horizon}, {PHYSICAL_ACTION_DIM})"
            )
        if self._action_smoothing_enabled:
            physical = self._smooth_arm_actions(physical, current)
        if self._gripper_debounce_enabled:
            physical = self._debounce_grippers(physical, current)
        if self._output_action_dim < PHYSICAL_ACTION_DIM:
            raise RuntimeError(
                f"output_action_dim={self._output_action_dim} is smaller than {PHYSICAL_ACTION_DIM}"
            )
        actions = np.zeros(
            (self._action_horizon, self._output_action_dim), dtype=np.float32
        )
        actions[:, :PHYSICAL_ACTION_DIM] = physical

        self._request_index += 1
        self._observed_since_predict = False
        return actions

    def _close_owned(self) -> None:
        self._flush_prediction_video_owned()
        self._reset_policy_state_owned()
        self._session_id = None
        self._frame_history = {}
        self._request_index = 0
        self._latest_obs = None
        self._observed_since_predict = False
        self._reset_gripper_debounce_state()
        self._last_arm_command = None

    def _accumulate_prediction_video_owned(self, video_pred: Any) -> None:
        """Keep predicted video latents on CPU when diagnostic export is enabled."""
        if not self._save_video_pred or video_pred is None:
            return
        try:
            if not self._torch.is_tensor(video_pred):
                video_pred = self._torch.as_tensor(video_pred)
            self._video_pred_latents.append(video_pred.detach().to("cpu"))
        except Exception as exc:
            # Video export is diagnostic-only and must never fail action serving.
            print(
                f"[DreamZero X2Robot] WARNING: failed to buffer predicted video: {exc}"
            )

    def _flush_prediction_video_owned(self) -> None:
        """Decode and persist one episode's predicted video, best effort."""
        if not self._save_video_pred or not self._video_pred_latents:
            return

        latents = self._video_pred_latents
        self._video_pred_latents = []
        session_id = self._session_id or f"session-{self._video_episode_index:04d}"
        episode_dir = (
            self._video_output_dir
            / f"episode_{self._video_episode_index:04d}_{session_id[:8]}"
        )
        self._video_episode_index += 1
        try:
            import imageio.v2 as imageio

            action_head = self._policy.trained_model.action_head
            joined = self._torch.cat(latents, dim=2).to(self._device)
            with self._torch.inference_mode():
                frames = action_head.vae.decode(
                    joined,
                    tiled=action_head.tiled,
                    tile_size=(
                        action_head.tile_size_height,
                        action_head.tile_size_width,
                    ),
                    tile_stride=(
                        action_head.tile_stride_height,
                        action_head.tile_stride_width,
                    ),
                )
            frames = frames.permute(0, 2, 3, 4, 1)
            frames = (
                ((frames.float() + 1.0) * 127.5)
                .clip(0, 255)
                .to("cpu")
                .numpy()
                .astype(np.uint8)
            )

            episode_dir.mkdir(parents=True, exist_ok=True)
            view_names = [
                model_key.rsplit(".", 1)[-1] for model_key in self._camera_map.values()
            ]
            for view_index, view_frames in enumerate(frames):
                view_name = (
                    view_names[view_index]
                    if view_index < len(view_names)
                    else f"view{view_index}"
                )
                imageio.mimsave(
                    episode_dir / f"pred_{view_name}.mp4",
                    list(view_frames),
                    fps=self._video_fps,
                    codec="libx264",
                )
            metadata = {
                "session_id": session_id,
                "instruction": self._instruction,
                "fps": self._video_fps,
                "latent_chunks": len(latents),
                "views": view_names[: int(frames.shape[0])],
                "frames_per_view": int(frames.shape[1]),
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            with (episode_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
            print(f"[DreamZero X2Robot] Saved predicted video -> {episode_dir}")
        except Exception as exc:
            # Decoding adds GPU work and is deliberately isolated from serving.
            print(f"[DreamZero X2Robot] WARNING: failed to save predicted video: {exc}")

    def _resolve_path(self, value: str) -> Path:
        return resolve_data_path(value, base=self._dreamzero_root)

    def _append_frames(self, obs: dict[str, Any]) -> None:
        images = obs.get("images", {})
        for sdk_name, model_key in self._camera_map.items():
            image = images.get(sdk_name)
            if image is None:
                continue
            history = self._frame_history.setdefault(
                model_key,
                collections.deque(maxlen=self._history_window),
            )
            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                arr = np.clip(
                    arr * 255 if arr.size and arr.max() <= 1.0 else arr, 0, 255
                ).astype(np.uint8)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"DreamZero camera {sdk_name!r} must be HWC RGB, got {arr.shape}"
                )
            target_h, target_w = self._image_resolution
            if arr.shape[:2] != (target_h, target_w):
                arr = cv2.resize(
                    arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            history.append(np.ascontiguousarray(arr))

    def _build_request(self, obs: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {}
        for model_key in self._camera_map.values():
            history = self._frame_history.get(model_key)
            if not history:
                raise RuntimeError(f"No frame history for {model_key}")
            frames = list(history)
            if self._request_index == 0 or self._replan_context_mode == "single":
                selected = [frames[-1]]
            else:
                selected = [
                    frames[max(0, len(frames) - 1 + offset)]
                    for offset in self._relative_offsets
                ]
            request[model_key] = np.stack(selected, axis=0)

        current = self._current_joint_state(obs)
        request[STATE_MODEL_KEYS["follow1_pos"]] = current[:7].reshape(1, 7)
        request[STATE_MODEL_KEYS["follow2_pos"]] = current[7:].reshape(1, 7)
        instruction = obs.get("instruction") or self._instruction
        if self._require_instruction and not instruction:
            raise ValueError("DreamZero X2Robot requires a task instruction.")
        self._instruction = str(instruction or "")
        request[self._language_key] = self._instruction
        return request

    def _smooth_arm_actions(
        self, actions: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Smooth arm targets and make consecutive 24-step blocks continuous.

        DreamZero's trained block remains intact in time: the trajectory is
        linearly upsampled, Savitzky-Golay filtered, and downsampled back to the
        same horizon. The start of each block is then connected to the last
        command over a short prefix. Grippers are deliberately excluded.
        """
        out = actions.copy()
        prefix = min(self._execute_horizon, int(out.shape[0]))
        if prefix < 2:
            return out

        arm = out[:prefix, ARM_ACTION_INDICES]
        interpolated_steps = max(prefix, prefix * self._action_interpolation_multiplier)
        arm = _resample_rows_linear(arm, interpolated_steps)
        window = min(
            self._action_smoothing_window,
            interpolated_steps if interpolated_steps % 2 else interpolated_steps - 1,
        )
        if window > self._action_smoothing_polyorder:
            arm = _savgol_smooth_rows(
                arm,
                window_length=window,
                polyorder=self._action_smoothing_polyorder,
            )
        arm = _resample_rows_linear(arm, prefix)

        anchor = self._last_arm_command
        if anchor is None:
            anchor = np.asarray(current[ARM_ACTION_INDICES], dtype=np.float32)
        blend_steps = min(prefix, self._action_boundary_blend_steps)
        weights = np.zeros(prefix, dtype=np.float32)
        weights[:blend_steps] = np.linspace(1.0, 0.0, blend_steps, dtype=np.float32)
        arm += weights[:, None] * (anchor - arm[0])[None, :]

        out[:prefix, ARM_ACTION_INDICES] = arm
        self._last_arm_command = arm[-1].copy()
        return out

    def _debounce_grippers(
        self, actions: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Suppress short open/close reversals in the prefix the sim executes.

        DreamZero emits continuous absolute gripper joints. Values below the
        close threshold and above the open threshold express discrete intent;
        the dead band preserves the current intent. A new intent must persist
        for ``min_stable_steps`` predicted rows before it is applied. Accepted
        targets and pending transitions persist across chunks: the measured
        joint may still lag when a refill arrives, so using it as the next
        chunk's hold target would turn a valid transition at the previous
        chunk boundary into a one-frame reversal. Once an intent is accepted,
        move toward its canonical normalized target at a bounded rate instead
        of either passing through same-state analog noise or jumping to the
        endpoint in one simulator step. Only the executed prefix updates the
        filter -- discarded look-ahead rows must not leak into the next replan.
        """
        if self._gripper_filter_mode in {"continuous_slew", "continuous_hysteresis"}:
            return self._filter_continuous_grippers(actions, current)

        out = actions.copy()
        prefix = min(self._execute_horizon, int(out.shape[0]))
        for index in (6, 13):
            accepted = self._gripper_accepted_target.get(index, float(current[index]))
            is_open = self._gripper_open_state.get(
                index, accepted >= self._gripper_open_threshold
            )
            pending = self._gripper_pending_state.get(index)
            pending_steps = self._gripper_pending_steps.get(index, 0)
            for step in range(prefix):
                target = float(out[step, index])
                candidate: bool | None
                if target <= self._gripper_close_threshold:
                    candidate = False
                elif target >= self._gripper_open_threshold:
                    candidate = True
                else:
                    candidate = None

                hold_target = (
                    self._gripper_open_target
                    if is_open
                    else self._gripper_closed_target
                )
                if candidate is None or candidate == is_open:
                    pending = None
                    pending_steps = 0
                    accepted = self._slew_gripper_target(accepted, hold_target)
                    out[step, index] = accepted
                    continue

                if pending != candidate:
                    pending = candidate
                    pending_steps = 1
                else:
                    pending_steps += 1
                if pending_steps < self._gripper_min_stable_steps:
                    out[step, index] = accepted
                    continue

                is_open = candidate
                pending = None
                pending_steps = 0
                hold_target = (
                    self._gripper_open_target
                    if is_open
                    else self._gripper_closed_target
                )
                accepted = self._slew_gripper_target(accepted, hold_target)
                out[step, index] = accepted
            self._gripper_open_state[index] = is_open
            self._gripper_accepted_target[index] = accepted
            if pending is None:
                self._gripper_pending_state.pop(index, None)
                self._gripper_pending_steps.pop(index, None)
            else:
                self._gripper_pending_state[index] = pending
                self._gripper_pending_steps[index] = pending_steps
        return out

    def _filter_continuous_grippers(
        self, actions: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Smooth continuous gripper targets without delaying grasp intent.

        DreamZero was trained with normalized continuous gripper joints, not a
        binary open/closed label. Preserve graded targets while closing and
        reject small same-state diffusion noise. In ``continuous_hysteresis``
        mode, reaching the closed region latches the grasp until an explicit
        release target persists for several executed commands. Filter state
        persists across refills, and only the prefix ManaEnv executes updates
        it.
        """
        out = actions.copy()
        prefix = min(self._execute_horizon, int(out.shape[0]))
        for index in (6, 13):
            accepted = float(
                np.clip(
                    self._gripper_accepted_target.get(index, float(current[index])),
                    self._gripper_closed_target,
                    self._gripper_open_target,
                )
            )
            closed_latched = self._gripper_closed_latched.get(index, False)
            release_pending_steps = self._gripper_release_pending_steps.get(index, 0)
            for step in range(prefix):
                target = float(
                    np.clip(
                        out[step, index],
                        self._gripper_closed_target,
                        self._gripper_open_target,
                    )
                )
                if (
                    self._gripper_filter_mode == "continuous_hysteresis"
                    and closed_latched
                ):
                    if target >= self._gripper_release_threshold:
                        release_pending_steps += 1
                    else:
                        release_pending_steps = 0
                    if release_pending_steps < self._gripper_release_stable_steps:
                        target = self._gripper_closed_target
                    else:
                        closed_latched = False
                        release_pending_steps = 0

                error = target - accepted
                if abs(error) <= self._gripper_deadband:
                    out[step, index] = accepted
                    if (
                        target <= self._gripper_close_threshold
                        and accepted <= self._gripper_close_threshold
                    ):
                        closed_latched = True
                    continue

                max_delta = (
                    self._gripper_max_open_delta_per_step
                    if error > 0
                    else self._gripper_max_close_delta_per_step
                )
                if abs(error) <= max_delta + 1e-6:
                    delta = error
                else:
                    delta = float(
                        np.clip(self._gripper_ema_alpha * error, -max_delta, max_delta)
                    )
                accepted = float(
                    np.clip(
                        accepted + delta,
                        self._gripper_closed_target,
                        self._gripper_open_target,
                    )
                )
                if abs(target - accepted) <= self._gripper_deadband:
                    accepted = target
                out[step, index] = accepted
                if (
                    target <= self._gripper_close_threshold
                    and accepted <= self._gripper_close_threshold
                ):
                    closed_latched = True

            self._gripper_accepted_target[index] = accepted
            self._gripper_closed_latched[index] = closed_latched
            if release_pending_steps:
                self._gripper_release_pending_steps[index] = release_pending_steps
            else:
                self._gripper_release_pending_steps.pop(index, None)
            self._gripper_open_state.pop(index, None)
            self._gripper_pending_state.pop(index, None)
            self._gripper_pending_steps.pop(index, None)
        return out

    def _slew_gripper_target(self, current: float, target: float) -> float:
        delta = float(
            np.clip(
                target - current,
                -self._gripper_max_delta_per_step,
                self._gripper_max_delta_per_step,
            )
        )
        return float(current + delta)

    def _reset_gripper_debounce_state(self) -> None:
        self._gripper_open_state: dict[int, bool] = {}
        self._gripper_accepted_target: dict[int, float] = {}
        self._gripper_pending_state: dict[int, bool] = {}
        self._gripper_pending_steps: dict[int, int] = {}
        self._gripper_closed_latched: dict[int, bool] = {}
        self._gripper_release_pending_steps: dict[int, int] = {}

    @staticmethod
    def _current_joint_state(obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("state", {})
        parts = []
        for field in ("follow1_pos", "follow2_pos"):
            if field not in state:
                raise KeyError(
                    f"DreamZero X2Robot observation is missing state field {field!r}"
                )
            value = np.asarray(state[field], dtype=np.float32).reshape(-1)
            if value.shape != (7,):
                raise ValueError(
                    f"DreamZero X2Robot state {field!r} must be 7-D, got {value.shape}"
                )
            parts.append(value)
        current = np.concatenate(parts)
        if not np.isfinite(current).all():
            raise ValueError("DreamZero X2Robot state contains NaN/Inf")
        return current
