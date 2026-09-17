"""WallX PolicyModel loaded directly by the Policy Space service.

WallXPolicy is instantiated in ``__init__`` and its inference method is called
directly from ``infer_actions``.

Requires ``robot_type=desktop`` on the policy: that mode prepends the
request's own state as row 0 of the trajectory, which is what the ManaEnv-side
``first_frame: prediction_only`` expects to skip.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from policy_space.base import PolicyModel
from policy_space.paths import external_source_root

logger = logging.getLogger(__name__)

_VIEWS_KEY = "views"
_STATE_KEY = "state"
_INSTRUCTION_KEY = "instruction"

_WIRE_ACTION_DIM = 14
_RTC_PROTOCOL = "wall_x_rtc.v1"

_RTC_REQUEST_KEYS = (
    "use_rtc",
    "prev_action_chunk",
    "inference_delay",
    "num_flow_steps",
    "num_steps",
    "rtc_mode",
)

_ROBOT_TYPE_TO_EMBODIMENT = {
    "desktop": "X2 ex001_desktop",
    "turtle": "X2 turtle_car",
    "ex001": "X2 EX001",
}


def _model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the effective knob dict from a deploy.yaml payload."""
    nested = (
        ((cfg.get("model") or {}).get("cfg") or {})
        if isinstance(cfg.get("model"), dict)
        else {}
    )
    merged = dict(cfg)
    merged.update(nested)
    return merged


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip():
        return raw
    if cfg.get(cfg_key) is not None:
        return cfg[cfg_key]
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _encode_image_b64(
    rgb: np.ndarray,
    *,
    wire_encoding: str = "lossless_png",
    jpeg_quality: int = 95,
) -> str | None:
    """Encode an RGB image for WallX's base64 image wire field.

    ``lossless_png`` is the production default so the tensor entering the
    inference preprocessor sees exactly the same uint8 pixels as the caller.
    ``jpeg95`` remains available only for backwards-compatible comparisons.
    WallX decodes both formats with ``cv2.imdecode``.
    """
    import cv2

    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    encoding = str(wire_encoding).strip().lower()
    if encoding == "lossless_png":
        success, encoded = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif encoding == "jpeg95":
        success, encoded = cv2.imencode(
            ".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        )
    else:
        raise ValueError(
            "image_wire_encoding must be 'lossless_png' or 'jpeg95', "
            f"got {wire_encoding!r}"
        )
    if not success:
        return None
    return base64.b64encode(encoded).decode("utf-8")


def _ensure_wallx_imports(wallx_root: Path) -> None:
    """Add wall_x and harrix to sys.path if not already importable."""
    wallx_path = str(wallx_root)
    harrix_path = os.path.join(wallx_path, "third_party", "harrix", "python")
    for p in (wallx_path, harrix_path):
        if p not in sys.path:
            sys.path.insert(0, p)


class Model(PolicyModel):
    """WallX bimanual policy loaded directly in the service process."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        knobs = _model_cfg(cfg)
        self._wallx_root = external_source_root(
            knobs,
            env_name="WALLX_ROOT",
            cfg_key="wallx_root",
            project="Wall-X",
        )

        self.instruction = str(_env_or(knobs, "WALLX_INSTRUCTION", "instruction", ""))
        self.image_wire_encoding = (
            str(
                _env_or(
                    knobs,
                    "WALLX_IMAGE_WIRE_ENCODING",
                    "image_wire_encoding",
                    "lossless_png",
                )
            )
            .strip()
            .lower()
        )
        if self.image_wire_encoding not in {"lossless_png", "jpeg95"}:
            raise ValueError(
                "image_wire_encoding must be 'lossless_png' or 'jpeg95', "
                f"got {self.image_wire_encoding!r}"
            )
        self.jpeg_quality = int(knobs.get("jpeg_quality", 95))
        self.response_left_key = str(knobs.get("response_left_key", "follow1_pos"))
        self.response_right_key = str(knobs.get("response_right_key", "follow2_pos"))
        self.control_rate_hz = int(
            _env_or(knobs, "WALLX_CONTROL_RATE_HZ", "control_rate_hz", 20)
        )
        if self.control_rate_hz < 1:
            raise ValueError(
                f"control_rate_hz must be positive, got {self.control_rate_hz}"
            )

        checkpoint_path = str(
            _env_or(knobs, "WALLX_CHECKPOINT_PATH", "checkpoint_path", "")
        )
        if not checkpoint_path:
            raise ValueError(
                "WallX checkpoint_path is required. Set model.cfg.checkpoint_path "
                "in deploy.yaml or WALLX_CHECKPOINT_PATH env var."
            )

        robot_type = str(_env_or(knobs, "WALLX_ROBOT_TYPE", "robot_type", "desktop"))
        robot_id = str(_env_or(knobs, "WALLX_ROBOT_ID", "robot_id", "10158"))
        norm_key = str(_env_or(knobs, "WALLX_NORM_KEY", "norm_key", "ex_normal"))
        action_dim = int(_env_or(knobs, "WALLX_ACTION_DIM", "action_dim", 26))
        action_horizon = int(
            _env_or(knobs, "WALLX_ACTION_HORIZON", "action_horizon", 32)
        )
        num_inference_timesteps = int(
            _env_or(
                knobs, "WALLX_NUM_INFERENCE_TIMESTEPS", "num_inference_timesteps", 10
            )
        )
        model_device = str(_env_or(knobs, "WALLX_MODEL_DEVICE", "model_device", "cuda"))
        image_passing_mode = str(knobs.get("image_passing_mode", "base64"))
        robot_action_start_ratio = float(knobs.get("robot_action_start_ratio", 0.0))
        robot_action_end_ratio = float(knobs.get("robot_action_end_ratio", 1.0))
        robot_action_interpolate_multiplier = int(
            knobs.get("robot_action_interpolate_multiplier", 70)
        )

        cfg_scale = float(knobs.get("cfg_scale", 1.0))
        cfg_uncond_instruction = knobs.get("cfg_uncond_instruction")
        cfg_uncond_metadata = knobs.get("cfg_uncond_metadata")
        cfg_drop = knobs.get("cfg_drop")

        _ensure_wallx_imports(self._wallx_root)
        self._load_policy(
            checkpoint_path=checkpoint_path,
            robot_type=robot_type,
            robot_id=robot_id,
            norm_key=norm_key,
            action_dim=action_dim,
            action_horizon=action_horizon,
            num_inference_timesteps=num_inference_timesteps,
            model_device=model_device,
            image_passing_mode=image_passing_mode,
            robot_action_start_ratio=robot_action_start_ratio,
            robot_action_end_ratio=robot_action_end_ratio,
            robot_action_interpolate_multiplier=robot_action_interpolate_multiplier,
            cfg_scale=cfg_scale,
            cfg_uncond_instruction=cfg_uncond_instruction,
            cfg_uncond_metadata=cfg_uncond_metadata,
            cfg_drop=cfg_drop,
        )

        self._latest_obs: dict[str, Any] | None = None
        logger.info(
            "[WallX] model loaded on %s from %s (image wire: %s)",
            model_device,
            checkpoint_path,
            self.image_wire_encoding,
        )

    def _load_policy(
        self,
        checkpoint_path: str,
        robot_type: str,
        robot_id: str,
        norm_key: str,
        action_dim: int,
        action_horizon: int,
        num_inference_timesteps: int,
        model_device: str,
        image_passing_mode: str,
        robot_action_start_ratio: float,
        robot_action_end_ratio: float,
        robot_action_interpolate_multiplier: int,
        cfg_scale: float,
        cfg_uncond_instruction: str | None,
        cfg_uncond_metadata: str | None,
        cfg_drop: str | None,
    ) -> None:
        """Construct InferConfig + WallXPolicy (mirrors launch_wallx_local.py)."""
        import time

        print(f"[WallX] checkpoint: {checkpoint_path}")
        print(
            f"[WallX] device: {model_device}, robot: {robot_type}/{robot_id}, norm: {norm_key}"
        )
        print(
            f"[WallX] action_dim={action_dim}, horizon={action_horizon}, timesteps={num_inference_timesteps}"
        )
        if abs(cfg_scale - 1.0) > 1e-6:
            print(f"[WallX] CFG enabled: scale={cfg_scale}")

        _cfg_active = abs(cfg_scale - 1.0) > 1e-6
        if _cfg_active:
            cur_bs = int(os.environ.get("WALLX_MAX_BATCH_SIZE", "1"))
            if cur_bs < 2:
                os.environ["WALLX_MAX_BATCH_SIZE"] = "2"
            cur_graph_bs = int(os.environ.get("CUDA_GRAPH_MAX_BS", "128"))
            if cur_graph_bs < 2:
                os.environ["CUDA_GRAPH_MAX_BS"] = "2"

        robot_type_map = os.path.join(
            str(self._wallx_root),
            "wall_x",
            "infer",
            "data",
            "robot_type_map_inference.json",
        )
        if "X2ROBOT_ID_CONFIG_PATH" not in os.environ:
            os.environ["X2ROBOT_ID_CONFIG_PATH"] = robot_type_map

        print("[WallX] importing wall_x modules...")
        t0 = time.time()
        import x2robot_dataset.constant as _x2c

        if robot_id not in _x2c._X2ROBOT_ID_TO_NAME:
            embodiment = _ROBOT_TYPE_TO_EMBODIMENT.get(robot_type, f"X2 {robot_type}")
            _x2c._X2ROBOT_ID_TO_NAME[robot_id] = embodiment
            if norm_key in ("x2_normal", "ex_normal"):
                _x2c._ROBOT_ID_TO_DATASET_TYPE[robot_id] = norm_key
            logger.info("Patched robot_id %s -> %s", robot_id, embodiment)

        from wall_x.infer.infer_config import InferConfig
        from wall_x.serving.policy.wall_x_policy import (
            WallXPolicy,
            parse_cfg_drop_targets,
        )

        print(f"[WallX] imports done ({time.time() - t0:.1f}s)")

        config = InferConfig(
            checkpoint_path=checkpoint_path,
            robot_type=robot_type,
            robot_id=robot_id,
            norm_key=norm_key,
            action_dim=action_dim,
            action_horizon=action_horizon,
            robot_action_start_ratio=robot_action_start_ratio,
            robot_action_end_ratio=robot_action_end_ratio,
            robot_action_interpolate_multiplier=robot_action_interpolate_multiplier,
            model_device=model_device,
            num_inference_timesteps=num_inference_timesteps,
        )

        cfg_drop_targets = parse_cfg_drop_targets(cfg_drop)

        print("[WallX] loading model weights (this may take a minute)...")
        t1 = time.time()
        self._policy = WallXPolicy(
            config=config,
            image_passing_mode=image_passing_mode,
            serialize_actions=True,
            cfg_scale=cfg_scale,
            cfg_uncond_instruction=cfg_uncond_instruction,
            cfg_uncond_metadata=cfg_uncond_metadata,
            cfg_drop_targets=cfg_drop_targets,
        )
        print(f"[WallX] model ready! weights loaded in {time.time() - t1:.1f}s")

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        if episode_info:
            instr = episode_info.get("instruction")
            if isinstance(instr, str) and instr.strip():
                self.instruction = instr
        self._latest_obs = None
        self._policy.reset()

    def runtime_metadata(self) -> dict[str, Any]:
        """Advertise RTC only when the loaded checkpoint was trained for it."""
        config = self._policy.config
        train_config = getattr(config, "train_config", {}) or {}
        rtc_train = (
            train_config.get("rtc_train", {}) if isinstance(train_config, dict) else {}
        )
        if not isinstance(rtc_train, dict) or rtc_train.get("enabled") is not True:
            return {}

        if _env_flag("ENABLE_CUDA_GRAPH", default=False) or _env_flag(
            "ENABLE_EXPERIMENTAL_INFERENCE_ENGINE", default=False
        ):
            raise RuntimeError(
                "This checkpoint enables rtc_train, but WallX RTC requires "
                "ENABLE_CUDA_GRAPH=False and ENABLE_EXPERIMENTAL_INFERENCE_ENGINE=False"
            )

        max_delay = rtc_train.get("max_delay")
        if (
            isinstance(max_delay, bool)
            or not isinstance(max_delay, int)
            or max_delay < 1
        ):
            raise RuntimeError(
                f"rtc_train.max_delay must be a positive integer, got {max_delay!r}"
            )
        model_horizon = int(config.action_horizon)
        num_flow_steps = int(config.num_inference_timesteps)
        if max_delay > model_horizon:
            raise RuntimeError(
                f"rtc_train.max_delay={max_delay} exceeds model action_horizon={model_horizon}"
            )
        if num_flow_steps < 1:
            raise RuntimeError(
                f"num_inference_timesteps must be positive, got {num_flow_steps}"
            )

        return {
            "capabilities": {
                "wall_x_rtc": {
                    "enabled": True,
                    "protocol": _RTC_PROTOCOL,
                    "model_action_horizon": model_horizon,
                    "max_delay": max_delay,
                    "num_flow_steps": num_flow_steps,
                    "control_rate_hz": self.control_rate_hz,
                }
            }
        }

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._latest_obs = obs

    def infer_actions(self) -> np.ndarray:
        if self._latest_obs is None:
            raise RuntimeError(
                "No observation buffered. Call ingest_observation first."
            )
        envelope = self._build_envelope(self._latest_obs)
        try:
            response = self._policy.infer(envelope)
        except Exception as exc:
            # Preserve the input-shape context in the wire error; this is vital for
            # diagnosing remote clients whose payload schema may drift.
            view_keys = sorted(map(str, envelope.get("views", {}).keys()))
            cam_names = getattr(getattr(self._policy, "model_wrapper", None), "cam_names", None)
            raise RuntimeError(
                f"{exc}; envelope_views={view_keys}; policy_cam_names={cam_names}"
            ) from exc
        if not hasattr(self, "_infer_count"):
            self._infer_count = 0
        self._infer_count += 1
        if self._infer_count <= 3 or self._infer_count % 100 == 0:
            resp_summary = (
                {k: type(v).__name__ for k, v in response.items()}
                if isinstance(response, dict)
                else type(response).__name__
            )
            print(f"[WallX] infer #{self._infer_count} response keys: {resp_summary}")
        return self._parse_response(response)

    def close(self) -> None:
        pass

    def _build_envelope(self, obs: dict[str, Any]) -> dict[str, Any]:
        # Policy Space v2 normally uses `images`; accept `views` as well for
        # older multiplexed clients and make an empty visual payload explicit.
        images = obs.get("images") or obs.get("views") or {}
        views: dict[str, Any] = {}
        for name, img in images.items():
            if img is None:
                continue
            encoded = _encode_image_b64(
                np.asarray(img),
                wire_encoding=self.image_wire_encoding,
                jpeg_quality=self.jpeg_quality,
            )
            if encoded is not None:
                views[name] = encoded

        state_in = obs.get("state") or {}
        state: dict[str, Any] = {}
        for name, val in state_in.items():
            if val is None:
                continue
            state[name] = np.asarray(val, dtype=np.float32)

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = self.instruction

        if not views:
            raise RuntimeError(f"WallX received no visual inputs (obs keys={sorted(map(str, obs.keys()))})")
        envelope = {_VIEWS_KEY: views, _STATE_KEY: state, _INSTRUCTION_KEY: instruction}
        # RTC is a per-request option owned by the ManaEnv client. Keep the
        # Policy Space wrapper stateless and forward only the explicit
        # WallX RTC protocol fields; absent keys preserve vanilla flow behavior.
        for key in _RTC_REQUEST_KEYS:
            if key in obs:
                envelope[key] = obs[key]
        return envelope

    def _parse_response(self, response: Any) -> np.ndarray:
        """Concat dual-arm wire trajectories into (H, 14)."""
        if isinstance(response, str):
            raise RuntimeError(f"WallX error: {response!r}")
        if not isinstance(response, dict):
            raise RuntimeError(
                f"WallX returned {type(response).__name__}, expected dict"
            )
        if "error" in response:
            raise RuntimeError(f"WallX error: {response['error']!r}")

        left = response.get(self.response_left_key)
        right = response.get(self.response_right_key)
        if left is not None and right is not None:
            left_arr = np.asarray(left, dtype=np.float32)
            right_arr = np.asarray(right, dtype=np.float32)
            if left_arr.size == 0 or right_arr.size == 0:
                print(
                    f"[WallX] WARNING: {self.response_left_key} or {self.response_right_key} is empty, falling through"
                )
            else:
                if left_arr.ndim == 1:
                    left_arr = left_arr.reshape(1, -1)
                if right_arr.ndim == 1:
                    right_arr = right_arr.reshape(1, -1)
                horizon = min(left_arr.shape[0], right_arr.shape[0])
                chunk = np.concatenate(
                    [left_arr[:horizon], right_arr[:horizon]], axis=-1
                ).astype(np.float32)
                return self._validate_chunk(chunk)

        for key in ("action_pred", "actions", "action", "trajectory"):
            val = response.get(key)
            if val is not None:
                arr = np.asarray(val, dtype=np.float32)
                if arr.size == 0:
                    continue
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return self._validate_chunk(arr)

        raise RuntimeError(
            f"WallX response has no usable action keys; got: {sorted(map(str, response.keys()))}"
        )

    @staticmethod
    def _validate_chunk(chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim != 2 or chunk.shape[0] < 1:
            raise RuntimeError(
                f"WallX chunk must be (H, D) with H>=1, got shape {chunk.shape}"
            )
        if chunk.shape[1] != _WIRE_ACTION_DIM:
            raise RuntimeError(
                f"WallX chunk width must be {_WIRE_ACTION_DIM}, got {chunk.shape[1]}"
            )
        if not np.isfinite(chunk).all():
            raise RuntimeError("WallX returned NaN/Inf in the action chunk")
        return np.ascontiguousarray(chunk, dtype=np.float32)
