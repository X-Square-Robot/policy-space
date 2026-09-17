"""In-process FastWAM X2Robot joint adapter for Policy Space v2."""

from __future__ import annotations

import inspect
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from policy_space.base import PolicyModel
from policy_space.paths import external_source_root

LOGGER = logging.getLogger(__name__)
PHYSICAL_ACTION_DIM = 14
WIRE_ACTION_DIM = 26
ACTION_HORIZON = 32
CAMERA_KEYS = ("camera_front", "camera_left", "camera_right")
STATE_KEYS = ("follow1_pos", "follow2_pos")
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _env_or(cfg: dict[str, Any], env_name: str, cfg_key: str, default: Any) -> Any:
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip():
        return raw
    if cfg.get(cfg_key) not in (None, ""):
        return cfg[cfg_key]
    return default


def _resolve_file(value: Any, *, name: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _model_dtype(torch: Any, value: Any) -> Any:
    key = str(value or "bf16").strip().lower()
    choices = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if key not in choices:
        raise ValueError(
            f"mixed_precision must be one of {sorted(choices)}, got {value!r}"
        )
    return choices[key]


def _validate_rgb(value: Any, *, camera: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"FastWAM camera {camera!r} must be a numpy.ndarray, "
            f"got {type(value).__name__}"
        )
    if value.dtype != np.uint8:
        raise TypeError(
            f"FastWAM camera {camera!r} must have dtype uint8, got {value.dtype}"
        )
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or value.shape[0] < 1
        or value.shape[1] < 1
    ):
        raise ValueError(
            f"FastWAM camera {camera!r} must be non-empty HWC RGB, "
            f"got {value.shape!r}"
        )
    return np.ascontiguousarray(value)


def _resize_rgb(value: np.ndarray, *, width: int, height: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(value, mode="RGB")
    return np.asarray(
        image.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8
    )


def pack_x2real_action(raw_actions: Any) -> np.ndarray:
    """Validate physical H32x14 and pack the ArtiXon Arm-6A H32x26 wire layout."""
    try:
        physical = np.asarray(raw_actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("FastWAM actions must be a numeric array") from exc
    expected = (ACTION_HORIZON, PHYSICAL_ACTION_DIM)
    if physical.shape != expected:
        raise ValueError(f"FastWAM output must be {expected}, got {physical.shape}")
    if not np.isfinite(physical).all():
        raise ValueError("FastWAM output contains NaN/Inf")
    packed = np.zeros((ACTION_HORIZON, WIRE_ACTION_DIM), dtype=np.float32)
    packed[:, :PHYSICAL_ACTION_DIM] = physical
    return packed


class Model(PolicyModel):
    """Load one full joint-DiT checkpoint and emit absolute H32 chunks."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self._fastwam_root = external_source_root(
            cfg,
            env_name="FASTWAM_ROOT",
            cfg_key="fastwam_root",
            project="FastWAM",
        )
        for path in (self._fastwam_root, self._fastwam_root / "src"):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

        training_config = _resolve_file(
            _env_or(
                cfg,
                "FASTWAM_TRAINING_CONFIG_PATH",
                "training_config_path",
                "",
            ),
            name="training_config_path",
        )
        checkpoint = _resolve_file(
            _env_or(cfg, "FASTWAM_CHECKPOINT_PATH", "checkpoint_path", ""),
            name="checkpoint_path",
        )
        dataset_stats = _resolve_file(
            _env_or(
                cfg,
                "FASTWAM_DATASET_STATS_PATH",
                "dataset_stats_path",
                "",
            ),
            name="dataset_stats_path",
        )

        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from fastwam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )

        self._torch = torch
        device_name = str(cfg.get("device", "cuda:0"))
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("FastWAM Policy Space requires CUDA")
        self._device = torch.device(device_name)
        self._dtype = _model_dtype(torch, cfg.get("mixed_precision", "bf16"))

        saved_cfg = OmegaConf.load(training_config)
        model_cfg = OmegaConf.create(
            OmegaConf.to_container(saved_cfg.model, resolve=True)
        )
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        self._model = instantiate(
            model_cfg, model_dtype=self._dtype, device=str(self._device)
        )
        self._model.load_checkpoint(str(checkpoint))
        self._model = self._model.to(self._device).eval()

        self._processor = instantiate(saved_cfg.data.train.processor).eval()
        stats = load_dataset_stats_from_json(str(dataset_stats))
        self._processor.set_normalizer_from_stats(stats)

        self._action_horizon = int(cfg.get("action_horizon", ACTION_HORIZON))
        if self._action_horizon != ACTION_HORIZON:
            raise ValueError("The verified FastWAM checkpoint requires H32")
        self._num_video_frames = int(cfg.get("num_video_frames", 9))
        self._num_inference_steps = int(cfg.get("num_inference_steps", 10))
        self._seed = int(cfg.get("seed", 42))
        self._rand_device = str(cfg.get("rand_device", "cpu"))
        self._text_cfg_scale = float(cfg.get("text_cfg_scale", 1.0))
        self._negative_prompt = str(cfg.get("negative_prompt", ""))
        self._tiled = bool(cfg.get("tiled", False))
        self._checkpoint = str(checkpoint)
        self._dataset_stats = str(dataset_stats)
        self._latest_obs: dict[str, Any] | None = None
        self._episode_id = ""
        self._lock = threading.RLock()

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._latest_obs = None
            self._episode_id = str((episode_info or {}).get("episode_id") or "")

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict):
            raise TypeError("FastWAM observation must be a mapping")
        images = obs.get("images")
        state = obs.get("state")
        if not isinstance(images, dict) or not isinstance(state, dict):
            raise TypeError("FastWAM observation requires images and state mappings")

        validated_images: dict[str, np.ndarray] = {}
        for camera in CAMERA_KEYS:
            if camera not in images:
                raise KeyError(f"FastWAM observation is missing camera {camera!r}")
            validated_images[camera] = _validate_rgb(
                images[camera], camera=camera
            ).copy()

        validated_state: dict[str, np.ndarray] = {}
        for key in STATE_KEYS:
            if key not in state:
                raise KeyError(f"FastWAM observation is missing state field {key!r}")
            try:
                value = np.asarray(state[key], dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"FastWAM state {key!r} must be numeric") from exc
            if value.shape != (7,):
                raise ValueError(
                    f"FastWAM state {key!r} must have shape (7,), got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"FastWAM state {key!r} contains NaN/Inf")
            validated_state[key] = value.copy()

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("FastWAM observation requires the current instruction")
        with self._lock:
            self._latest_obs = {
                "images": validated_images,
                "state": validated_state,
                "instruction": instruction.strip(),
                "episode_id": str(obs.get("episode_id") or self._episode_id),
                "step": int(obs.get("step") or 0),
            }

    def infer_actions(self) -> np.ndarray:
        with self._lock:
            if self._latest_obs is None:
                raise RuntimeError(
                    "No FastWAM observation buffered. Call ingest_observation first."
                )
            obs = self._latest_obs
            kwargs: dict[str, Any] = {
                "prompt": DEFAULT_PROMPT.format(task=obs["instruction"]),
                "input_image": self._image_tensor(obs),
                "action_horizon": self._action_horizon,
                "proprio": self._normalized_state(obs),
                "negative_prompt": self._negative_prompt,
                "text_cfg_scale": self._text_cfg_scale,
                "num_inference_steps": self._num_inference_steps,
                "seed": self._seed,
                "rand_device": self._rand_device,
                "tiled": self._tiled,
            }
            if (
                "num_video_frames"
                in inspect.signature(self._model.infer_action).parameters
            ):
                kwargs["num_video_frames"] = self._num_video_frames
            started = time.perf_counter()
            with self._torch.inference_mode():
                result = self._model.infer_action(**kwargs)
            if not isinstance(result, dict) or "action" not in result:
                raise RuntimeError("FastWAM response is missing 'action'")
            action14 = self._denormalized_action(result["action"])
            action26 = pack_x2real_action(action14)
            LOGGER.info(
                "FastWAM inference episode=%s step=%s shape=%s elapsed=%.3fs",
                obs["episode_id"],
                obs["step"],
                action26.shape,
                time.perf_counter() - started,
            )
            return action26

    def finalize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        del episode_info
        with self._lock:
            self._latest_obs = None
            self._episode_id = ""

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "fastwam_runtime": "joint_full_dit",
            "fastwam_checkpoint": self._checkpoint,
            "fastwam_dataset_stats": self._dataset_stats,
            "fastwam_num_video_frames": self._num_video_frames,
            "fastwam_num_inference_steps": self._num_inference_steps,
            "fastwam_model_output_shape": [ACTION_HORIZON, PHYSICAL_ACTION_DIM],
            "fastwam_wire_output_shape": [ACTION_HORIZON, WIRE_ACTION_DIM],
        }

    def close(self) -> None:
        self._latest_obs = None

    def _image_tensor(self, obs: dict[str, Any]) -> Any:
        images = obs["images"]
        # Matches RobotVideoDataset(concat_multi_camera="robotwin") exactly.
        front = _resize_rgb(images["camera_front"], width=320, height=256)
        left = _resize_rgb(images["camera_left"], width=160, height=128)
        right = _resize_rgb(images["camera_right"], width=160, height=128)
        mosaic = np.concatenate([front, np.concatenate([left, right], axis=1)], axis=0)
        tensor = self._torch.from_numpy(mosaic.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=self._dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def _normalized_state(self, obs: dict[str, Any]) -> Any:
        state = np.concatenate(
            [obs["state"][key] for key in STATE_KEYS], axis=0
        ).astype(np.float32, copy=False)
        key = self._processor.shape_meta["state"][0]["key"]
        batch = {"state": {key: self._torch.from_numpy(state).unsqueeze(0)}}
        batch = self._processor.action_state_transform(batch)
        batch = self._processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalized_action(self, value: Any) -> np.ndarray:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(
                "FastWAM action must be [B,T,D] or [T,D], " f"got {tuple(value.shape)}"
            )
        key = self._processor.shape_meta["action"][0]["key"]
        normalizer = self._processor.normalizer.normalizers["action"][key]
        return normalizer.backward(value.detach().float().cpu()).numpy()[0]
