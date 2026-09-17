#!/usr/bin/env python3
"""Strict Cosmos 3 inference server for the X2Robot 14-D EE contract.

This module extends the upstream RoboLab service without changing its DROID
code path.  It requires an X2Robot-post-trained checkpoint; the public DROID
checkpoint is deliberately rejected by contract/metadata rather than being
silently reinterpreted as a bimanual policy.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
from cosmos_framework.data.generator.action.action_processing import (
    ActionAffineNormalization,
    load_action_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.pose_utils import convert_rotation
from cosmos_framework.inference.common.args import tyro_cli
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _build_data_batch_from_sample,
    _ensure_rgb_uint8_image,
    _load_checkpoint_metadata,
    _load_openpi_websocket_policy_server,
    _resize_rgb_uint8,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.utils import log
from policy import (
    ACTION_DIM,
    ACTION_SPACE,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ROTATION_COMPOSITION,
    ROTATION_FORMAT,
    STATE_SPACE,
    TRANSLATION_RELATIVE_FRAME,
    WIRE_POSE_REFERENCE,
)

_X2_VIEW_DESCRIPTION = (
    "The top row is the X2Robot head camera. The bottom row contains the left "
    "and right wrist-mounted cameras, respectively."
)


class X2RobotServerArgs(RobolabServerArgs):
    """Upstream server arguments plus strict X2Robot contract fields."""

    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint_path: str
    """Required X2Robot-post-trained checkpoint; there is no DROID default on this server."""
    port: int = 8002
    domain_name: str = "x2robot"
    action_space: Literal["x2_bimanual_ee"] = "x2_bimanual_ee"
    # Kept as an internal upstream field. X2Robot sets it from
    # ``model_action_dim`` before constructing RobolabPolicyService.
    action_dim: int | None = None
    conditioning_fps: float | None = 15.0
    action_chunk_size: int | None = 32
    history_length: int = 1
    use_state: bool = True

    model_action_dim: Literal[14, 20]
    """Frozen post-training contract: EE Euler14 or EE rotation-6D20."""
    domain_id: int
    """Domain embedding ID used during X2Robot post-training; no guessed default."""
    normalization_method: Literal["none", "quantile", "meanstd", "minmax"] = "quantile"
    """Action/state normalizer used during post-training."""
    normalization_stats_path: Path | None = None
    """JSON stats used for both state conditioning and generated action inverse transform."""
    normalization_stats_key: str = "global"
    """Stats block within the JSON file."""
    allow_transform_fallback: bool = False
    """Unsafe escape hatch. Default false: checkpoint must restore its training transform."""
    require_json_prompt: bool = True
    """Require the same structured JSON prompt formatter used by the training recipe."""
    checkpoint_contract_path: Path
    """Machine-checkable X2Robot representation/training contract JSON."""
    training_config_path: Path
    """Exact training config artifact whose SHA256 is frozen in the contract."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_x2_contract(args: X2RobotServerArgs) -> dict[str, Any]:
    path = args.checkpoint_contract_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"X2Robot checkpoint contract not found: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "action_space": ACTION_SPACE,
        "wire_action_dim": ACTION_DIM,
        "wire_pose_reference": WIRE_POSE_REFERENCE,
        "wire_rotation": ROTATION_FORMAT,
        "translation_relative_frame": TRANSLATION_RELATIVE_FRAME,
        "rotation_composition": ROTATION_COMPOSITION,
        "model_action_dim": int(args.model_action_dim),
        "model_rotation": "euler_xyz" if args.model_action_dim == 14 else "rot6d",
        "domain_id": int(args.domain_id),
        "model_chunk_size": int(args.action_chunk_size or 32),
        "model_fps": float(args.conditioning_fps or 15.0),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "resolution": args.resolution,
        "normalization_method": args.normalization_method,
        "camera_layout": "head_top_left_wrist_bottom_left_right_wrist_bottom_right",
        "prompt_format": "json",
        "state_space": STATE_SPACE,
    }
    mismatches = {key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value}
    if mismatches:
        raise ValueError(f"incompatible Cosmos3 X2Robot checkpoint contract: {mismatches}")

    if args.allow_dcp_checkpoint:
        raise ValueError(
            "Cosmos3 X2Robot v1 requires a consolidated local checkpoint with checkpoint.json; "
            "direct DCP loading cannot prove the frozen training-transform identity"
        )
    checkpoint_metadata = _load_checkpoint_metadata(str(args.checkpoint_path))
    if checkpoint_metadata is None:
        raise ValueError(
            "X2Robot checkpoint must contain checkpoint.json so the server can verify the exact "
            "training config/experiment used to restore preprocessing"
        )
    metadata_expected = {
        "config_file": contract.get("checkpoint_config_file"),
        "experiment": contract.get("checkpoint_experiment"),
        "experiment_overrides": contract.get("checkpoint_experiment_overrides"),
    }
    metadata_actual = {
        "config_file": checkpoint_metadata.get("config_file"),
        "experiment": checkpoint_metadata.get("experiment", ""),
        "experiment_overrides": checkpoint_metadata.get("experiment_overrides", []),
    }
    if metadata_actual != metadata_expected:
        raise ValueError(
            "checkpoint.json training-transform identity does not match the X2 contract: "
            f"contract={metadata_expected}, checkpoint={metadata_actual}"
        )
    required_hashes = ["checkpoint_id", "training_config_sha256"]
    if args.normalization_method != "none":
        required_hashes.append("normalization_stats_sha256")
    for key in required_hashes:
        value = contract.get(key)
        if not isinstance(value, str) or not value or value.startswith("REPLACE_"):
            raise ValueError(f"contract.{key} must be filled with a real value")

    training_config_path = args.training_config_path.expanduser().resolve()
    if not training_config_path.is_file():
        raise FileNotFoundError(f"training config not found: {training_config_path}")
    actual_training_sha = _file_sha256(training_config_path)
    if actual_training_sha != contract["training_config_sha256"]:
        raise ValueError(
            "training config SHA256 mismatch: "
            f"contract={contract['training_config_sha256']} actual={actual_training_sha}"
        )
    if args.normalization_method == "none":
        if args.normalization_stats_path is not None:
            raise ValueError("Do not pass --normalization-stats-path when the frozen method is none")
        if contract.get("normalization_stats_sha256") not in (None, ""):
            raise ValueError("contract.normalization_stats_sha256 must be null/empty for method none")
        contract["_contract_sha256"] = _file_sha256(path)
        return contract
    if args.normalization_stats_path is None:
        raise ValueError("checkpoint contract requires --normalization-stats-path")
    stats_path = args.normalization_stats_path.expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"normalization stats not found: {stats_path}")
    actual_stats_sha = _file_sha256(stats_path)
    if actual_stats_sha != contract["normalization_stats_sha256"]:
        raise ValueError(
            "normalization stats SHA256 mismatch: "
            f"contract={contract['normalization_stats_sha256']} actual={actual_stats_sha}"
        )
    contract["_contract_sha256"] = _file_sha256(path)
    return contract


class X2RobotPolicyService(RobolabPolicyService):
    def __init__(self, args: X2RobotServerArgs) -> None:
        if args.action_dim not in (None, args.model_action_dim):
            raise ValueError(
                f"--action-dim={args.action_dim} conflicts with " f"--model-action-dim={args.model_action_dim}"
            )
        if not args.use_state or args.history_length != 1:
            raise ValueError("X2Robot v1 requires --use-state true and --history-length 1")
        if args.domain_id < 0:
            raise ValueError("--domain-id must be non-negative")
        self._checkpoint_contract = _load_x2_contract(args)
        args = args.model_copy(update={"action_dim": int(args.model_action_dim)})
        self.x2_args = args
        self._action_normalizer = self._build_action_normalizer(args)
        super().__init__(args)

    @staticmethod
    def _build_action_normalizer(args: X2RobotServerArgs) -> ActionAffineNormalization | None:
        if args.normalization_method == "none":
            if args.normalization_stats_path is not None:
                raise ValueError("Do not pass --normalization-stats-path with --normalization-method none")
            return None
        if args.normalization_stats_path is None:
            raise ValueError(
                "X2Robot inference requires --normalization-stats-path unless the checkpoint "
                "was explicitly trained with --normalization-method none"
            )
        path = args.normalization_stats_path.expanduser().resolve()
        stats_np = load_action_stats(str(path), stats_key=args.normalization_stats_key)
        stats = {key: torch.from_numpy(value).float() for key, value in stats_np.items()}
        normalizer = resolve_action_normalization(args.normalization_method, stats)
        for name, value in (("offset", normalizer.offset), ("scale", normalizer.scale)):
            if value.shape != (args.model_action_dim,):
                raise ValueError(
                    f"normalization {name} must have shape ({args.model_action_dim},), " f"got {value.shape}"
                )
        return normalizer

    def _build_transform(self, training_config: Any | None, args: X2RobotServerArgs):
        if training_config is None and not args.allow_transform_fallback:
            raise RuntimeError(
                "Could not restore the checkpoint training config. Refusing the default transform "
                "fallback because that can change prompt/action preprocessing. Export the X2Robot "
                "checkpoint with valid config metadata, or pass --allow-transform-fallback only "
                "for a diagnostic smoke test."
            )
        transform, inferred = super()._build_transform(training_config, args)
        if args.require_json_prompt and getattr(transform, "prompt_json_formatter", None) is None:
            raise RuntimeError(
                "X2Robot server requires the training JSON prompt formatter, but the restored "
                "transform has format_prompt_as_json=False. Fix checkpoint metadata/config; do "
                "not benchmark with a mismatched formatter."
            )
        return transform, inferred

    @staticmethod
    def _compose_views(obs: dict[str, Any]) -> np.ndarray:
        keys = (
            "observation/head_image",
            "observation/left_wrist_image",
            "observation/right_wrist_image",
        )
        missing = [key for key in keys if key not in obs]
        if missing:
            raise ValueError(f"X2Robot request is missing camera keys: {missing}")
        head = _ensure_rgb_uint8_image(obs[keys[0]], keys[0])
        left_raw = _ensure_rgb_uint8_image(obs[keys[1]], keys[1])
        right_raw = _ensure_rgb_uint8_image(obs[keys[2]], keys[2])
        half_h, half_w = head.shape[0] // 2, head.shape[1] // 2
        left = _resize_rgb_uint8(left_raw, (half_h, half_w))
        right = _resize_rgb_uint8(right_raw, (half_h, half_w))
        return np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)

    @staticmethod
    def _eef_state(obs: dict[str, Any], key: str) -> np.ndarray:
        value = np.asarray(obs.get(key), dtype=np.float32).reshape(-1)
        if value.shape != (7,):
            raise ValueError(f"{key!r} must be [x,y,z,roll,pitch,yaw,gripper], got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key!r} contains NaN/Inf")
        return value

    def _wire_to_model_action(self, wire: np.ndarray) -> np.ndarray:
        """Convert EE Euler14 wire/state rows to the frozen model contract."""
        wire = np.asarray(wire, dtype=np.float32)
        if wire.shape[-1] != ACTION_DIM:
            raise ValueError(f"EE wire must end in {ACTION_DIM} channels, got {wire.shape}")
        if self.x2_args.model_action_dim == 14:
            return wire
        arms = []
        for start in (0, 7):
            arm = wire[..., start : start + 7]
            rot6d = convert_rotation(arm[..., 3:6], "euler_xyz", "rot6d", normalize_matrix=True)
            arms.append(np.concatenate([arm[..., :3], rot6d, arm[..., 6:7]], axis=-1))
        return np.concatenate(arms, axis=-1).astype(np.float32, copy=False)

    def _model_to_wire_action(self, action: np.ndarray) -> np.ndarray:
        """Convert generated EE Euler14/rotation-6D20 rows to ManaEnv EE14."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1] != self.x2_args.model_action_dim:
            raise ValueError(
                f"model action must end in {self.x2_args.model_action_dim} channels, " f"got {action.shape}"
            )
        if self.x2_args.model_action_dim == 14:
            return action
        arms = []
        for start in (0, 10):
            arm = action[..., start : start + 10]
            euler = convert_rotation(arm[..., 3:9], "rot6d", "euler_xyz", normalize_matrix=True)
            arms.append(np.concatenate([arm[..., :3], euler, arm[..., 9:10]], axis=-1))
        return np.concatenate(arms, axis=-1).astype(np.float32, copy=False)

    def _build_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("X2Robot 'prompt' must be a non-empty string")

        image = self._compose_views(obs)
        image_h, image_w = self.cfg.image_height, self.cfg.image_width
        if image.shape[:2] != (image_h, image_w):
            image = _resize_rgb_uint8(image, (image_h, image_w))

        t_frames = self.cfg.action_chunk_size + 1
        video = torch.zeros((3, t_frames, image_h, image_w), dtype=torch.uint8)
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)

        left = self._eef_state(obs, "observation/left_eef")
        right = self._eef_state(obs, "observation/right_eef")
        current_wire = np.concatenate([left, right])
        current_model = self._wire_to_model_action(current_wire)
        action = torch.zeros(
            (self.cfg.action_chunk_size + 1, self.x2_args.model_action_dim),
            dtype=torch.float32,
        )
        action[0] = torch.from_numpy(current_model)

        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.cfg.conditioning_fps, dtype=torch.long),
            "mode": "policy",
            "domain_id": torch.tensor(self.x2_args.domain_id, dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": _X2_VIEW_DESCRIPTION,
        }
        return self._transform(sample, self.cfg.resolution, action_normalizer=self._action_normalizer)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        sample = self._build_sample(obs)
        data_batch = _build_data_batch_from_sample(sample)
        seed = self._next_seed()
        log.info(f"[x2robot-policy-server] prompt={data_batch['ai_caption'][0]!r} seed={seed}")

        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                )

        # ActionProcessor externalizes the 14 real channels and applies inverse
        # normalization because _build_sample attached the exact processing record.
        action = samples["action"][0]
        action = action[self.cfg.history_length :]
        model_action = action.detach().cpu().numpy().astype(np.float32, copy=False)
        expected_model = (self.cfg.action_chunk_size, self.x2_args.model_action_dim)
        if model_action.shape != expected_model or not np.isfinite(model_action).all():
            raise RuntimeError(
                f"Invalid generated X2Robot model action: " f"shape={model_action.shape}, expected={expected_model}"
            )
        action_np = self._model_to_wire_action(model_action)
        expected_wire = (self.cfg.action_chunk_size, ACTION_DIM)
        if action_np.shape != expected_wire or not np.isfinite(action_np).all():
            raise RuntimeError(
                f"Invalid converted X2Robot wire action: " f"shape={action_np.shape}, expected={expected_wire}"
            )

        outputs: dict[str, Any] = {"action": np.ascontiguousarray(action_np)}
        if self.cfg.decode_video:
            pred_vision_latent = samples["vision"][0]
            decoded = self.model.decode(pred_vision_latent)
            video = ((decoded[0].clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
            outputs["video"] = video.permute(1, 2, 3, 0).detach().cpu().numpy()
        return outputs


def _stats_sha256(args: X2RobotServerArgs) -> str | None:
    if args.normalization_stats_path is None:
        return None
    path = args.normalization_stats_path.expanduser().resolve()
    return _file_sha256(path)


def build_metadata(args: X2RobotServerArgs, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "embodiment": "ArtiXon Arm-6A",
        "action_space": ACTION_SPACE,
        "action_dim": ACTION_DIM,
        "model_action_dim": int(args.model_action_dim),
        "model_rotation": "euler_xyz" if args.model_action_dim == 14 else "rot6d",
        "model_chunk_size": int(args.action_chunk_size or 32),
        "model_fps": float(args.conditioning_fps or 15.0),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "resolution": args.resolution,
        "wire_pose_reference": WIRE_POSE_REFERENCE,
        "rotation": ROTATION_FORMAT,
        "translation_relative_frame": TRANSLATION_RELATIVE_FRAME,
        "rotation_composition": ROTATION_COMPOSITION,
        "state_space": STATE_SPACE,
        "domain_id": int(args.domain_id),
        "normalization_method": args.normalization_method,
        "normalization_stats_sha256": _stats_sha256(args),
        "training_config_sha256": contract["training_config_sha256"],
        "checkpoint_contract_sha256": contract["_contract_sha256"],
        "checkpoint_id": contract["checkpoint_id"],
        "checkpoint_config_file": contract["checkpoint_config_file"],
        "checkpoint_experiment": contract["checkpoint_experiment"],
        "checkpoint_experiment_overrides": contract["checkpoint_experiment_overrides"],
        "camera_layout": contract["camera_layout"],
        "checkpoint_path": str(args.checkpoint_path),
        "backend": "cosmos3",
    }


def serve_x2robot(args: X2RobotServerArgs) -> None:
    hostname = socket.gethostname()
    log.info(f"[x2robot-policy-server] starting host={hostname} bind={args.host}:{args.port}")
    service = X2RobotPolicyService(args)
    metadata = build_metadata(args, service._checkpoint_contract)
    local_ip = get_local_ip()
    log.info(f"[x2robot-policy-server] websocket ws://{local_ip}:{args.port}/ metadata={metadata}")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata=metadata).serve_forever()


def main() -> None:
    args = tyro_cli(X2RobotServerArgs, description=__doc__)
    serve_x2robot(args)


if __name__ == "__main__":
    main()
