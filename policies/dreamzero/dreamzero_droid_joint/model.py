"""DreamZero DROID single-arm model loaded by Policy Space.

Loads GrootSimPolicy (oxe_droid embodiment) directly in-process and
performs causal AR inference without a separate GPU server.

Ground-truth frame injection (Algorithm 2):
  - Request 0: single frame per camera -> KV cache init + prefill
  - Request N>0: 4 frames per camera at history offsets -> GT injection
    into KV cache before denoising next action chunk
"""

from __future__ import annotations

import collections
import concurrent.futures
import functools
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

_DEFAULT_RELATIVE_OFFSETS = [-23, -16, -8, 0]
_DEFAULT_HISTORY_WINDOW = 24
_DEFAULT_CAMERA_MAP = {
    "exterior_1": "video.exterior_image_1_left",
    "exterior_2": "video.exterior_image_2_left",
    "wrist": "video.wrist_image_left",
}
_STATE_SPEC = (
    ("state.joint_position", "joint_position", 7),
    ("state.gripper_position", "gripper_position", 1),
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
    """Strip ``mode="reduce-overhead"`` from every ``torch.compile`` call.

    That mode's only effect is enabling CUDA Graphs
    (``list_mode_options("reduce-overhead")`` -> ``{"triton.cudagraphs": True}``),
    and dreamzero applies it to the text encoder, image encoder, VAE and
    scheduler. CUDA Graphs cannot cope with this policy's input shapes: request 0
    sends one frame per camera for prefill, later requests send four for
    ground-truth injection, so the graph tree has to rewind and
    ``_cuda_setCheckpointPoolState`` dies with
    ``Expected curr_block->next == nullptr``.

    Because the mode is passed explicitly at each call site, the
    ``TORCHINDUCTOR_CUDAGRAPHS`` env var and ``config.triton.cudagraphs`` are both
    overridden — the only lever is the decorator itself. Compilation still
    happens; we lose only the kernel-launch overhead saving, which is noise
    against a ~30s inference.

    Set ``DREAMZERO_CUDAGRAPHS=1`` to keep the upstream behaviour.
    """
    if os.environ.get("DREAMZERO_CUDAGRAPHS", "0") == "1":
        return

    import torch

    if getattr(torch.compile, "_dreamzero_no_cudagraphs", False):
        return

    original = torch.compile

    @functools.wraps(original)
    def compile_without_cudagraphs(*args: Any, **kwargs: Any) -> Any:
        # mode= and options= are mutually exclusive, so drop mode outright rather
        # than rewriting it to options={"triton.cudagraphs": False}. Since
        # reduce-overhead sets nothing but that one flag, dropping it is exact.
        if kwargs.get("mode") == "reduce-overhead":
            kwargs.pop("mode")
        return original(*args, **kwargs)

    compile_without_cudagraphs._dreamzero_no_cudagraphs = True  # type: ignore[attr-defined]
    torch.compile = compile_without_cudagraphs  # type: ignore[assignment]
    print("[DreamZero DROID] CUDA Graphs disabled (shape-varying input)")


def _configure_torch_dynamo() -> None:
    try:
        import torch._dynamo.config as dynamo_cfg
    except Exception:
        return
    for key, default in [
        ("cache_size_limit", 1000),
        ("recompile_limit", 800),
        ("accumulated_cache_size_limit", 1000),
        ("accumulated_recompile_limit", 2000),
    ]:
        val = int(os.environ.get(f"DREAMZERO_DYNAMO_{key.upper()}", str(default)))
        if hasattr(dynamo_cfg, key):
            setattr(dynamo_cfg, key, val)


class Model(PolicyModel):
    """DreamZero DROID model that loads GrootSimPolicy directly."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self._dreamzero_root = external_source_root(
            cfg,
            env_name="DREAMZERO_ROOT",
            cfg_key="dreamzero_root",
            project="DreamZero",
        )
        self._camera_map: dict[str, str] = dict(
            cfg.get("camera_map") or _DEFAULT_CAMERA_MAP
        )
        res = cfg.get("image_resolution") or (180, 320)
        self._image_resolution = (int(res[0]), int(res[1]))
        self._relative_offsets: list[int] = list(
            cfg.get("relative_offsets") or _DEFAULT_RELATIVE_OFFSETS
        )
        self._history_window: int = int(
            cfg.get("history_window", _DEFAULT_HISTORY_WINDOW)
        )
        self._instruction = str(cfg.get("instruction", ""))
        self._external_camera_2_mode: str = (
            str(cfg.get("external_camera_2_mode", "right")).strip().lower()
        )
        self._inference_method = cfg.get(
            "inference_method", "lazy_joint_forward_causal"
        )

        checkpoint_path = cfg.get("checkpoint_path", "") or os.environ.get(
            "DREAMZERO_CHECKPOINT_PATH", ""
        )
        self._checkpoint_path = str(self._resolve_path(checkpoint_path))

        tokenizer_path = cfg.get("tokenizer_path") or os.environ.get(
            "DREAMZERO_TOKENIZER_PATH"
        )
        self._tokenizer_path = (
            str(self._resolve_path(tokenizer_path)) if tokenizer_path else None
        )

        device = cfg.get("device", "cuda:0")
        skip_img_transform = bool(cfg.get("skip_img_transform", False))

        # Every torch touch — load, compile, inference — happens on this one
        # thread. See _run() for why.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dreamzero-droid"
        )
        self._run(lambda: self._load_owned(device, skip_img_transform))

    def _load_owned(self, device: str, skip_img_transform: bool) -> None:
        cfg = self.cfg

        _ensure_dreamzero_imports(self._dreamzero_root)
        _configure_torch_dynamo()
        _ensure_dist_initialized()

        # Must precede the groot imports: flow_unipc_multistep_scheduler applies
        # @torch.compile(mode="reduce-overhead") at class-body level, i.e. at
        # module-import time, so patching afterwards would miss it.
        _disable_cudagraphs()

        import torch
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from tianshou.data import Batch as _Batch

        self._torch = torch
        self._Batch = _Batch

        embodiment = cfg.get("embodiment_tag", "oxe_droid")
        print(f"[DreamZero DROID] Loading checkpoint: {self._checkpoint_path}")
        print(f"[DreamZero DROID] Embodiment: {embodiment}, device: {device}")

        self._policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(embodiment),
            model_path=self._checkpoint_path,
            device=device,
            tokenizer_path_override=self._tokenizer_path,
            skip_img_transform=skip_img_transform,
        )
        print("[DreamZero DROID] Model loaded successfully")

        self._session_id: str | None = None
        self._frame_history: dict[str, collections.deque[np.ndarray]] = {}
        self._step: int = 0
        self._latest_obs: dict[str, Any] | None = None
        self._observed_since_predict: bool = False

    def _run(self, fn: Callable[[], _T]) -> _T:
        """Execute ``fn`` on the single owner thread.

        Two reasons this has to be one fixed thread, not just a lock:

        1. ``torch.compile(mode="reduce-overhead")`` enables CUDA Graphs, and
           ``torch._inductor.cudagraph_trees`` stashes its tree-manager dict in a
           ``threading.local()`` at module-import time — i.e. only on whichever
           thread compiled first. Any other thread trips the bare
           ``assert torch._C._is_key_in_tls(...)`` in ``get_obj``.
        2. The policy carries mutable per-episode state (KV cache,
           ``current_start_frame``, ``_frame_history``), which is not safe to
           interleave across connections anyway.

        ``websockets.sync.server`` hands each connection its own thread, so
        without this every connection after the first would hit both problems.
        """
        return self._executor.submit(fn).result()

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        self._run(lambda: self._initialize_episode_owned(episode_info))

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._run(lambda: self._ingest_observation_owned(obs))

    def infer_actions(self) -> np.ndarray:
        return self._run(self._infer_actions_owned)

    def close(self) -> None:
        # Process cleanup remains on the model-owned worker thread.
        self._run(self._close_owned)

    def _initialize_episode_owned(
        self, episode_info: dict[str, Any] | None = None
    ) -> None:
        if episode_info:
            instr = episode_info.get("instruction")
            if isinstance(instr, str) and instr.strip():
                self._instruction = instr
        self._reset_policy_state_owned()
        self._session_id = str(uuid.uuid4())
        self._step = 0
        self._frame_history = {}
        self._observed_since_predict = False
        self._latest_obs = None

    def _reset_policy_state_owned(self) -> None:
        """Re-arm DreamZero's causal sequence for a new PolicySpace episode.

        Clearing only this wrapper's frame deque is insufficient: the action
        head owns the causal-frame cursor, cached language, and KV caches.  A
        zero cursor makes the next single-frame request rebuild those caches;
        clearing language guarantees it takes the fresh-sequence path even
        when consecutive endpoint probes/evaluations use the same prompt.
        """
        head = getattr(
            getattr(self._policy, "trained_model", None), "action_head", None
        )
        if head is None:
            return
        if hasattr(head, "current_start_frame"):
            head.current_start_frame = 0
        if hasattr(head, "language"):
            head.language = None

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

        request = self._build_request(self._latest_obs)

        batch = self._Batch(obs=request)
        with self._torch.inference_mode():
            if self._inference_method == "lazy_joint_forward_causal":
                result, _ = self._policy.lazy_joint_forward_causal(batch)
            elif self._inference_method == "lazy_joint_forward_causal_gt_cond":
                result, _ = self._policy.lazy_joint_forward_causal_gt_cond(batch)
            else:
                result = self._policy.forward(batch)

        action_dict = result.act

        joint_action = None
        gripper_action = None
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper" in key:
                gripper_action = value

        if joint_action is None:
            actions = np.zeros((1, 8), dtype=np.float32)
        else:
            if self._torch.is_tensor(joint_action):
                joint_action = joint_action.detach().cpu().numpy()
            joint_action = np.asarray(joint_action, dtype=np.float32)
            if joint_action.ndim == 1:
                joint_action = joint_action.reshape(1, -1)

            if gripper_action is not None:
                if self._torch.is_tensor(gripper_action):
                    gripper_action = gripper_action.detach().cpu().numpy()
                gripper_action = np.asarray(gripper_action, dtype=np.float32)
                if gripper_action.ndim == 0:
                    gripper_action = gripper_action.reshape(1, 1)
                elif gripper_action.ndim == 1:
                    gripper_action = gripper_action.reshape(-1, 1)
            else:
                gripper_action = np.zeros((joint_action.shape[0], 1), dtype=np.float32)

            actions = np.concatenate([joint_action, gripper_action], axis=-1)

        self._step += 1
        self._observed_since_predict = False
        return actions.astype(np.float32)

    def _close_owned(self) -> None:
        self._reset_policy_state_owned()
        self._session_id = None
        self._frame_history = {}
        self._step = 0
        self._latest_obs = None
        self._observed_since_predict = False

    def _resolve_path(self, value: str) -> Path:
        return resolve_data_path(value, base=self._dreamzero_root)

    def _append_frames(self, obs: dict[str, Any]) -> None:
        images = obs.get("images") or {}
        for sdk_name, dz_key in self._camera_map.items():
            img = images.get(sdk_name)
            if img is None:
                continue
            if sdk_name == "exterior_2" and self._external_camera_2_mode == "black":
                img = np.zeros_like(np.asarray(img))
            frame = self._prep_frame(img)
            buf = self._frame_history.setdefault(
                dz_key, collections.deque(maxlen=self._history_window)
            )
            buf.append(frame)

    def _prep_frame(self, img: Any) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        target_h, target_w = self._image_resolution
        if arr.ndim == 3 and arr.shape[:2] != (target_h, target_w):
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(arr)

    def _build_request(self, obs: dict[str, Any]) -> dict[str, Any]:
        first = self._step == 0
        request: dict[str, Any] = {}

        for dz_key, buf in self._frame_history.items():
            if not buf:
                continue
            if first:
                request[dz_key] = buf[-1][np.newaxis, ...]
            else:
                length = len(buf)
                selected = [
                    buf[max(0, length - 1 + off)] for off in self._relative_offsets
                ]
                request[dz_key] = np.stack(selected, axis=0)

        state = obs.get("state") or {}
        for dz_key, src_key, dim in _STATE_SPEC:
            val = state.get(src_key)
            if val is None:
                request[dz_key] = np.zeros((1, dim), dtype=np.float32)
            else:
                arr = np.asarray(val, dtype=np.float32).reshape(-1)[:dim]
                if arr.shape[0] < dim:
                    arr = np.concatenate(
                        [arr, np.zeros(dim - arr.shape[0], dtype=np.float32)]
                    )
                request[dz_key] = arr.reshape(1, -1)

        instruction = obs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = self._instruction or "complete the task"
        request["annotation.language.action_text"] = instruction

        return request
