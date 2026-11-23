"""Wall-X public API gateway for the Policy Space bimanual EE contract.

Public named-arm codec adapted from the private public_runner/public_rtc_codec.
History is owned by the upstream API, isolated by connection and episode UUID.
No checkpoints or credentials belong in this integration.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import time
import uuid

import msgpack
import msgpack_numpy
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from websockets.sync.client import connect
from policy_space.base import PolicyModel


def named_rows(rows, scale):
    a = np.asarray(rows, dtype=np.float32)
    if a.shape[-1] != 14 or not np.isfinite(a).all():
        raise ValueError("Expected finite Euler-14 rows")
    result = {}
    for name, offset in (("left_ee", 0), ("right_ee", 7)):
        result[name] = dict(position=a[..., offset:offset+3].tolist(),
                            rotation_rpy=a[..., offset+3:offset+6].tolist(),
                            gripper=(a[..., offset+6] * scale).tolist())
    return result


def action_rows(reply, scale):
    arms = []
    for name in ("left_ee", "right_ee"):
        b = reply["actions"][name]
        xyz, rpy, grip = (np.asarray(b[k], dtype=np.float32)
                          for k in ("position", "rotation_rpy", "gripper"))
        if xyz.shape != (32, 3) or rpy.shape != (32, 3) or grip.shape != (32,):
            raise ValueError("API must return 32 rows per arm")
        # The handshake requires valid_rows=31; row 31 is unused padding.
        valid_grip = grip[:31]
        if (valid_grip < -1e-5).any() or (valid_grip > scale + 1e-5).any():
            bad = np.flatnonzero((valid_grip < -1e-5) | (valid_grip > scale + 1e-5))
            raise ValueError(f"API gripper outside declared range: arm={name} "
                             f"range=[0,{scale}] min={float(grip.min()):.9g} "
                             f"max={float(grip.max()):.9g} indices={bad.tolist()} "
                             f"values={grip[bad].tolist()}")
        arms.append(np.column_stack((xyz, rpy, grip / scale)))
    rows = np.concatenate(arms, axis=1)
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite API action")
    return rows


class Model(PolicyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        knobs = {**cfg, **((cfg.get("model") or {}).get("cfg") or {})}
        self.gripper = str(knobs.get("gripper", "G")).upper()
        self.scale = {"G": 1.526, "H": 5.5}[self.gripper]
        self.embodiment = "quanta_x1_pro/gripper-" + self.gripper.lower()
        self.url = os.environ.get("WALLX_ENDPOINT", "wss://maas-api.x2robot.com/v1/vla/wallx")
        self.timeout = float(knobs.get("timeout_s", 90))
        self.ws = None
        self.upstream = {}
        self.initialize_episode()

    def initialize_episode(self, episode_info=None):
        self.close()
        self.latest = None
        self.calls = 0
        self.session = uuid.uuid4().hex

    def _connect(self):
        key = os.environ.get("WALLX_API_KEY", "")
        if not key and os.environ.get("WALLX_API_KEY_FILE"):
            key = Path(os.environ["WALLX_API_KEY_FILE"]).read_text().strip()
        if not key:
            raise RuntimeError("Set WALLX_API_KEY or WALLX_API_KEY_FILE")
        try:
            self.ws = connect(self.url, additional_headers={"Authorization": "Bearer " + key},
                              proxy=None, open_timeout=30, close_timeout=2,
                              max_size=64*1024*1024, ping_interval=None, compression=None)
            raw = self.ws.recv(timeout=self.timeout)
            self.upstream = json.loads(raw) if isinstance(raw, str) else msgpack.unpackb(raw, raw=False)
        except Exception as exc:
            self.close()
            raise RuntimeError("Wall-X connection failed: " + type(exc).__name__) from None
        if self.upstream.get("model") != "wall-x" or self.upstream.get("valid_rows") != 31:
            raise ValueError("Unexpected Wall-X model/valid-row contract")
        if not self.upstream.get("cb_history", {}).get("server_auto_history", {}).get("enabled"):
            raise ValueError("API does not advertise server-owned history")

    def runtime_metadata(self):
        return {"api_embodiment": self.embodiment, "history_owner": "upstream_session",
                "rtc_max_delay": 4, "valid_prediction_rows": 31}

    def ingest_observation(self, obs):
        self.latest = obs

    def infer_actions(self):
        if self.latest is None:
            raise RuntimeError("No observation")
        if self.ws is None:
            self._connect()
        obs = self.latest
        state = obs["state"]
        anchor = np.concatenate([state["follow1_pos"], state["follow2_pos"]]).astype(np.float32)
        images = {}
        for source, target in (("camera_front", "front"), ("camera_left", "left_wrist"), ("camera_right", "right_wrist")):
            frame = np.asarray(obs["images"][source])
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError("Expected RGB uint8 image")
            buf = io.BytesIO()
            image = Image.fromarray(frame)
            if min(image.size) < 32:
                image = image.resize((max(32, image.width), max(32, image.height)))
            image.save(buf, format="JPEG", quality=95)
            images[target] = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = dict(embodiment=self.embodiment, normalization="provided",
                       instruction=obs["instruction"],
                       observation=dict(images=images, state=named_rows(anchor, self.scale)),
                       use_history=True, reset_history=self.calls == 0, session_id=self.session)
        prefix = None
        if obs.get("use_rtc"):
            delay = int(obs["inference_delay"])
            if not 1 <= delay <= 4:
                raise ValueError("RTC delay must be 1..4")
            prefix = np.asarray(obs["prev_action_chunk"], dtype=np.float32)
            if len(prefix) < delay + 1:
                raise ValueError("RTC prefix too short")
            wire_prefix = prefix.copy()
            wire_prefix[:, [6, 13]] *= self.scale
            payload.update(use_rtc=True, inference_delay=delay, rtc_mode="training_time", source_phase_ns=0, s_model=int(obs["s_model"]),
                           prev_action_chunk=wire_prefix)
        started = time.monotonic()
        try:
            self.ws.send(msgpack.packb(payload, use_bin_type=True, default=msgpack_numpy.encode))
            raw = self.ws.recv(timeout=self.timeout)
            reply = json.loads(raw) if isinstance(raw, str) else msgpack.unpackb(raw, raw=False, object_hook=msgpack_numpy.decode)
        except Exception as exc:
            raise RuntimeError("Wall-X request failed: " + type(exc).__name__) from None
        if "error" in reply:
            err = reply["error"]
            code = err.get("code", "unknown") if isinstance(err, dict) else str(err)[:160]
            raise RuntimeError("Wall-X API error: " + code)
        if reply.get("embodiment") != self.embodiment:
            raise ValueError("API embodiment mismatch")
        rows = action_rows(reply, self.scale)
        if prefix is not None:
            ack = reply.get("rtc", {})
            if ack.get("enabled") is not True or ack.get("effective_delay") != delay:
                raise ValueError("Missing RTC acknowledgement")
            for o in (0, 7):
                np.testing.assert_allclose(rows[:delay, o:o+3], prefix[1:delay+1, o:o+3], atol=3e-5, rtol=0)
                np.testing.assert_allclose(rows[:delay, o+6], prefix[1:delay+1, o+6], atol=5e-5, rtol=0)
                rotation_error = Rotation.from_euler("xyz", rows[:delay, o+3:o+6]) * Rotation.from_euler("xyz", prefix[1:delay+1, o+3:o+6]).inv()
                if rotation_error.magnitude().max() > 3e-5:
                    raise ValueError("API changed committed rotation prefix")
        self.calls += 1
        print("WALLX_API " + json.dumps(dict(session=self.session, call=self.calls,
              rtc=reply.get("rtc"), history=reply.get("history"),
              seconds=round(time.monotonic()-started, 4), checkpoint=self.upstream.get("checkpoint"))), flush=True)
        # API's last prediction is padding (valid_rows=31). Preserve it on the
        # 33-row wire for existing decoder compatibility; executor never uses it.
        return np.vstack((anchor, rows)).astype(np.float32)

    def finalize_episode(self, episode_info=None):
        self.close()
        self.latest = None

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None
