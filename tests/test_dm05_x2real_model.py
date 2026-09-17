"""DM0.5 X2Real PolicySpace gateway contract tests."""

from __future__ import annotations

import base64
import json
import struct
import threading
import zlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import numpy as np
import pytest

from policies.dm05.dm05_artixon_arm_6a_joint.model import (
    CAMERA_KEYS,
    ROBOT_TYPE,
    Model,
    pack_x2real_action,
)
from policy_space.server import load_policy_definition, load_policy_model


def _actions() -> list[list[float]]:
    return np.arange(50 * 14, dtype=np.float32).reshape(50, 14).tolist()


def _observation() -> dict[str, Any]:
    return {
        "images": {
            camera: np.full((2 + index, 4 + index, 3), index + 1, dtype=np.uint8)
            for index, camera in enumerate(CAMERA_KEYS)
        },
        "state": {
            "follow1_pos": np.arange(7, dtype=np.float32),
            "follow2_pos": 10 + np.arange(7, dtype=np.float32),
        },
        "instruction": "pick up the red block",
    }


def _decode_rgb_png(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        kind = raw[offset + 4 : offset + 8]
        payload = raw[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            assert (bit_depth, color_type, compression, filtering, interlace) == (
                8,
                2,
                0,
                0,
                0,
            )
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    scanlines = zlib.decompress(bytes(compressed))
    rows = []
    stride = width * 3
    for row in range(height):
        start = row * (stride + 1)
        assert scanlines[start] == 0
        rows.append(
            np.frombuffer(
                scanlines[start + 1 : start + 1 + stride], dtype=np.uint8
            ).reshape(width, 3)
        )
    return np.stack(rows)


@contextmanager
def _backend(
    *, response: Any = None, status: int = 200, raw_body: bytes | None = None
) -> Iterator[tuple[str, list]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(
                {
                    "path": self.path,
                    "content_type": self.headers.get("Content-Type"),
                    "body": json.loads(self.rfile.read(length).decode("utf-8")),
                }
            )
            body = (
                raw_body
                if raw_body is not None
                else json.dumps(response).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/infer", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_request_preserves_camera_order_resolution_state_and_instruction() -> None:
    observation = _observation()
    with _backend(
        response={"actions": _actions(), "metadata": {"latency_ms": 12.5}}
    ) as (endpoint, requests):
        model = Model({"backend_url": endpoint})
        model.ingest_observation(observation)
        result = model.infer_actions()

    assert len(requests) == 1
    sent = requests[0]
    assert sent["path"] == "/v1/infer"
    assert sent["content_type"] == "application/json"
    assert "sampling" not in sent["body"]
    request_obs = sent["body"]["observation"]
    assert list(request_obs["images"]) == ["1", "2", "3"]
    for slot, camera in zip(("1", "2", "3"), CAMERA_KEYS, strict=True):
        assert not request_obs["images"][slot].startswith("data:")
        np.testing.assert_array_equal(
            _decode_rgb_png(request_obs["images"][slot]), observation["images"][camera]
        )
    assert request_obs["state"] == list(range(7)) + list(range(10, 17))
    assert request_obs["prompt"] == observation["instruction"]
    assert request_obs["robot_type"] == ROBOT_TYPE
    assert result["actions"].shape == (50, 26)
    np.testing.assert_array_equal(
        result["actions"][:, :14], np.asarray(_actions(), dtype=np.float32)
    )
    np.testing.assert_array_equal(result["actions"][:, 14:], 0.0)
    assert result["dm05_timing"]["backend_latency_ms"] == 12.5


def test_explicit_sampling_seed_is_sent() -> None:
    with _backend(response={"actions": _actions()}) as (endpoint, requests):
        model = Model({"backend_url": endpoint, "sampling_seed": 17})
        model.ingest_observation(_observation())
        model.infer_actions()
    assert requests[0]["body"]["sampling"] == {"seed": 17}


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda obs: obs["images"].pop("camera_left"), "missing camera"),
        (
            lambda obs: obs["images"].__setitem__("camera_front", [[[]]]),
            "numpy.ndarray",
        ),
        (
            lambda obs: obs["images"].__setitem__(
                "camera_front", np.zeros((2, 3, 3), dtype=np.float32)
            ),
            "dtype uint8",
        ),
        (
            lambda obs: obs["images"].__setitem__(
                "camera_front", np.zeros((2, 3), dtype=np.uint8)
            ),
            "HWC RGB",
        ),
        (lambda obs: obs["state"].pop("follow2_pos"), "missing state field"),
        (
            lambda obs: obs["state"].__setitem__("follow1_pos", np.zeros(14)),
            r"shape \(7,\)",
        ),
        (lambda obs: obs["state"]["follow1_pos"].__setitem__(0, np.nan), "NaN/Inf"),
        (lambda obs: obs.__setitem__("instruction", ""), "current episode instruction"),
    ],
)
def test_invalid_observations_fail_before_http(mutator, error: str) -> None:
    model = Model({})
    observation = _observation()
    mutator(observation)
    with pytest.raises((KeyError, TypeError, ValueError), match=error):
        model.ingest_observation(observation)


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({}, "missing 'actions'"),
        ({"actions": np.zeros((49, 14)).tolist()}, "shape"),
        ({"actions": np.zeros((50, 13)).tolist()}, "shape"),
        ({"actions": np.full((50, 14), np.nan).tolist()}, "NaN/Inf"),
    ],
)
def test_invalid_backend_actions_fail_without_fallback(response, error: str) -> None:
    with _backend(response=response) as (endpoint, _requests):
        model = Model({"backend_url": endpoint})
        model.ingest_observation(_observation())
        with pytest.raises((RuntimeError, ValueError), match=error):
            model.infer_actions()


def test_http_error_and_non_json_response_fail() -> None:
    with _backend(response={"error": "bad request"}, status=400) as (
        endpoint,
        _requests,
    ):
        model = Model({"backend_url": endpoint})
        model.ingest_observation(_observation())
        with pytest.raises(RuntimeError, match="OpenDM HTTP 400"):
            model.infer_actions()

    with _backend(raw_body=b"not-json") as (endpoint, _requests):
        model = Model({"backend_url": endpoint})
        model.ingest_observation(_observation())
        with pytest.raises(RuntimeError, match="not valid UTF-8 JSON"):
            model.infer_actions()


def test_reset_clears_episode_observation_and_action_state() -> None:
    with _backend(response={"actions": _actions()}) as (endpoint, requests):
        model = Model({"backend_url": endpoint})
        with pytest.raises(RuntimeError, match="ingest_observation first"):
            model.infer_actions()
        model.ingest_observation(_observation())
        model.infer_actions()
        model.initialize_episode()
        with pytest.raises(RuntimeError, match="ingest_observation first"):
            model.infer_actions()
    assert len(requests) == 1


def test_packing_requires_exact_finite_absolute_50x14() -> None:
    physical = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
    packed = pack_x2real_action(physical)
    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed[:, :14], physical)
    np.testing.assert_array_equal(packed[:, 14:], 0.0)
    with pytest.raises(ValueError, match="shape"):
        pack_x2real_action(np.zeros((32, 14), dtype=np.float32))
    physical[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        pack_x2real_action(physical)


def test_policy_definition_and_runtime_metadata_are_non_overlapping(
    monkeypatch,
) -> None:
    definition = load_policy_definition("dm05_x2real")
    assert definition.model_cfg["backend_url"].endswith("/v1/infer")
    assert definition.metadata["action_dim"] == 26
    assert definition.metadata["physical_action_dim"] == 14
    assert definition.metadata["action_horizon"] == 50
    assert definition.model_cfg["checkpoint_id"] == ""
    assert definition.metadata["recommended_execute_horizon"] == 50

    monkeypatch.setenv("DM05_X2REAL_BACKEND_URL", "http://gpu-server:8002/v1/infer")
    model, metadata = load_policy_model("dm05_x2real")
    runtime = model.runtime_metadata()
    assert not (definition.metadata.keys() & runtime.keys())
    assert metadata["dm05_backend"]["url"] == "http://gpu-server:8002/v1/infer"
    assert metadata["dm05_input_contract"]["state_shape"] == [14]
    assert metadata["dm05_output_contract"]["raw_shape"] == [50, 14]
    assert metadata["dm05_robot_type"] == ROBOT_TYPE
    assert metadata["dm05_action_semantics"] == "absolute_joint"
