"""Wall-X RTC metadata and request-shaping tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from policies.wallx.wall_oss_05_artixon_arm_6a_eef.model import Model


def test_wallx_model_forwards_only_rtc_protocol_fields() -> None:
    model = Model.__new__(Model)
    model.instruction = "fallback"
    model.image_wire_encoding = "lossless_png"
    model.jpeg_quality = 95
    prefix = np.zeros((9, 14), dtype=np.float32)

    dummy_image = np.zeros((8, 8, 3), dtype=np.uint8)
    envelope = model._build_envelope(
        {
            "images": {"camera_front": dummy_image},
            "state": {"follow1_pos": np.zeros(7), "follow2_pos": np.zeros(7)},
            "instruction": "rtc task",
            "use_rtc": True,
            "prev_action_chunk": prefix,
            "inference_delay": 8,
            "num_flow_steps": 10,
            "rtc_mode": "training_time",
            "unrelated": "must not leak",
        }
    )

    assert envelope["use_rtc"] is True
    assert envelope["prev_action_chunk"] is prefix
    assert envelope["inference_delay"] == 8
    assert envelope["num_flow_steps"] == 10
    assert "unrelated" not in envelope


def test_wallx_model_advertises_loaded_checkpoint_rtc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_CUDA_GRAPH", "False")
    monkeypatch.setenv("ENABLE_EXPERIMENTAL_INFERENCE_ENGINE", "False")
    model = Model.__new__(Model)
    model.control_rate_hz = 20
    model._policy = SimpleNamespace(
        config=SimpleNamespace(
            train_config={"rtc_train": {"enabled": True, "max_delay": 8}},
            action_horizon=32,
            num_inference_timesteps=10,
        )
    )

    assert model.runtime_metadata() == {
        "capabilities": {
            "wall_x_rtc": {
                "enabled": True,
                "protocol": "wall_x_rtc.v1",
                "model_action_horizon": 32,
                "max_delay": 8,
                "num_flow_steps": 10,
                "control_rate_hz": 20,
            }
        }
    }


def test_wallx_rtc_rejects_incompatible_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_CUDA_GRAPH", "True")
    monkeypatch.setenv("ENABLE_EXPERIMENTAL_INFERENCE_ENGINE", "False")
    model = Model.__new__(Model)
    model._policy = SimpleNamespace(
        config=SimpleNamespace(
            train_config={"rtc_train": {"enabled": True, "max_delay": 8}},
            action_horizon=32,
            num_inference_timesteps=10,
        )
    )

    with pytest.raises(RuntimeError, match="ENABLE_CUDA_GRAPH=False"):
        model.runtime_metadata()
