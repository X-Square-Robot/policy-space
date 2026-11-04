"""Compatibility checks for the public Wall-X 1.1 serving interface."""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from policies.wallx.wall_oss_05_artixon_arm_6a_eef.model import Model


def _install_public_wallx(monkeypatch: pytest.MonkeyPatch) -> None:
    config_module = types.ModuleType(
        "wall_x._vendor.harrix.serving._wallx_infer.infer_config"
    )
    policy_module = types.ModuleType(
        "wall_x._vendor.harrix.serving.policy.wall_x_policy"
    )

    class InferConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.train_config = {"rtc_train": {"enabled": True, "max_delay": 8}}

    class WallXPolicy:
        def __init__(self, config, image_passing_mode, serialize_actions):
            self.config = config
            self.image_passing_mode = image_passing_mode
            self.serialize_actions = serialize_actions

        def reset(self):
            pass

        def infer(self, obs):
            assert obs["views"] and "follow1_pos" in obs["state"]
            return {"follow1_pos": [[0.0] * 7] * 33, "follow2_pos": [[0.0] * 7] * 33}

    config_module.InferConfig = InferConfig
    policy_module.WallXPolicy = WallXPolicy
    for name in (
        "wall_x",
        "wall_x._vendor",
        "wall_x._vendor.harrix",
        "wall_x._vendor.harrix.serving",
        "wall_x._vendor.harrix.serving._wallx_infer",
        "wall_x._vendor.harrix.serving.policy",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, policy_module.__name__, policy_module)


def test_public_backend_loads_and_infers_without_internal_dataset(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("X2ROBOT_ID_CONFIG_PATH", raising=False)
    _install_public_wallx(monkeypatch)
    model = Model(
        {
            "wallx_root": str(tmp_path),
            "checkpoint_path": "checkpoint",
        }
    )
    assert model._policy.config.kwargs["checkpoint_path"] == "checkpoint"
    assert model._policy.serialize_actions is True
    assert model.runtime_metadata() == {}
    model.initialize_episode({"instruction": "pick"})
    model.ingest_observation(
        {
            "images": {"camera_front": np.zeros((8, 8, 3), dtype=np.uint8)},
            "state": {"follow1_pos": np.zeros(7)},
        }
    )
    assert model.infer_actions().shape == (33, 14)
    assert "X2ROBOT_ID_CONFIG_PATH" not in os.environ


@pytest.mark.parametrize(
    "knob",
    [{"cfg_scale": 2.0}, {"cfg_drop": "state"}, {"cfg_uncond_instruction": "empty"}],
)
def test_public_backend_rejects_unsupported_cfg(monkeypatch, tmp_path, knob):
    _install_public_wallx(monkeypatch)
    with pytest.raises(ValueError, match="public.*CFG"):
        Model(
            {
                "wallx_root": str(tmp_path),
                "checkpoint_path": "checkpoint",
                "backend": "public",
                **knob,
            }
        )


def test_public_backend_rejects_rtc_requests(monkeypatch, tmp_path):
    _install_public_wallx(monkeypatch)
    model = Model(
        {
            "wallx_root": str(tmp_path),
            "checkpoint_path": "checkpoint",
            "backend": "public",
        }
    )
    with pytest.raises(ValueError, match="RTC"):
        model._build_envelope(
            {
                "images": {"camera_front": np.zeros((8, 8, 3), dtype=np.uint8)},
                "use_rtc": True,
            }
        )


def test_internal_backend_keeps_legacy_loading(monkeypatch, tmp_path):
    for name in (
        "x2robot_dataset",
        "wall_x",
        "wall_x.infer",
        "wall_x.serving",
        "wall_x.serving.policy",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    constants = types.ModuleType("x2robot_dataset.constant")
    constants._X2ROBOT_ID_TO_NAME = {}
    constants._ROBOT_ID_TO_DATASET_TYPE = {}
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    config_module = types.ModuleType("wall_x.infer.infer_config")
    config_module.InferConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    policy_module = types.ModuleType("wall_x.serving.policy.wall_x_policy")
    policy_module.WallXPolicy = lambda **kwargs: kwargs
    policy_module.parse_cfg_drop_targets = lambda value: value
    monkeypatch.setitem(sys.modules, policy_module.__name__, policy_module)
    monkeypatch.delenv("X2ROBOT_ID_CONFIG_PATH", raising=False)

    model = Model(
        {
            "wallx_root": str(tmp_path),
            "checkpoint_path": "checkpoint",
            "backend": "internal",
            "cfg_drop": "state",
        }
    )

    assert model._policy["cfg_drop_targets"] == "state"
    assert constants._X2ROBOT_ID_TO_NAME["10158"] == "X2 ex001_desktop"
    assert constants._ROBOT_ID_TO_DATASET_TYPE["10158"] == "ex_normal"
