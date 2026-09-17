"""Example PolicyModel — minimal proportional controller.

Shows how to implement the PolicyModel interface with a pure-NumPy
closed-loop controller that drives joints toward a zero target.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_space.base import PolicyModel


class Model(PolicyModel):
    """Proportional controller: action = gain * (target - current_pos)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        action_layout = cfg.get("action_layout") or []
        self.action_dim: int = len(action_layout) or int(cfg.get("action_dim", 16))
        self.action_horizon: int = int(cfg.get("action_horizon", 32))
        self._gain: float = float(cfg.get("gain", 0.0))
        self._target = np.zeros(self.action_dim, dtype=np.float32)
        self._current_pos = np.zeros(self.action_dim, dtype=np.float32)

    def initialize_episode(self, episode_info=None) -> None:
        self._target = np.zeros(self.action_dim, dtype=np.float32)
        self._current_pos = np.zeros(self.action_dim, dtype=np.float32)

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        joint_pos = obs.get("state", {}).get("joint_pos")
        if joint_pos is not None:
            self._current_pos = np.asarray(joint_pos, dtype=np.float32)[
                : self.action_dim
            ]

    def infer_actions(self) -> np.ndarray:
        row = self._gain * (self._target - self._current_pos)
        return np.repeat(row[None, :], self.action_horizon, axis=0)

    def close(self) -> None:
        pass
