"""Hold-position stand-in on the Franka/DROID 8-D joint wire."""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_space import PolicyModel


class Model(PolicyModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.action_dim = int(cfg.get("action_dim", 8))
        self.action_horizon = int(cfg.get("action_horizon", 24))

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._observation = obs

    def infer_actions(self) -> np.ndarray:
        return np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)


__all__ = ["Model"]
