"""Hold-position stand-in on the public Quanta X1 whole-body policy wire.

Returns a self-describing envelope rather than the legacy ``[T, D]`` array so
the sim-side implementation validates the same wire it validates for a real
model. Every column is zero, which the semantic20 adapter reads as: keep both
end-effectors at their reset-captured home pose, closed grippers, no base
velocity, lift and head at their zero targets.

Deliberately carries no model code and no torch dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_space.base import PolicyModel
from policy_space.wholebody_policy_wire import (
    ACTION_DIM,
    BASE_REPRESENTATION,
    EE_REFERENCE,
    EE_ROTATION_REPRESENTATION,
    WIRE_SCHEMA,
)


class Model(PolicyModel):
    """Emits a zero whole-body policy-wire chunk for every request."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.action_dim: int = ACTION_DIM
        self.action_horizon: int = int(cfg.get("action_horizon", 32))

    def initialize_episode(self, episode_info=None) -> None:
        pass

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        pass

    def infer_actions(self) -> dict[str, Any]:
        chunk = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
        return {
            "wire_schema": WIRE_SCHEMA,
            "action_dim": ACTION_DIM,
            "base_representation": BASE_REPRESENTATION,
            "ee_rotation_representation": EE_ROTATION_REPRESENTATION,
            "ee_reference": EE_REFERENCE,
            "action_chunk": chunk,
        }

    def close(self) -> None:
        pass
