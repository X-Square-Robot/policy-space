"""PolicyModel base class for PolicySpace policies.

All PolicySpace policies implement this interface. The generic WebSocket
server (``policy_space/server.py``) wraps any ``PolicyModel`` subclass
and exposes it to ManaEnv via the standardized msgpack protocol.

Model-specific implementations inherit from this process-local interface.
"""

from __future__ import annotations

from typing import Any


class PolicyModel:
    """Base interface for PolicySpace model implementations.

    Subclasses must implement at minimum:
      - ``ingest_observation(obs)`` or ``ingest_observation_batch(obs_list)``
      - ``infer_actions()`` or ``infer_actions_batch(env_idx_list)``

    Lifecycle:
      1. ``__init__(cfg)`` — load checkpoint, build model
      2. ``initialize_episode()`` — prepare episode-specific state
      3. ``ingest_observation(obs)`` — receive one observation frame
      4. ``infer_actions()`` — produce an action or action sequence
      5. ``finalize_episode()`` — release episode-specific state
      6. ``close()`` — release process resources
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def initialize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        """Prepare internal state for a new episode."""
        pass

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        """Buffer a single observation frame."""
        raise NotImplementedError

    def ingest_observation_batch(self, obs_list: list[dict[str, Any]]) -> None:
        """Buffer observations for multiple envs."""
        for obs in obs_list:
            self.ingest_observation(obs)

    def infer_actions(self) -> Any:
        """Return an action or action sequence for the latest observation.

        Returns either:
          - list of action dicts (one per timestep), or
          - ndarray of shape (T, D)
        """
        raise NotImplementedError

    def infer_actions_batch(self, env_idx_list: list[int] | None = None) -> list[Any]:
        """Return actions for multiple envs. Default calls infer_actions."""
        return [self.infer_actions()]

    def finalize_episode(self, episode_info: dict[str, Any] | None = None) -> None:
        """Release state owned by the active episode."""
        pass

    def runtime_metadata(self) -> dict[str, Any]:
        """Return capabilities derived from the loaded model/runtime.

        Deployment YAML describes the static wire contract.  Capabilities that
        depend on the actual checkpoint must be reported only after the model is
        loaded, so clients never have to guess them from a policy name or port.
        """
        return {}

    def close(self) -> None:
        """Release model resources."""
        pass
