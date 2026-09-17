"""Generic PolicySpace WebSocket server.

Wraps any ``PolicyModel`` subclass and serves it over msgpack WebSocket,
compatible with ``PolicySpaceImpl`` on the ManaEnv side.

Usage:
    policy-space serve policies/dreamzero/dreamzero_artixon_arm_6a_joint/deploy.yaml --port 8001

Or programmatically:
    from policy_space.server import PolicySpaceServer
    server = PolicySpaceServer(model, port=8001)
    server.serve_forever()
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from websockets.sync.server import ServerConnection, serve

from policy_space.base import PolicyModel
from policy_space.conformance import normalize_action_output, validate_observation_input
from policy_space.contracts import (
    SchemaPair,
    schema_pairs_from_metadata,
    select_schema_pair,
)
from policy_space.contracts.registry import ContractRegistry
from policy_space.protocol import (
    SESSION_MODE_MULTIPLEXED_SERIAL,
    SESSION_OWNER_BUSY_MARKER,
    V2_METADATA,
    normalize_session_mode,
)
from policy_space.transport import unpack_ndarray

# Some policy implementations call ``msgpack_numpy.patch()`` while they are
# imported.  Keep the unpatched codec classes before loading any policy module;
# looking them up on ``msgpack`` at request time would pick up the monkey-patch.
_ORIGINAL_PACKER = msgpack.Packer
_ORIGINAL_UNPACKER = msgpack.Unpacker


@dataclass(frozen=True)
class PolicyDefinition:
    """Validated policy deployment definition without a constructed model."""

    policy_name: str
    model_path: Path
    model_class_name: str
    model_cfg: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class _MultiplexedSession:
    """Reconnectable protocol state for one multiplexed client episode."""

    episode_id: str
    schema_pair: SchemaPair | None = None
    owner: object | None = None
    last_request_id: str | None = None
    last_response: Any = None


_MAX_DISCONNECTED_MULTIPLEXED_SESSIONS = 64
DEFAULT_MAX_MESSAGE_SIZE = 64 * 1024 * 1024
DEFAULT_MAX_CONNECTIONS = 32
DEFAULT_MAX_SESSIONS = 64
DEFAULT_MAX_REQUESTS_PER_MINUTE = 3600
DEFAULT_MAX_PAYLOAD_DEPTH = 32

_POLICY_TEMPLATE_ALIASES = {
    "example_policy": Path("templates/bimanual_ee"),
    "example_franka_policy": Path("templates/franka_joint"),
    "example_mobile_policy": Path("templates/mobile_manipulation"),
}


def _safe_packb(obj: Any, default: Any = None) -> bytes:
    """Pack using a fresh Packer to bypass any msgpack_numpy monkey-patch."""
    packer = _ORIGINAL_PACKER(default=default, use_bin_type=True)
    return packer.pack(obj)


def _safe_unpackb(data: bytes, object_hook: Any = None) -> Any:
    """Unpack using explicit Unpacker to bypass any msgpack_numpy monkey-patch."""

    def checked_object_pairs_hook(pairs: list[tuple[Any, Any]]) -> Any:
        result: dict[Any, Any] = {}
        for key, item in pairs:
            if isinstance(key, bool) or not isinstance(key, (str, bytes, int)):
                raise ValueError(
                    "Policy Space map keys must be strings, bytes, or integer environment IDs"
                )
            if key in result:
                raise ValueError(
                    f"Policy Space payload contains duplicate map key {key!r}"
                )
            result[key] = item
        return object_hook(result) if object_hook is not None else result

    # Protocol v2 uses string object keys, but early clients encoded integer
    # environment identifiers inside lifecycle metadata. Accept those valid
    # MsgPack maps so a final notification cannot tear down the connection.
    unpacker = _ORIGINAL_UNPACKER(
        object_pairs_hook=checked_object_pairs_hook,
        raw=False,
        strict_map_key=False,
    )
    unpacker.feed(data)
    value = unpacker.unpack()
    _validate_legacy_integer_key_paths(value)
    return value


def _validate_legacy_integer_key_paths(value: Any) -> None:
    """Limit integer-key compatibility to legacy lifecycle environment maps."""
    pending: list[tuple[str | None, Any]] = [(None, value)]
    while pending:
        parent_key, current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if isinstance(key, int) and parent_key not in {"env_steps", "outcomes"}:
                    raise ValueError(
                        "Policy Space integer map keys are allowed only in env_steps or outcomes"
                    )
                pending.append((key if isinstance(key, str) else None, item))
        elif isinstance(current, (list, tuple)):
            pending.extend((None, item) for item in current)


def _pack_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict) -> Any:
    return unpack_ndarray(value)


class PolicySpaceServer:
    """Synchronous msgpack WebSocket server wrapping a PolicyModel."""

    def __init__(
        self,
        model: PolicyModel,
        host: str = "127.0.0.1",
        port: int = 8001,
        metadata: dict[str, Any] | None = None,
        sock: socket.socket | None = None,
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE,
        max_payload_depth: int = DEFAULT_MAX_PAYLOAD_DEPTH,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self._metadata = metadata or {}
        self._session_mode = normalize_session_mode(self._metadata.get("session_mode"))
        self._sock = sock
        self._max_message_size = max_message_size
        self._max_sessions = max(1, int(max_sessions))
        self._max_requests_per_minute = max(1, int(max_requests_per_minute))
        self._max_payload_depth = max(1, int(max_payload_depth))
        self._connection_slots = threading.BoundedSemaphore(
            max(1, int(max_connections))
        )
        self._request_rate_lock = threading.Lock()
        self._request_times: deque[float] = deque()
        raw_pairs = self._metadata.get("supported_schema_pairs")
        self._schema_pairs = (
            schema_pairs_from_metadata(self._metadata) if raw_pairs is not None else []
        )
        self._contract_registry = (
            ContractRegistry.builtin() if self._schema_pairs else None
        )
        if self._contract_registry is not None:
            self._contract_registry.validate_references()
            for pair in self._schema_pairs:
                if pair.observation_schema not in self._contract_registry.observations:
                    raise ValueError(
                        f"Unknown observation schema: {pair.observation_schema}"
                    )
                if pair.action_schema not in self._contract_registry.actions:
                    raise ValueError(f"Unknown action schema: {pair.action_schema}")
        # PolicyModel instances may own causal caches, frame history, and other
        # episode-scoped state.  websockets.sync serves connections on separate
        # threads, so a second connection must never initialize or close the model
        # while the active connection is using it.
        self._session_lock = threading.Lock()
        self._model_inference_lock = threading.Lock()
        self._multiplexed_sessions_lock = threading.Lock()
        self._multiplexed_sessions: OrderedDict[str, _MultiplexedSession] = (
            OrderedDict()
        )
        # The identifier distinguishes a transient socket reconnect from a
        # policy-service process restart. Episode state can be resumed only in
        # the former case.
        self._instance_id = uuid.uuid4().hex
        self._active_episode_id: str | None = None
        self._active_schema_pair: SchemaPair | None = None
        self._observation_buffered = False
        self._last_request_id: str | None = None
        self._last_response: Any = None

    def serve_forever(self) -> None:
        print(f"[PolicySpaceServer] listening on ws://{self.host}:{self.port}")
        if self._sock is not None:
            ctx = serve(
                self._handle_client,
                sock=self._sock,
                max_size=self._max_message_size,
                compression=None,
                # Simulation may execute an action chunk longer than the
                # websockets keepalive window while the client is not reading.
                ping_interval=None,
                ping_timeout=None,
            )
        else:
            ctx = serve(
                self._handle_client,
                self.host,
                self.port,
                max_size=self._max_message_size,
                compression=None,
                # Simulation may execute an action chunk longer than the
                # websockets keepalive window while the client is not reading.
                ping_interval=None,
                ping_timeout=None,
            )
        with ctx as server:
            server.serve_forever()

    def _handle_client(self, ws: ServerConnection) -> None:
        if not self._connection_slots.acquire(blocking=False):
            try:
                ws.send(
                    _safe_packb(
                        {
                            **self._metadata,
                            **V2_METADATA,
                            "session_mode": self._session_mode,
                            "server_instance_id": self._instance_id,
                            "error": "connection_limit",
                            "retryable": True,
                        },
                        default=_pack_array,
                    )
                )
                ws.close(code=1013, reason="Policy Space connection limit reached")
            finally:
                return
        try:
            self._handle_admitted_client(ws)
        finally:
            self._connection_slots.release()

    def _handle_admitted_client(self, ws: ServerConnection) -> None:
        metadata = {
            **self._metadata,
            **V2_METADATA,
            "session_mode": self._session_mode,
            "server_instance_id": self._instance_id,
        }

        if self._session_mode == SESSION_MODE_MULTIPLEXED_SERIAL:
            self._handle_multiplexed_client(ws, metadata)
            return

        self._handle_exclusive_client(ws, metadata)

    def _handle_exclusive_client(
        self, ws: ServerConnection, metadata: dict[str, Any]
    ) -> None:
        """Serve one connection while preserving process-global episode state."""

        if not self._session_lock.acquire(blocking=False):
            busy_metadata = {
                **metadata,
                "error": "server_busy",
                "retryable": True,
            }
            try:
                ws.send(_safe_packb(busy_metadata, default=_pack_array))
            finally:
                try:
                    ws.close(
                        code=1013,
                        reason="PolicySpace model already has an active session",
                    )
                except Exception:
                    pass
            return

        try:
            episode_active = False
            ws.send(_safe_packb(metadata, default=_pack_array))

            for raw in ws:
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                self._check_request_rate()
                request = _safe_unpackb(raw, object_hook=_unpack_array)
                self._validate_payload_depth(request)
                endpoint = request.pop("endpoint", None)
                request_id = str(request.pop("request_id", "") or "").strip()
                try:
                    if endpoint == "initialize_episode" and episode_active:
                        raise RuntimeError(
                            "Cannot initialize a new episode before finalizing the active episode"
                        )
                    if endpoint != "initialize_episode" and not episode_active:
                        raise RuntimeError(
                            "No active episode. Call initialize_episode first."
                        )
                    if request_id and request_id == self._last_request_id:
                        response = self._last_response
                    else:
                        response = self._dispatch(
                            endpoint,
                            request,
                            observation_buffered=self._observation_buffered,
                        )
                        if request_id:
                            # Store the result before sending it. If the reply is
                            # lost, a reconnect can retry the same request id
                            # without applying the observation or inference twice.
                            self._last_request_id = request_id
                            self._last_response = response
                except Exception as exc:
                    import traceback

                    print(f"[PolicySpaceServer] ERROR in {endpoint}: {exc}")
                    traceback.print_exc()
                    ws.send(f"ERROR: {type(exc).__name__}: {exc}")
                    continue
                if endpoint == "initialize_episode":
                    episode_active = True
                elif endpoint == "ingest_observation":
                    self._observation_buffered = True
                elif endpoint == "finalize_episode":
                    episode_active = False
                elif endpoint == "infer_actions":
                    self._observation_buffered = False
                ws.send(_safe_packb(response, default=_pack_array))
        finally:
            # Keep episode-scoped model state across a transient socket drop.
            # A reconnect claims the same episode id; a different episode id
            # explicitly supersedes and finalizes this abandoned state.
            self._session_lock.release()

    def _handle_multiplexed_client(
        self, ws: ServerConnection, metadata: dict[str, Any]
    ) -> None:
        """Serve one request-isolated client on a shared serialized model."""
        owner = object()
        session: _MultiplexedSession | None = None
        ws.send(_safe_packb(metadata, default=_pack_array))

        try:
            for raw in ws:
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                self._check_request_rate()
                request = _safe_unpackb(raw, object_hook=_unpack_array)
                self._validate_payload_depth(request)
                endpoint = request.pop("endpoint", None)
                request_id = str(request.pop("request_id", "") or "").strip()
                finalized = False
                try:
                    if endpoint == "initialize_episode":
                        if session is not None:
                            raise RuntimeError(
                                "Cannot initialize a new episode before finalizing the active episode"
                            )
                        session, resumed = self._claim_multiplexed_session(
                            request, owner
                        )
                        response: Any = {"ok": True, "resumed": resumed}
                    else:
                        if session is None:
                            raise RuntimeError(
                                "No active episode. Call initialize_episode first."
                            )
                        if request_id and request_id == session.last_request_id:
                            response = session.last_response
                        else:
                            response = self._dispatch_multiplexed(
                                endpoint, request, session
                            )
                            if request_id and endpoint != "finalize_episode":
                                session.last_request_id = request_id
                                session.last_response = response
                        finalized = endpoint == "finalize_episode"
                except Exception as exc:
                    import traceback

                    print(f"[PolicySpaceServer] ERROR in {endpoint}: {exc}")
                    traceback.print_exc()
                    ws.send(f"ERROR: {type(exc).__name__}: {exc}")
                    continue
                ws.send(_safe_packb(response, default=_pack_array))
                if finalized:
                    self._remove_multiplexed_session(session, owner)
                    session = None
        finally:
            if session is not None:
                self._release_multiplexed_session(session, owner)

    def _claim_multiplexed_session(
        self,
        payload: dict[str, Any],
        owner: object,
    ) -> tuple[_MultiplexedSession, bool]:
        episode_id = str(payload.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError("initialize_episode requires a non-empty episode_id")
        resume_only = payload.get("resume_only") is True
        with self._multiplexed_sessions_lock:
            session = self._multiplexed_sessions.get(episode_id)
            resumed = session is not None
            if session is None:
                if resume_only:
                    raise RuntimeError(
                        f"Episode {episode_id!r} resume state is unavailable"
                    )
                self._evict_disconnected_multiplexed_sessions_locked(
                    max_disconnected=self._max_sessions - 1
                )
                if len(self._multiplexed_sessions) >= self._max_sessions:
                    raise RuntimeError("Policy Space session limit reached")
                session = _MultiplexedSession(
                    episode_id=episode_id,
                    schema_pair=self._resolve_schema_pair(payload),
                )
                self._multiplexed_sessions[episode_id] = session
            elif session.owner is not None:
                raise RuntimeError(
                    f"Episode {episode_id!r} {SESSION_OWNER_BUSY_MARKER}"
                )
            elif self._has_schema_selection(payload):
                selected = self._resolve_schema_pair(payload)
                if selected != session.schema_pair:
                    raise RuntimeError(
                        "reconnect schema pair does not match the episode"
                    )
            session.owner = owner
            self._multiplexed_sessions.move_to_end(episode_id)
            return session, resumed

    def _release_multiplexed_session(
        self, session: _MultiplexedSession, owner: object
    ) -> None:
        with self._multiplexed_sessions_lock:
            if (
                self._multiplexed_sessions.get(session.episode_id) is not session
                or session.owner is not owner
            ):
                return
            session.owner = None
            self._multiplexed_sessions.move_to_end(session.episode_id)
            self._evict_disconnected_multiplexed_sessions_locked(
                max_disconnected=min(
                    _MAX_DISCONNECTED_MULTIPLEXED_SESSIONS,
                    self._max_sessions,
                )
            )

    def _remove_multiplexed_session(
        self, session: _MultiplexedSession, owner: object
    ) -> None:
        with self._multiplexed_sessions_lock:
            if (
                self._multiplexed_sessions.get(session.episode_id) is not session
                or session.owner is not owner
            ):
                raise RuntimeError(
                    f"Episode {session.episode_id!r} is not owned by this client"
                )
            del self._multiplexed_sessions[session.episode_id]

    def _evict_disconnected_multiplexed_sessions_locked(
        self,
        *,
        max_disconnected: int = _MAX_DISCONNECTED_MULTIPLEXED_SESSIONS,
    ) -> None:
        disconnected = sum(
            session.owner is None for session in self._multiplexed_sessions.values()
        )
        if disconnected <= max_disconnected:
            return
        for episode_id, session in list(self._multiplexed_sessions.items()):
            if session.owner is not None:
                continue
            del self._multiplexed_sessions[episode_id]
            disconnected -= 1
            if disconnected <= max_disconnected:
                break

    def _dispatch_multiplexed(
        self,
        endpoint: str,
        payload: dict[str, Any],
        session: _MultiplexedSession,
    ) -> Any:
        if endpoint == "ingest_observation":
            raise RuntimeError(
                "ingest_observation is unavailable in multiplexed_serial mode; "
                "use exclusive mode for policies with service-side observation history"
            )
        if endpoint == "infer_actions":
            with self._model_inference_lock:
                return self._infer_actions(
                    payload,
                    observation_buffered=False,
                    schema_pair=session.schema_pair,
                )
        if endpoint == "finalize_episode":
            requested_episode_id = str(payload.get("episode_id") or "").strip()
            if requested_episode_id and requested_episode_id != session.episode_id:
                raise RuntimeError(
                    f"Cannot finalize episode {requested_episode_id!r}; active episode is {session.episode_id!r}"
                )
            return {"ok": True}
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def _dispatch(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        observation_buffered: bool = False,
    ) -> Any:
        if endpoint == "initialize_episode":
            episode_id = str(payload.get("episode_id") or "").strip()
            if not episode_id:
                raise ValueError("initialize_episode requires a non-empty episode_id")
            if self._active_episode_id == episode_id:
                if self._has_schema_selection(payload):
                    selected = self._resolve_schema_pair(payload)
                    if selected != self._active_schema_pair:
                        raise RuntimeError(
                            "reconnect schema pair does not match the active episode"
                        )
                return {"ok": True, "resumed": True}
            selected_schema_pair = self._resolve_schema_pair(payload)
            if self._active_episode_id is not None:
                self.model.finalize_episode(
                    {
                        "episode_id": self._active_episode_id,
                        "reason": "superseded_by_new_episode",
                    }
                )
                self._active_episode_id = None
                self._active_schema_pair = None
                self._observation_buffered = False
            self.model.initialize_episode(payload)
            self._active_episode_id = episode_id
            self._active_schema_pair = selected_schema_pair
            self._observation_buffered = False
            self._last_request_id = None
            self._last_response = None
            return {"ok": True, "resumed": False}

        if endpoint == "ingest_observation":
            self._validate_observation(payload, self._active_schema_pair)
            self.model.ingest_observation(payload)
            return {"ok": True}

        if endpoint == "infer_actions":
            return self._infer_actions(
                payload,
                observation_buffered=observation_buffered,
                schema_pair=self._active_schema_pair,
            )

        if endpoint == "finalize_episode":
            requested_episode_id = str(payload.get("episode_id") or "").strip()
            if self._active_episode_id is None:
                raise RuntimeError("No active episode to finalize")
            if requested_episode_id and requested_episode_id != self._active_episode_id:
                raise RuntimeError(
                    f"Cannot finalize episode {requested_episode_id!r}; "
                    f"active episode is {self._active_episode_id!r}"
                )
            self.model.finalize_episode(payload)
            self._active_episode_id = None
            self._active_schema_pair = None
            self._observation_buffered = False
            self._last_request_id = None
            self._last_response = None
            return {"ok": True}

        if endpoint == "ingest_observation_batch":
            observations = payload.get("observations", [])
            self.model.ingest_observation_batch(observations)
            return {"ok": True}

        if endpoint == "infer_actions_batch":
            env_idx_list = payload.get("env_idx_list")
            actions = self.model.infer_actions_batch(env_idx_list)
            return {"actions": actions}

        raise ValueError(f"Unknown endpoint: {endpoint}")

    def _infer_actions(
        self,
        payload: dict[str, Any],
        *,
        observation_buffered: bool,
        schema_pair: SchemaPair | None = None,
    ) -> Any:
        # Stateful policies receive every control frame through ingest_observation.
        # Re-appending the refill frame here corrupts their fixed-size history
        # window. Inference-only and multiplexed clients carry the observation
        # with the request and ingest it immediately before model execution.
        if not observation_buffered:
            self._validate_observation(payload, schema_pair)
            self.model.ingest_observation(payload)
        actions = self.model.infer_actions()
        if schema_pair is not None:
            assert self._contract_registry is not None
            return normalize_action_output(
                actions,
                self._contract_registry.actions[schema_pair.action_schema],
                expected_horizon=self._metadata.get("action_horizon"),
            )
        # A model-neutral wire may need to return a self-describing envelope.
        if isinstance(actions, dict):
            return actions
        if isinstance(actions, np.ndarray):
            if actions.ndim != 2:
                print(
                    f"[PolicySpaceServer] WARNING: infer_actions returned ndim={actions.ndim}, shape={actions.shape}"
                )
            return {"actions": actions}
        if isinstance(actions, list):
            return {"actions": np.array(actions, dtype=np.float32)}
        print(
            f"[PolicySpaceServer] WARNING: infer_actions returned type={type(actions).__name__}"
        )
        return {"actions": actions}

    def _validate_observation(
        self,
        payload: dict[str, Any],
        schema_pair: SchemaPair | None,
    ) -> None:
        if schema_pair is None:
            return
        assert self._contract_registry is not None
        validate_observation_input(
            payload,
            self._contract_registry.observations[schema_pair.observation_schema],
        )

    @staticmethod
    def _has_schema_selection(payload: dict[str, Any]) -> bool:
        return any(
            payload.get(key)
            for key in ("observation_schema", "action_schema", "embodiment_schema")
        )

    def _resolve_schema_pair(self, payload: dict[str, Any]) -> SchemaPair | None:
        if not self._schema_pairs:
            return None
        observation = str(payload.get("observation_schema") or "").strip()
        action = str(payload.get("action_schema") or "").strip()
        embodiment = str(payload.get("embodiment_schema") or "").strip()
        if (
            not observation
            and not action
            and not embodiment
            and len(self._schema_pairs) == 1
        ):
            return self._schema_pairs[0]
        if not observation or not action or not embodiment:
            raise ValueError(
                "initialize_episode requires observation_schema, action_schema, "
                "and embodiment_schema"
            )
        return select_schema_pair(
            self._schema_pairs,
            [SchemaPair(observation, action, (embodiment,))],
            embodiment=embodiment,
        )

    def _check_request_rate(self) -> None:
        now = time.monotonic()
        with self._request_rate_lock:
            while self._request_times and now - self._request_times[0] >= 60.0:
                self._request_times.popleft()
            if len(self._request_times) >= self._max_requests_per_minute:
                raise RuntimeError("Policy Space request rate limit reached")
            self._request_times.append(now)

    def _validate_payload_depth(self, value: Any, depth: int = 0) -> None:
        if depth > self._max_payload_depth:
            raise ValueError("Policy Space request exceeds maximum nesting depth")
        if isinstance(value, dict):
            for key, item in value.items():
                self._validate_payload_depth(key, depth + 1)
                self._validate_payload_depth(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._validate_payload_depth(item, depth + 1)


def _resolve_policy_dir(policy: str | Path) -> Path:
    """Resolve a policy name, directory, or deploy file without repository coupling."""

    requested = Path(policy).expanduser()
    if requested.name == "deploy.yaml" and requested.is_file():
        return requested.resolve().parent
    if requested.is_dir():
        return requested.resolve()
    configured_root = os.environ.get("POLICY_SPACE_POLICIES_DIR")
    roots = [Path(configured_root).expanduser()] if configured_root else []
    roots.append(Path.cwd() / "policies")

    # Preserve pre-catalog example names and paths without duplicating files.
    compatibility_key = (
        requested.parent.name if requested.name == "deploy.yaml" else requested.name
    )
    template_relative = _POLICY_TEMPLATE_ALIASES.get(compatibility_key)
    if template_relative is not None:
        for root in roots:
            template_dir = (root / template_relative).resolve()
            if template_dir.is_dir():
                return template_dir

    for root in roots:
        candidate = root / requested
        if candidate.is_dir():
            return candidate.resolve()

    # Family-grouped catalogs keep the stable policy identity in deploy.yaml
    # while allowing the on-disk layout to evolve independently. Resolve a
    # simple policy name recursively so existing CLI and API callers do not
    # need to know the catalog's directory hierarchy.
    if len(requested.parts) == 1 and requested.name:
        matches: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            for deploy_yaml in sorted(root.rglob("deploy.yaml")):
                try:
                    import yaml

                    document = (
                        yaml.safe_load(deploy_yaml.read_text(encoding="utf-8")) or {}
                    )
                except (OSError, yaml.YAMLError):
                    continue
                if str(document.get("policy_name") or "") == requested.name:
                    matches.append(deploy_yaml.parent.resolve())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            locations = ", ".join(str(path) for path in matches)
            raise RuntimeError(
                f"Policy name {requested.name!r} is ambiguous across catalog entries: {locations}"
            )
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"No policy directory for {policy!r}; searched: {searched}")


def load_policy_definition(policy: str | Path) -> PolicyDefinition:
    """Load and validate a policy deployment without constructing its model."""
    policy_dir = _resolve_policy_dir(policy)
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"No policy directory: {policy_dir}")

    deploy_yaml = policy_dir / "deploy.yaml"
    cfg: dict[str, Any] = {}
    if deploy_yaml.is_file():
        import yaml

        with deploy_yaml.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    metadata = cfg.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{deploy_yaml}: metadata must be an object")
    metadata = dict(metadata)
    policy_name = str(cfg.get("policy_name", policy_dir.name))
    metadata["policy_name"] = policy_name
    server_section = cfg.get("server", {})
    if not isinstance(server_section, dict):
        raise ValueError(f"{deploy_yaml}: server must be an object")
    metadata["session_mode"] = normalize_session_mode(
        server_section.get("session_mode")
    )
    pairs = schema_pairs_from_metadata(metadata)
    registry = ContractRegistry.builtin()
    registry.validate_references()
    for pair in pairs:
        if pair.observation_schema not in registry.observations:
            raise ValueError(
                f"{deploy_yaml}: unknown observation schema {pair.observation_schema!r}"
            )
        if pair.action_schema not in registry.actions:
            raise ValueError(
                f"{deploy_yaml}: unknown action schema {pair.action_schema!r}"
            )

    model_path = policy_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.py in {policy_dir}")

    model_section = cfg.get("model", {})
    model_class_name = (
        model_section.get("class", "Model")
        if isinstance(model_section, dict)
        else "Model"
    )
    model_cfg = model_section.get("cfg", {}) if isinstance(model_section, dict) else {}
    if not isinstance(model_cfg, dict):
        raise ValueError(f"{deploy_yaml}: model.cfg must be an object")
    return PolicyDefinition(
        policy_name=policy_name,
        model_path=model_path,
        model_class_name=model_class_name,
        model_cfg=model_cfg,
        metadata=metadata,
    )


def load_policy_model(policy: str | Path) -> tuple[PolicyModel, dict[str, Any]]:
    """Dynamically import and instantiate a model from a policy directory.

    Returns (model_instance, metadata_dict).
    """
    definition = load_policy_definition(policy)
    model_path = definition.model_path
    spec = importlib.util.spec_from_file_location(
        f"policy_space_policy.{definition.policy_name}.model", model_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    model_class = getattr(module, definition.model_class_name, None)
    if model_class is None:
        raise AttributeError(
            f"{model_path}: must define a class named {definition.model_class_name!r}"
        )
    model = model_class(definition.model_cfg)
    runtime_metadata = model.runtime_metadata()
    if not isinstance(runtime_metadata, dict):
        raise TypeError(
            f"{model_path}: runtime_metadata() must return a dict, "
            f"got {type(runtime_metadata).__name__}"
        )
    metadata = dict(definition.metadata)
    overlapping_keys = sorted(metadata.keys() & runtime_metadata.keys())
    if overlapping_keys:
        raise ValueError(
            f"{model_path}: runtime_metadata() cannot override static metadata keys: {overlapping_keys}"
        )
    metadata.update(runtime_metadata)
    return model, metadata


def main() -> None:
    import logging
    import time

    parser = argparse.ArgumentParser(description="PolicySpace model server")
    parser.add_argument("--policy", required=True, help="Policy directory name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Bind the listening socket *before* loading the model. The forwarded port
    # range lives inside the kernel's ephemeral range, so an unbound port can be
    # grabbed as the source port of an outbound connection while the model loads
    # (which takes ~10 min) -- leaving us with EADDRINUSE after all that work.
    sock = socket.create_server((args.host, args.port))
    print(f"[PolicySpaceServer] bound ws://{args.host}:{args.port}")

    print(f"[PolicySpaceServer] loading policy: {args.policy}")
    t0 = time.time()
    try:
        model, metadata = load_policy_model(args.policy)
    except BaseException:
        sock.close()
        raise
    print(f"[PolicySpaceServer] model loaded in {time.time() - t0:.1f}s")
    print(f"[PolicySpaceServer] metadata: {metadata}")

    server = PolicySpaceServer(
        model, host=args.host, port=args.port, metadata=metadata, sock=sock
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PolicySpaceServer] shutting down")
    finally:
        model.close()


if __name__ == "__main__":
    main()
