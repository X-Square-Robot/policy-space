"""Customer-facing commands for creating, checking, and serving policies."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from policy_space.conformance import (
    make_observation_fixture,
    normalize_action_output,
    validate_observation_input,
)
from policy_space.contracts import schema_pairs_from_metadata
from policy_space.contracts.registry import ContractRegistry
from policy_space.protocol import PROTOCOL_VERSION
from policy_space.runtime.server import PolicySpaceServer, load_policy_model

_MODEL_TEMPLATE = '''"""Policy Space model adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_space import PolicyModel


class Model(PolicyModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.action_dim = int(cfg.get("action_dim", 1))

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self._observation = obs

    def infer_actions(self) -> np.ndarray:
        return np.zeros((1, self.action_dim), dtype=np.float32)
'''


def _init_policy(args: argparse.Namespace) -> int:
    registry = ContractRegistry.builtin()
    if args.observation not in registry.observations:
        raise ValueError(f"Unknown observation schema: {args.observation}")
    if args.action not in registry.actions:
        raise ValueError(f"Unknown action schema: {args.action}")
    action_dim = int(registry.actions[args.action]["action_dim"])
    catalog_root = Path(args.root).expanduser().resolve()
    if args.family:
        family = Path(args.family)
        if family.name != args.family or args.family in {".", ".."}:
            raise ValueError("Policy family must be one directory name")
        catalog_root = catalog_root / family
    policy_dir = catalog_root / args.name
    policy_dir.mkdir(parents=True, exist_ok=False)
    deploy = {
        "policy_name": args.name,
        "model": {"class": "Model", "cfg": {"action_dim": action_dim}},
        "server": {
            "host": "127.0.0.1",
            "port": 8001,
            "session_mode": "exclusive",
            "max_connections": 32,
            "max_sessions": 64,
            "max_requests_per_minute": 3600,
            "max_message_size": 67108864,
            "max_payload_depth": 32,
        },
        "metadata": {
            "protocol": "policy_space",
            "protocol_version": str(PROTOCOL_VERSION),
            "supported_schema_pairs": [
                {
                    "observation_schema": args.observation,
                    "action_schema": args.action,
                    "embodiment_constraints": [],
                }
            ],
        },
    }
    (policy_dir / "model.py").write_text(_MODEL_TEMPLATE, encoding="utf-8")
    (policy_dir / "deploy.yaml").write_text(
        yaml.safe_dump(deploy, sort_keys=False), encoding="utf-8"
    )
    (policy_dir / "requirements.txt").write_text("numpy>=1.24\n", encoding="utf-8")
    (policy_dir / "README.md").write_text(
        f"# {args.name}\n\nImplement the lifecycle in `model.py`, then run `policy-space check deploy.yaml`.\n",
        encoding="utf-8",
    )
    print(policy_dir)
    return 0


def _validate_actions(value: object) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("actions")
    actions = np.asarray(value)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
        raise ValueError(
            f"Policy output must have shape [T, D] with T,D > 0; got {actions.shape}"
        )
    if not np.issubdtype(actions.dtype, np.number):
        raise ValueError(f"Policy output must be numeric; got {actions.dtype}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy output contains NaN or infinity")
    return actions


def _check_policy(args: argparse.Namespace) -> int:
    model, metadata = load_policy_model(args.deploy)
    pairs = schema_pairs_from_metadata(metadata)
    registry = ContractRegistry.builtin()
    registry.validate_references()
    requested = [args.observation, args.action, args.embodiment]
    if any(requested) and not all(requested):
        raise ValueError(
            "--observation, --action, and --embodiment must be provided together"
        )
    if all(requested):
        matches = [
            pair
            for pair in pairs
            if pair.observation_schema == args.observation
            and pair.action_schema == args.action
            and pair.supports_embodiment(args.embodiment)
        ]
        if len(matches) != 1:
            raise ValueError("requested schema pair is not uniquely supported")
        pair = matches[0]
        embodiment = args.embodiment
    else:
        if len(pairs) != 1:
            raise ValueError(
                "multiple schema pairs are declared; select one with "
                "--observation, --action, and --embodiment"
            )
        pair = pairs[0]
        constraints = pair.embodiment_constraints
        if len(constraints) > 1:
            raise ValueError("select one declared embodiment with --embodiment")
        embodiment = constraints[0] if constraints else ""
    if pair.observation_schema not in registry.observations:
        raise ValueError(f"Unknown observation schema: {pair.observation_schema}")
    if pair.action_schema not in registry.actions:
        raise ValueError(f"Unknown action schema: {pair.action_schema}")
    observation_contract = registry.observations[pair.observation_schema]
    observation = make_observation_fixture(observation_contract)
    validate_observation_input(observation, observation_contract)
    initialization = {
        "episode_id": "policy-space-check",
        "observation_schema": pair.observation_schema,
        "action_schema": pair.action_schema,
    }
    if embodiment:
        initialization["embodiment_schema"] = embodiment
    lifecycle: list[str] = []
    try:
        model.initialize_episode(initialization)
        lifecycle.append("initialize_episode")
        model.ingest_observation(observation)
        lifecycle.append("ingest_observation")
        response = normalize_action_output(
            model.infer_actions(),
            registry.actions[pair.action_schema],
            expected_horizon=metadata.get("action_horizon"),
        )
        response_key = str(
            registry.actions[pair.action_schema].get("response_key") or "actions"
        )
        actions = _validate_actions(response[response_key])
        lifecycle.append("infer_actions")
        model.finalize_episode({"episode_id": "policy-space-check"})
        lifecycle.append("finalize_episode")
    finally:
        model.close()
    report = {
        "status": "passed",
        "policy_name": metadata.get("policy_name"),
        "protocol_version": str(metadata.get("protocol_version", PROTOCOL_VERSION)),
        "session_mode": metadata.get("session_mode"),
        "schema_pair": {
            "observation": pair.observation_schema,
            "action": pair.action_schema,
            "embodiment": embodiment or None,
        },
        "action": {"shape": list(actions.shape), "dtype": str(actions.dtype)},
        "lifecycle": lifecycle,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _serve_policy(args: argparse.Namespace) -> int:
    deploy = Path(args.deploy).expanduser().resolve()
    document = yaml.safe_load(deploy.read_text(encoding="utf-8")) or {}
    server_cfg = document.get("server") or {}
    host = args.host or str(server_cfg.get("host", "127.0.0.1"))
    port = args.port if args.port is not None else int(server_cfg.get("port", 8001))
    sock = socket.create_server((host, port))
    try:
        model, metadata = load_policy_model(deploy)
    except BaseException:
        sock.close()
        raise
    server = PolicySpaceServer(
        model,
        host=host,
        port=port,
        metadata=metadata,
        sock=sock,
        max_message_size=int(server_cfg.get("max_message_size", 64 * 1024 * 1024)),
        max_connections=int(server_cfg.get("max_connections", 32)),
        max_sessions=int(server_cfg.get("max_sessions", 64)),
        max_requests_per_minute=int(server_cfg.get("max_requests_per_minute", 3600)),
        max_payload_depth=int(server_cfg.get("max_payload_depth", 32)),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        model.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-space")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="create a four-file policy template"
    )
    init_parser.add_argument("name")
    init_parser.add_argument("--root", default="policies")
    init_parser.add_argument("--family")
    init_parser.add_argument("--observation", required=True)
    init_parser.add_argument("--action", required=True)
    init_parser.set_defaults(handler=_init_policy)

    check_parser = subparsers.add_parser(
        "check", help="run a local lifecycle and action check"
    )
    check_parser.add_argument("deploy")
    check_parser.add_argument("--observation")
    check_parser.add_argument("--action")
    check_parser.add_argument("--embodiment")
    check_parser.set_defaults(handler=_check_policy)

    serve_parser = subparsers.add_parser("serve", help="start a Policy Space service")
    serve_parser.add_argument("deploy")
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    serve_parser.set_defaults(handler=_serve_policy)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
