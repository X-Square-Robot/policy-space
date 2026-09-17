"""Test-only PolicySpace gateway for the XReal-to-Factory delivery smoke.

This server deliberately does not load DreamZero. It exposes the exact locked
X2Robot metadata and returns finite 24x14 zero-action chunks so the delivery
path can prove that a Factory workload reaches a public PolicySpace endpoint.
"""

from __future__ import annotations

import argparse
from typing import Any

import msgpack
from websockets.sync.server import ServerConnection, serve

METADATA: dict[str, object] = {
    "server": "policy_space",
    "version": 2,
    "policy_name": "dreamzero_x2robot",
    "protocol": "policy_space",
    "protocol_version": "2",
    "server_instance_id": "xreal-dummy-gateway",
    "session_mode": "exclusive",
    "embodiment": "ArtiXon Arm-6A",
    "action_dim": 14,
    "action_mode": "delta_ee",
    "action_horizon": 24,
}
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def zero_action_chunk() -> list[list[float]]:
    return [[0.0] * 14 for _ in range(24)]


def dispatch(endpoint: object, _payload: dict[object, object]) -> dict[str, object]:
    if endpoint in {"initialize_episode", "ingest_observation", "finalize_episode"}:
        return {"ok": True}
    if endpoint == "infer_actions":
        return {"actions": zero_action_chunk()}
    raise ValueError(f"Unknown endpoint: {endpoint!r}")


def handle_client(ws: ServerConnection) -> None:
    ws.send(msgpack.packb(METADATA))
    for raw in ws:
        if not isinstance(raw, bytes):
            ws.send("ERROR: request must be msgpack bytes")
            continue
        try:
            request: Any = msgpack.unpackb(raw, raw=False)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            endpoint = request.pop("endpoint", None)
            response = dispatch(endpoint, request)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - protocol errors are returned to the smoke client
            ws.send(f"ERROR: {type(exc).__name__}: {exc}")
            continue
        ws.send(msgpack.packb(response))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-only XReal PolicySpace dummy gateway"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    print(f"[xreal-dummy] listening on ws://{args.host}:{args.port}", flush=True)
    with serve(
        handle_client,
        args.host,
        args.port,
        max_size=MAX_MESSAGE_BYTES,
        compression=None,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
