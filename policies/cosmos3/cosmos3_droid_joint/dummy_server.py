#!/usr/bin/env python3
"""Zero-GPU dummy Cosmos3 RoboLab policy server for protocol-layer verification.

The real ``cosmos_framework.scripts.action_policy_server_robolab`` hard-requires
CUDA + the 16B world model and is deployed via Docker/uv, so it cannot run here
cheaply. This dummy speaks the **exact same wire protocol** (OpenPI
``WebsocketPolicyServer`` = ``msgpack_numpy`` over ``websockets``) so we can
verify ``policy_space/cosmos3/policy.py``'s request/response translation without
a GPU:

  * on connect -> sends an EMPTY metadata dict ``{}`` (matches the real server's
    ``WebsocketPolicyServer(..., metadata={})``);
  * each request -> unpacked, its keys/prompt logged, lightly validated the same
    way the real server validates (needs a prompt + an image + joint_position +
    gripper_position), then answered with ``{"action": zeros[(chunk, dim)]}``.

It intentionally does NOT run the model, so returned actions are zeros — this
verifies the transport + field names + shapes, NOT the action semantics.

Run it in an env that has ``openpi-client`` + ``websockets`` (the ``dreamzero``
conda env already does)::

    /path/to/dreamzero/bin/python policy_space/cosmos3/dummy_server.py --port 8000

Then point the policy at it via COSMOS3_SERVER_PORT and run smoke_predict.py.
"""

from __future__ import annotations

import argparse

import numpy as np
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

# Same defaults as RobolabServerArgs (verified from the real server source).
_DEFAULT_ACTION_CHUNK_SIZE = 32
_DEFAULT_ACTION_DIM = 8

_REQUIRED_IMAGE_ANY = ("observation/image",)
_REQUIRED_TRIPLET = (
    "observation/wrist_image_left",
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
)


def _validate_like_real_server(obs: dict) -> None:
    """Mirror the real server's request validation so the dummy rejects the
    same malformed requests (helps catch client bugs early)."""
    if not isinstance(obs.get("prompt"), str):
        raise ValueError("'prompt' must be a string")
    has_composed = any(k in obs for k in _REQUIRED_IMAGE_ANY)
    has_triplet = all(k in obs for k in _REQUIRED_TRIPLET)
    if not (has_composed or has_triplet):
        raise ValueError(
            "Observation must contain 'observation/image' or the RoBoArena "
            "wrist/exterior image triplet"
        )
    if "observation/joint_position" not in obs:
        raise ValueError("missing 'observation/joint_position'")
    jp = np.asarray(obs["observation/joint_position"])
    if jp.reshape(-1).shape[0] % 7 != 0:
        raise ValueError(
            f"'observation/joint_position' width must be 7, got shape {jp.shape}"
        )
    if "observation/gripper_position" not in obs:
        raise ValueError("missing 'observation/gripper_position'")


def make_handler(chunk: int, dim: int, decode_video: bool):
    packer = msgpack_numpy.Packer()

    def handler(websocket) -> None:
        # 1) Handshake: send empty metadata exactly like the real server.
        websocket.send(packer.pack({}))
        print("[dummy-cosmos3] client connected; sent empty metadata {}")
        # 2) Inference loop. A client disconnecting at process exit is normal.
        try:
            for message in websocket:
                obs = msgpack_numpy.unpackb(message)
                keys = sorted(obs.keys())
                prompt = obs.get("prompt")
                img_keys = [
                    k for k in keys if k.startswith("observation/") and "image" in k
                ]
                print(
                    f"[dummy-cosmos3] infer: prompt={prompt!r} image_keys={img_keys} all_keys={keys}"
                )
                try:
                    _validate_like_real_server(obs)
                except ValueError as exc:
                    # The real server would raise; surface it as an error string so
                    # the client's WebsocketClientPolicy raises RuntimeError.
                    websocket.send(f"dummy-cosmos3 validation error: {exc}")
                    continue
                action = np.zeros((chunk, dim), dtype=np.float32)
                outputs = {"action": action}
                if decode_video:
                    outputs["video"] = np.zeros((chunk, 8, 8, 3), dtype=np.uint8)
                websocket.send(packer.pack(outputs))
        except ConnectionClosed:
            print("[dummy-cosmos3] client disconnected")

    return handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--action-chunk-size", type=int, default=_DEFAULT_ACTION_CHUNK_SIZE
    )
    parser.add_argument("--action-dim", type=int, default=_DEFAULT_ACTION_DIM)
    parser.add_argument(
        "--decode-video", action="store_true", help="also return a dummy 'video' array"
    )
    args = parser.parse_args()

    handler = make_handler(args.action_chunk_size, args.action_dim, args.decode_video)
    print(
        f"[dummy-cosmos3] serving ws://{args.host}:{args.port}  "
        f"(action chunk={args.action_chunk_size}, dim={args.action_dim}, "
        f"decode_video={args.decode_video})"
    )
    # max_size=None + compression=None mirror OpenPI's WebsocketPolicyServer so
    # multi-image observation frames (>1 MiB) are not rejected as "too big".
    with serve(
        handler, args.host, args.port, max_size=None, compression=None
    ) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
