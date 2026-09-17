"""Regression tests for PolicySpace's msgpack codec isolation."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from policy_space.runtime.server import _unpack_array


def test_codec_survives_msgpack_numpy_global_patch() -> None:
    """The server must keep its wire format after a policy patches msgpack."""
    pytest.importorskip("msgpack_numpy")

    script = textwrap.dedent(
        """
        import msgpack
        import numpy as np

        from policy_space import server

        original_packer = msgpack.Packer
        original_unpacker = msgpack.Unpacker

        import msgpack_numpy
        msgpack_numpy.patch()

        assert msgpack.Packer is not original_packer
        assert msgpack.Unpacker is not original_unpacker
        assert server._ORIGINAL_PACKER is original_packer
        assert server._ORIGINAL_UNPACKER is original_unpacker

        actions = np.zeros((33, 14), dtype=np.float32)
        payload = server._safe_packb(
            {"actions": actions},
            default=server._pack_array,
        )

        wire = server._safe_unpackb(payload)
        assert b"__ndarray__" in wire["actions"]
        assert b"nd" not in wire["actions"]

        decoded = server._safe_unpackb(payload, object_hook=server._unpack_array)
        assert isinstance(decoded["actions"], np.ndarray)
        assert decoded["actions"].shape == (33, 14)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "payload",
    [
        {
            b"__ndarray__": True,
            b"data": b"x",
            b"dtype": np.dtype(object).str,
            b"shape": (1,),
        },
        {
            b"__ndarray__": True,
            b"data": np.zeros(2, dtype=np.float32).tobytes(),
            b"dtype": np.dtype(np.float32).str,
            b"shape": (3,),
        },
        {
            b"__ndarray__": True,
            b"data": b"",
            b"dtype": np.dtype(np.float32).str,
            b"shape": (1,) * 17,
        },
    ],
)
def test_server_rejects_unsafe_or_malformed_ndarray_envelopes(payload) -> None:
    assert _unpack_array(payload) is payload
