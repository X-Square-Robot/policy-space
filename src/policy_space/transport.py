"""Small transport helpers shared by standalone policy adapters."""

from __future__ import annotations

import socket
import threading
from typing import Any

import numpy as np

_MAX_ARRAY_NDIM = 16


def _numeric_dtype(value: Any) -> np.dtype[Any] | None:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        return None
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufc":
        return None
    return dtype


def _numeric_array(data: Any, dtype_value: Any, shape_value: Any) -> Any:
    dtype = _numeric_dtype(dtype_value)
    if dtype is None or not isinstance(data, (bytes, bytearray, memoryview)):
        return None
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) > _MAX_ARRAY_NDIM:
        return None
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in shape_value
    ):
        return None
    shape = tuple(shape_value)
    element_count = 1
    for width in shape:
        element_count *= width
    if int(dtype.itemsize) * element_count != len(data):
        return None
    try:
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    except (TypeError, ValueError):
        return None


def unpack_ndarray(value: dict[Any, Any]) -> Any:
    """Decode canonical and legacy numeric ndarray msgpack envelopes."""

    if b"__ndarray__" in value:
        decoded = _numeric_array(
            value.get(b"data"), value.get(b"dtype"), value.get(b"shape")
        )
        return value if decoded is None else decoded
    if b"__npgeneric__" in value:
        dtype = _numeric_dtype(value.get(b"dtype"))
        if dtype is None:
            return value
        try:
            return dtype.type(value[b"data"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return value
    if b"nd" not in value or value.get(b"kind") in (b"O", b"V"):
        return value
    try:
        dtype = _numeric_dtype(value[b"type"])
        if dtype is None:
            return value
        if value[b"nd"] is True:
            decoded = _numeric_array(value.get(b"data"), dtype, value.get(b"shape"))
            return value if decoded is None else decoded
        if value[b"nd"] is False:
            data = value[b"data"]
            if (
                not isinstance(data, (bytes, bytearray, memoryview))
                or len(data) != dtype.itemsize
            ):
                return value
            return np.frombuffer(data, dtype=dtype)[0]
    except (KeyError, TypeError, ValueError):
        return value
    return value


def abort_websocket(websocket: Any) -> None:
    close_socket = getattr(websocket, "close_socket", None)
    if callable(close_socket):
        try:
            close_socket()
        except Exception:
            pass
        return
    raw_socket = getattr(websocket, "socket", None)
    if raw_socket is not None:
        try:
            raw_socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            raw_socket.close()
        except Exception:
            pass


def send_websocket_bounded(
    websocket: Any,
    payload: bytes,
    *,
    timeout_seconds: float | None,
    operation: str = "Policy Space",
) -> None:
    """Bound a synchronous WebSocket send that has no native deadline."""

    timeout = 10.0 if timeout_seconds is None else max(0.0, float(timeout_seconds))
    completed = threading.Event()
    outcome: list[BaseException] = []

    def send() -> None:
        try:
            websocket.send(payload)
        except BaseException as exc:  # noqa: BLE001 - replay transport failures
            outcome.append(exc)
        finally:
            completed.set()

    threading.Thread(target=send, name="policy-space-send", daemon=True).start()
    if not completed.wait(timeout):
        abort_websocket(websocket)
        raise TimeoutError(f"{operation} send timed out after {timeout:g} seconds")
    if outcome:
        raise outcome[0]


__all__ = ["abort_websocket", "send_websocket_bounded", "unpack_ndarray"]
