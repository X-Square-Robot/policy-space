import base64

import cv2
import numpy as np
import pytest

from policies.wallx.wall_oss_05_artixon_arm_6a_eef.model import Model, _encode_image_b64


def _decode_rgb(payload: str) -> np.ndarray:
    encoded = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert bgr is not None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_lossless_png_round_trip_is_pixel_exact():
    rng = np.random.default_rng(2602)
    rgb = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)

    payload = _encode_image_b64(rgb)

    assert payload is not None
    assert base64.b64decode(payload).startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(_decode_rgb(payload), rgb)


def test_build_envelope_uses_configured_lossless_png():
    model = object.__new__(Model)
    model.image_wire_encoding = "lossless_png"
    model.jpeg_quality = 95
    model.instruction = "fallback"
    rgb = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)

    envelope = model._build_envelope(
        {
            "images": {"camera_front": rgb},
            "state": {"follow1_pos": np.zeros(7, dtype=np.float32)},
            "instruction": "test instruction",
        }
    )

    payload = envelope["views"]["camera_front"]
    assert base64.b64decode(payload).startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(_decode_rgb(payload), rgb)


def test_unknown_wire_encoding_is_rejected():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="image_wire_encoding"):
        _encode_image_b64(rgb, wire_encoding="unknown")
