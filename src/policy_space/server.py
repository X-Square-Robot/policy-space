"""Compatibility module for ``python -m policy_space.server``."""

from policy_space.runtime.server import *  # noqa: F403
from policy_space.runtime.server import (
    _ORIGINAL_PACKER,
    _ORIGINAL_UNPACKER,
    _pack_array,
    _safe_packb,
    _safe_unpackb,
    _unpack_array,
    main,
)

if __name__ == "__main__":
    main()
