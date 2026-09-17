"""Compatibility imports for the standalone protocol package."""

from policy_space_protocol.protocol import (
    PROTOCOL_VERSION,
    SESSION_MODE_EXCLUSIVE,
    SESSION_MODE_MULTIPLEXED_SERIAL,
    SESSION_MODES,
    SESSION_OWNER_BUSY_MARKER,
    V2_METADATA,
    metadata_mismatches,
    normalize_session_mode,
    validate_server_metadata,
)

__all__ = [
    "PROTOCOL_VERSION",
    "SESSION_MODE_EXCLUSIVE",
    "SESSION_MODE_MULTIPLEXED_SERIAL",
    "SESSION_MODES",
    "SESSION_OWNER_BUSY_MARKER",
    "V2_METADATA",
    "metadata_mismatches",
    "normalize_session_mode",
    "validate_server_metadata",
]
