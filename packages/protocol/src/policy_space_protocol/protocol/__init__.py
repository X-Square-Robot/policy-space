"""Shared Policy Space v2 wire metadata contract."""

from __future__ import annotations

from typing import Any

from policy_space_protocol.contracts import schema_pairs_from_metadata

PROTOCOL_VERSION = 2

V2_METADATA: dict[str, Any] = {
    "server": "policy_space",
    "version": PROTOCOL_VERSION,
    "protocol": "policy_space",
    "protocol_version": str(PROTOCOL_VERSION),
}

SESSION_MODE_EXCLUSIVE = "exclusive"
SESSION_MODE_MULTIPLEXED_SERIAL = "multiplexed_serial"
SESSION_OWNER_BUSY_MARKER = "is already owned by another live client"
SESSION_MODES = frozenset({SESSION_MODE_EXCLUSIVE, SESSION_MODE_MULTIPLEXED_SERIAL})


def normalize_session_mode(value: Any) -> str:
    """Return one supported server session mode, defaulting fail-closed."""
    mode = SESSION_MODE_EXCLUSIVE if value is None else str(value).strip()
    if mode not in SESSION_MODES:
        allowed = ", ".join(sorted(SESSION_MODES))
        raise ValueError(
            f"Policy Space server.session_mode must be one of {allowed}; got {value!r}"
        )
    return mode


def metadata_mismatches(metadata: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return ``(field, expected, actual)`` rows for an invalid v2 handshake."""
    rows = [
        (field, expected, metadata.get(field))
        for field, expected in V2_METADATA.items()
        if metadata.get(field) != expected
    ]
    instance_id = metadata.get("server_instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        rows.append(("server_instance_id", "non-empty string", instance_id))
    session_mode = metadata.get("session_mode")
    if session_mode not in SESSION_MODES:
        rows.append(("session_mode", sorted(SESSION_MODES), session_mode))
    return rows


def validate_server_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate required v2 fields while allowing optional extensions."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    mismatches = metadata_mismatches(metadata)
    if mismatches:
        details = "; ".join(
            f"{field}: expected {expected!r}, got {actual!r}"
            for field, expected, actual in mismatches
        )
        raise ValueError(f"Invalid Policy Space metadata: {details}")
    schema_pairs_from_metadata(metadata)
    return dict(metadata)


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
