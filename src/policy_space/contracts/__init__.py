"""Compatibility imports for the standalone protocol package."""

from policy_space_protocol.contracts import (
    SchemaPair,
    schema_pairs_from_metadata,
    select_schema_pair,
)

__all__ = ["SchemaPair", "schema_pairs_from_metadata", "select_schema_pair"]
