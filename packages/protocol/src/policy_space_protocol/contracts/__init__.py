"""Versioned public contract selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SchemaPair:
    """One compatible public observation/action representation."""

    observation_schema: str
    action_schema: str
    embodiment_constraints: tuple[str, ...] = ()

    def supports_embodiment(self, embodiment: str) -> bool:
        return not self.embodiment_constraints or embodiment in self.embodiment_constraints

    @classmethod
    def from_mapping(cls, value: Any) -> "SchemaPair":
        if not isinstance(value, dict):
            raise ValueError("schema pair must be an object")
        observation = str(value.get("observation_schema") or "").strip()
        action = str(value.get("action_schema") or "").strip()
        constraints = value.get("embodiment_constraints") or []
        if not observation or not action:
            raise ValueError("schema pair requires observation_schema and action_schema")
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) and item.strip() for item in constraints
        ):
            raise ValueError("embodiment_constraints must be a list of non-empty strings")
        return cls(observation, action, tuple(constraints))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "embodiment_constraints": list(self.embodiment_constraints),
        }


def schema_pairs_from_metadata(metadata: dict[str, Any]) -> list[SchemaPair]:
    raw_pairs = metadata.get("supported_schema_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("metadata requires a non-empty supported_schema_pairs list")
    pairs = [SchemaPair.from_mapping(value) for value in raw_pairs]
    identities = [
        (pair.observation_schema, pair.action_schema, pair.embodiment_constraints)
        for pair in pairs
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("supported_schema_pairs contains duplicates")
    return pairs


def select_schema_pair(
    policy_pairs: Iterable[SchemaPair],
    bridge_pairs: Iterable[SchemaPair],
    *,
    embodiment: str,
) -> SchemaPair:
    """Select exactly one schema pair shared by a policy and environment bridge."""

    bridge_pairs = list(bridge_pairs)
    matches = [
        pair
        for pair in policy_pairs
        if any(
            pair.observation_schema == bridge.observation_schema
            and pair.action_schema == bridge.action_schema
            and bridge.supports_embodiment(embodiment)
            for bridge in bridge_pairs
        )
        and pair.supports_embodiment(embodiment)
    ]
    if not matches:
        raise ValueError(f"No compatible schema pair satisfies embodiment {embodiment!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Contract negotiation selected multiple schema pairs for embodiment {embodiment!r}"
        )
    return matches[0]


__all__ = ["SchemaPair", "schema_pairs_from_metadata", "select_schema_pair"]
