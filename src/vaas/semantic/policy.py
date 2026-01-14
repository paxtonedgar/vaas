"""Policy defaults for semantic gating and precedence behavior."""

from __future__ import annotations

from typing import Dict


DEFAULT_PRECEDENCE_SENTENCE_DISTANCE = 2
DEFAULT_PRECEDENCE_SCOPE = "paragraph"

# Predicate-specific overrides. Keys are lower-case predicate strings.
PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE: Dict[str, int] = {
    "excludes": 1,
    "applies_if": 2,
    "defines": 2,
    "qualifies": 2,
    "requires": 2,
    "aggregates": 2,
    "subset_of": 2,
    "includes": 2,
    "portion_of": 2,
    "gated_by": 1,
    "fallback_include": 1,
}

PRECEDENCE_SCOPE_BY_PREDICATE: Dict[str, str] = {
    "defines": "anchor",
    "excludes": "anchor",
}


def precedence_sentence_distance_limit(predicate: str) -> int:
    """Return the sentence distance limit for a predicate."""
    if not predicate:
        return DEFAULT_PRECEDENCE_SENTENCE_DISTANCE
    key = str(predicate).strip().lower()
    return PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE.get(
        key,
        DEFAULT_PRECEDENCE_SENTENCE_DISTANCE,
    )


def precedence_scope_preference(predicate: str) -> str:
    if not predicate:
        return DEFAULT_PRECEDENCE_SCOPE
    key = str(predicate).strip().lower()
    return PRECEDENCE_SCOPE_BY_PREDICATE.get(key, DEFAULT_PRECEDENCE_SCOPE)


__all__ = [
    "DEFAULT_PRECEDENCE_SENTENCE_DISTANCE",
    "PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE",
    "precedence_sentence_distance_limit",
    "DEFAULT_PRECEDENCE_SCOPE",
    "PRECEDENCE_SCOPE_BY_PREDICATE",
    "precedence_scope_preference",
]
