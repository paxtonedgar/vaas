"""
Semantic KG quality enforcement utilities.

These helpers validate claim rows and provide structured rejection metadata so
gates can fail deterministically when semantic completeness degrades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from vaas.schemas.semantic_contract import ClaimRow, REQUIRED_FIELDS_BY_ORIGIN


@dataclass
class ClaimRejection:
    """Structured rejection emitted when a claim cannot be materialized."""

    predicate: str
    source_node_id: str
    reason: str
    edge_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "predicate": self.predicate,
            "source_node_id": self.source_node_id,
            "reason": self.reason,
            "edge_id": self.edge_id,
        }


def validate_claim_row(row: ClaimRow) -> Tuple[bool, Optional[str]]:
    """Check that a ClaimRow satisfies required field completeness."""
    required_fields = REQUIRED_FIELDS_BY_ORIGIN.get(row.created_by, [])
    for field_name in required_fields:
        value = getattr(row, field_name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, f"missing_field:{field_name}"

    if row.sentence_idx is None or row.sentence_idx < 0:
        return False, "invalid_sentence_idx"
    if row.char_start is None or row.char_end is None or row.char_start > row.char_end:
        return False, "invalid_char_span"
    if not row.evidence_text:
        return False, "missing_evidence_text"
    return True, None


__all__ = ["validate_claim_row", "ClaimRejection"]
