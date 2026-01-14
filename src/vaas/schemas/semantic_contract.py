"""
Semantic KG contract definitions.

These dataclasses describe the canonical row layout for semKG v1 artifacts.
Keeping them centralized allows emitters, audits, and downstream consumers to
share a single source of truth for required fields and schema evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

SchemaVersion = "semkg_v1"


@dataclass
class ClaimRow:
    """Row emitted for each semantic claim extracted from instructions."""

    claim_id: str
    doc_id: str
    scope_node_id: str
    source_element_id: str
    sentence_idx: int
    char_start: int
    char_end: int
    evidence_text: str
    pattern_id: str
    predicate: str
    polarity: str
    rule_class: str
    created_by: str = "typed_edges"
    precedence_rank: Optional[int] = None
    subject_canonical_id: Optional[str] = None
    object_canonical_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Convert to serialisable dict for DataFrame construction."""
        return asdict(self)


@dataclass
class AuthorityRow:
    """Normalized authority reference emitted from references and citations."""

    authority_id: str
    authority_type: str
    label: str
    citation: str
    raw_text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ConstraintRow:
    """Typed constraint extracted from a sentence."""

    constraint_id: str
    constraint_type: str
    doc_id: str
    source_element_id: str
    sentence_idx: int
    raw_text: str
    comparator: Optional[str] = None
    unit: Optional[str] = None
    anchor_event: Optional[str] = None
    direction: Optional[str] = None
    value_int: Optional[int] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    inclusive_min: Optional[bool] = None
    inclusive_max: Optional[bool] = None
    amount_value: Optional[int] = None
    amount_currency: Optional[str] = None
    percent_value: Optional[float] = None
    percent_basis: Optional[str] = None
    date_year: Optional[int] = None
    date_month: Optional[str] = None
    date_day: Optional[int] = None
    atom_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ClaimAuthorityRow:
    """Join table connecting claims to the authorities that govern them."""

    claim_id: str
    authority_id: str
    relation: str
    evidence_text: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ClaimAuthorityMentionRow:
    """Sentence-scoped authority mention aligned to a claim."""

    claim_id: str
    authority_id: str
    ref_occurrence_id: Optional[str]
    relation: str
    claim_sentence_idx: Optional[int]
    claim_char_start: Optional[int]
    claim_char_end: Optional[int]
    mention_char_start: Optional[int]
    mention_char_end: Optional[int]
    evidence_text: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ClaimPrecedenceRow:
    """Edges that encode precedence or exception relationships between claims."""

    from_claim_id: str
    to_claim_id: str
    relation: str
    scope_struct_node_id: Optional[str] = None
    sentence_distance: Optional[int] = None
    topic_key: Optional[str] = None
    precedence_topic_key: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ResolvedClaimRow:
    """Resolved claim status after precedence resolution."""

    claim_id: str
    precedence_scope_id: str
    group_kind: str
    group_value: str
    topic_key: str
    precedence_topic_key: Optional[str] = None
    rank: int
    status: str
    suppressed_by_claim_id: Optional[str] = None
    reason: Optional[str] = None
    predicate: Optional[str] = None
    subject_canonical_id: Optional[str] = None
    object_canonical_id: Optional[str] = None
    source_element_id: Optional[str] = None
    sentence_idx: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    evidence_text: Optional[str] = None
    missing_provenance_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ResolutionGroupRow:
    """Group-level status for precedence resolution."""

    precedence_scope_id: str
    group_kind: str
    group_value: str
    topic_key: str
    precedence_topic_key: Optional[str] = None
    group_size: int
    status: str
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
    applies_count: int = 0
    suppressed_count: int = 0
    exception_count: int = 0
    undecidable_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CompiledDirectiveRow:
    """Directive IR derived from resolved claims."""

    directive_id: str
    precedence_scope_id: str
    topic_key: str
    precedence_topic_key: Optional[str]
    target_box_id: Optional[str]
    op: str
    condition_ir: str
    supporting_claim_ids: List[str] = field(default_factory=list)
    supporting_spans: List[Dict[str, object]] = field(default_factory=list)
    resolution_reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ResolvedConstraintRow:
    """Resolved constraint derived from resolved claims."""

    constraint_id: str
    directive_id: str
    precedence_scope_id: str
    topic_key: str
    precedence_topic_key: Optional[str]
    target_box_id: Optional[str]
    op: str
    condition_ir: Optional[str]
    supporting_claim_ids: List[str] = field(default_factory=list)
    supporting_spans: List[Dict[str, object]] = field(default_factory=list)
    resolution_reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


REQUIRED_FIELDS_BY_ORIGIN: Dict[str, Sequence[str]] = {
    "typed_edges": [
        "claim_id",
        "doc_id",
        "scope_node_id",
        "source_element_id",
        "sentence_idx",
        "char_start",
        "char_end",
        "pattern_id",
        "predicate",
        "polarity",
        "rule_class",
    ],
}

__all__ = [
    "SchemaVersion",
    "ClaimRow",
    "AuthorityRow",
    "ConstraintRow",
    "ClaimAuthorityRow",
    "ClaimAuthorityMentionRow",
    "ClaimPrecedenceRow",
    "ResolvedClaimRow",
    "ResolutionGroupRow",
    "CompiledDirectiveRow",
    "ResolvedConstraintRow",
    "REQUIRED_FIELDS_BY_ORIGIN",
]
