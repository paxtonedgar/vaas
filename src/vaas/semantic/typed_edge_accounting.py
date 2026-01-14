"""Typed edge accounting helpers."""

from __future__ import annotations

from hashlib import sha256
from typing import Dict, List, Tuple

import pandas as pd

from vaas.graph.edges import Edge

CLAIMABLE_RULE_CLASSES = {
    "GATING",
    "PROHIBITION",
    "FALLBACK",
    "POPULATION",
}

CLAIMABLE_EDGE_TYPES = {
    "gated_by",
    "excludes",
    "fallback_include",
    "applies_if",
    "defines",
    "qualifies",
    "portion_of",
}

TYPED_EDGE_COLUMNS = [
    "typed_edge_key",
    "doc_id",
    "edge_id",
    "source_node_id",
    "target_node_id",
    "edge_type",
    "rule_class",
    "precedence",
    "pattern_matched",
    "polarity",
    "created_by",
    "source_element_id",
    "sentence_idx",
    "char_start",
    "char_end",
    "claimable",
    "claimable_reason",
    "missing_required_fields",
]


def make_typed_edge_key(doc_id: str, edge: Edge) -> str:
    """Create a stable key for a typed edge."""
    parts = [
        doc_id or "",
        edge.source_node_id or "",
        edge.target_node_id or "",
        edge.edge_type or "",
        edge.rule_class or "",
        str(edge.precedence) if edge.precedence is not None else "",
        str(edge.source_element_id) if edge.source_element_id is not None else "",
        str(edge.evidence_sentence_idx) if edge.evidence_sentence_idx is not None else "",
        str(edge.evidence_char_start) if edge.evidence_char_start is not None else "",
        str(edge.evidence_char_end) if edge.evidence_char_end is not None else "",
        edge.polarity or "",
    ]
    payload = "|".join(parts)
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return f"ted_{digest[:16]}"


def is_claimable(edge: Edge) -> Tuple[bool, str]:
    """Determine if an edge should emit a claim."""
    if not edge.source_element_id:
        return False, "missing_source_element"
    if edge.evidence_sentence_idx is None:
        return False, "missing_sentence_idx"
    if edge.evidence_char_start is None or edge.evidence_char_end is None:
        return False, "missing_char_span"
    if edge.rule_class in CLAIMABLE_RULE_CLASSES:
        return True, "rule_class"
    if edge.edge_type in CLAIMABLE_EDGE_TYPES:
        return True, "edge_type"
    return False, "non_semantic_edge"


def get_missing_required_fields(edge: Edge) -> List[str]:
    missing: List[str] = []
    if not edge.source_element_id:
        missing.append("source_element_id")
    if edge.evidence_sentence_idx is None:
        missing.append("sentence_idx")
    if edge.evidence_char_start is None or edge.evidence_char_end is None:
        missing.append("char_span")
    if not edge.pattern_matched:
        missing.append("pattern_id")
    if not edge.polarity:
        missing.append("polarity")
    if not edge.rule_class:
        missing.append("rule_class")
    return missing


def build_typed_edge_dataframe(doc_id: str, typed_edges: List[Edge]) -> pd.DataFrame:
    """Materialize typed edge records with claimable metadata."""
    rows: List[Dict[str, object]] = []
    for edge in typed_edges:
        key = make_typed_edge_key(doc_id, edge)
        claimable, reason = is_claimable(edge)
        rows.append({
            "typed_edge_key": key,
            "doc_id": doc_id,
            "edge_id": edge.edge_id,
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
            "edge_type": edge.edge_type,
            "rule_class": edge.rule_class,
            "precedence": edge.precedence,
            "pattern_matched": edge.pattern_matched,
            "polarity": edge.polarity,
            "created_by": edge.created_by,
            "source_element_id": edge.source_element_id,
            "sentence_idx": edge.evidence_sentence_idx,
            "char_start": edge.evidence_char_start,
            "char_end": edge.evidence_char_end,
            "claimable": claimable,
            "claimable_reason": reason,
            "missing_required_fields": get_missing_required_fields(edge),
        })
    if rows:
        return pd.DataFrame(rows, columns=TYPED_EDGE_COLUMNS)
    return pd.DataFrame(columns=TYPED_EDGE_COLUMNS)


__all__ = [
    "build_typed_edge_dataframe",
    "make_typed_edge_key",
    "is_claimable",
    "CLAIMABLE_EDGE_TYPES",
    "CLAIMABLE_RULE_CLASSES",
]
