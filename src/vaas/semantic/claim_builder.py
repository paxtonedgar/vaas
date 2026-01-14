"""
Claim assembly helpers.

Converts typed semantic edges into normalized ClaimRow objects with constraint
and canonicalization metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from vaas.graph.edges import Edge
from vaas.schemas.semantic_contract import ClaimRow, ConstraintRow
from vaas.semantic.canonical_ids import (
    Scope,
    canonicalize_box,
    canonicalize_subject_object,
)
from vaas.semantic.claim_id import make_claim_id
from vaas.semantic.constraints import extract_constraints
from vaas.semantic.quality import ClaimRejection, validate_claim_row


def _canonicalize_target(edge: Edge, form_id: str, canonical_map: Dict[str, str]) -> Optional[str]:
    target = edge.target_node_id
    if not target:
        return None
    if canonical_map:
        canonical = canonical_map.get(target)
        if canonical:
            return canonical
    if ":box_" in target:
        box_key = target.split(":")[-1].replace("box_", "")
        return canonicalize_box(f"Box {box_key}", form_id=form_id)
    return target


def build_claim_from_edge(
    edge: Edge,
    doc_id: str,
    form_id: str,
    element_text_map: Dict[str, str],
    sentence_lookup: Dict[Tuple[str, int], Dict[str, object]],
    canonical_node_map: Dict[str, str],
) -> Tuple[Optional[ClaimRow], List[ConstraintRow], Optional[Dict[str, str]], Optional[ClaimRejection]]:
    """Create a ClaimRow + auxiliary rows from a typed edge."""
    predicate = edge.edge_type
    source_node_id = edge.source_node_id or ""
    subject_canonical_id = canonicalize_subject_object(
        source_node_id,
        canonical_node_map,
        Scope(doc_id=doc_id, form_id=form_id, anchor_id=None),
    ) or source_node_id
    object_canonical_id = _canonicalize_target(edge, form_id=form_id, canonical_map=canonical_node_map) or edge.target_node_id
    sentence_idx = edge.evidence_sentence_idx
    element_char_start = edge.evidence_char_start
    element_char_end = edge.evidence_char_end
    pattern_id = edge.pattern_matched
    polarity = edge.polarity
    rule_class = edge.rule_class
    source_element_id = edge.source_element_id

    missing = None
    if source_node_id == "":
        missing = "missing_source_node"
    elif source_element_id is None:
        missing = "missing_source_element"
    elif sentence_idx is None:
        missing = "missing_sentence_idx"
    elif element_char_start is None or element_char_end is None:
        missing = "missing_char_span"
    elif not pattern_id:
        missing = "missing_pattern_id"
    elif not polarity:
        missing = "missing_polarity"
    elif not rule_class:
        missing = "missing_rule_class"

    if missing:
        return None, [], None, ClaimRejection(
            predicate=predicate,
            source_node_id=source_node_id,
            reason=missing,
            edge_id=edge.edge_id,
        )

    element_id_str = str(source_element_id)
    element_text = element_text_map.get(element_id_str)
    if element_text is None:
        return None, [], None, ClaimRejection(
            predicate=predicate,
            source_node_id=source_node_id,
            reason="missing_element_text",
            edge_id=edge.edge_id,
        )

    try:
        element_char_start = int(element_char_start)
        element_char_end = int(element_char_end)
    except Exception:
        return None, [], None, ClaimRejection(
            predicate=predicate,
            source_node_id=source_node_id,
            reason="invalid_char_span",
            edge_id=edge.edge_id,
        )

    if element_char_start < 0 or element_char_end > len(element_text) or element_char_start >= element_char_end:
        return None, [], None, ClaimRejection(
            predicate=predicate,
            source_node_id=source_node_id,
            reason="invalid_span_bounds",
            edge_id=edge.edge_id,
        )

    evidence_text = element_text[element_char_start:element_char_end]
    sentence_char_start = None
    sentence_char_end = None

    sentence_key = (element_id_str, int(sentence_idx)) if sentence_idx is not None else None
    if sentence_key and sentence_key in sentence_lookup:
        info = sentence_lookup[sentence_key]
        sent_start = info.get("sentence_char_start")
        sent_end = info.get("sentence_char_end")
        try:
            sent_start = int(sent_start)
            sent_end = int(sent_end)
        except Exception:
            sent_start = None
            sent_end = None
        if sent_start is None or sent_end is None:
            return None, [], None, ClaimRejection(
                predicate=predicate,
                source_node_id=source_node_id,
                reason="missing_sentence_index",
                edge_id=edge.edge_id,
            )
        if element_char_start < sent_start or element_char_end > sent_end:
            return None, [], None, ClaimRejection(
                predicate=predicate,
                source_node_id=source_node_id,
                reason="sentence_span_mismatch",
                edge_id=edge.edge_id,
            )
        sentence_char_start = max(0, element_char_start - sent_start)
        span_len = element_char_end - element_char_start
        sentence_char_end = sentence_char_start + span_len
    else:
        return None, [], None, ClaimRejection(
            predicate=predicate,
            source_node_id=source_node_id,
            reason="missing_sentence_index",
            edge_id=edge.edge_id,
        )

    char_start = sentence_char_start
    char_end = sentence_char_end

    constraint_rows = extract_constraints(
        doc_id=doc_id,
        source_element_id=element_id_str,
        sentence_idx=sentence_idx,
        sentence=evidence_text,
    )
    claim_id = make_claim_id(
        doc_id=doc_id,
        source_element_id=str(source_element_id),
        sentence_idx=sentence_idx,
        char_start=sentence_char_start,
        char_end=sentence_char_end,
        predicate=predicate,
        subject_canonical_id=subject_canonical_id or "",
        object_canonical_id=object_canonical_id or "",
        polarity=polarity,
    )

    claim = ClaimRow(
        claim_id=claim_id,
        doc_id=doc_id,
        scope_node_id=source_node_id,
        source_element_id=str(source_element_id),
        sentence_idx=sentence_idx,
        char_start=char_start,
        char_end=char_end,
        evidence_text=evidence_text,
        pattern_id=pattern_id,
        predicate=predicate,
        polarity=polarity,
        rule_class=rule_class,
        created_by=edge.created_by,
        precedence_rank=edge.precedence,
        subject_canonical_id=subject_canonical_id,
        object_canonical_id=object_canonical_id,
    )

    ok, reason = validate_claim_row(claim)
    if not ok:
        return None, constraint_rows, None, ClaimRejection(
            predicate=predicate,
            source_node_id=source_node_id,
            reason=reason or "invalid_claim",
            edge_id=edge.edge_id,
        )

    claim_edge = {
        "claim_id": claim_id,
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "predicate": predicate,
        "edge_id": edge.edge_id,
    }

    return claim, constraint_rows, claim_edge, None


__all__ = ["build_claim_from_edge"]
