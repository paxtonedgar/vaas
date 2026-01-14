"""
Precedence edge emission for claims.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from vaas.schemas.semantic_contract import ClaimPrecedenceRow
from vaas.semantic.policy import precedence_sentence_distance_limit

EXCEPTION_KEYWORDS = ("exception", "however", "but not", "except that")


def build_claim_precedence(
    claims_df: pd.DataFrame,
) -> List[ClaimPrecedenceRow]:
    """Emit precedence edges grouped by structural scope and topic key."""
    if claims_df.empty:
        return []

    required_cols = {
        "precedence_scope_id",
        "topic_key",
        "topic_key_source",
        "predicate",
        "source_element_id",
        "sentence_idx",
        "char_start",
        "claim_id",
    }
    missing = required_cols - set(claims_df.columns)
    if missing:
        raise ValueError(f"Claims dataframe missing columns: {missing}")

    precedence_rows: List[ClaimPrecedenceRow] = []
    scoped = claims_df.dropna(subset=["precedence_scope_id", "topic_key", "predicate"])
    scoped = scoped.copy()
    scoped["group_kind"] = scoped.apply(_group_kind, axis=1)
    scoped["group_key"] = scoped.apply(_group_key, axis=1)

    grouped = scoped.groupby(["precedence_scope_id", "group_kind", "group_key"], dropna=True)
    for (scope_node, group_kind, group_key), group in grouped:
        if len(group) < 2:
            continue
        ordered = sorted(
            group.to_dict("records"),
            key=_claim_sort_key,
        )
        sentence_order: Dict[Tuple[str, int], int] = {}
        next_index = 0
        for claim in ordered:
            element_id = claim.get("source_element_id")
            sentence_idx = claim.get("sentence_idx")
            if element_id is None or sentence_idx is None:
                continue
            key = (str(element_id), int(sentence_idx))
            if key not in sentence_order:
                sentence_order[key] = next_index
                next_index += 1
            claim["_sentence_order"] = sentence_order[key]
        for idx in range(len(ordered) - 1):
            higher = ordered[idx]
            lower = ordered[idx + 1]
            sentence_distance = _sentence_distance(higher, lower)
            predicate = str(higher.get("predicate") or lower.get("predicate") or "")
            limit = precedence_sentence_distance_limit(predicate)
            from_id = higher["claim_id"]
            to_id = lower["claim_id"]
            if sentence_distance is not None and sentence_distance <= limit:
                precedence_rows.append(ClaimPrecedenceRow(
                    from_claim_id=from_id,
                    to_claim_id=to_id,
                    relation="precedes",
                    scope_struct_node_id=str(scope_node),
                    sentence_distance=sentence_distance,
                    topic_key=_group_topic_key(group_kind, group_key, higher, scope_node),
                ))
            if _is_exception(lower):
                precedence_rows.append(ClaimPrecedenceRow(
                    from_claim_id=lower["claim_id"],
                    to_claim_id=higher["claim_id"],
                    relation="exception_of",
                    scope_struct_node_id=str(scope_node),
                    sentence_distance=sentence_distance,
                    topic_key=_group_topic_key(group_kind, group_key, higher, scope_node),
                ))

    return precedence_rows


def _claim_sort_key(claim: Dict[str, object]) -> Tuple[str, int, int, str]:
    element_id = claim.get("source_element_id")
    element_key = str(element_id) if element_id else "~"
    sentence_idx = claim.get("sentence_idx")
    char_start = claim.get("char_start")
    claim_id = str(claim.get("claim_id") or "")
    return (
        element_key,
        int(sentence_idx) if sentence_idx is not None else 10**9,
        int(char_start) if char_start is not None else 10**9,
        claim_id,
    )


def _group_kind(claim: Dict[str, object]) -> str:
    source = str(claim.get("topic_key_source") or "").strip().lower()
    return "canonical" if source == "canonical_ids" else "predicate"


def _group_key(claim: Dict[str, object]) -> str:
    if _group_kind(claim) == "canonical":
        return str(claim.get("topic_key") or "")
    return str(claim.get("predicate") or "")


def _group_topic_key(
    kind: str,
    key: str,
    claim: Dict[str, object],
    scope_node: object,
) -> str:
    if kind == "canonical":
        return key
    predicate = str(claim.get("predicate") or key or "")
    return f"{predicate}|scope:{str(scope_node)}"


def _is_exception(lower: Dict[str, object]) -> bool:
    evidence = (lower.get("evidence_text") or "").lower()
    return any(kw in evidence for kw in EXCEPTION_KEYWORDS)


def _sentence_distance(a: Dict[str, object], b: Dict[str, object]) -> int:
    if "_sentence_order" in a and "_sentence_order" in b:
        try:
            return abs(int(a["_sentence_order"]) - int(b["_sentence_order"]))
        except Exception:
            return 0
    sa = a.get("sentence_idx")
    sb = b.get("sentence_idx")
    try:
        return abs(int(sa) - int(sb))
    except Exception:
        return 0


__all__ = ["build_claim_precedence"]
