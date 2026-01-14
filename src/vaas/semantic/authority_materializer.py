"""
Authority materialization utilities.

Transforms raw references extracted from instruction text into a canonical
authority registry plus paragraph-to-authority link tables.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd

from vaas.schemas.semantic_contract import AuthorityRow
from vaas.semantic.canonical_ids import canonicalize_authority
from vaas.utils.text import stable_hash


class AuthorityMaterializationError(RuntimeError):
    """Raised when materialization fails due to missing context."""


REFERENCE_COLUMN_ALIASES = {
    "ref_text": ["ref_text", "reference_text"],
    "ref_id": ["ref_id", "reference_id"],
    "evidence_text": ["evidence_text", "evidence"],
    "page": ["page", "page_number"],
    "source_element_id": ["source_element_id", "element_id"],
}


def _normalize_reference_df(references_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to canonical identifiers."""
    rename_map: Dict[str, str] = {}
    for canonical, options in REFERENCE_COLUMN_ALIASES.items():
        found = None
        for option in options:
            if option in references_df.columns:
                found = option
                break
        if not found:
            raise AuthorityMaterializationError(
                f"Missing required reference column '{canonical}'. Expected one of {options}"
            )
        rename_map[found] = canonical
    return references_df.rename(columns=rename_map)


def _element_to_node_map(nodes_df: pd.DataFrame) -> Dict[str, str]:
    """Build element_id → node_id mapping for paragraph nodes."""
    if nodes_df is None or nodes_df.empty:
        return {}
    paragraph_nodes = nodes_df[nodes_df["node_type"] == "paragraph"]
    paragraph_nodes = paragraph_nodes.dropna(subset=["element_id"])
    return dict(zip(paragraph_nodes["element_id"], paragraph_nodes["node_id"]))


def _anchor_to_node_map(nodes_df: pd.DataFrame) -> Dict[str, str]:
    """Build anchor_id → node_id mapping for section/box/concept nodes."""
    if nodes_df is None or nodes_df.empty:
        return {}
    has_anchor = nodes_df.dropna(subset=["anchor_id"])
    if has_anchor.empty:
        return {}
    non_paragraph = has_anchor[has_anchor["node_type"] != "paragraph"]
    if non_paragraph.empty:
        return {}
    return dict(zip(non_paragraph["anchor_id"], non_paragraph["node_id"]))


def _build_authority_row(authority_id: str, raw_text: str) -> AuthorityRow:
    """Create an AuthorityRow instance."""
    authority_type = authority_id.split(":")[1] if ":" in authority_id else "unknown"
    label = raw_text.strip() or authority_id
    citation = label
    return AuthorityRow(
        authority_id=authority_id,
        authority_type=authority_type,
        label=label,
        citation=citation,
        raw_text=raw_text,
    )


def materialize_authorities(
    doc_id: str,
    references_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Build authority registry and paragraph-authority edges from references.

    Returns:
        authorities_df: Unique authority rows
        paragraph_authority_df: Edges linking paragraph nodes to authorities
        stats: Summary counts for audits
    """
    if references_df is None or references_df.empty:
        empty_authorities = pd.DataFrame(
            columns=["authority_id", "authority_type", "label", "citation", "raw_text"]
        )
        empty_edges = pd.DataFrame(
            columns=[
                "edge_id",
                "doc_id",
                "paragraph_node_id",
                "source_node_type",
                "authority_id",
                "relation",
                "ref_id",
                "ref_occurrence_id",
                "evidence_text",
                "page",
                "source_element_id",
                "source_anchor_id",
                "evidence_sentence_idx",
                "evidence_char_start",
                "evidence_char_end",
            ]
        )
        empty_mentions = pd.DataFrame(
            columns=[
                "doc_id",
                "authority_id",
                "source_element_id",
                "sentence_idx",
                "char_start",
                "char_end",
                "evidence_text",
                "ref_occurrence_id",
            ]
        )
        return empty_authorities, empty_edges, empty_mentions, {"total": 0, "canonicalized": 0, "joined": 0}

    normalized_refs = _normalize_reference_df(references_df)
    element_map = _element_to_node_map(nodes_df)
    anchor_map = _anchor_to_node_map(nodes_df)
    authorities: Dict[str, AuthorityRow] = {}
    edge_rows: List[Dict[str, object]] = []
    mention_rows: List[Dict[str, object]] = []

    total_refs = len(normalized_refs)
    canonicalized_refs = 0
    joined_refs = 0

    for _, ref in normalized_refs.iterrows():
        raw_text = str(ref.get("ref_text", "") or "")
        authority_id = canonicalize_authority(raw_text)
        if not authority_id:
            authority_id = f"authority:raw:{stable_hash([raw_text], length=12)}"
        canonicalized_refs += 1
        if authority_id not in authorities:
            authorities[authority_id] = _build_authority_row(authority_id, raw_text)

        element_id = ref.get("source_element_id")
        paragraph_node_id = element_map.get(element_id)
        anchor_node_id = None
        source_anchor_id = ref.get("source_anchor_id")
        if not paragraph_node_id and source_anchor_id:
            anchor_node_id = anchor_map.get(source_anchor_id)
        context_node_id = paragraph_node_id or anchor_node_id
        start_pos = ref.get("char_start")
        end_pos = ref.get("char_end")
        if start_pos is None or end_pos is None:
            elem_start = ref.get("element_char_start")
            elem_end = ref.get("element_char_end")
            sent_start = ref.get("sentence_char_start_in_element")
            try:
                if start_pos is None and elem_start is not None and sent_start is not None:
                    start_pos = int(elem_start) - int(sent_start)
                    if start_pos < 0:
                        start_pos = 0
                if end_pos is None and elem_start is not None and elem_end is not None and sent_start is not None:
                    span_len = int(elem_end) - int(elem_start)
                    end_pos = max(0, start_pos or 0) + max(span_len, 0)
            except Exception:
                start_pos = start_pos or None
                end_pos = end_pos or None
        if context_node_id:
            joined_refs += 1
            hash_suffix = stable_hash(
                [doc_id, context_node_id, authority_id, str(ref.get("ref_id", ""))], length=12
            )
            edge_rows.append(
                {
                    "edge_id": f"auth_link_{hash_suffix}",
                    "doc_id": doc_id,
                    "paragraph_node_id": context_node_id,
                    "source_node_type": "paragraph" if paragraph_node_id else "anchor",
                    "authority_id": authority_id,
                    "relation": "mentions_authority",
                    "ref_id": ref.get("ref_id"),
                    "ref_occurrence_id": ref.get("ref_occurrence_id"),
                    "evidence_text": ref.get("evidence_text"),
                    "page": ref.get("page"),
                    "source_element_id": element_id,
                    "source_anchor_id": source_anchor_id,
                    "evidence_sentence_idx": ref.get("sentence_idx"),
                    "evidence_char_start": start_pos,
                    "evidence_char_end": end_pos,
                }
            )
        mention_rows.append(
            {
                "doc_id": doc_id,
                "authority_id": authority_id,
                "source_element_id": element_id,
                "sentence_idx": ref.get("sentence_idx"),
                "char_start": start_pos,
                "char_end": end_pos,
                "evidence_text": ref.get("evidence_text"),
                "ref_occurrence_id": ref.get("ref_occurrence_id"),
            }
        )

    authorities_df = pd.DataFrame([asdict(row) for row in authorities.values()]) if authorities else pd.DataFrame(
        columns=["authority_id", "authority_type", "label", "citation", "raw_text"]
    )
    paragraph_authority_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=[
            "edge_id",
            "doc_id",
            "paragraph_node_id",
            "source_node_type",
            "authority_id",
            "relation",
            "ref_id",
            "ref_occurrence_id",
            "evidence_text",
            "page",
            "source_element_id",
            "source_anchor_id",
            "evidence_sentence_idx",
            "evidence_char_start",
            "evidence_char_end",
        ]
    )
    authority_mentions_df = pd.DataFrame(mention_rows) if mention_rows else pd.DataFrame(
        columns=[
            "doc_id",
            "authority_id",
            "source_element_id",
            "sentence_idx",
            "char_start",
            "char_end",
            "evidence_text",
            "ref_occurrence_id",
        ]
    )

    stats = {
        "total": total_refs,
        "canonicalized": canonicalized_refs,
        "joined": joined_refs,
    }
    return authorities_df, paragraph_authority_df, authority_mentions_df, stats


__all__ = ["materialize_authorities", "AuthorityMaterializationError"]
