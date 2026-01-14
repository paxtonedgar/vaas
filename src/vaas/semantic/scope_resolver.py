"""Utilities for deriving structural scope for claims from structural nodes."""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import pandas as pd

from vaas.semantic.policy import precedence_scope_preference

ELEMENT_SENTENCE_TYPE = "element_sentence"
PARAGRAPH_TYPE = "paragraph"
ANCHOR_TYPE = "anchor"
SECTION_TYPE = "section"


def build_struct_scope_overlay(
    nodes_df: pd.DataFrame,
    sentence_index_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a structural overlay DataFrame with scope-ready nodes.

    Output columns:
        struct_node_id
        doc_id
        struct_type (element_sentence, paragraph, anchor, section)
        source_element_id
        sentence_idx
        parent_struct_node_id
    """
    overlay_rows = []

    def _prefix_anchor_id(doc_id: str, anchor_id: Optional[str]) -> Optional[str]:
        if not anchor_id:
            return None
        anchor_id_str = str(anchor_id)
        if anchor_id_str.startswith(f"{doc_id}:"):
            return anchor_id_str
        return f"{doc_id}:{anchor_id_str}"

    paragraph_parent_map: Dict[str, str] = {}
    element_anchor_map: Dict[str, str] = {}
    paragraph_nodes = nodes_df[nodes_df.get("node_type") == "paragraph"] if nodes_df is not None else pd.DataFrame()
    if not paragraph_nodes.empty:
        for _, row in paragraph_nodes.iterrows():
            node_id = str(row.get("node_id"))
            element_id = str(row.get("element_id"))
            doc_id = str(row.get("doc_id"))
            overlay_rows.append({
                "struct_node_id": node_id,
                "doc_id": doc_id,
                "struct_type": PARAGRAPH_TYPE,
                "source_element_id": element_id,
                "sentence_idx": None,
                "parent_struct_node_id": _prefix_anchor_id(doc_id, row.get("anchor_id")),
            })
            paragraph_parent_map[(doc_id, element_id)] = node_id

    anchor_nodes = nodes_df[nodes_df.get("node_type").isin({"section", "box_section", "concept"})] if nodes_df is not None else pd.DataFrame()
    if not anchor_nodes.empty:
        for _, row in anchor_nodes.iterrows():
            node_id = str(row.get("node_id"))
            element_ids = row.get("element_ids")
            if isinstance(element_ids, (list, tuple)):
                for element_id in element_ids:
                    element_anchor_map[str(element_id)] = node_id

    if sentence_index_df is not None and not sentence_index_df.empty:
        for _, row in sentence_index_df.iterrows():
            element_id = str(row.get("source_element_id"))
            sentence_idx = int(row.get("sentence_idx"))
            doc_id = row.get("doc_id")
            node_id = f"{doc_id}:{element_id}:sent:{sentence_idx}"
            parent = paragraph_parent_map.get((doc_id, element_id))
            if not parent:
                parent = element_anchor_map.get(element_id)
            overlay_rows.append({
                "struct_node_id": node_id,
                "doc_id": doc_id,
                "struct_type": ELEMENT_SENTENCE_TYPE,
                "source_element_id": element_id,
                "sentence_idx": sentence_idx,
                "parent_struct_node_id": parent,
            })

    if not anchor_nodes.empty:
        for _, row in anchor_nodes.iterrows():
            node_id = str(row.get("node_id"))
            overlay_rows.append({
                "struct_node_id": node_id,
                "doc_id": row.get("doc_id"),
                "struct_type": ANCHOR_TYPE if row.get("node_type") != "section" else SECTION_TYPE,
                "source_element_id": None,
                "sentence_idx": None,
                "parent_struct_node_id": _prefix_anchor_id(str(row.get("doc_id")), row.get("anchor_id")),
            })

    if not overlay_rows:
        return pd.DataFrame(columns=[
            "struct_node_id",
            "doc_id",
            "struct_type",
            "source_element_id",
            "sentence_idx",
            "parent_struct_node_id",
        ])
    return pd.DataFrame(overlay_rows)


def resolve_claim_scopes(
    claims_df: pd.DataFrame,
    struct_overlay_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach scope_struct_node_id and scope_struct_type to claims."""
    if claims_df.empty or struct_overlay_df.empty:
        claims_df["scope_struct_node_id"] = None
        claims_df["scope_struct_type"] = None
        return claims_df

    overlay_by_sentence: Dict[Tuple[str, str, int], str] = {}
    overlay_types: Dict[str, str] = {}
    overlay_by_element: Dict[Tuple[str, str], str] = {}

    for _, row in struct_overlay_df.iterrows():
        struct_node_id = str(row.get("struct_node_id"))
        doc_id = row.get("doc_id")
        overlay_types[struct_node_id] = row.get("struct_type")
        source_element_id = row.get("source_element_id")
        sentence_idx = row.get("sentence_idx")
        if (
            row.get("struct_type") == ELEMENT_SENTENCE_TYPE
            and source_element_id
            and sentence_idx is not None
            and doc_id
        ):
            overlay_by_sentence[(str(doc_id), str(source_element_id), int(sentence_idx))] = struct_node_id
        elif row.get("struct_type") == PARAGRAPH_TYPE and source_element_id and doc_id:
            overlay_by_element[(str(doc_id), str(source_element_id))] = struct_node_id

    scope_ids = []
    scope_types = []
    for _, claim in claims_df.iterrows():
        element_id = str(claim.get("source_element_id"))
        sentence_idx = claim.get("sentence_idx")
        if sentence_idx is not None and pd.isna(sentence_idx):
            sentence_idx = None
        doc_id = claim.get("doc_id")
        scope_node = None
        scope_type = None
        if sentence_idx is not None and doc_id is not None and (str(doc_id), element_id, int(sentence_idx)) in overlay_by_sentence:
            scope_node = overlay_by_sentence[(str(doc_id), element_id, int(sentence_idx))]
            scope_type = ELEMENT_SENTENCE_TYPE
        elif doc_id is not None and (str(doc_id), element_id) in overlay_by_element:
            scope_node = overlay_by_element[(str(doc_id), element_id)]
            scope_type = PARAGRAPH_TYPE
        else:
            scope_node = str(claim.get("scope_node_id"))
            scope_type = overlay_types.get(scope_node)
        scope_ids.append(scope_node)
        scope_types.append(scope_type)
    claims_df = claims_df.copy()
    claims_df["scope_struct_node_id"] = scope_ids
    claims_df["scope_struct_type"] = scope_types
    return claims_df


def resolve_precedence_scopes(
    claims_df: pd.DataFrame,
    struct_overlay_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach precedence_scope_id/type using paragraph/anchor scopes."""
    if claims_df.empty or struct_overlay_df.empty:
        claims_df["precedence_scope_id"] = None
        claims_df["precedence_scope_type"] = None
        return claims_df

    overlay_types: Dict[str, str] = {}
    parent_map: Dict[str, Optional[str]] = {}
    for _, row in struct_overlay_df.iterrows():
        struct_node_id = str(row.get("struct_node_id"))
        overlay_types[struct_node_id] = row.get("struct_type")
        parent_map[struct_node_id] = row.get("parent_struct_node_id")

    precedence_ids = []
    precedence_types = []
    for _, claim in claims_df.iterrows():
        scope_id = claim.get("scope_struct_node_id")
        precedence_id = None
        precedence_type = None

        if scope_id:
            scope_id_str = str(scope_id)
            preference = precedence_scope_preference(claim.get("predicate"))
            if preference == "anchor":
                preferred_order = [ANCHOR_TYPE, SECTION_TYPE, PARAGRAPH_TYPE]
            else:
                preferred_order = [PARAGRAPH_TYPE, ANCHOR_TYPE, SECTION_TYPE]

            path = []
            cur = scope_id_str
            seen = set()
            while cur and cur not in seen:
                seen.add(cur)
                path.append(cur)
                parent_id = parent_map.get(cur)
                cur = str(parent_id) if parent_id else None

            for desired in preferred_order:
                for node_id in path:
                    node_type = overlay_types.get(node_id)
                    if node_type == desired:
                        precedence_id = node_id
                        precedence_type = node_type
                        break
                if precedence_id:
                    break

        precedence_ids.append(precedence_id)
        precedence_types.append(precedence_type)

    claims_df = claims_df.copy()
    claims_df["precedence_scope_id"] = precedence_ids
    claims_df["precedence_scope_type"] = precedence_types
    return claims_df


__all__ = [
    "build_struct_scope_overlay",
    "resolve_claim_scopes",
    "resolve_precedence_scopes",
]
