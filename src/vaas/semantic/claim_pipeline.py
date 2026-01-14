"""
Semantic claim emission orchestration.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from vaas.graph.edges import Edge
from vaas.schemas.semantic_contract import (
    ClaimAuthorityMentionRow,
    ClaimPrecedenceRow,
)
from vaas.semantic.canonical_ids import canonicalize_box
from vaas.semantic.claim_builder import build_claim_from_edge
from vaas.semantic.precedence import build_claim_precedence
from vaas.semantic.scope_resolver import resolve_claim_scopes, resolve_precedence_scopes
from vaas.semantic.typed_edge_accounting import make_typed_edge_key, is_claimable
from vaas.utils import slug_title

CLAIM_COLUMNS = [
    "claim_id",
    "doc_id",
    "scope_node_id",
    "scope_struct_node_id",
    "scope_struct_type",
    "precedence_scope_id",
    "precedence_scope_type",
    "source_element_id",
    "sentence_idx",
    "char_start",
    "char_end",
    "evidence_text",
    "pattern_id",
    "predicate",
    "polarity",
    "rule_class",
    "created_by",
    "precedence_rank",
    "subject_canonical_id",
    "object_canonical_id",
    "topic_key",
    "topic_key_source",
    "topic_key_parts",
]
CONSTRAINT_COLUMNS = [
    "constraint_id",
    "constraint_type",
    "doc_id",
    "source_element_id",
    "sentence_idx",
    "raw_text",
]
CLAIM_EDGE_COLUMNS = ["claim_id", "source_node_id", "target_node_id", "predicate", "edge_id"]
CLAIM_AUTH_MENTION_COLUMNS = [
    "claim_id",
    "authority_id",
    "ref_occurrence_id",
    "relation",
    "claim_sentence_idx",
    "claim_char_start",
    "claim_char_end",
    "mention_char_start",
    "mention_char_end",
    "evidence_text",
]
CLAIM_PRECEDENCE_COLUMNS = [
    "from_claim_id",
    "to_claim_id",
    "relation",
    "scope_struct_node_id",
    "sentence_distance",
    "topic_key",
]
REJECTION_COLUMNS = ["predicate", "source_node_id", "reason", "edge_id"]
CLAIM_CONSTRAINT_COLUMNS = ["claim_id", "constraint_id"]
CONSTRAINT_ATTRIBUTE_COLUMNS = ["constraint_id", "key", "value_json"]
CROSSWALK_COLUMNS = ["typed_edge_key", "edge_id", "claim_id", "status", "reason"]


def build_canonical_node_map(nodes_df: pd.DataFrame, form_id: str) -> Dict[str, str]:
    """Create canonical IDs for nodes based on type/labels."""
    canonical: Dict[str, str] = {}
    if nodes_df is None or nodes_df.empty:
        return canonical

    for _, node in nodes_df.iterrows():
        node_id = node.get("node_id")
        if not node_id:
            continue
        node_id_str = str(node_id)
        node_type = (node.get("node_type") or "").lower()
        label = node.get("label") or ""
        anchor_id = node.get("anchor_id") or ""
        box_key = node.get("box_key") or ""
        concept_role = node.get("concept_role") or ""

        canonical_value: Optional[str] = None
        if node_type == "box_section":
            if box_key:
                canonical_value = canonicalize_box(f"Box {box_key}", form_id=form_id)
            else:
                canonical_value = node_id_str
        elif node_type == "authority":
            canonical_value = node_id_str
        elif node_type in {"section", "preamble"}:
            slug_source = label or anchor_id or node_id_str
            canonical_value = f"concept:{slug_title(str(slug_source))}"
        elif node_type == "concept" or concept_role:
            role = concept_role or "concept"
            slug_source = label or anchor_id or node_id_str
            canonical_value = f"{role}:{slug_title(str(slug_source))}"
        elif node_type == "paragraph":
            slug_source = anchor_id or label or node_id_str
            canonical_value = f"concept:{slug_title(str(slug_source))}"
        elif node_type == "doc_root":
            canonical_value = f"doc:{form_id}"
        else:
            canonical_value = node_id_str

        canonical[node_id_str] = canonical_value

    return canonical


def _to_dataframe(rows: List[dict], columns: List[str]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=columns)


def _make_topic_fields(claim: pd.Series) -> Tuple[str, str, List[str]]:
    def _normalize_topic_part(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        if pd.isna(value):
            return ""
        return str(value).strip()

    predicate = _normalize_topic_part(claim.get("predicate"))
    subject = _normalize_topic_part(claim.get("subject_canonical_id"))
    obj = _normalize_topic_part(claim.get("object_canonical_id"))
    parts = [predicate, subject, obj]
    if subject or obj:
        return f"{predicate}|{subject}|{obj}", "canonical_ids", parts
    precedence_scope = _normalize_topic_part(claim.get("precedence_scope_id")) or "none"
    return f"{predicate}|scope:{precedence_scope}", "fallback_scope", parts


def emit_claim_artifacts(
    doc_id: str,
    form_id: str,
    typed_edges: List[Edge],
    nodes_df: pd.DataFrame,
    paragraph_authority_df: pd.DataFrame,
    sentence_index_df: pd.DataFrame,
    authority_mentions_df: pd.DataFrame,
    struct_overlay_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, int]:
    """Materialize claims, constraints, and related join tables."""
    claim_rows = []
    constraint_rows = {}
    claim_edge_rows = []
    rejection_rows = []
    claim_constraint_rows: List[Dict[str, object]] = []
    constraint_attribute_rows: List[Dict[str, object]] = []
    typed_edge_claim_rows: List[Dict[str, object]] = []

    element_text_map: Dict[str, str] = {}
    if nodes_df is not None and not nodes_df.empty:
        nodes_with_elements = nodes_df[nodes_df.get("element_id").notna()] if "element_id" in nodes_df.columns else nodes_df
        for _, node in nodes_with_elements.iterrows():
            element_id = node.get("element_id")
            if element_id is None:
                continue
            text = node.get("text", "")
            element_text_map[str(element_id)] = text if isinstance(text, str) else ""

        if "element_ids" in nodes_df.columns:
            for _, node in nodes_df.iterrows():
                element_ids = node.get("element_ids")
                if not isinstance(element_ids, (list, tuple)):
                    continue
                text = node.get("text", "")
                if not isinstance(text, str) or not text:
                    continue
                for element_id in element_ids:
                    key = str(element_id)
                    if key not in element_text_map:
                        element_text_map[key] = text

    sentence_lookup: Dict[Tuple[str, int], Dict[str, object]] = {}
    if sentence_index_df is not None and not sentence_index_df.empty:
        for _, row in sentence_index_df.iterrows():
            element_id = str(row.get("source_element_id"))
            sentence_idx = row.get("sentence_idx")
            if element_id and sentence_idx is not None:
                sentence_lookup[(element_id, int(sentence_idx))] = row

    canonical_node_map = build_canonical_node_map(nodes_df, form_id)

    mentions_by_sentence: Dict[Tuple[str, int], List[pd.Series]] = {}
    if authority_mentions_df is not None and not authority_mentions_df.empty:
        for _, mention in authority_mentions_df.iterrows():
            element_id = str(mention.get("source_element_id"))
            sentence_idx = mention.get("sentence_idx")
            if not element_id or sentence_idx is None:
                continue
            key = (element_id, int(sentence_idx))
            mentions_by_sentence.setdefault(key, []).append(mention)

    for edge in typed_edges:
        typed_edge_key = make_typed_edge_key(doc_id, edge)
        claimable, _ = is_claimable(edge)
        claim, constraints, claim_edge, rejection = build_claim_from_edge(
            edge,
            doc_id,
            form_id,
            element_text_map,
            sentence_lookup,
            canonical_node_map,
        )
        if rejection:
            rejection_rows.append(rejection.to_dict())
            if claimable:
                typed_edge_claim_rows.append({
                    "typed_edge_key": typed_edge_key,
                    "edge_id": edge.edge_id,
                    "claim_id": None,
                    "status": "rejected",
                    "reason": rejection.reason,
                })
            continue
        if claim:
            claim_dict = claim.to_dict()
            for constraint in constraints:
                claim_constraint_rows.append({
                    "claim_id": claim_dict["claim_id"],
                    "constraint_id": constraint.constraint_id,
                })
            claim_rows.append(claim_dict)
            if claimable:
                typed_edge_claim_rows.append({
                    "typed_edge_key": typed_edge_key,
                    "edge_id": edge.edge_id,
                    "claim_id": claim_dict["claim_id"],
                    "status": "emitted",
                    "reason": None,
                })
        elif claimable:
            typed_edge_claim_rows.append({
                "typed_edge_key": typed_edge_key,
                "edge_id": edge.edge_id,
                "claim_id": None,
                "status": "rejected",
                "reason": "claim_builder_returned_none",
            })
        for constraint in constraints:
            constraint_dict = constraint.to_dict()
            normalized = constraint_dict.pop("normalized_values", {}) or {}
            constraint_rows[constraint.constraint_id] = constraint_dict
            for key, value in normalized.items():
                constraint_attribute_rows.append(
                    {
                        "constraint_id": constraint.constraint_id,
                        "key": key,
                        "value_json": json.dumps(value),
                    }
                )
        if claim_edge:
            claim_edge_rows.append(claim_edge)

    claims_df = _to_dataframe(claim_rows, CLAIM_COLUMNS)
    claims_df = resolve_claim_scopes(claims_df, struct_overlay_df)
    claims_df = resolve_precedence_scopes(claims_df, struct_overlay_df)
    topic_fields = claims_df.apply(_make_topic_fields, axis=1, result_type="expand")
    claims_df["topic_key"] = topic_fields[0]
    claims_df["topic_key_source"] = topic_fields[1]
    claims_df["topic_key_parts"] = topic_fields[2]
    claims_df = claims_df.reindex(columns=CLAIM_COLUMNS)
    constraints_df = _to_dataframe(list(constraint_rows.values()), CONSTRAINT_COLUMNS)
    claim_edges_df = _to_dataframe(claim_edge_rows, CLAIM_EDGE_COLUMNS)
    rejections_df = _to_dataframe(rejection_rows, REJECTION_COLUMNS)
    claim_constraints_df = _to_dataframe(claim_constraint_rows, CLAIM_CONSTRAINT_COLUMNS)
    constraint_attributes_df = _to_dataframe(constraint_attribute_rows, CONSTRAINT_ATTRIBUTE_COLUMNS)

    authority_mentions_rows = []
    if not claims_df.empty and mentions_by_sentence:
        for _, claim in claims_df.iterrows():
            element_id = str(claim["source_element_id"])
            sentence_idx = claim.get("sentence_idx")
            if sentence_idx is None:
                continue
            key = (element_id, int(sentence_idx))
            mentions = mentions_by_sentence.get(key, [])
            if not mentions:
                continue
            claim_start = claim.get("char_start")
            claim_end = claim.get("char_end")
            claim_text = claim.get("evidence_text") or ""
            for mention in mentions:
                mention_start = mention.get("char_start")
                mention_end = mention.get("char_end")
                overlaps = False
                if (
                    claim_start is not None and claim_end is not None
                    and mention_start is not None and mention_end is not None
                ):
                    try:
                        cs = int(claim_start)
                        ce = int(claim_end)
                        ms = int(mention_start)
                        me = int(mention_end)
                        overlaps = ce > ms and me > cs
                    except (TypeError, ValueError):
                        overlaps = False
                substring_match = False
                mention_text = mention.get("evidence_text") or ""
                if mention_text and claim_text:
                    norm_claim = " ".join(str(claim_text).split()).lower()
                    norm_mention = " ".join(str(mention_text).split()).lower()
                    substring_match = norm_mention and norm_mention in norm_claim
                if overlaps or substring_match:
                    authority_mentions_rows.append(ClaimAuthorityMentionRow(
                        claim_id=claim["claim_id"],
                        authority_id=mention.get("authority_id"),
                        ref_occurrence_id=mention.get("ref_occurrence_id"),
                        relation="mentions_authority",
                        claim_sentence_idx=sentence_idx,
                        claim_char_start=claim_start,
                        claim_char_end=claim_end,
                        mention_char_start=mention_start,
                        mention_char_end=mention_end,
                        evidence_text=mention_text,
                    ).to_dict())
    claim_authority_mentions_df = _to_dataframe(authority_mentions_rows, CLAIM_AUTH_MENTION_COLUMNS)

    claim_precedence_rows: List[ClaimPrecedenceRow] = []
    if not claims_df.empty:
        for row in build_claim_precedence(claims_df):
            claim_precedence_rows.append(asdict(row))
    claim_precedence_df = _to_dataframe(claim_precedence_rows, CLAIM_PRECEDENCE_COLUMNS)
    typed_edge_claims_df = _to_dataframe(typed_edge_claim_rows, CROSSWALK_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    claims_path = output_dir / "claims.parquet"
    constraints_path = output_dir / "constraints.parquet"
    claim_edges_path = output_dir / "claim_edges.parquet"
    claim_authority_mentions_path = output_dir / "claim_authority_mentions.parquet"
    claim_precedence_path = output_dir / "claim_precedence.parquet"
    rejections_path = output_dir / "claim_rejections.parquet"
    typed_edge_claims_path = output_dir / "typed_edge_claims.parquet"
    claim_constraints_path = output_dir / "claim_constraints.parquet"
    constraint_attributes_path = output_dir / "constraint_attributes.parquet"

    claims_df.to_parquet(claims_path, index=False)
    constraints_df.to_parquet(constraints_path, index=False)
    claim_edges_df.to_parquet(claim_edges_path, index=False)
    claim_authority_mentions_df.to_parquet(claim_authority_mentions_path, index=False)
    claim_precedence_df.to_parquet(claim_precedence_path, index=False)
    rejections_df.to_parquet(rejections_path, index=False)
    claim_constraints_df.to_parquet(claim_constraints_path, index=False)
    constraint_attributes_df.to_parquet(constraint_attributes_path, index=False)
    typed_edge_claims_df.to_parquet(typed_edge_claims_path, index=False)

    rejection_reason_counts: Dict[str, int] = {}
    if not rejections_df.empty and "reason" in rejections_df.columns:
        reason_counts = rejections_df["reason"].value_counts()
        rejection_reason_counts = {str(reason): int(count) for reason, count in reason_counts.items()}

    claim_predicate_counts: Dict[str, int] = {}
    if not claims_df.empty and "predicate" in claims_df.columns:
        claim_predicate_counts = {
            str(predicate): int(count)
            for predicate, count in claims_df["predicate"].value_counts().items()
        }

    return {
        "claims": len(claims_df),
        "claim_edges": len(claim_edges_df),
        "constraints": len(constraints_df),
        "claim_authority_mentions": len(claim_authority_mentions_df),
        "claim_precedence": len(claim_precedence_df),
        "claim_rejections": len(rejections_df),
        "claim_constraints": len(claim_constraints_df),
        "constraint_attributes": len(constraint_attributes_df),
        "typed_edge_claims": len(typed_edge_claims_df),
        "rejection_reasons": rejection_reason_counts,
        "claim_predicate_counts": claim_predicate_counts,
    }


__all__ = ["emit_claim_artifacts"]
