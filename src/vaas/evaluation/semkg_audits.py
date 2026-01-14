"""
Semantic KG deterministic audits.

These gates check the semantic layer outputs and ensure invariants such as
typed-edge accounting, sentence-level coverage, and precedence integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from vaas.semantic.policy import (
    DEFAULT_PRECEDENCE_SENTENCE_DISTANCE,
    precedence_sentence_distance_limit,
)

RESOLUTION_NO_EDGES_WARN_THRESHOLD = 0


@dataclass
class AuditResult:
    gate_id: str
    passed: bool
    total: int
    succeeded: int
    threshold: float
    details: str = ""

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 1.0


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_claims(output_dir: Path) -> pd.DataFrame:
    return _load_parquet(output_dir / "claims.parquet")


def _load_rejections(output_dir: Path) -> pd.DataFrame:
    return _load_parquet(output_dir / "claim_rejections.parquet")


def _load_typed_edges(output_dir: Path) -> pd.DataFrame:
    return _load_parquet(output_dir / "typed_edges.parquet")


def _load_typed_edge_claims(output_dir: Path) -> pd.DataFrame:
    return _load_parquet(output_dir / "typed_edge_claims.parquet")


def _load_claim_authority_mentions(output_dir: Path) -> pd.DataFrame:
    return _load_parquet(output_dir / "claim_authority_mentions.parquet")


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return bool(pd.isna(value))


def _precedence_group_kind(row: pd.Series) -> str:
    source = str(row.get("topic_key_source") or "").strip().lower()
    return "canonical" if source == "canonical_ids" else "predicate"


def _precedence_group_value(row: pd.Series) -> str:
    if _precedence_group_kind(row) == "canonical":
        return str(row.get("topic_key") or "")
    return str(row.get("predicate") or "")


def _build_claim_index(claims_df: pd.DataFrame) -> Dict[str, Tuple[str, str, str, str, str]]:
    index: Dict[str, Tuple[str, str, str, str, str]] = {}
    if claims_df.empty:
        return index
    for _, row in claims_df.iterrows():
        claim_id = row.get("claim_id")
        scope = row.get("precedence_scope_id")
        topic = row.get("topic_key")
        if _is_missing(claim_id) or _is_missing(scope) or _is_missing(topic):
            continue
        predicate = row.get("predicate") or ""
        group_kind = _precedence_group_kind(row)
        group_value = _precedence_group_value(row)
        index[str(claim_id)] = (str(scope), group_kind, str(group_value), str(predicate), str(topic))
    return index


def _build_group_claim_map(
    claims_df: pd.DataFrame,
) -> Dict[Tuple[str, str, str], List[str]]:
    group_map: Dict[Tuple[str, str, str], List[str]] = {}
    if claims_df.empty:
        return group_map
    scoped = claims_df.dropna(subset=["precedence_scope_id", "predicate", "topic_key"]).copy()
    scoped["group_kind"] = scoped.apply(_precedence_group_kind, axis=1)
    scoped["group_value"] = scoped.apply(_precedence_group_value, axis=1)
    for _, row in scoped.iterrows():
        key = (
            str(row.get("precedence_scope_id")),
            str(row.get("group_kind")),
            str(row.get("group_value")),
        )
        claim_id = str(row.get("claim_id") or "")
        if not claim_id:
            continue
        group_map.setdefault(key, []).append(claim_id)
    return group_map


def _validate_precedence_edges(
    edges_df: pd.DataFrame,
    claim_index: Dict[str, Tuple[str, str, str, str, str]],
) -> Tuple[List[Tuple[str, str, str, str, str, str, Optional[int]]], List[str]]:
    valid_edges: List[Tuple[str, str, str, str, str, str, Optional[int]]] = []
    invalid_reasons: List[str] = []
    if edges_df.empty:
        return valid_edges, invalid_reasons
    for _, row in edges_df.iterrows():
        from_id = row.get("from_claim_id")
        to_id = row.get("to_claim_id")
        if _is_missing(from_id) or _is_missing(to_id):
            invalid_reasons.append("missing_claim_id")
            continue
        from_key = claim_index.get(str(from_id))
        to_key = claim_index.get(str(to_id))
        if not from_key or not to_key:
            invalid_reasons.append("unknown_claim_id")
            continue
        if from_key[:3] != to_key[:3]:
            invalid_reasons.append("cross_scope_or_group")
            continue
        scope, group_kind, group_value, predicate, topic = from_key
        edge_scope = row.get("scope_struct_node_id")
        edge_topic = row.get("topic_key")
        if _is_missing(edge_scope) or _is_missing(edge_topic):
            invalid_reasons.append("missing_scope")
            continue
        if str(edge_scope) != scope:
            invalid_reasons.append("scope_mismatch")
            continue
        if str(edge_topic) != topic:
            invalid_reasons.append("topic_mismatch")
            continue
        relation = str(row.get("relation") or "")
        sentence_distance = row.get("sentence_distance")
        if _is_missing(sentence_distance):
            sentence_distance = None
        valid_edges.append(
            (scope, group_kind, group_value, predicate, str(from_id), str(to_id), relation, sentence_distance)
        )
    return valid_edges, invalid_reasons


def run_semkg_audits(output_dir: Path) -> List[AuditResult]:
    gates = [
        reference_join_gate(output_dir),
        claim_scope_gate(output_dir),
        claim_group_competition_gate(output_dir),
        precedence_nonempty_gate(output_dir),
        claim_completeness_gate(output_dir),
        claim_duplicate_gate(output_dir),
        authority_presence_gate(output_dir),
        authority_mentions_gate(output_dir),
        constraint_coverage_gate(output_dir),
        precedence_cross_group_gate(output_dir),
        precedence_links_gate(output_dir),
        precedence_chain_gate(output_dir),
        precedence_dag_gate(output_dir),
        precedence_locality_gate(output_dir),
        paragraph_node_alignment_gate(output_dir),
        claim_authority_coverage_gate(output_dir),
        claim_authority_fanout_gate(output_dir),
        reference_span_gate(output_dir),
        reference_span_exactness_gate(output_dir),
        canonicalization_gate(output_dir),
        sentence_span_gate(output_dir),
        resolution_nonempty_gate(output_dir),
        resolution_group_integrity_gate(output_dir),
        resolution_trace_completeness_gate(output_dir),
        compiler_ir_present_gate(output_dir),
        compiler_support_gate(output_dir),
        compiler_ir_schema_gate(output_dir),
        compiler_directive_join_gate(output_dir),
        compiler_scope_topic_consistency_gate(output_dir),
        predicate_outcome_coverage_gate(output_dir),
        predicate_health_report(output_dir),
    ]
    return gates


def claim_group_competition_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_claim_group_competition",
            passed=False,
            total=1,
            succeeded=0,
            threshold=2.0,
            details="No claims available to evaluate competition",
        )
    required_cols = {"precedence_scope_id", "topic_key", "predicate", "topic_key_source"}
    missing = required_cols - set(claims_df.columns)
    if missing:
        return AuditResult(
            gate_id="semkg_claim_group_competition",
            passed=False,
            total=1,
            succeeded=0,
            threshold=2.0,
            details=f"claims.parquet missing columns: {sorted(missing)}",
        )
    scoped = claims_df.dropna(subset=["precedence_scope_id", "predicate", "topic_key"])
    scoped = scoped.copy()
    scoped["group_kind"] = scoped.apply(_precedence_group_kind, axis=1)
    scoped["group_value"] = scoped.apply(_precedence_group_value, axis=1)
    grouped = scoped.groupby(["precedence_scope_id", "group_kind", "group_value"]).size()
    max_group = int(grouped.max()) if not grouped.empty else 0
    passed = max_group >= 2
    details = f"max_group_size={max_group}"
    return AuditResult(
        gate_id="semkg_claim_group_competition",
        passed=passed,
        total=1,
        succeeded=1 if passed else 0,
        threshold=2.0,
        details=details,
    )


def precedence_nonempty_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    edges_df = _load_parquet(output_dir / "claim_precedence.parquet")
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_precedence_nonempty",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no claims available",
        )
    required_cols = {"precedence_scope_id", "topic_key", "predicate", "topic_key_source"}
    missing = required_cols - set(claims_df.columns)
    if missing:
        return AuditResult(
            gate_id="semkg_precedence_nonempty",
            passed=False,
            total=len(claims_df),
            succeeded=0,
            threshold=1.0,
            details=f"claims.parquet missing columns: {sorted(missing)}",
        )
    scoped = claims_df.dropna(subset=["precedence_scope_id", "predicate", "topic_key"]).copy()
    scoped["group_kind"] = scoped.apply(_precedence_group_kind, axis=1)
    scoped["group_value"] = scoped.apply(_precedence_group_value, axis=1)
    grouped = scoped.groupby(["precedence_scope_id", "group_kind", "group_value"]).size()
    competing_groups = {key for key, size in grouped.items() if size >= 2}
    if not competing_groups:
        return AuditResult(
            gate_id="semkg_precedence_nonempty",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no competing claim groups",
        )
    claim_index = _build_claim_index(claims_df)
    valid_edges, _ = _validate_precedence_edges(edges_df, claim_index)
    groups_with_edges = {(scope, kind, value) for scope, kind, value, _, _, _, _, _ in valid_edges}
    missing_groups = competing_groups - groups_with_edges
    passed = len(missing_groups) == 0 and len(valid_edges) > 0
    details = f"groups_with_edges={len(groups_with_edges)}/{len(competing_groups)}"
    if missing_groups:
        details += f"; missing_groups={list(missing_groups)[:3]}"
    return AuditResult(
        gate_id="semkg_precedence_nonempty",
        passed=passed,
        total=len(competing_groups),
        succeeded=len(competing_groups) - len(missing_groups),
        threshold=1.0,
        details=details,
    )


def reference_join_gate(output_dir: Path) -> AuditResult:
    refs = _load_parquet(output_dir / "references.parquet")
    nodes = _load_parquet(output_dir / "graph_nodes.parquet")
    sentences = _load_parquet(output_dir / "element_sentences.parquet")
    if refs.empty:
        return AuditResult(
            gate_id="semkg_reference_join",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.99,
            details="references.parquet missing or empty; gate skipped",
        )

    paragraph_nodes = nodes[nodes.get("node_type") == "paragraph"] if not nodes.empty else pd.DataFrame()
    paragraph_elements = set(paragraph_nodes["element_id"].dropna().astype(str)) if not paragraph_nodes.empty else set()
    anchor_nodes = set(nodes[nodes["anchor_id"].notna()]["anchor_id"].astype(str)) if not nodes.empty else set()
    sentence_keys = {
        (str(row["source_element_id"]), int(row["sentence_idx"]))
        for _, row in sentences.iterrows()
        if row.get("source_element_id") and row.get("sentence_idx") is not None
    }

    paragraph_hits = anchor_hits = sentence_hits = unresolved = 0
    for _, ref in refs.iterrows():
        element_id = str(ref.get("source_element_id"))
        anchor_id = ref.get("source_anchor_id")
        sentence_idx = ref.get("sentence_idx")
        if element_id in paragraph_elements:
            paragraph_hits += 1
        elif anchor_id and anchor_id in anchor_nodes:
            anchor_hits += 1
        elif (element_id, sentence_idx) in sentence_keys:
            sentence_hits += 1
        else:
            unresolved += 1

    total = len(refs)
    success_rate = paragraph_hits / total if total else 1.0
    anchor_rate = anchor_hits / total if total else 0.0
    passed = success_rate >= 0.99 and anchor_rate <= 0.05 and unresolved == 0
    details = (
        f"{paragraph_hits}/{total} references mapped to paragraph or sentence "
        f"(anchor_fallback={anchor_hits}, sentence_fallback={sentence_hits}, unresolved={unresolved})"
    )
    return AuditResult(
        gate_id="semkg_reference_join",
        passed=passed,
        total=total,
        succeeded=paragraph_hits,
        threshold=0.99,
        details=details,
    )


def claim_scope_gate(output_dir: Path) -> AuditResult:
    claims = _load_claims(output_dir)
    if claims.empty:
        return AuditResult(
            gate_id="semkg_claim_scope",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.99,
            details="No claims emitted",
        )
    if "scope_struct_node_id" not in claims.columns:
        return AuditResult(
            gate_id="semkg_claim_scope",
            passed=False,
            total=len(claims),
            succeeded=0,
            threshold=0.99,
            details="claims.parquet missing scope_struct_node_id column",
        )
    total = len(claims)
    resolved = claims["scope_struct_node_id"].notna().sum()
    success_rate = resolved / total if total else 1.0
    passed = success_rate >= 0.99
    details = f"{resolved}/{total} claims have scope_struct_node_id ({success_rate:.2%})"
    return AuditResult(
        gate_id="semkg_claim_scope",
        passed=passed,
        total=total,
        succeeded=resolved,
        threshold=0.99,
        details=details,
    )


def claim_completeness_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    typed_edges_df = _load_typed_edges(output_dir)
    typed_edge_claims_df = _load_typed_edge_claims(output_dir)
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_claim_completeness",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.99,
            details="No claims emitted",
        )

    claimable_df = typed_edges_df[typed_edges_df.get("claimable") == True] if not typed_edges_df.empty else pd.DataFrame()
    claimable_keys = set(claimable_df["typed_edge_key"]) if not claimable_df.empty else set()
    accounted = typed_edge_claims_df[typed_edge_claims_df["typed_edge_key"].isin(claimable_keys)] if not typed_edge_claims_df.empty else pd.DataFrame()

    missing = claimable_keys - set(accounted["typed_edge_key"]) if claimable_keys else set()
    emitted = accounted[accounted["status"] == "emitted"] if not accounted.empty else pd.DataFrame()
    total = len(claimable_keys)
    if total == 0:
        return AuditResult(
            gate_id="semkg_claim_completeness",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.99,
            details="No claimable typed edges detected; gate skipped",
        )

    accounting_gap = len(missing) > 0
    success_rate = len(emitted) / total if total else 1.0
    passed = success_rate >= 0.99 and not accounting_gap
    details = f"{len(emitted)}/{total} claimable edges emitted"
    if accounting_gap:
        details += f"; {len(missing)} claimable edges missing crosswalk rows"
    return AuditResult(
        gate_id="semkg_claim_completeness",
        passed=passed,
        total=total,
        succeeded=len(emitted),
        threshold=0.99,
        details=details,
    )


def claim_duplicate_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_claim_uniqueness",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No claims to audit",
        )
    unique_ids = claims_df["claim_id"].nunique()
    total = len(claims_df)
    passed = unique_ids == total
    details = f"{unique_ids}/{total} unique claim_ids"
    return AuditResult(
        gate_id="semkg_claim_uniqueness",
        passed=passed,
        total=total,
        succeeded=unique_ids,
        threshold=1.0,
        details=details,
    )


def authority_presence_gate(output_dir: Path) -> AuditResult:
    refs_df = _load_parquet(output_dir / "references.parquet")
    edges_df = _load_parquet(output_dir / "paragraph_authority_edges.parquet")
    if refs_df.empty:
        return AuditResult(
            gate_id="semkg_authority_presence",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.0,
            details="No references extracted",
        )
    total = len(refs_df)
    succeeded = len(edges_df)
    passed = succeeded >= total
    details = f"{succeeded}/{total} references linked to authority nodes"
    return AuditResult(
        gate_id="semkg_authority_presence",
        passed=passed,
        total=total,
        succeeded=min(succeeded, total),
        threshold=1.0,
        details=details,
    )


def authority_mentions_gate(output_dir: Path) -> AuditResult:
    mentions_df = _load_parquet(output_dir / "authority_mentions.parquet")
    refs_df = _load_parquet(output_dir / "references.parquet")
    if refs_df.empty:
        return AuditResult(
            gate_id="semkg_authority_mentions",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.0,
            details="No references extracted",
        )
    passed = not mentions_df.empty
    details = f"{len(mentions_df)} authority mentions recorded"
    return AuditResult(
        gate_id="semkg_authority_mentions",
        passed=passed,
        total=len(refs_df),
        succeeded=len(mentions_df),
        threshold=0.01,
        details=details,
    )


def constraint_coverage_gate(output_dir: Path) -> AuditResult:
    constraints_df = _load_parquet(output_dir / "constraints.parquet")
    claim_constraints_df = _load_parquet(output_dir / "claim_constraints.parquet")
    if constraints_df.empty:
        return AuditResult(
            gate_id="semkg_constraint_coverage",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.0,
            details="No constraints extracted",
        )
    passed = not claim_constraints_df.empty
    details = f"{len(claim_constraints_df)} claim-constraint links for {len(constraints_df)} constraints"
    return AuditResult(
        gate_id="semkg_constraint_coverage",
        passed=passed,
        total=len(constraints_df),
        succeeded=len(claim_constraints_df),
        threshold=0.01,
        details=details,
    )


def precedence_cross_group_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    edges_df = _load_parquet(output_dir / "claim_precedence.parquet")
    if claims_df.empty or edges_df.empty:
        return AuditResult(
            gate_id="semkg_precedence_cross_group_edges",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.0,
            details="SKIP: no claims or precedence edges",
        )
    claim_index = _build_claim_index(claims_df)
    cross_group = 0
    checked = 0
    for _, row in edges_df.iterrows():
        from_id = row.get("from_claim_id")
        to_id = row.get("to_claim_id")
        if _is_missing(from_id) or _is_missing(to_id):
            continue
        from_key = claim_index.get(str(from_id))
        to_key = claim_index.get(str(to_id))
        if not from_key or not to_key:
            continue
        checked += 1
        if from_key[:3] != to_key[:3]:
            cross_group += 1
    passed = cross_group == 0
    details = f"{checked - cross_group}/{checked} edges within group"
    return AuditResult(
        gate_id="semkg_precedence_cross_group_edges",
        passed=passed,
        total=checked,
        succeeded=checked - cross_group,
        threshold=0.0,
        details=details,
    )


def precedence_links_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    edges_df = _load_parquet(output_dir / "claim_precedence.parquet")
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_precedence_links",
            passed=True,
            total=0,
            succeeded=0,
            threshold=0.0,
            details="No claims emitted",
        )
    required_cols = {"precedence_scope_id", "topic_key", "predicate", "topic_key_source"}
    missing = required_cols - set(claims_df.columns)
    if missing:
        return AuditResult(
            gate_id="semkg_precedence_links",
            passed=False,
            total=len(claims_df),
            succeeded=0,
            threshold=1.0,
            details=f"claims.parquet missing columns: {sorted(missing)}",
        )
    scoped = claims_df.dropna(subset=["precedence_scope_id", "predicate", "topic_key"]).copy()
    scoped["group_kind"] = scoped.apply(_precedence_group_kind, axis=1)
    scoped["group_value"] = scoped.apply(_precedence_group_value, axis=1)
    group_sizes = scoped.groupby(["precedence_scope_id", "group_kind", "group_value"]).size()
    groups_with_competition = {key for key, size in group_sizes.items() if size > 1}
    claim_index = _build_claim_index(claims_df)
    valid_edges, invalid_reasons = _validate_precedence_edges(edges_df, claim_index)
    if not groups_with_competition:
        passed = not invalid_reasons and edges_df.empty
        details = "No competing claims detected"
        if invalid_reasons:
            details += f"; invalid_edges={len(invalid_reasons)}"
        elif not edges_df.empty:
            details += "; unexpected precedence edges"
        return AuditResult(
            gate_id="semkg_precedence_links",
            passed=passed,
            total=0,
            succeeded=0,
            threshold=0.0,
            details=details,
        )
    precedes_edges = [
        edge for edge in valid_edges
        if edge[6] == "precedes"
    ]
    groups_with_edges = {(scope, kind, value) for scope, kind, value, _, _, _, _, _ in precedes_edges}
    missing_groups = groups_with_competition - groups_with_edges
    expected_precedes = {key: max(0, size - 1) for key, size in group_sizes.items() if size > 1}
    precedes_counts: Dict[Tuple[str, str, str], int] = {}
    for scope, kind, value, _, _, _, _, _ in precedes_edges:
        precedes_counts[(scope, kind, value)] = precedes_counts.get((scope, kind, value), 0) + 1
    bad_counts = [
        key for key, expected in expected_precedes.items()
        if precedes_counts.get(key, 0) != expected
    ]
    passed = not invalid_reasons and not missing_groups and not bad_counts
    details = f"precedes_groups={len(groups_with_edges)}/{len(groups_with_competition)}"
    if invalid_reasons:
        details += f"; invalid_edges={len(invalid_reasons)}"
    if missing_groups:
        sample = list(missing_groups)[:3]
        details += f"; missing_groups={sample}"
    if bad_counts:
        sample = bad_counts[:3]
        details += f"; bad_precedes_counts={sample}"
    return AuditResult(
        gate_id="semkg_precedence_links",
        passed=passed,
        total=len(groups_with_competition),
        succeeded=len(groups_with_competition) - len(missing_groups),
        threshold=1.0,
        details=details,
    )


def precedence_chain_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    edges_df = _load_parquet(output_dir / "claim_precedence.parquet")
    groups_df = _load_parquet(output_dir / "resolution_groups.parquet")
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_precedence_chain",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no claims available",
        )
    group_map = _build_group_claim_map(claims_df)
    competing_groups = {key for key, ids in group_map.items() if len(ids) >= 2}
    if not competing_groups:
        return AuditResult(
            gate_id="semkg_precedence_chain",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no competing claim groups",
        )
    if groups_df.empty:
        return AuditResult(
            gate_id="semkg_precedence_chain",
            passed=False,
            total=len(competing_groups),
            succeeded=0,
            threshold=1.0,
            details="resolution_groups.parquet empty",
        )
    claim_index = _build_claim_index(claims_df)
    valid_edges, _ = _validate_precedence_edges(edges_df, claim_index)
    precedes_edges = [edge for edge in valid_edges if edge[6] == "precedes"]
    edges_by_group: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
    for scope, kind, value, _, from_id, to_id, _, _ in precedes_edges:
        edges_by_group.setdefault((scope, kind, value), []).append((from_id, to_id))

    failures = 0
    for group_key in competing_groups:
        scope_id, kind, value = group_key
        group_row = groups_df[
            (groups_df["precedence_scope_id"].astype(str) == str(scope_id))
            & (groups_df["group_kind"].astype(str) == str(kind))
            & (groups_df["group_value"].astype(str) == str(value))
        ]
        if group_row.empty:
            failures += 1
            continue
        status = str(group_row["status"].iloc[0])
        if status == "SKIP_SINGLETON":
            continue
        if status != "RESOLVED":
            failures += 1
            continue
        expected_ids = group_map.get(group_key, [])
        edges = edges_by_group.get(group_key, [])
        if len(edges) != len(expected_ids) - 1:
            failures += 1
            continue
        order = _chain_order_from_edges(edges)
        resolved_subset = resolved_df[
            (resolved_df["precedence_scope_id"].astype(str) == str(scope_id))
            & (resolved_df["group_kind"].astype(str) == str(kind))
            & (resolved_df["group_value"].astype(str) == str(value))
        ]
        ranks = resolved_subset.sort_values("rank")["claim_id"].astype(str).tolist()
        if order != ranks:
            failures += 1
    passed = failures == 0
    details = f"groups_with_valid_chain={len(competing_groups) - failures}/{len(competing_groups)}"
    return AuditResult(
        gate_id="semkg_precedence_chain",
        passed=passed,
        total=len(competing_groups),
        succeeded=len(competing_groups) - failures,
        threshold=1.0,
        details=details,
    )


def precedence_dag_gate(output_dir: Path) -> AuditResult:
    edges_df = _load_parquet(output_dir / "claim_precedence.parquet")
    if edges_df.empty:
        return AuditResult(
            gate_id="semkg_precedence_dag",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No precedence edges",
        )
    claims_df = _load_claims(output_dir)
    claim_index = _build_claim_index(claims_df)
    valid_edges, invalid_reasons = _validate_precedence_edges(edges_df, claim_index)
    edges_by_group: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
    for scope, kind, value, _, from_id, to_id, relation, _ in valid_edges:
        if relation != "precedes":
            continue
        edges_by_group.setdefault((scope, kind, value), []).append((from_id, to_id))

    def has_cycle(edge_list: List[Tuple[str, str]]) -> bool:
        adjacency: Dict[str, List[str]] = {}
        for src, dst in edge_list:
            adjacency.setdefault(src, []).append(dst)

        visited: Dict[str, int] = {}

        def dfs(node: str) -> bool:
            state = visited.get(node, 0)
            if state == 1:
                return True
            if state == 2:
                return False
            visited[node] = 1
            for neighbor in adjacency.get(node, []):
                if dfs(neighbor):
                    return True
            visited[node] = 2
            return False

        return any(dfs(node) for node in adjacency)

    cyclic_groups = [group for group, edges in edges_by_group.items() if has_cycle(edges)]
    passed = len(cyclic_groups) == 0 and not invalid_reasons
    details = "Precedence graph is a DAG"
    if invalid_reasons:
        details = f"Invalid precedence edges detected ({len(invalid_reasons)})"
    elif cyclic_groups:
        details = f"Cycles detected in groups: {cyclic_groups[:3]}"
    return AuditResult(
        gate_id="semkg_precedence_dag",
        passed=passed,
        total=len(edges_df),
        succeeded=len(edges_df) if passed else 0,
        threshold=1.0,
        details=details,
    )


def precedence_locality_gate(output_dir: Path) -> AuditResult:
    edges_df = _load_parquet(output_dir / "claim_precedence.parquet")
    if edges_df.empty:
        return AuditResult(
            gate_id="semkg_precedence_locality",
            passed=True,
            total=0,
            succeeded=0,
            threshold=DEFAULT_PRECEDENCE_SENTENCE_DISTANCE,
            details="No precedence edges",
        )
    distances = edges_df.get("sentence_distance")
    if distances is None or distances.empty:
        return AuditResult(
            gate_id="semkg_precedence_locality",
            passed=False,
            total=len(edges_df),
            succeeded=0,
            threshold=DEFAULT_PRECEDENCE_SENTENCE_DISTANCE,
            details="No sentence distance metadata",
        )
    claims_df = _load_claims(output_dir)
    claim_index = _build_claim_index(claims_df)
    valid_edges, invalid_reasons = _validate_precedence_edges(edges_df, claim_index)
    by_predicate: Dict[str, List[int]] = {}
    invalid_distances = 0
    too_long = 0
    for scope, kind, value, predicate, _, _, relation, distance in valid_edges:
        if relation != "precedes":
            continue
        if distance is None:
            invalid_distances += 1
            continue
        dist_value = int(distance)
        by_predicate.setdefault(str(predicate), []).append(dist_value)
        if dist_value > precedence_sentence_distance_limit(str(predicate)):
            too_long += 1

    failures = {}
    for predicate, values in by_predicate.items():
        if not values:
            continue
        p95 = float(pd.Series(values).quantile(0.95))
        limit = precedence_sentence_distance_limit(predicate)
        if p95 > limit:
            failures[predicate] = {"p95": p95, "limit": limit}

    passed = not failures and invalid_distances == 0 and too_long == 0 and not invalid_reasons
    details = "P95 sentence distance within limits"
    if invalid_reasons:
        details = f"Invalid precedence edges detected ({len(invalid_reasons)})"
    elif invalid_distances:
        details = f"{invalid_distances} edges missing sentence_distance"
    elif too_long:
        details = f"{too_long} edges exceed sentence distance limit"
    if failures:
        samples = list(failures.items())[:3]
        parts = [f"{pred}:p95={vals['p95']:.2f}>limit={vals['limit']}" for pred, vals in samples]
        details = f"Predicate locality violations: {', '.join(parts)}"
        if invalid_distances:
            details += f"; missing_distance={invalid_distances}"
        if too_long:
            details += f"; too_long={too_long}"
    return AuditResult(
        gate_id="semkg_precedence_locality",
        passed=passed,
        total=len(distances),
        succeeded=len(distances) if passed else 0,
        threshold=DEFAULT_PRECEDENCE_SENTENCE_DISTANCE,
        details=details,
    )


def paragraph_node_alignment_gate(output_dir: Path) -> AuditResult:
    edges_df = _load_parquet(output_dir / "paragraph_authority_edges.parquet")
    nodes_df = _load_parquet(output_dir / "graph_nodes.parquet")
    if edges_df.empty:
        return AuditResult(
            gate_id="semkg_paragraph_node_alignment",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No paragraph-authority edges to audit",
        )
    paragraph_nodes = nodes_df[nodes_df.get("node_type") == "paragraph"] if not nodes_df.empty else pd.DataFrame()
    node_map = paragraph_nodes.set_index("node_id").to_dict(orient="index") if not paragraph_nodes.empty else {}
    missing_node = element_mismatch = 0
    for _, row in edges_df.iterrows():
        node_id = row.get("paragraph_node_id")
        if node_id not in node_map:
            missing_node += 1
            continue
        node_element = str(node_map[node_id].get("element_id"))
        edge_element = str(row.get("source_element_id"))
        if node_element != edge_element:
            element_mismatch += 1
    failures = missing_node + element_mismatch
    total = len(edges_df)
    succeeded = total - failures
    passed = failures == 0
    details = (
        f"{succeeded}/{total} paragraph edges aligned "
        f"(missing_nodes={missing_node}, element_mismatches={element_mismatch})"
    )
    return AuditResult(
        gate_id="semkg_paragraph_node_alignment",
        passed=passed,
        total=total,
        succeeded=succeeded,
        threshold=1.0,
        details=details,
    )


def claim_authority_coverage_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    refs_df = _load_parquet(output_dir / "references.parquet")
    mentions_df = _load_claim_authority_mentions(output_dir)
    if claims_df.empty or refs_df.empty:
        return AuditResult(
            gate_id="semkg_claim_authority_coverage",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No claims or references to evaluate",
        )
    reference_keys = {
        (str(row.get("doc_id")), str(row.get("source_element_id")), int(row.get("sentence_idx")))
        for _, row in refs_df.iterrows()
        if row.get("source_element_id") and row.get("sentence_idx") is not None
    }
    claims_df = claims_df.dropna(subset=["sentence_idx"])
    claims_df["sentence_key"] = list(
        zip(
            claims_df["doc_id"].astype(str),
            claims_df["source_element_id"].astype(str),
            claims_df["sentence_idx"].astype(int),
        )
    )
    applicable = claims_df[claims_df["sentence_key"].isin(reference_keys)]
    total = len(applicable)
    if total == 0:
        return AuditResult(
            gate_id="semkg_claim_authority_coverage",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No claims share sentences with references",
        )
    mention_claim_ids = set(mentions_df["claim_id"]) if not mentions_df.empty else set()
    covered = int(applicable["claim_id"].isin(mention_claim_ids).sum())
    passed = covered == total
    details = f"{covered}/{total} claims with same-sentence references linked to authorities"
    return AuditResult(
        gate_id="semkg_claim_authority_coverage",
        passed=passed,
        total=total,
        succeeded=covered,
        threshold=1.0,
        details=details,
    )


def claim_authority_fanout_gate(output_dir: Path) -> AuditResult:
    mentions_df = _load_claim_authority_mentions(output_dir)
    if mentions_df.empty:
        return AuditResult(
            gate_id="semkg_claim_authority_fanout",
            passed=True,
            total=0,
            succeeded=0,
            threshold=5.0,
            details="No claim authority mentions recorded",
        )
    counts = mentions_df["claim_id"].value_counts()
    p95 = float(counts.quantile(0.95))
    threshold = 5.0
    passed = p95 <= threshold
    details = f"95th percentile authorities per claim: {p95:.2f}"
    return AuditResult(
        gate_id="semkg_claim_authority_fanout",
        passed=passed,
        total=len(counts),
        succeeded=len(counts[counts <= threshold]),
        threshold=threshold,
        details=details,
    )


def reference_span_gate(output_dir: Path) -> AuditResult:
    refs_df = _load_parquet(output_dir / "references.parquet")
    sentences_df = _load_parquet(output_dir / "element_sentences.parquet")
    if refs_df.empty or sentences_df.empty:
        return AuditResult(
            gate_id="semkg_reference_spans",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No references or sentence index found",
        )
    sentence_map = sentences_df.set_index(["source_element_id", "sentence_idx"])
    checked = invalid = 0
    for _, ref in refs_df.iterrows():
        element_id = ref.get("source_element_id")
        sentence_idx = ref.get("sentence_idx")
        if element_id is None or sentence_idx is None:
            continue
        key = (str(element_id), int(sentence_idx))
        if key not in sentence_map.index:
            invalid += 1
            checked += 1
            continue
        sentence_text = sentence_map.loc[key]["sentence_text"] or ""
        start = ref.get("char_start")
        end = ref.get("char_end")
        try:
            start = int(start)
            end = int(end)
        except Exception:
            invalid += 1
            checked += 1
            continue
        checked += 1
        if start < 0 or end > len(sentence_text) or start >= end:
            invalid += 1
    passed = invalid == 0
    details = f"{checked - invalid}/{checked} sentence spans valid" if checked else "No spans to audit"
    return AuditResult(
        gate_id="semkg_reference_spans",
        passed=passed,
        total=checked,
        succeeded=checked - invalid,
        threshold=1.0,
        details=details,
    )


def reference_span_exactness_gate(output_dir: Path, sample_size: int = 25) -> AuditResult:
    refs_df = _load_parquet(output_dir / "references.parquet")
    claims_df = _load_claims(output_dir)
    sentences_df = _load_parquet(output_dir / "element_sentences.parquet")
    if sentences_df.empty or (refs_df.empty and claims_df.empty):
        return AuditResult(
            gate_id="semkg_span_exactness",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No references or claims available for spot-check",
        )
    sentence_map = sentences_df.set_index(["source_element_id", "sentence_idx"])

    def _check(df: pd.DataFrame, text_column: str) -> Tuple[int, int]:
        if df.empty:
            return 0, 0
        sample_n = min(sample_size, len(df))
        sample_df = df.sample(n=sample_n, random_state=42) if sample_n else df
        mismatches = checked = 0
        for _, row in sample_df.iterrows():
            element_id = row.get("source_element_id")
            sentence_idx = row.get("sentence_idx")
            if element_id is None or sentence_idx is None:
                continue
            key = (str(element_id), int(sentence_idx))
            if key not in sentence_map.index:
                mismatches += 1
                checked += 1
                continue
            start = row.get("char_start")
            end = row.get("char_end")
            try:
                start = int(start)
                end = int(end)
            except Exception:
                mismatches += 1
                checked += 1
                continue
            sentence_text = sentence_map.loc[key]["sentence_text"] or ""
            snippet = sentence_text[start:end]
            target = row.get(text_column) or ""
            norm_snippet = " ".join(snippet.split()).lower()
            norm_target = " ".join(str(target).split()).lower()
            checked += 1
            if norm_target and norm_snippet:
                if norm_target not in norm_snippet and norm_snippet not in norm_target:
                    mismatches += 1
            elif norm_target:
                mismatches += 1
        return mismatches, checked

    ref_mismatch, ref_checked = _check(refs_df, "ref_text")
    claim_mismatch, claim_checked = _check(claims_df, "evidence_text")
    total = ref_checked + claim_checked
    mismatches = ref_mismatch + claim_mismatch
    passed = mismatches == 0
    details = (
        f"references: {ref_checked - ref_mismatch}/{ref_checked} exact, "
        f"claims: {claim_checked - claim_mismatch}/{claim_checked} exact"
    )
    return AuditResult(
        gate_id="semkg_span_exactness",
        passed=passed,
        total=total,
        succeeded=total - mismatches,
        threshold=1.0,
        details=details,
    )


def canonicalization_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_canonicalization",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No claims emitted",
        )
    missing_subject = claims_df["subject_canonical_id"].isna().sum()
    missing_object = claims_df["object_canonical_id"].isna().sum()
    total = len(claims_df)
    passed = (missing_subject + missing_object) == 0
    details = f"{total - missing_subject}/{total} claims have subject canonical IDs; {total - missing_object}/{total} have object IDs"
    return AuditResult(
        gate_id="semkg_canonicalization",
        passed=passed,
        total=total,
        succeeded=total - (missing_subject + missing_object),
        threshold=1.0,
        details=details,
    )


def sentence_span_gate(output_dir: Path) -> AuditResult:
    rejections_df = _load_rejections(output_dir)
    if rejections_df.empty:
        return AuditResult(
            gate_id="semkg_sentence_spans",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No claim rejections recorded",
        )
    invalid = rejections_df[rejections_df["reason"].str.contains("span", na=False)]
    passed = invalid.empty
    details = f"{len(invalid)} span-related rejections"
    return AuditResult(
        gate_id="semkg_sentence_spans",
        passed=passed,
        total=len(rejections_df),
        succeeded=len(rejections_df) - len(invalid),
        threshold=1.0,
        details=details,
    )


def resolution_nonempty_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_resolution_nonempty",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no claims available",
        )
    required_cols = {"precedence_scope_id", "topic_key", "predicate", "topic_key_source"}
    missing = required_cols - set(claims_df.columns)
    if missing:
        return AuditResult(
            gate_id="semkg_resolution_nonempty",
            passed=False,
            total=len(claims_df),
            succeeded=0,
            threshold=1.0,
            details=f"claims.parquet missing columns: {sorted(missing)}",
        )
    scoped = claims_df.dropna(subset=["precedence_scope_id", "predicate", "topic_key"]).copy()
    scoped["group_kind"] = scoped.apply(_precedence_group_kind, axis=1)
    scoped["group_value"] = scoped.apply(_precedence_group_value, axis=1)
    group_sizes = scoped.groupby(["precedence_scope_id", "group_kind", "group_value"]).size()
    competing_groups = {key for key, size in group_sizes.items() if size >= 2}
    if not competing_groups:
        return AuditResult(
            gate_id="semkg_resolution_nonempty",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no competing claim groups",
        )
    groups_df = _load_parquet(output_dir / "resolution_groups.parquet")
    if groups_df.empty:
        return AuditResult(
            gate_id="semkg_resolution_nonempty",
            passed=False,
            total=len(competing_groups),
            succeeded=0,
            threshold=1.0,
            details="resolution_groups.parquet empty",
        )
    groups_df = groups_df.copy()
    groups_df["group_key"] = list(
        zip(
            groups_df["precedence_scope_id"].astype(str),
            groups_df["group_kind"].astype(str),
            groups_df["group_value"].astype(str),
        )
    )
    competing_keys = set((str(scope), str(kind), str(value)) for scope, kind, value in competing_groups)
    competing_rows = groups_df[groups_df["group_key"].isin(competing_keys)]
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    if resolved_df.empty:
        statuses = competing_rows["status"].astype(str).tolist()
        if any(status == "RESOLVED" for status in statuses):
            return AuditResult(
                gate_id="semkg_resolution_nonempty",
                passed=False,
                total=len(competing_groups),
                succeeded=0,
                threshold=1.0,
                details="resolved_claims.parquet empty for resolved groups",
            )
        no_edges = sum(1 for status in statuses if status == "SKIP_NO_EDGES")
        details = "SKIP: no resolved rows"
        if no_edges > RESOLUTION_NO_EDGES_WARN_THRESHOLD:
            details = f"WARN: {no_edges} groups missing precedes edges; {details}"
        return AuditResult(
            gate_id="semkg_resolution_nonempty",
            passed=True,
            total=len(competing_groups),
            succeeded=0,
            threshold=1.0,
            details=details,
        )
    resolved_df = resolved_df.copy()
    group_cols = {"precedence_scope_id", "group_kind", "group_value", "rank"}
    if not group_cols.issubset(resolved_df.columns):
        return AuditResult(
            gate_id="semkg_resolution_nonempty",
            passed=False,
            total=len(competing_groups),
            succeeded=0,
            threshold=1.0,
            details="resolved_claims.parquet missing group columns",
        )
    failures = 0
    no_edges = 0
    for key in competing_groups:
        scope_id, kind, value = key
        group_row = groups_df[
            (groups_df["precedence_scope_id"].astype(str) == str(scope_id))
            & (groups_df["group_kind"].astype(str) == str(kind))
            & (groups_df["group_value"].astype(str) == str(value))
        ]
        if group_row.empty:
            failures += 1
            continue
        status = str(group_row["status"].iloc[0])
        if status == "SKIP_SINGLETON":
            continue
        if status == "SKIP_NO_EDGES":
            no_edges += 1
            continue
        subset = resolved_df[
            (resolved_df["precedence_scope_id"].astype(str) == str(scope_id))
            & (resolved_df["group_kind"].astype(str) == str(kind))
            & (resolved_df["group_value"].astype(str) == str(value))
        ]
        expected_size = int(group_sizes[key])
        ranks = sorted(subset["rank"].dropna().astype(int).tolist())
        if ranks != list(range(1, expected_size + 1)):
            failures += 1
    passed = failures == 0
    details = f"groups_with_valid_ranks={len(competing_groups) - failures}/{len(competing_groups)}"
    if no_edges > RESOLUTION_NO_EDGES_WARN_THRESHOLD:
        details = f"WARN: {no_edges} groups missing precedes edges; {details}"
    return AuditResult(
        gate_id="semkg_resolution_nonempty",
        passed=passed,
        total=len(competing_groups),
        succeeded=len(competing_groups) - failures,
        threshold=1.0,
        details=details,
    )


def resolution_group_integrity_gate(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    groups_df = _load_parquet(output_dir / "resolution_groups.parquet")
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    if claims_df.empty:
        return AuditResult(
            gate_id="semkg_resolution_group_integrity",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no claims available",
        )
    group_map = _build_group_claim_map(claims_df)
    if not group_map:
        return AuditResult(
            gate_id="semkg_resolution_group_integrity",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no precedence groups",
        )
    if groups_df.empty:
        return AuditResult(
            gate_id="semkg_resolution_group_integrity",
            passed=False,
            total=len(group_map),
            succeeded=0,
            threshold=1.0,
            details="resolution_groups.parquet empty",
        )
    failures = 0
    for group_key, claim_ids in group_map.items():
        scope_id, kind, value = group_key
        group_row = groups_df[
            (groups_df["precedence_scope_id"].astype(str) == str(scope_id))
            & (groups_df["group_kind"].astype(str) == str(kind))
            & (groups_df["group_value"].astype(str) == str(value))
        ]
        if group_row.empty:
            failures += 1
            continue
        status = str(group_row["status"].iloc[0])
        subset = resolved_df[
            (resolved_df["precedence_scope_id"].astype(str) == str(scope_id))
            & (resolved_df["group_kind"].astype(str) == str(kind))
            & (resolved_df["group_value"].astype(str) == str(value))
        ]
        if status in {"SKIP_SINGLETON", "SKIP_NO_EDGES"}:
            if not subset.empty:
                failures += 1
            continue
        resolved_ids = set(subset["claim_id"].astype(str).tolist())
        expected_ids = set(str(cid) for cid in claim_ids)
        if resolved_ids != expected_ids:
            failures += 1
            continue
        ranks = sorted(subset["rank"].dropna().astype(int).tolist())
        if ranks != list(range(1, len(expected_ids) + 1)):
            failures += 1
            continue
        if subset["topic_key"].astype(str).nunique() > 1:
            failures += 1
    passed = failures == 0
    details = f"groups_with_integrity={len(group_map) - failures}/{len(group_map)}"
    return AuditResult(
        gate_id="semkg_resolution_group_integrity",
        passed=passed,
        total=len(group_map),
        succeeded=len(group_map) - failures,
        threshold=1.0,
        details=details,
    )


def resolution_determinism_gate(output_dir: Path) -> AuditResult:
    metadata = _load_json(output_dir / "resolution_run_metadata.json")
    compiler_meta = _load_json(output_dir / "compiler_run_metadata.json")
    if not metadata or not compiler_meta:
        return AuditResult(
            gate_id="semkg_resolution_determinism",
            passed=False,
            total=1,
            succeeded=0,
            threshold=1.0,
            details="Missing resolution or compiler metadata",
        )
    claims_df = _load_claims(output_dir)
    precedence_df = _load_parquet(output_dir / "claim_precedence.parquet")
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    groups_df = _load_parquet(output_dir / "resolution_groups.parquet")
    directives_df = _load_parquet(output_dir / "compiled_directives.parquet")
    constraints_df = _load_parquet(output_dir / "constraints_resolved.parquet")
    input_hash = _hash_dataframe(claims_df) + _hash_dataframe(precedence_df)
    output_hash = (
        _hash_dataframe(resolved_df)
        + _hash_dataframe(groups_df)
        + _hash_dataframe(directives_df)
        + _hash_dataframe(constraints_df)
        + _hash_traces(output_dir / "resolution_traces")
    )
    prev_input = metadata.get("previous_input_hash")
    prev_output = (
        str(metadata.get("previous_output_hash") or "")
        + str(compiler_meta.get("previous_directives_hash") or "")
        + str(compiler_meta.get("previous_constraints_hash") or "")
        + str(metadata.get("previous_trace_hash") or "")
    )
    if not prev_input or not prev_output:
        return AuditResult(
            gate_id="semkg_resolution_determinism",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no prior metadata to compare",
        )
    if prev_input != input_hash:
        return AuditResult(
            gate_id="semkg_resolution_determinism",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: input hash changed between runs",
        )
    passed = prev_output == output_hash
    details = "Resolution outputs stable across runs"
    if not passed:
        details = "Resolution outputs changed for identical inputs"
    return AuditResult(
        gate_id="semkg_resolution_determinism",
        passed=passed,
        total=1,
        succeeded=1 if passed else 0,
        threshold=1.0,
        details=details,
    )


def resolution_trace_completeness_gate(output_dir: Path) -> AuditResult:
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    if resolved_df.empty:
        return AuditResult(
            gate_id="semkg_resolution_trace_completeness",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="No resolved claims",
        )
    required = {"status", "missing_provenance_reason", "source_element_id", "sentence_idx", "char_start", "char_end"}
    if not required.issubset(resolved_df.columns):
        return AuditResult(
            gate_id="semkg_resolution_trace_completeness",
            passed=False,
            total=len(resolved_df),
            succeeded=0,
            threshold=1.0,
            details="resolved_claims.parquet missing provenance columns",
        )
    active = resolved_df[resolved_df["status"].isin({"applies", "suppressed", "exception"})]
    missing = active[
        active["missing_provenance_reason"].isna()
        & (
            active["source_element_id"].isna()
            | active["sentence_idx"].isna()
            | active["char_start"].isna()
            | active["char_end"].isna()
        )
    ]
    passed = missing.empty
    details = f"{len(active) - len(missing)}/{len(active)} resolved rows have provenance or reason"
    return AuditResult(
        gate_id="semkg_resolution_trace_completeness",
        passed=passed,
        total=len(active),
        succeeded=len(active) - len(missing),
        threshold=1.0,
        details=details,
    )


def compiler_ir_present_gate(output_dir: Path) -> AuditResult:
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    directives_df = _load_parquet(output_dir / "compiled_directives.parquet")
    if resolved_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_ir_present",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no resolved claims",
        )
    active = resolved_df[resolved_df["status"].isin({"applies", "exception"})]
    if active.empty:
        return AuditResult(
            gate_id="semkg_compiler_ir_present",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no active resolved claims",
        )
    required = {"directive_id", "precedence_scope_id", "topic_key", "op", "supporting_claim_ids"}
    if directives_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_ir_present",
            passed=False,
            total=len(active),
            succeeded=0,
            threshold=1.0,
            details="compiled_directives.parquet empty",
        )
    if not required.issubset(directives_df.columns):
        return AuditResult(
            gate_id="semkg_compiler_ir_present",
            passed=False,
            total=len(active),
            succeeded=0,
            threshold=1.0,
            details="compiled_directives.parquet missing required columns",
        )
    passed = len(directives_df) > 0
    details = f"compiled_directives_rows={len(directives_df)}"
    return AuditResult(
        gate_id="semkg_compiler_ir_present",
        passed=passed,
        total=len(active),
        succeeded=len(directives_df),
        threshold=1.0,
        details=details,
    )


def compiler_support_gate(output_dir: Path) -> AuditResult:
    directives_df = _load_parquet(output_dir / "compiled_directives.parquet")
    if directives_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_support_nonempty",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no compiled directives",
        )
    def _has_support(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        try:
            return len(value) > 0
        except Exception:
            return False

    missing_support = directives_df[
        ~directives_df["supporting_claim_ids"].apply(_has_support)
        | ~directives_df["supporting_spans"].apply(_has_support)
    ]
    passed = missing_support.empty
    details = f"{len(directives_df) - len(missing_support)}/{len(directives_df)} directives have support"
    return AuditResult(
        gate_id="semkg_compiler_support_nonempty",
        passed=passed,
        total=len(directives_df),
        succeeded=len(directives_df) - len(missing_support),
        threshold=1.0,
        details=details,
    )


def compiler_ir_schema_gate(output_dir: Path) -> AuditResult:
    directives_df = _load_parquet(output_dir / "compiled_directives.parquet")
    if directives_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_ir_schema",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no compiled directives",
        )
    required_cols = {"condition_ir", "supporting_spans"}
    if not required_cols.issubset(directives_df.columns):
        return AuditResult(
            gate_id="semkg_compiler_ir_schema",
            passed=False,
            total=len(directives_df),
            succeeded=0,
            threshold=1.0,
            details="compiled_directives.parquet missing condition_ir/supporting_spans",
        )
    required_condition_keys = {"kind", "predicate", "subject_canonical_id", "object_canonical_id"}
    required_span_keys = {"source_element_id", "sentence_idx", "char_start", "char_end"}
    invalid = 0
    for _, row in directives_df.iterrows():
        condition_ir = row.get("condition_ir")
        try:
            condition_payload = json.loads(condition_ir) if isinstance(condition_ir, str) else None
        except Exception:
            invalid += 1
            continue
        if not isinstance(condition_payload, dict):
            invalid += 1
            continue
        if set(condition_payload.keys()) != required_condition_keys:
            invalid += 1
            continue
        spans = row.get("supporting_spans")
        if isinstance(spans, str):
            invalid += 1
            continue
        if hasattr(spans, "tolist") and not isinstance(spans, (list, tuple, str)):
            spans = spans.tolist()
        if not isinstance(spans, list):
            invalid += 1
            continue
        span_ok = True
        for span in spans:
            if not isinstance(span, dict) or set(span.keys()) != required_span_keys:
                span_ok = False
                break
            if not isinstance(span.get("source_element_id"), str) or not span.get("source_element_id"):
                span_ok = False
                break
            for key in ("sentence_idx", "char_start", "char_end"):
                value = span.get(key)
                if not isinstance(value, Integral) or isinstance(value, bool):
                    span_ok = False
                    break
            if not span_ok:
                break
        if not span_ok:
            invalid += 1
            continue
    passed = invalid == 0
    details = f"{len(directives_df) - invalid}/{len(directives_df)} directives have valid IR schema"
    return AuditResult(
        gate_id="semkg_compiler_ir_schema",
        passed=passed,
        total=len(directives_df),
        succeeded=len(directives_df) - invalid,
        threshold=1.0,
        details=details,
    )


def compiler_directive_join_gate(output_dir: Path) -> AuditResult:
    directives_df = _load_parquet(output_dir / "compiled_directives.parquet")
    constraints_df = _load_parquet(output_dir / "constraints_resolved.parquet")
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    if directives_df.empty and constraints_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_directive_join",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no compiled directives or constraints",
        )
    if directives_df.empty or constraints_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_directive_join",
            passed=False,
            total=len(constraints_df),
            succeeded=0,
            threshold=1.0,
            details="Missing compiled_directives or constraints_resolved",
        )
    directive_ids = set(directives_df["directive_id"].dropna().astype(str)) if "directive_id" in directives_df.columns else set()
    missing_constraint = 0
    for _, row in constraints_df.iterrows():
        if str(row.get("directive_id") or "") not in directive_ids:
            missing_constraint += 1
    claim_ids = set(resolved_df["claim_id"].dropna().astype(str)) if not resolved_df.empty else set()
    missing_claims = 0
    for _, row in directives_df.iterrows():
        claims = row.get("supporting_claim_ids")
        if isinstance(claims, str):
            missing_claims += 1
            continue
        if hasattr(claims, "tolist") and not isinstance(claims, (list, tuple, str)):
            claims = claims.tolist()
        if not isinstance(claims, list):
            missing_claims += 1
            continue
        if any(str(cid) not in claim_ids for cid in claims):
            missing_claims += 1
    failures = missing_constraint + missing_claims
    passed = failures == 0
    details = f"constraint_directive_missing={missing_constraint}, directive_claim_missing={missing_claims}"
    return AuditResult(
        gate_id="semkg_compiler_directive_join",
        passed=passed,
        total=len(constraints_df) + len(directives_df),
        succeeded=len(constraints_df) + len(directives_df) - failures,
        threshold=1.0,
        details=details,
    )


def compiler_scope_topic_consistency_gate(output_dir: Path) -> AuditResult:
    directives_df = _load_parquet(output_dir / "compiled_directives.parquet")
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    if directives_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_scope_topic_consistency",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no compiled directives",
        )
    if resolved_df.empty:
        return AuditResult(
            gate_id="semkg_compiler_scope_topic_consistency",
            passed=False,
            total=len(directives_df),
            succeeded=0,
            threshold=1.0,
            details="resolved_claims.parquet empty",
        )
    resolved_index = resolved_df.set_index("claim_id")[["precedence_scope_id", "topic_key"]].to_dict("index")
    mismatches = 0
    checked = 0
    for _, row in directives_df.iterrows():
        directive_scope = str(row.get("precedence_scope_id") or "")
        directive_topic = str(row.get("topic_key") or "")
        claims = row.get("supporting_claim_ids")
        if isinstance(claims, str):
            continue
        if hasattr(claims, "tolist") and not isinstance(claims, (list, tuple, str)):
            claims = claims.tolist()
        if not isinstance(claims, list) or not claims:
            continue
        for claim_id in claims:
            info = resolved_index.get(str(claim_id))
            if not info:
                mismatches += 1
                checked += 1
                continue
            checked += 1
            if str(info.get("precedence_scope_id")) != directive_scope or str(info.get("topic_key")) != directive_topic:
                mismatches += 1
    passed = mismatches == 0
    details = f"{checked - mismatches}/{checked} claims match directive scope/topic"
    return AuditResult(
        gate_id="semkg_compiler_scope_topic_consistency",
        passed=passed,
        total=checked,
        succeeded=checked - mismatches,
        threshold=1.0,
        details=details,
    )


def predicate_outcome_coverage_gate(output_dir: Path) -> AuditResult:
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    groups_df = _load_parquet(output_dir / "resolution_groups.parquet")
    if resolved_df.empty:
        return AuditResult(
            gate_id="semkg_predicate_outcome_coverage",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no resolved claims",
        )
    if groups_df.empty:
        return AuditResult(
            gate_id="semkg_predicate_outcome_coverage",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no resolution groups",
        )
    competing = groups_df[(groups_df["group_size"] >= 2) & (groups_df["status"] == "RESOLVED")]
    if competing.empty:
        return AuditResult(
            gate_id="semkg_predicate_outcome_coverage",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="SKIP: no competing resolved groups",
        )
    statuses = resolved_df["status"].astype(str)
    has_suppressed = int((statuses == "suppressed").any())
    has_exception = int((statuses == "exception").any())
    passed = has_suppressed == 1 and has_exception == 1
    details = f"has_suppressed={bool(has_suppressed)}, has_exception={bool(has_exception)}"
    return AuditResult(
        gate_id="semkg_predicate_outcome_coverage",
        passed=passed,
        total=2,
        succeeded=has_suppressed + has_exception,
        threshold=2.0,
        details=details,
    )


def predicate_health_report(output_dir: Path) -> AuditResult:
    claims_df = _load_claims(output_dir)
    precedence_df = _load_parquet(output_dir / "claim_precedence.parquet")
    resolved_df = _load_parquet(output_dir / "resolved_claims.parquet")
    groups_df = _load_parquet(output_dir / "resolution_groups.parquet")
    directives_df = _load_parquet(output_dir / "compiled_directives.parquet")
    typed_edges_df = _load_parquet(output_dir / "typed_edges.parquet")

    input_hash = _hash_dataframe(claims_df) + _hash_dataframe(precedence_df)
    payload: Dict[str, object] = {
        "version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_hash": input_hash,
    }

    dead_predicates = []
    if not typed_edges_df.empty and "edge_type" in typed_edges_df.columns:
        emitted = set(typed_edges_df["edge_type"].dropna().astype(str))
        claimed = set(claims_df["predicate"].dropna().astype(str)) if "predicate" in claims_df.columns else set()
        dead_predicates = sorted(emitted - claimed)

    if resolved_df.empty:
        payload.update({
            "skipped_reason": "no_resolved_claims",
            "dead_predicates": dead_predicates,
        })
        _write_predicate_health(output_dir, payload)
        return AuditResult(
            gate_id="semkg_predicate_health_report",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="Predicate health skipped: no resolved claims",
        )

    competing_groups = groups_df[
        (groups_df["group_size"] >= 2) & (groups_df["status"] == "RESOLVED")
    ] if not groups_df.empty else pd.DataFrame()
    if competing_groups.empty:
        payload.update({
            "skipped_reason": "no_competing_groups",
            "dead_predicates": dead_predicates,
        })
        _write_predicate_health(output_dir, payload)
        return AuditResult(
            gate_id="semkg_predicate_health_report",
            passed=True,
            total=0,
            succeeded=0,
            threshold=1.0,
            details="Predicate health skipped: no competing groups",
        )

    status_counts = (
        resolved_df.groupby(["predicate", "status"])
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = status_counts.pivot(
        index="predicate",
        columns="status",
        values="count",
    ).fillna(0)
    predicate_stats = pivot.copy()
    for col in ["applies", "suppressed", "exception", "undecidable"]:
        if col not in predicate_stats.columns:
            predicate_stats[col] = 0
    predicate_stats["total"] = predicate_stats[
        ["applies", "suppressed", "exception", "undecidable"]
    ].sum(axis=1)
    predicate_stats["apply_rate"] = predicate_stats.apply(
        lambda row: row["applies"] / row["total"] if row["total"] else 0.0,
        axis=1,
    )
    predicate_stats["suppressed_rate"] = predicate_stats.apply(
        lambda row: row["suppressed"] / row["total"] if row["total"] else 0.0,
        axis=1,
    )
    predicate_stats["exception_rate"] = predicate_stats.apply(
        lambda row: row["exception"] / row["total"] if row["total"] else 0.0,
        axis=1,
    )

    claims_copy = claims_df.copy()
    claims_copy["group_kind"] = claims_copy["topic_key_source"].apply(
        lambda v: "canonical" if str(v or "").strip().lower() == "canonical_ids" else "predicate"
    )
    claims_copy["group_value"] = claims_copy.apply(
        lambda row: row.get("topic_key") if row.get("group_kind") == "canonical" else row.get("predicate"),
        axis=1,
    )
    resolved_keys = set(
        zip(
            competing_groups["precedence_scope_id"].astype(str),
            competing_groups["group_kind"].astype(str),
            competing_groups["group_value"].astype(str),
        )
    )
    claims_copy["group_key"] = list(
        zip(
            claims_copy["precedence_scope_id"].astype(str),
            claims_copy["group_kind"].astype(str),
            claims_copy["group_value"].astype(str),
        )
    )
    competing_predicates = set(
        claims_copy[claims_copy["group_key"].isin(resolved_keys)]["predicate"].dropna().astype(str)
    )

    min_total_for_never_wins = 5
    predicate_never_wins = predicate_stats[
        (predicate_stats["apply_rate"] == 0.0)
        & (predicate_stats["total"] >= min_total_for_never_wins)
        & predicate_stats.index.isin(competing_predicates)
    ]

    resolved_active_predicates = set(
        resolved_df[resolved_df["status"].isin(["applies", "exception"])]["predicate"].dropna().astype(str)
    )
    directives_predicates = set(directives_df["op"].dropna().astype(str)) if not directives_df.empty else set()
    never_compiled = sorted(resolved_active_predicates - directives_predicates)

    resolved_predicates = set(resolved_df["predicate"].dropna().astype(str))
    claims_predicates = set(claims_df["predicate"].dropna().astype(str)) if "predicate" in claims_df.columns else set()
    never_resolved = sorted(claims_predicates - resolved_predicates)

    previous_health = _load_json(output_dir / "predicate_health.json")
    previous_signature = previous_health.get("input_hash")
    previous_stats = {
        row["predicate"]: row
        for row in previous_health.get("predicate_stats", [])
    }
    unstable = []
    min_total_for_stability = 50
    if previous_signature == input_hash:
        for predicate, row in predicate_stats.reset_index().set_index("predicate").iterrows():
            prev = previous_stats.get(predicate)
            if not prev:
                continue
            if row["total"] < min_total_for_stability or prev.get("total", 0) < min_total_for_stability:
                continue
            if abs(row["apply_rate"] - prev.get("apply_rate", 0.0)) > 0.05:
                unstable.append(predicate)

    payload.update({
        "predicate_stats": predicate_stats.reset_index().to_dict(orient="records"),
        "predicate_never_wins": predicate_never_wins.reset_index().to_dict(orient="records"),
        "dead_predicates": dead_predicates,
        "never_compiled_predicates": never_compiled,
        "never_resolved_predicates": never_resolved,
        "unstable_predicates": unstable,
        "min_total_for_never_wins": min_total_for_never_wins,
        "min_total_for_stability": min_total_for_stability,
    })
    _write_predicate_health(output_dir, payload)

    return AuditResult(
        gate_id="semkg_predicate_health_report",
        passed=True,
        total=0,
        succeeded=0,
        threshold=1.0,
        details="Predicate health report written",
    )


def _hash_dataframe(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    norm = df.copy()
    for col in norm.columns:
        norm[col] = norm[col].apply(_normalize_cell)
    norm = norm.sort_values(list(norm.columns)).reset_index(drop=True)
    payload = norm.to_csv(index=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if hasattr(value, "tolist") and not isinstance(value, str):
        return json.dumps(value.tolist(), sort_keys=True)
    if isinstance(value, float) and pd.isna(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _hash_traces(trace_dir: Path) -> str:
    if not trace_dir.exists():
        return ""
    digests: List[str] = []
    for path in sorted(trace_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        digests.append(digest)
    return sha256("".join(digests).encode("utf-8")).hexdigest()


def _chain_order_from_edges(edges: List[Tuple[str, str]]) -> List[str]:
    adjacency: Dict[str, str] = {}
    indegree: Dict[str, int] = {}
    for src, dst in edges:
        adjacency[src] = dst
        indegree[dst] = indegree.get(dst, 0) + 1
        indegree.setdefault(src, 0)
    starts = [cid for cid, deg in indegree.items() if deg == 0]
    if len(starts) != 1:
        return []
    order: List[str] = []
    cur = starts[0]
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        order.append(cur)
        cur = adjacency.get(cur)
    return order


def _write_predicate_health(output_dir: Path, payload: Dict[str, object]) -> None:
    path = output_dir / "predicate_health.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


__all__ = ["run_semkg_audits", "AuditResult"]
