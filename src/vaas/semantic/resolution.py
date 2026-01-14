"""
Claim resolution and precedence application.

Resolver Contract
1) Input: claims.parquet, claim_precedence.parquet
2) Group key: (precedence_scope_id, topic_key); if topic_key_source != canonical_ids,
   group by predicate within precedence_scope_id and use topic_key = predicate|scope:<scope>
3) Required edges: precedes forms a chain (N-1) when competition exists; else SKIP_NO_COMPETITION
4) Statuses: applies | suppressed | exception | undecidable
5) Tie-breaker: (source_element_id, sentence_idx, char_start, claim_id)
6) Failure: malformed graphs => ERROR_GRAPH and no applies without error recording
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from vaas.schemas.semantic_contract import ResolvedClaimRow, ResolutionGroupRow
from vaas.semantic.policy import precedence_sentence_distance_limit


ALLOWED_STATUSES = {"applies", "suppressed", "exception", "undecidable"}
GROUP_STATUS_ERROR = "ERROR_GRAPH"
GROUP_STATUS_SKIP = "SKIP_NO_COMPETITION"
GROUP_STATUS_RESOLVED = "RESOLVED"


def resolve_claims(
    claims_df: pd.DataFrame,
    precedence_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, int]:
    """Resolve claims into ordered groups + statuses."""
    required = {"claim_id", "precedence_scope_id", "topic_key", "topic_key_source", "predicate"}
    missing = required - set(claims_df.columns)
    if missing:
        raise ValueError(f"claims.parquet missing columns: {sorted(missing)}")

    claims = claims_df.copy()
    claims["group_kind"] = claims.apply(_group_kind, axis=1)
    claims["group_value"] = claims.apply(_group_value, axis=1)
    claims["group_topic_key"] = claims.apply(_group_topic_key, axis=1)

    resolved_rows: List[ResolvedClaimRow] = []
    group_rows: List[ResolutionGroupRow] = []
    trace_index: List[Dict[str, object]] = []

    for (scope_id, kind, value), group in claims.groupby(
        ["precedence_scope_id", "group_kind", "group_value"],
        dropna=True,
    ):
        group_topic_key = group["group_topic_key"].iloc[0]
        group_claims = group.sort_values(
            ["source_element_id", "sentence_idx", "char_start", "claim_id"]
        ).copy()
        group_ids = set(group_claims["claim_id"].astype(str))
        group_edges = _filter_group_edges(precedence_df, scope_id, group_topic_key)

        status = GROUP_STATUS_RESOLVED
        error_code = None
        error_detail = None
        chain_order: List[str] = []

        if len(group_claims) < 2:
            status = GROUP_STATUS_SKIP
            chain_order = list(group_claims["claim_id"].astype(str))
        else:
            error_code, error_detail = _validate_group_edges(group_claims, group_edges, group_ids)
            if error_code:
                status = GROUP_STATUS_ERROR
                chain_order = list(group_claims["claim_id"].astype(str))
            else:
                chain_order = _build_chain(group_edges["precedes"], group_ids)

        resolved = _resolve_group_rows(
            group_claims,
            chain_order,
            group_edges["exception_of"],
            status,
            error_code,
        )
        resolved_rows.extend(resolved)

        counts = _status_counts(resolved)
        group_rows.append(ResolutionGroupRow(
            precedence_scope_id=str(scope_id),
            group_kind=str(kind),
            group_value=str(value),
            topic_key=str(group_topic_key),
            group_size=len(group_claims),
            status=status,
            error_code=error_code,
            error_detail=error_detail,
            applies_count=counts.get("applies", 0),
            suppressed_count=counts.get("suppressed", 0),
            exception_count=counts.get("exception", 0),
            undecidable_count=counts.get("undecidable", 0),
        ))

        trace_index.append(_write_trace(
            output_dir=output_dir,
            scope_id=scope_id,
            group_key=str(group_topic_key),
            group_status=status,
            error_code=error_code,
            claims=resolved,
            edges=group_edges,
        ))

    resolved_df = _rows_to_df(resolved_rows)
    group_df = _rows_to_df(group_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_df.to_parquet(output_dir / "resolved_claims.parquet", index=False)
    group_df.to_parquet(output_dir / "resolution_groups.parquet", index=False)
    _write_trace_index(output_dir, trace_index)
    _write_metadata(output_dir, claims_df, precedence_df, resolved_df, group_df)

    return {
        "resolved_claims": len(resolved_df),
        "resolution_groups": len(group_df),
        "resolution_traces": len(trace_index),
    }


def _group_kind(row: pd.Series) -> str:
    source = str(row.get("topic_key_source") or "").strip().lower()
    return "canonical" if source == "canonical_ids" else "predicate"


def _group_value(row: pd.Series) -> str:
    if _group_kind(row) == "canonical":
        return str(row.get("topic_key") or "")
    return str(row.get("predicate") or "")


def _group_topic_key(row: pd.Series) -> str:
    if _group_kind(row) == "canonical":
        return str(row.get("topic_key") or "")
    scope = str(row.get("precedence_scope_id") or "")
    predicate = str(row.get("predicate") or "")
    return f"{predicate}|scope:{scope}"


def _filter_group_edges(
    precedence_df: pd.DataFrame,
    scope_id: object,
    topic_key: str,
) -> Dict[str, List[Tuple[str, str, Optional[int]]]]:
    edges = {"precedes": [], "exception_of": []}
    if precedence_df.empty:
        return edges
    mask = (
        precedence_df["scope_struct_node_id"].astype(str) == str(scope_id)
    ) & (
        precedence_df["topic_key"].astype(str) == str(topic_key)
    )
    for _, row in precedence_df[mask].iterrows():
        relation = str(row.get("relation") or "")
        from_id = str(row.get("from_claim_id") or "")
        to_id = str(row.get("to_claim_id") or "")
        if not from_id or not to_id:
            continue
        if relation in edges:
            distance = row.get("sentence_distance")
            edges[relation].append((from_id, to_id, distance))
    return edges


def _validate_group_edges(
    claims_df: pd.DataFrame,
    edges: Dict[str, List[Tuple[str, str, Optional[int]]]],
    group_ids: Iterable[str],
) -> Tuple[Optional[str], Optional[str]]:
    group_set = set(group_ids)
    precedes = edges.get("precedes", [])
    exceptions = edges.get("exception_of", [])
    if not precedes:
        return "missing_precedes", "No precedes edges for competing group"
    for relation, edge_list in edges.items():
        for from_id, to_id, _ in edge_list:
            if from_id not in group_set or to_id not in group_set:
                return "cross_group_edge", f"{relation} edge links outside group"

    max_distance_by_pred = {}
    for _, row in claims_df.iterrows():
        pred = str(row.get("predicate") or "")
        max_distance_by_pred[pred] = precedence_sentence_distance_limit(pred)

    for from_id, to_id, distance in precedes:
        if distance is None:
            return "missing_distance", "Precedes edge missing distance"
        pred = str(claims_df.loc[claims_df["claim_id"] == from_id, "predicate"].iloc[0])
        if int(distance) > max_distance_by_pred.get(pred, precedence_sentence_distance_limit(pred)):
            return "distance_limit", "Precedes edge exceeds distance limit"

    if not _chain_ok(precedes, group_set):
        return "invalid_chain", "Precedes edges do not form a chain"
    return None, None


def _chain_ok(precedes: List[Tuple[str, str, Optional[int]]], group_ids: Iterable[str]) -> bool:
    group_list = list(group_ids)
    if len(precedes) != max(0, len(group_list) - 1):
        return False
    adjacency: Dict[str, List[str]] = {}
    indegree: Dict[str, int] = {cid: 0 for cid in group_list}
    outdegree: Dict[str, int] = {cid: 0 for cid in group_list}
    for src, dst, _ in precedes:
        adjacency.setdefault(src, []).append(dst)
        indegree[dst] = indegree.get(dst, 0) + 1
        outdegree[src] = outdegree.get(src, 0) + 1
    if any(val > 1 for val in indegree.values()) or any(val > 1 for val in outdegree.values()):
        return False
    starts = [cid for cid, deg in indegree.items() if deg == 0]
    if len(starts) != 1:
        return False
    order = []
    cur = starts[0]
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        order.append(cur)
        next_nodes = adjacency.get(cur, [])
        if len(next_nodes) > 1:
            return False
        cur = next_nodes[0] if next_nodes else None
    return len(order) == len(group_list)


def _build_chain(
    precedes: List[Tuple[str, str, Optional[int]]],
    group_ids: Iterable[str],
) -> List[str]:
    adjacency: Dict[str, str] = {}
    indegree: Dict[str, int] = {cid: 0 for cid in group_ids}
    for src, dst, _ in precedes:
        adjacency[src] = dst
        indegree[dst] = indegree.get(dst, 0) + 1
    start = [cid for cid, deg in indegree.items() if deg == 0]
    if not start:
        return list(group_ids)
    order = []
    cur = start[0]
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        order.append(cur)
        cur = adjacency.get(cur)
    return order


def _resolve_group_rows(
    group_claims: pd.DataFrame,
    chain_order: List[str],
    exception_edges: List[Tuple[str, str, Optional[int]]],
    group_status: str,
    error_code: Optional[str],
) -> List[ResolvedClaimRow]:
    rows: List[ResolvedClaimRow] = []
    claim_map = {str(row["claim_id"]): row for _, row in group_claims.iterrows()}
    status_map = {cid: "suppressed" for cid in claim_map}
    reason_map: Dict[str, str] = {}
    suppressed_by: Dict[str, Optional[str]] = {}

    if group_status == GROUP_STATUS_SKIP:
        if chain_order:
            status_map[chain_order[0]] = "applies"
            reason_map[chain_order[0]] = "single_claim"
    elif group_status == GROUP_STATUS_ERROR:
        for cid in status_map:
            status_map[cid] = "undecidable"
            reason_map[cid] = error_code or "error_graph"
    else:
        for idx, cid in enumerate(chain_order):
            if idx == 0:
                status_map[cid] = "applies"
                reason_map[cid] = "top_rank"
            else:
                status_map[cid] = "suppressed"
                suppressed_by[cid] = chain_order[idx - 1]
                reason_map[cid] = "precedes"

        for exc_id, base_id, _ in exception_edges:
            if exc_id not in status_map or base_id not in status_map:
                continue
            status_map[exc_id] = "exception"
            reason_map[exc_id] = "exception_of"
            status_map[base_id] = "suppressed"
            suppressed_by[base_id] = exc_id
            reason_map[base_id] = "exception_override"

    for rank, cid in enumerate(chain_order, start=1):
        row = claim_map.get(cid)
        if row is None:
            continue
        status = status_map.get(cid, "undecidable")
        reason = reason_map.get(cid)
        missing_reason = _missing_provenance_reason(row)
        rows.append(ResolvedClaimRow(
            claim_id=cid,
            precedence_scope_id=str(row.get("precedence_scope_id")),
            group_kind=str(row.get("group_kind")),
            group_value=str(row.get("group_value")),
            topic_key=str(row.get("group_topic_key")),
            rank=rank,
            status=status,
            suppressed_by_claim_id=suppressed_by.get(cid),
            reason=reason,
            predicate=row.get("predicate"),
            subject_canonical_id=row.get("subject_canonical_id"),
            object_canonical_id=row.get("object_canonical_id"),
            source_element_id=row.get("source_element_id"),
            sentence_idx=row.get("sentence_idx"),
            char_start=row.get("char_start"),
            char_end=row.get("char_end"),
            evidence_text=row.get("evidence_text"),
            missing_provenance_reason=missing_reason,
        ))
    return rows


def _missing_provenance_reason(row: pd.Series) -> Optional[str]:
    missing = []
    if row.get("source_element_id") is None:
        missing.append("source_element_id")
    if row.get("sentence_idx") is None:
        missing.append("sentence_idx")
    if row.get("char_start") is None or row.get("char_end") is None:
        missing.append("char_span")
    if not missing:
        return None
    return ",".join(missing)


def _status_counts(rows: List[ResolvedClaimRow]) -> Dict[str, int]:
    counts = {status: 0 for status in ALLOWED_STATUSES}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def _rows_to_df(rows: List[object]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([asdict(row) for row in rows])


def _write_trace(
    output_dir: Path,
    scope_id: object,
    group_key: str,
    group_status: str,
    error_code: Optional[str],
    claims: List[ResolvedClaimRow],
    edges: Dict[str, List[Tuple[str, str, Optional[int]]]],
) -> Dict[str, object]:
    trace_dir = output_dir / "resolution_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "precedence_scope_id": str(scope_id),
        "group_key": group_key,
        "status": group_status,
        "error_code": error_code,
        "claims": [row.to_dict() for row in claims],
        "edges": edges,
    }
    name_seed = f"{scope_id}|{group_key}"
    digest = sha256(name_seed.encode("utf-8")).hexdigest()[:12]
    filename = f"group_{digest}.json"
    path = trace_dir / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return {"group_key": group_key, "file": str(path.name)}


def _write_trace_index(output_dir: Path, entries: List[Dict[str, object]]) -> None:
    trace_dir = output_dir / "resolution_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    index_path = trace_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump({"traces": entries}, fh, indent=2, sort_keys=True)


def _write_metadata(
    output_dir: Path,
    claims_df: pd.DataFrame,
    precedence_df: pd.DataFrame,
    resolved_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> None:
    input_hash = _hash_dataframe(claims_df) + _hash_dataframe(precedence_df)
    output_hash = _hash_dataframe(resolved_df) + _hash_dataframe(group_df)
    payload = {
        "input_hash": input_hash,
        "output_hash": output_hash,
        "claims_rows": int(len(claims_df)),
        "precedence_rows": int(len(precedence_df)),
        "resolved_rows": int(len(resolved_df)),
        "group_rows": int(len(group_df)),
    }
    with open(output_dir / "resolution_run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


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


__all__ = ["resolve_claims"]
