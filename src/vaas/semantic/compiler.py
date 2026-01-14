"""Compile resolved claims into executable constraints."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from vaas.schemas.semantic_contract import ResolvedConstraintRow


def compile_constraints(
    resolved_claims_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, int]:
    """Compile resolved claims into constraint directives."""
    if resolved_claims_df.empty:
        empty_df = pd.DataFrame(columns=[
            "constraint_id",
            "precedence_scope_id",
            "topic_key",
            "target_box_id",
            "action",
            "condition",
            "supporting_claim_ids",
            "supporting_evidence_spans",
        ])
        empty_df.to_parquet(output_dir / "constraints_resolved.parquet", index=False)
        return {"constraints_resolved": 0}

    active = resolved_claims_df[
        resolved_claims_df["status"].isin({"applies", "exception"})
    ].copy()

    rows: List[ResolvedConstraintRow] = []
    for _, row in active.iterrows():
        claim_id = str(row.get("claim_id") or "")
        predicate = str(row.get("predicate") or "")
        target_box = row.get("object_canonical_id")
        scope_id = str(row.get("precedence_scope_id") or "")
        topic_key = str(row.get("topic_key") or "")
        evidence = {
            "source_element_id": row.get("source_element_id"),
            "sentence_idx": row.get("sentence_idx"),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
        }
        constraint_id = _make_constraint_id(scope_id, topic_key, claim_id, predicate)
        rows.append(ResolvedConstraintRow(
            constraint_id=constraint_id,
            precedence_scope_id=scope_id,
            topic_key=topic_key,
            target_box_id=target_box,
            action=predicate,
            condition=None,
            supporting_claim_ids=[claim_id],
            supporting_evidence_spans=[evidence],
        ))

    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_parquet(output_dir / "constraints_resolved.parquet", index=False)
    return {"constraints_resolved": len(df)}


def _make_constraint_id(scope_id: str, topic_key: str, claim_id: str, action: str) -> str:
    payload = json.dumps(
        {
            "scope": scope_id,
            "topic": topic_key,
            "claim": claim_id,
            "action": action,
        },
        sort_keys=True,
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"rc_{digest}"


__all__ = ["compile_constraints"]
