"""Compile resolved claims into directive IR and constraints."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from vaas.schemas.semantic_contract import CompiledDirectiveRow, ResolvedConstraintRow


def compile_constraints(
    resolved_claims_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, int]:
    """Compile resolved claims into directive IR, then serialize constraints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if resolved_claims_df.empty:
        empty_directives = pd.DataFrame(columns=[
            "directive_id",
            "precedence_scope_id",
            "topic_key",
            "target_box_id",
            "op",
            "condition_ir",
            "supporting_claim_ids",
            "supporting_spans",
            "resolution_reason_codes",
        ])
        empty_constraints = pd.DataFrame(columns=[
            "constraint_id",
            "directive_id",
            "precedence_scope_id",
            "topic_key",
            "target_box_id",
            "op",
            "condition_ir",
            "supporting_claim_ids",
            "supporting_spans",
            "resolution_reason_codes",
        ])
        empty_directives.to_parquet(output_dir / "compiled_directives.parquet", index=False)
        empty_constraints.to_parquet(output_dir / "constraints_resolved.parquet", index=False)
        _write_metadata(output_dir, empty_directives, empty_constraints)
        return {"compiled_directives": 0, "constraints_resolved": 0}

    active = resolved_claims_df[
        resolved_claims_df["status"].isin({"applies", "exception"})
    ].copy()

    directive_rows: List[CompiledDirectiveRow] = []
    constraint_rows: List[ResolvedConstraintRow] = []
    for _, row in active.iterrows():
        claim_id = str(row.get("claim_id") or "")
        predicate = str(row.get("predicate") or "")
        target_box = row.get("object_canonical_id")
        scope_id = str(row.get("precedence_scope_id") or "")
        topic_key = str(row.get("topic_key") or "")
        evidence_span = {
            "source_element_id": row.get("source_element_id"),
            "sentence_idx": row.get("sentence_idx"),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
        }
        condition_ir = json.dumps(
            {
                "kind": "predicate",
                "predicate": predicate,
                "subject_canonical_id": row.get("subject_canonical_id"),
                "object_canonical_id": row.get("object_canonical_id"),
            },
            sort_keys=True,
        )
        reason_codes = [str(row.get("reason") or ""), str(row.get("status") or "")]
        directive_id = _make_directive_id(scope_id, topic_key, claim_id, predicate, condition_ir)
        directive_rows.append(CompiledDirectiveRow(
            directive_id=directive_id,
            precedence_scope_id=scope_id,
            topic_key=topic_key,
            target_box_id=target_box,
            op=predicate,
            condition_ir=condition_ir,
            supporting_claim_ids=[claim_id],
            supporting_spans=[evidence_span],
            resolution_reason_codes=[code for code in reason_codes if code],
        ))
        constraint_id = _make_constraint_id(directive_id)
        constraint_rows.append(ResolvedConstraintRow(
            constraint_id=constraint_id,
            directive_id=directive_id,
            precedence_scope_id=scope_id,
            topic_key=topic_key,
            target_box_id=target_box,
            op=predicate,
            condition_ir=condition_ir,
            supporting_claim_ids=[claim_id],
            supporting_spans=[evidence_span],
            resolution_reason_codes=[code for code in reason_codes if code],
        ))

    directives_df = pd.DataFrame([asdict(row) for row in directive_rows])
    constraints_df = pd.DataFrame([asdict(row) for row in constraint_rows])
    directives_df.to_parquet(output_dir / "compiled_directives.parquet", index=False)
    constraints_df.to_parquet(output_dir / "constraints_resolved.parquet", index=False)
    _write_metadata(output_dir, directives_df, constraints_df)
    return {
        "compiled_directives": len(directives_df),
        "constraints_resolved": len(constraints_df),
    }


def _make_directive_id(
    scope_id: str,
    topic_key: str,
    claim_id: str,
    op: str,
    condition_ir: str,
) -> str:
    payload = json.dumps(
        {
            "scope": scope_id,
            "topic": topic_key,
            "claim": claim_id,
            "op": op,
            "condition_ir": condition_ir,
        },
        sort_keys=True,
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"dir_{digest}"


def _make_constraint_id(directive_id: str) -> str:
    digest = sha256(directive_id.encode("utf-8")).hexdigest()[:16]
    return f"cr_{digest}"


def _write_metadata(
    output_dir: Path,
    directives_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
) -> None:
    metadata_path = output_dir / "compiler_run_metadata.json"
    previous = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            previous = json.load(fh)
    directives_hash = _hash_dataframe(directives_df)
    constraints_hash = _hash_dataframe(constraints_df)
    payload = {
        "directives_hash": directives_hash,
        "constraints_hash": constraints_hash,
        "directive_rows": int(len(directives_df)),
        "constraint_rows": int(len(constraints_df)),
        "previous_directives_hash": previous.get("directives_hash"),
        "previous_constraints_hash": previous.get("constraints_hash"),
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
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


__all__ = ["compile_constraints"]
