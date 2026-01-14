from __future__ import annotations

from pathlib import Path

import pandas as pd

from hashlib import sha256
import json

from vaas.evaluation.semkg_audits import (
    predicate_outcome_coverage_gate,
    resolution_group_integrity_gate,
    resolution_nonempty_gate,
)
from vaas.semantic.compiler import compile_constraints
from vaas.semantic.resolution import resolve_claims


def _claims(rows):
    return pd.DataFrame(rows)


def _precedence(rows):
    return pd.DataFrame(rows)


def test_resolver_emits_exception_and_chain(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Rule applies.",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
            "source_element_id": "el1",
            "sentence_idx": 1,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Exception: special case.",
        },
    ])
    precedence_df = _precedence([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "excludes|subj|obj",
        },
        {
            "from_claim_id": "c2",
            "to_claim_id": "c1",
            "relation": "exception_of",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "excludes|subj|obj",
        },
    ])

    resolve_claims(claims_df, precedence_df, tmp_path)
    resolved_df = pd.read_parquet(tmp_path / "resolved_claims.parquet")
    compile_constraints(resolved_df, tmp_path)
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    precedence_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)
    resolved = pd.read_parquet(tmp_path / "resolved_claims.parquet")
    compile_constraints(resolved, tmp_path)
    groups = pd.read_parquet(tmp_path / "resolution_groups.parquet")

    assert set(resolved["status"]) == {"exception", "suppressed"}
    assert groups.loc[0, "status"] == "RESOLVED"

    assert resolution_nonempty_gate(tmp_path).passed is True
    assert resolution_group_integrity_gate(tmp_path).passed is True
    assert predicate_outcome_coverage_gate(tmp_path).passed is True


def test_resolver_flags_error_graph(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "includes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "includes",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Rule applies.",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-1",
            "topic_key": "includes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "includes",
            "source_element_id": "el1",
            "sentence_idx": 1,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Rule applies too.",
        },
    ])
    precedence_df = _precedence([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "includes|subj|obj",
        },
        {
            "from_claim_id": "c2",
            "to_claim_id": "c1",
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "includes|subj|obj",
        },
    ])

    resolve_claims(claims_df, precedence_df, tmp_path)
    resolved = pd.read_parquet(tmp_path / "resolved_claims.parquet")
    groups = pd.read_parquet(tmp_path / "resolution_groups.parquet")

    assert set(resolved["status"]) == {"undecidable"}
    assert groups.loc[0, "status"] == "ERROR_GRAPH"


def test_resolver_chain_emits_single_directive(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "char_end": 5,
            "evidence_text": "Rule applies.",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-1",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
            "source_element_id": "el1",
            "sentence_idx": 1,
            "char_start": 0,
            "char_end": 5,
            "evidence_text": "Rule continues.",
        },
        {
            "claim_id": "c3",
            "precedence_scope_id": "scope-1",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
            "source_element_id": "el1",
            "sentence_idx": 2,
            "char_start": 0,
            "char_end": 5,
            "evidence_text": "Rule ends.",
        },
    ])
    precedence_df = _precedence([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "defines|subj|obj",
        },
        {
            "from_claim_id": "c2",
            "to_claim_id": "c3",
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "defines|subj|obj",
        },
    ])

    resolve_claims(claims_df, precedence_df, tmp_path)
    resolved = pd.read_parquet(tmp_path / "resolved_claims.parquet")
    compile_constraints(resolved, tmp_path)

    assert set(resolved["status"]) == {"applies", "suppressed"}
    constraints = pd.read_parquet(tmp_path / "constraints_resolved.parquet")
    assert len(constraints) == 1


def test_resolver_skips_when_no_precedes_edges(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "gated_by|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "gated_by",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Rule applies.",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-1",
            "topic_key": "gated_by|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "gated_by",
            "source_element_id": "el1",
            "sentence_idx": 10,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Rule applies later.",
        },
    ])
    precedence_df = _precedence([])

    resolve_claims(claims_df, precedence_df, tmp_path)
    resolved = pd.read_parquet(tmp_path / "resolved_claims.parquet")
    groups = pd.read_parquet(tmp_path / "resolution_groups.parquet")

    assert resolved.empty
    assert groups.loc[0, "status"] == "SKIP_NO_EDGES"


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
    digests = []
    for path in sorted(trace_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        digests.append(sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest())
    return sha256("".join(digests).encode("utf-8")).hexdigest()


def test_determinism_two_fresh_dirs(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Rule applies.",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
            "source_element_id": "el1",
            "sentence_idx": 1,
            "char_start": 0,
            "char_end": 10,
            "evidence_text": "Exception: special case.",
        },
    ])
    precedence_df = _precedence([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "excludes|subj|obj",
        },
        {
            "from_claim_id": "c2",
            "to_claim_id": "c1",
            "relation": "exception_of",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "excludes|subj|obj",
        },
    ])

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    resolve_claims(claims_df, precedence_df, out1)
    compile_constraints(pd.read_parquet(out1 / "resolved_claims.parquet"), out1)
    resolve_claims(claims_df, precedence_df, out2)
    compile_constraints(pd.read_parquet(out2 / "resolved_claims.parquet"), out2)

    def _hash_dir(path: Path) -> str:
        resolved = pd.read_parquet(path / "resolved_claims.parquet")
        groups = pd.read_parquet(path / "resolution_groups.parquet")
        directives = pd.read_parquet(path / "compiled_directives.parquet")
        constraints = pd.read_parquet(path / "constraints_resolved.parquet")
        parts = [
            _hash_dataframe(resolved),
            _hash_dataframe(groups),
            _hash_dataframe(directives),
            _hash_dataframe(constraints),
            _hash_traces(path / "resolution_traces"),
        ]
        return sha256("".join(parts).encode("utf-8")).hexdigest()

    assert _hash_dir(out1) == _hash_dir(out2)
