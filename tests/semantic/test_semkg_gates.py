from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaas.evaluation.semkg_audits import (
    predicate_health_report,
    precedence_chain_gate,
    precedence_cross_group_gate,
    precedence_dag_gate,
)
from vaas.semantic.resolution import resolve_claims


def test_precedence_chain_gate_fails_when_edge_missing(tmp_path: Path):
    claims_df = pd.DataFrame([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "evidence_text": "Rule one.",
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
            "evidence_text": "Rule two.",
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
            "evidence_text": "Rule three.",
        },
    ])
    full_edges = pd.DataFrame([
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
    resolve_claims(claims_df, full_edges, tmp_path)
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    full_edges.head(1).to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_chain_gate(tmp_path)
    assert result.passed is False


def test_precedence_dag_gate_fails_on_cycle(tmp_path: Path):
    claims_df = pd.DataFrame([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
        },
    ])
    edges_df = pd.DataFrame([
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
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "excludes|subj|obj",
        },
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_dag_gate(tmp_path)
    assert result.passed is False


def test_precedence_cross_group_gate_fails_on_cross_edge(tmp_path: Path):
    claims_df = pd.DataFrame([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-2",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
        },
    ])
    edges_df = pd.DataFrame([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "scope-1",
            "sentence_distance": 1,
            "topic_key": "defines|subj|obj",
        },
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_cross_group_gate(tmp_path)
    assert result.passed is False


def test_predicate_health_report_writes_json(tmp_path: Path):
    claims_df = pd.DataFrame([
        {
            "claim_id": f"c{i}",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
        }
        for i in range(5)
    ] + [
        {
            "claim_id": "d1",
            "precedence_scope_id": "scope-2",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
        },
        {
            "claim_id": "d2",
            "precedence_scope_id": "scope-2",
            "topic_key": "defines|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "defines",
        },
    ])
    resolved_df = pd.DataFrame([
        {
            "claim_id": f"c{i}",
            "precedence_scope_id": "scope-1",
            "group_kind": "canonical",
            "group_value": "excludes|subj|obj",
            "topic_key": "excludes|subj|obj",
            "rank": i + 1,
            "status": "suppressed",
            "predicate": "excludes",
        }
        for i in range(5)
    ] + [
        {
            "claim_id": "d1",
            "precedence_scope_id": "scope-2",
            "group_kind": "canonical",
            "group_value": "defines|subj|obj",
            "topic_key": "defines|subj|obj",
            "rank": 1,
            "status": "exception",
            "predicate": "defines",
        },
        {
            "claim_id": "d2",
            "precedence_scope_id": "scope-2",
            "group_kind": "canonical",
            "group_value": "defines|subj|obj",
            "topic_key": "defines|subj|obj",
            "rank": 2,
            "status": "suppressed",
            "predicate": "defines",
        },
    ])
    groups_df = pd.DataFrame([
        {
            "precedence_scope_id": "scope-1",
            "group_kind": "canonical",
            "group_value": "excludes|subj|obj",
            "topic_key": "excludes|subj|obj",
            "group_size": 5,
            "status": "RESOLVED",
        },
        {
            "precedence_scope_id": "scope-2",
            "group_kind": "canonical",
            "group_value": "defines|subj|obj",
            "topic_key": "defines|subj|obj",
            "group_size": 2,
            "status": "RESOLVED",
        },
    ])
    directives_df = pd.DataFrame([
        {
            "directive_id": "dir_1",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "op": "excludes",
            "supporting_claim_ids": ["c0"],
            "supporting_spans": [{"source_element_id": "el1"}],
        }
    ])
    typed_edges_df = pd.DataFrame([
        {"edge_type": "gated_by"},
        {"edge_type": "excludes"},
    ])

    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    pd.DataFrame([]).to_parquet(tmp_path / "claim_precedence.parquet", index=False)
    resolved_df.to_parquet(tmp_path / "resolved_claims.parquet", index=False)
    groups_df.to_parquet(tmp_path / "resolution_groups.parquet", index=False)
    directives_df.to_parquet(tmp_path / "compiled_directives.parquet", index=False)
    typed_edges_df.to_parquet(tmp_path / "typed_edges.parquet", index=False)

    result = predicate_health_report(tmp_path)
    assert result.passed is True

    health_path = tmp_path / "predicate_health.json"
    assert health_path.exists()
    import json
    with open(health_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "dead_predicates" in payload
    assert "never_compiled_predicates" in payload
