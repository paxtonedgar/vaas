from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaas.evaluation.semkg_audits import (
    claim_group_competition_gate,
    precedence_cross_group_gate,
    precedence_dag_gate,
    precedence_links_gate,
    precedence_locality_gate,
)
from vaas.semantic.policy import (
    DEFAULT_PRECEDENCE_SENTENCE_DISTANCE,
    PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE,
)
from vaas.semantic.precedence import build_claim_precedence
from vaas.semantic.claim_pipeline import _make_topic_fields


def _claims(rows):
    df = pd.DataFrame(rows)
    if "precedence_scope_id" not in df.columns and "scope_struct_node_id" in df.columns:
        df["precedence_scope_id"] = df["scope_struct_node_id"]
    if "topic_key_source" not in df.columns and "topic_key" in df.columns:
        df["topic_key_source"] = "canonical_ids"
    return df


def test_build_claim_precedence_orders_by_sentence_tuple():
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "predicate|sub|obj",
            "source_element_id": "el2",
            "sentence_idx": 0,
            "char_start": 5,
            "predicate": "predicate",
            "rule_class": "POPULATION",
            "precedence_rank": 2,
            "evidence_text": "alpha",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "predicate|sub|obj",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 1,
            "predicate": "predicate",
            "rule_class": "PROHIBITION",
            "precedence_rank": 1,
            "evidence_text": "beta",
        },
    ])

    rows = build_claim_precedence(claims_df)
    assert len(rows) == 1
    assert rows[0].from_claim_id == "c2"
    assert rows[0].to_claim_id == "c1"
    assert rows[0].sentence_distance == 1
    assert rows[0].relation == "precedes"


def test_build_claim_precedence_marks_exceptions():
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "predicate|sub|obj",
            "source_element_id": "el1",
            "sentence_idx": 1,
            "char_start": 1,
            "predicate": "predicate",
            "rule_class": "POPULATION",
            "precedence_rank": 1,
            "evidence_text": "rule applies",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "predicate|sub|obj",
            "source_element_id": "el1",
            "sentence_idx": 2,
            "char_start": 1,
            "predicate": "predicate",
            "rule_class": "POPULATION",
            "precedence_rank": 2,
            "evidence_text": "exception: special case",
        },
    ])

    rows = build_claim_precedence(claims_df)
    assert len(rows) == 2
    relations = {row.relation for row in rows}
    assert relations == {"precedes", "exception_of"}
    exception_row = next(row for row in rows if row.relation == "exception_of")
    assert exception_row.from_claim_id == "c2"
    assert exception_row.to_claim_id == "c1"


def test_precedence_fixture_emits_chain_and_exception(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
            "subject_canonical_id": "subj",
            "object_canonical_id": "obj",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "evidence_text": "Rule applies.",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "scope-1",
            "topic_key": "excludes|subj|obj",
            "topic_key_source": "canonical_ids",
            "predicate": "excludes",
            "subject_canonical_id": "subj",
            "object_canonical_id": "obj",
            "source_element_id": "el1",
            "sentence_idx": 1,
            "char_start": 0,
            "evidence_text": "Exception: special case.",
        },
    ])
    rows = build_claim_precedence(claims_df)
    assert {row.relation for row in rows} == {"precedes", "exception_of"}
    pd.DataFrame([row.to_dict() for row in rows]).to_parquet(
        tmp_path / "claim_precedence.parquet",
        index=False,
    )
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)

    assert precedence_links_gate(tmp_path).passed is True
    assert precedence_cross_group_gate(tmp_path).passed is True
    assert precedence_dag_gate(tmp_path).passed is True
    assert precedence_locality_gate(tmp_path).passed is True


def test_precedence_locality_gate_uses_policy(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "precedence_scope_id": "s1",
            "topic_key": "t1",
            "predicate": "predicate",
        },
        {
            "claim_id": "c2",
            "precedence_scope_id": "s1",
            "topic_key": "t1",
            "predicate": "predicate",
        },
    ])
    edges_df = _claims([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "s1",
            "sentence_distance": DEFAULT_PRECEDENCE_SENTENCE_DISTANCE + 3,
            "topic_key": "t1",
        }
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_locality_gate(tmp_path)
    assert result.passed is False


def test_precedence_locality_gate_honors_override(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "custom_predicate",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "custom_predicate",
        },
    ])
    edges_df = _claims([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "s1",
            "sentence_distance": DEFAULT_PRECEDENCE_SENTENCE_DISTANCE + 3,
            "topic_key": "t1",
        }
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    original = PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE.get("custom_predicate")
    PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE["custom_predicate"] = DEFAULT_PRECEDENCE_SENTENCE_DISTANCE + 5
    try:
        result = precedence_locality_gate(tmp_path)
    finally:
        if original is None:
            PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE.pop("custom_predicate", None)
        else:
            PRECEDENCE_SENTENCE_DISTANCE_BY_PREDICATE["custom_predicate"] = original
    assert result.passed is True


def test_topic_key_ignores_nan_values():
    series = pd.Series({
        "predicate": float("nan"),
        "subject_canonical_id": float("nan"),
        "object_canonical_id": float("nan"),
        "precedence_scope_id": "scope-1",
    })
    key, source, parts = _make_topic_fields(series)
    assert "nan" not in key.lower()
    assert source == "fallback_scope"
    assert parts == ["", "", ""]


def test_topic_key_uses_canonical_ids():
    series = pd.Series({
        "predicate": "excludes",
        "subject_canonical_id": "concept:subject",
        "object_canonical_id": "form:1099-div:box:1a",
        "precedence_scope_id": "scope-1",
    })
    key, source, parts = _make_topic_fields(series)
    assert key == "excludes|concept:subject|form:1099-div:box:1a"
    assert source == "canonical_ids"
    assert parts == ["excludes", "concept:subject", "form:1099-div:box:1a"]


def test_precedence_links_gate_rejects_cross_topic(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "t2",
            "predicate": "includes",
        },
    ])
    edges_df = _claims([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "s1",
            "sentence_distance": 0,
            "topic_key": "t1",
        }
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_links_gate(tmp_path)
    assert result.passed is False


def test_precedence_links_gate_allows_multiple_edges(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
        {
            "claim_id": "c3",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
    ])
    edges_df = _claims([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "s1",
            "sentence_distance": 1,
            "topic_key": "t1",
        },
        {
            "from_claim_id": "c1",
            "to_claim_id": "c3",
            "relation": "precedes",
            "scope_struct_node_id": "s1",
            "sentence_distance": 2,
            "topic_key": "t1",
        },
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_links_gate(tmp_path)
    assert result.passed is True


def test_precedence_dag_gate_detects_cycle(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
    ])
    edges_df = _claims([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": "s1",
            "sentence_distance": 0,
            "topic_key": "t1",
        },
        {
            "from_claim_id": "c2",
            "to_claim_id": "c1",
            "relation": "precedes",
            "scope_struct_node_id": "s1",
            "sentence_distance": 0,
            "topic_key": "t1",
        },
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_dag_gate(tmp_path)
    assert result.passed is False


def test_precedence_locality_gate_requires_scope(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
    ])
    edges_df = _claims([
        {
            "from_claim_id": "c1",
            "to_claim_id": "c2",
            "relation": "precedes",
            "scope_struct_node_id": None,
            "sentence_distance": 0,
            "topic_key": "t1",
        }
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    edges_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)

    result = precedence_locality_gate(tmp_path)
    assert result.passed is False


def test_claim_group_competition_gate_requires_competing_group(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s2",
            "topic_key": "t2",
            "predicate": "includes",
        },
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)

    result = claim_group_competition_gate(tmp_path)
    assert result.passed is False


def test_claim_group_competition_gate_passes_with_competition(tmp_path: Path):
    claims_df = _claims([
        {
            "claim_id": "c1",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
        {
            "claim_id": "c2",
            "scope_struct_node_id": "s1",
            "topic_key": "t1",
            "predicate": "includes",
        },
    ])
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)

    result = claim_group_competition_gate(tmp_path)
    assert result.passed is True
