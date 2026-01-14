from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaas.evaluation.semkg_audits import (
    resolution_determinism_gate,
    resolution_nonempty_gate,
)
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
    claims_df.to_parquet(tmp_path / "claims.parquet", index=False)
    precedence_df.to_parquet(tmp_path / "claim_precedence.parquet", index=False)
    resolved = pd.read_parquet(tmp_path / "resolved_claims.parquet")
    groups = pd.read_parquet(tmp_path / "resolution_groups.parquet")

    assert set(resolved["status"]) == {"exception", "suppressed"}
    assert groups.loc[0, "status"] == "RESOLVED"

    assert resolution_nonempty_gate(tmp_path).passed is True
    assert resolution_determinism_gate(tmp_path).passed is True


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
