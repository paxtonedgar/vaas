from __future__ import annotations

from vaas.semantic.claim_pipeline import CLAIM_COLUMNS, CLAIM_PRECEDENCE_COLUMNS


def test_claim_schema_columns_include_resolution_fields():
    required = {
        "precedence_scope_id",
        "precedence_scope_type",
        "topic_key",
        "topic_key_source",
        "topic_key_parts",
    }
    assert required.issubset(set(CLAIM_COLUMNS))


def test_precedence_schema_columns_include_topic():
    required = {"from_claim_id", "to_claim_id", "relation", "scope_struct_node_id", "sentence_distance", "topic_key"}
    assert required.issubset(set(CLAIM_PRECEDENCE_COLUMNS))
