from __future__ import annotations

from vaas.semantic.claim_pipeline import CLAIM_COLUMNS, CLAIM_PRECEDENCE_COLUMNS
from vaas.schemas.semantic_contract import CompiledDirectiveRow, ResolvedConstraintRow


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


def test_compiler_schema_columns_include_ir():
    directive_required = {
        "directive_id",
        "precedence_scope_id",
        "topic_key",
        "op",
        "condition_ir",
        "supporting_claim_ids",
        "supporting_spans",
        "resolution_reason_codes",
    }
    constraint_required = {
        "constraint_id",
        "directive_id",
        "precedence_scope_id",
        "topic_key",
        "op",
        "condition_ir",
        "supporting_claim_ids",
        "supporting_spans",
        "resolution_reason_codes",
    }
    assert directive_required.issubset(set(CompiledDirectiveRow.__annotations__.keys()))
    assert constraint_required.issubset(set(ResolvedConstraintRow.__annotations__.keys()))
