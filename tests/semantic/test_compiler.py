from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaas.evaluation.semkg_audits import (
    compiler_directive_join_gate,
    compiler_ir_present_gate,
    compiler_ir_schema_gate,
    compiler_scope_topic_consistency_gate,
    compiler_support_gate,
)
from vaas.semantic.compiler import compile_constraints


def test_compiler_emits_directive_ir(tmp_path: Path):
    resolved_df = pd.DataFrame([
        {
            "claim_id": "c1",
            "precedence_scope_id": "scope-1",
            "group_kind": "canonical",
            "group_value": "topic",
            "topic_key": "excludes|subj|obj",
            "rank": 1,
            "status": "applies",
            "predicate": "excludes",
            "subject_canonical_id": "subj",
            "object_canonical_id": "obj",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "char_start": 0,
            "char_end": 5,
            "evidence_text": "Rule applies.",
        },
    ])
    resolved_df.to_parquet(tmp_path / "resolved_claims.parquet", index=False)
    resolved_df = pd.read_parquet(tmp_path / "resolved_claims.parquet")
    compile_constraints(resolved_df, tmp_path)

    directives = pd.read_parquet(tmp_path / "compiled_directives.parquet")
    assert len(directives) == 1
    assert directives.loc[0, "directive_id"].startswith("dir_")
    assert directives.loc[0, "supporting_claim_ids"] == ["c1"]

    assert compiler_ir_present_gate(tmp_path).passed is True
    assert compiler_support_gate(tmp_path).passed is True
    assert compiler_ir_schema_gate(tmp_path).passed is True
    assert compiler_directive_join_gate(tmp_path).passed is True
    assert compiler_scope_topic_consistency_gate(tmp_path).passed is True
