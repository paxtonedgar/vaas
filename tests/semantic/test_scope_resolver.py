from __future__ import annotations

import pandas as pd

from vaas.semantic.scope_resolver import (
    build_struct_scope_overlay,
    resolve_claim_scopes,
    resolve_precedence_scopes,
)


def test_resolve_prefers_sentence_scope():
    nodes_df = pd.DataFrame([
        {
            "node_id": "doc:para1",
            "doc_id": "doc",
            "node_type": "paragraph",
            "element_id": "el1",
            "anchor_id": "doc:secA",
        },
        {
            "node_id": "doc:secA",
            "doc_id": "doc",
            "node_type": "section",
            "element_id": None,
            "anchor_id": None,
        },
    ])
    sentence_index_df = pd.DataFrame([
        {
            "doc_id": "doc",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "sentence_text": "Hello world.",
            "sentence_char_start": 0,
            "sentence_char_end": 12,
        }
    ])

    overlay = build_struct_scope_overlay(nodes_df, sentence_index_df)
    claims_df = pd.DataFrame(
        [
            {
                "claim_id": "c1",
                "doc_id": "doc",
                "source_element_id": "el1",
                "sentence_idx": 0,
                "char_start": 0,
                "char_end": 5,
                "scope_node_id": "doc:para1",
            },
            {
                "claim_id": "c2",
                "doc_id": "doc",
                "source_element_id": "el2",
                "sentence_idx": None,
                "char_start": 0,
                "char_end": 5,
                "scope_node_id": "doc:secA",
            },
        ]
    )

    resolved = resolve_claim_scopes(claims_df, overlay)
    resolved = resolve_precedence_scopes(resolved, overlay)

    sentence_scope = resolved.loc[resolved["claim_id"] == "c1", "scope_struct_node_id"].item()
    paragraph_scope_type = resolved.loc[resolved["claim_id"] == "c2", "scope_struct_type"].item()
    precedence_scope_type = resolved.loc[resolved["claim_id"] == "c1", "precedence_scope_type"].item()

    assert sentence_scope == "doc:el1:sent:0"
    assert paragraph_scope_type in {"anchor", "section"}
    assert precedence_scope_type == "paragraph"


def test_precedence_scope_prefers_anchor_for_defines():
    nodes_df = pd.DataFrame([
        {
            "node_id": "doc:para1",
            "doc_id": "doc",
            "node_type": "paragraph",
            "element_id": "el1",
            "anchor_id": "doc:anchor1",
        },
        {
            "node_id": "doc:anchor1",
            "doc_id": "doc",
            "node_type": "concept",
            "element_id": None,
            "anchor_id": None,
        },
    ])
    sentence_index_df = pd.DataFrame([
        {
            "doc_id": "doc",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "sentence_text": "Qualified dividends are dividends.",
            "sentence_char_start": 0,
            "sentence_char_end": 34,
        }
    ])
    overlay = build_struct_scope_overlay(nodes_df, sentence_index_df)
    claims_df = pd.DataFrame([
        {
            "claim_id": "c1",
            "doc_id": "doc",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "scope_struct_node_id": "doc:el1:sent:0",
            "scope_struct_type": "element_sentence",
            "predicate": "defines",
        },
        {
            "claim_id": "c2",
            "doc_id": "doc",
            "source_element_id": "el1",
            "sentence_idx": 0,
            "scope_struct_node_id": "doc:el1:sent:0",
            "scope_struct_type": "element_sentence",
            "predicate": "applies_if",
        },
    ])
    resolved = resolve_precedence_scopes(claims_df, overlay)
    define_scope = resolved.loc[resolved["claim_id"] == "c1", "precedence_scope_id"].item()
    applies_scope = resolved.loc[resolved["claim_id"] == "c2", "precedence_scope_id"].item()
    assert define_scope == "doc:anchor1"
    assert applies_scope == "doc:para1"
