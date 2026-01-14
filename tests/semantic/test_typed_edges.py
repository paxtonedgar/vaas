from __future__ import annotations

from vaas.graph.edges import Edge, _dedupe_edges
from vaas.semantic.typed_edges import extract_concept_to_box_edges


def test_gating_edges_are_sentence_scoped():
    text = (
        "Only RICs and REITs should complete boxes 2e and 2f. "
        "Only RICs and REITs should complete boxes 2e and 2f."
    )
    edges = extract_concept_to_box_edges(
        source_node_id="doc:para1",
        text=text,
        valid_box_keys={"2e", "2f"},
    )
    gating = [edge for edge in edges if edge.edge_type == "gated_by"]
    assert len(gating) == 4
    sentence_idxs = {edge.sentence_idx for edge in gating}
    assert sentence_idxs == {0, 1}


def test_fallback_edges_are_sentence_scoped():
    text = "Include dividends for which it is impractical to determine in box 1a."
    edges = extract_concept_to_box_edges(
        source_node_id="doc:para1",
        text=text,
        valid_box_keys={"1a"},
    )
    fallback = [edge for edge in edges if edge.edge_type == "fallback_include"]
    assert fallback
    assert fallback[0].sentence_idx == 0
    assert fallback[0].sentence_char_start == 0


def test_defines_term_maps_to_box_key():
    text = (
        "Qualified dividends are dividends paid during the tax year from domestic "
        "corporations and qualified foreign corporations."
    )
    edges = extract_concept_to_box_edges(
        source_node_id="doc:para1",
        text=text,
        valid_box_keys={"1b"},
        form_id="form:1099-div",
    )
    defines = [edge for edge in edges if edge.edge_type == "defines"]
    assert defines
    assert defines[0].target_box_key == "1b"


def test_excludes_term_maps_to_box_key():
    text = (
        "Dividends paid by a regulated investment company (RIC) that are not "
        "treated as qualified dividend income under section 854."
    )
    edges = extract_concept_to_box_edges(
        source_node_id="doc:para1",
        text=text,
        valid_box_keys={"1b"},
        form_id="form:1099-div",
    )
    excludes = [edge for edge in edges if edge.edge_type == "excludes"]
    assert excludes
    assert excludes[0].target_box_key == "1b"


def test_exception_context_emits_excludes():
    text = (
        "Dividends the recipient received on any share of stock held for less "
        "than 61 days during the 121-day period that began 60 days before the "
        "ex-dividend date."
    )
    edges = extract_concept_to_box_edges(
        source_node_id="doc:para1",
        text=text,
        valid_box_keys={"1b"},
        form_id="form:1099-div",
        exception_term="qualified dividends",
    )
    excludes = [edge for edge in edges if edge.edge_type == "excludes"]
    assert excludes
    assert excludes[0].pattern_matched == "exception_context"


def test_dedupe_preserves_sentence_variants():
    edges = [
        Edge(
            edge_id="e1",
            source_node_id="s1",
            target_node_id="t1",
            edge_type="excludes",
            confidence=0.5,
            evidence_sentence_idx=0,
            evidence_char_start=0,
            evidence_char_end=10,
        ),
        Edge(
            edge_id="e2",
            source_node_id="s1",
            target_node_id="t1",
            edge_type="excludes",
            confidence=0.6,
            evidence_sentence_idx=1,
            evidence_char_start=15,
            evidence_char_end=25,
        ),
    ]
    deduped = _dedupe_edges(edges)
    assert len(deduped) == 2
