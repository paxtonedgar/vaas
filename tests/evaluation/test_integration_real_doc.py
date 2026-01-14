"""
Integration tests that run the extraction pipeline on real documents.

These tests validate that the early pipeline stages produce outputs
that pass corpus-grounded validators.

Tests are skipped if the PDF is not present.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

# Skip all tests if PDF not present
PDF_PATH = Path(__file__).parent.parent.parent / "data" / "i1099div.pdf"
SKIP_REASON = f"PDF not found at {PDF_PATH}"


@pytest.fixture(scope="module")
def pipeline_outputs():
    """
    Run pipeline stages 1-5 on i1099div.pdf.

    Cached at module level for efficiency.
    """
    if not PDF_PATH.exists():
        pytest.skip(SKIP_REASON)

    from vaas.extraction import (
        extract_spans_from_pdf,
        infer_body_font_size,
        build_line_dataframe,
        add_block_geometry,
        add_page_geometry,
        add_emphasis_flags,
        add_list_item_flags,
        add_text_properties,
        add_page_marker_flags,
        add_header_pattern_flags,
        add_structural_filters,
        detect_columns_for_document,
        assign_line_columns,
        detect_subsection_candidates,
        assign_split_triggers,
        split_blocks_into_elements,
        classify_elements,
        extract_anchors,
        build_anchor_timeline,
        assign_elements_to_anchors,
        EXPECTED_BOXES_1099DIV,
    )

    # Stage 1: Extract spans
    result = extract_spans_from_pdf(str(PDF_PATH), "1099div_filer")
    spans_df = result.spans_df

    # Stage 2: Infer body font
    body_size = infer_body_font_size(spans_df)

    # Stage 3: Build lines
    line_df = build_line_dataframe(spans_df, body_size)
    line_df = add_block_geometry(line_df)
    line_df = add_page_geometry(line_df)
    line_df = add_emphasis_flags(line_df, body_size)
    line_df = add_list_item_flags(line_df)

    col_info_df = detect_columns_for_document(line_df)
    line_df = assign_line_columns(line_df, col_info_df)
    line_df.drop(columns=["_is_bullet", "_is_enum", "_is_list_item"], inplace=True, errors="ignore")

    line_df = add_text_properties(line_df)
    line_df = add_page_marker_flags(line_df)
    line_df = add_header_pattern_flags(line_df)
    line_df = add_structural_filters(line_df)
    line_df = detect_subsection_candidates(line_df)
    line_df = assign_split_triggers(line_df)

    # Stage 4: Split and classify
    elements_df = split_blocks_into_elements(line_df, 300.0)
    elements_df = classify_elements(elements_df)

    # Stage 5: Extract anchors
    extraction = extract_anchors(elements_df)
    anchors_df = extraction.anchors_df

    timeline = build_anchor_timeline(anchors_df, elements_df)
    elements_df = assign_elements_to_anchors(elements_df, timeline)

    return {
        "spans_df": spans_df,
        "line_df": line_df,
        "elements_df": elements_df,
        "anchors_df": anchors_df,
        "body_size": body_size,
        "expected_boxes": EXPECTED_BOXES_1099DIV,
    }


@pytest.mark.skipif(not PDF_PATH.exists(), reason=SKIP_REASON)
class TestRealDocExtraction:
    """Tests that extraction produces valid outputs."""

    def test_spans_extracted(self, pipeline_outputs):
        """Pipeline should extract spans from PDF."""
        spans_df = pipeline_outputs["spans_df"]
        assert len(spans_df) > 100, f"Expected >100 spans, got {len(spans_df)}"

    def test_body_font_detected(self, pipeline_outputs):
        """Body font should be in expected range (8-12 for IRS docs)."""
        body_size = pipeline_outputs["body_size"]
        assert 8.0 <= body_size <= 12.0, f"Body font {body_size} outside expected range"

    def test_elements_classified(self, pipeline_outputs):
        """Elements should have role classifications."""
        elements_df = pipeline_outputs["elements_df"]
        assert "role" in elements_df.columns
        assert elements_df["role"].notna().sum() > 0

    def test_anchors_extracted(self, pipeline_outputs):
        """Anchors should be extracted."""
        anchors_df = pipeline_outputs["anchors_df"]
        assert len(anchors_df) > 10, f"Expected >10 anchors, got {len(anchors_df)}"


@pytest.mark.skipif(not PDF_PATH.exists(), reason=SKIP_REASON)
class TestRealDocBoxCoverage:
    """Tests for box discovery and coverage."""

    def test_box_coverage_above_threshold(self, pipeline_outputs):
        """Box coverage should be >= 80%."""
        from vaas.extraction import validate_box_coverage

        anchors_df = pipeline_outputs["anchors_df"]
        expected = pipeline_outputs["expected_boxes"]

        validation = validate_box_coverage(anchors_df, expected)
        coverage = len(validation.found) / len(expected) if expected else 0

        assert coverage >= 0.80, (
            f"Box coverage {coverage:.1%} below 80% threshold. "
            f"Missing: {validation.missing}"
        )

    def test_critical_boxes_found(self, pipeline_outputs):
        """Critical boxes (1a, 1b, 2a) must be found."""
        anchors_df = pipeline_outputs["anchors_df"]

        if anchors_df.empty:
            pytest.fail("No anchors extracted")

        found_keys = set(anchors_df[anchors_df["anchor_type"] == "box"]["box_key"])
        critical = {"1a", "1b", "2a"}
        missing_critical = critical - found_keys

        assert not missing_critical, f"Missing critical boxes: {missing_critical}"


@pytest.mark.skipif(not PDF_PATH.exists(), reason=SKIP_REASON)
class TestRealDocCorpusValidation:
    """Tests that run corpus validators on real doc output."""

    def test_cg1a_anchor_discovery(self, pipeline_outputs):
        """CG1a: Anchor discovery should find boxes."""
        from vaas.evaluation.validate_corpus import check_cg1a_anchor_discovery

        result = check_cg1a_anchor_discovery(pipeline_outputs["line_df"])

        assert result.passed, f"CG1a failed: {[f.message for f in result.findings]}"
        assert result.metrics.get("discovered_count", 0) >= 18, (
            f"Expected >= 18 boxes, got {result.metrics.get('discovered_count')}"
        )

    def test_cg1b_registry_alignment(self, pipeline_outputs):
        """CG1b: Discovered boxes should align with registry."""
        from vaas.evaluation.validate_corpus import (
            check_cg1a_anchor_discovery,
            check_cg1b_registry_alignment,
        )

        cg1a = check_cg1a_anchor_discovery(pipeline_outputs["line_df"])
        discovered = set(cg1a.metrics.get("discovered_box_keys", []))

        result = check_cg1b_registry_alignment(discovered)

        assert result.metrics["alignment_rate"] >= 0.80, (
            f"Alignment {result.metrics['alignment_rate']:.1%} below 80%. "
            f"Missing: {result.metrics['missing']}"
        )


@pytest.mark.skipif(not PDF_PATH.exists(), reason=SKIP_REASON)
class TestRealDocFullValidation:
    """Run full validation suite on real document."""

    def test_corpus_validators_all_run(self, pipeline_outputs):
        """Corpus validators should all execute and provide metrics."""
        from vaas.evaluation.validate_corpus import validate_corpus_grounded

        # Create empty dataframes for nodes/edges since we don't build full graph
        nodes_df = pd.DataFrame({"body_text": []})
        edges_df = pd.DataFrame({"source_evidence": []})

        results = validate_corpus_grounded(
            pipeline_outputs["line_df"],
            nodes_df,
            edges_df,
            pipeline_outputs["anchors_df"],
            pipeline_outputs["elements_df"],
        )

        # Check that we got results
        assert len(results) >= 4, f"Expected >= 4 corpus checks, got {len(results)}"

        # Check all expected validators ran
        check_ids = {r.check_id for r in results}
        expected_ids = {"CG1a", "CG1b", "CG2", "CG3", "CG4"}
        missing = expected_ids - check_ids
        assert not missing, f"Missing validators: {missing}"

    def test_cg4_geometry_finds_anchors(self, pipeline_outputs):
        """CG4 should find geometry-based anchors in real document."""
        from vaas.evaluation.validate_corpus import (
            check_cg1a_anchor_discovery,
            check_cg4_geometry_discovery,
        )

        # Get text-discovered boxes first
        cg1a = check_cg1a_anchor_discovery(pipeline_outputs["line_df"])
        text_discovered = set(cg1a.metrics.get("discovered_box_keys", []))

        # Run geometry discovery
        result = check_cg4_geometry_discovery(pipeline_outputs["line_df"], text_discovered)

        # Should find some geometry candidates in a real PDF
        geom_count = result.metrics.get("geometry_candidates", 0)
        assert geom_count >= 5, f"Expected >= 5 geometry candidates, got {geom_count}"
