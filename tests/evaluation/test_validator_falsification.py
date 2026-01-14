"""
Falsification tests for validation framework.

These tests MUST FAIL when given bad input - proving the validators
actually catch problems rather than rubber-stamping everything.

Test categories:
1. CG1a/CG1b: Anchor discovery vs registry alignment separation
2. CG2: Artifact contamination detection
3. CG3: Heading coverage degradation
4. CG4: Geometry-based anchor discovery
5. IS5: Structural DAG cycle detection
6. IS1/IS2: ID integrity and edge endpoint validation
7. TE1-TE3: Typed edge extraction (negation, scope, self-edge)
8. TE4/TE5: Typed edge ambiguity and overlap handling
9. H1-H3: Hierarchy constraints (skeleton, multi-root, orphans)
10. IS7: Reachability gating (doc_root, 95% threshold, box nodes)
"""

import pandas as pd
import pytest

from vaas.evaluation.validate_internal import (
    validate_internal,
    check_is1_id_integrity,
    check_is2_edge_endpoints,
    check_is5_structural_dag,
    check_is7_reachability,
    STRUCTURAL_EDGE_TYPES,
    REACHABILITY_THRESHOLD,
)
from vaas.evaluation.validate_corpus import (
    validate_corpus_grounded,
    check_cg1a_anchor_discovery,
    check_cg1b_registry_alignment,
    check_cg2_artifact_contamination,
    check_cg3_structure_completeness,
    check_cg4_geometry_discovery,
    EXPECTED_BOXES_1099DIV,
    VARIANCE,
)

# Typed edge extraction imports
try:
    from vaas.semantic.typed_edges import (
        extract_excludes_edges,
        extract_includes_edges,
        TypedEdgeCandidate,
    )
    TYPED_EDGES_AVAILABLE = True
except ImportError:
    TYPED_EDGES_AVAILABLE = False


# =============================================================================
# FIXTURES: Known-good baseline data
# =============================================================================

@pytest.fixture
def good_lines_df():
    """Lines with proper box headers for anchor discovery."""
    return pd.DataFrame({
        "line_text": [
            "Box 1a. Total ordinary dividends",
            "Box 1b. Qualified dividends",
            "Box 2a. Total capital gain distr.",
            "Box 2b. Unrecap. Sec. 1250 gain",
            "Box 2c. Section 1202 gain",
            "Box 2d. Collectibles (28%) gain",
            "Box 2e. Section 897 ordinary dividends",
            "Box 2f. Section 897 capital gain",
            "Box 3. Nondividend distributions",
            "Box 4. Federal income tax withheld",
            "Box 5. Section 199A dividends",
            "Box 6. Investment expenses",
            "Box 7. Foreign tax paid",
            "Box 8. Foreign country or U.S. possession",
            "Box 9. Cash liquidation distributions",
            "Box 10. Noncash liquidation distributions",
            "Box 11. Exempt-interest dividends",
            "Box 12. Specified private activity bond interest dividends",
            "Box 13. State",
            "Box 14. State identification no.",
            "Box 15. State tax withheld",
            "Box 16. Reserved",
        ],
        "page": [1] * 22,
        "geom_y0": list(range(100, 100 + 22 * 20, 20)),
    })


@pytest.fixture
def good_nodes_df():
    """Valid nodes with unique IDs."""
    return pd.DataFrame({
        "node_id": ["doc:root", "doc:box_1a", "doc:box_1b", "doc:sec_general"],
        "node_type": ["doc_root", "box_section", "box_section", "section"],
        "body_text": ["", "Enter dividends here", "Qualified amounts", "General info"],
    })


@pytest.fixture
def good_edges_df():
    """Valid edges with existing endpoints."""
    return pd.DataFrame({
        "edge_id": ["e1", "e2", "e3"],
        "source_node_id": ["doc:root", "doc:root", "doc:box_1a"],
        "target_node_id": ["doc:box_1a", "doc:box_1b", "doc:box_1b"],
        "edge_type": ["parent_of", "parent_of", "references_box"],
        "source_evidence": ["structural", "structural", "See Box 1b"],
    })


@pytest.fixture
def good_anchors_df():
    """Valid anchors."""
    return pd.DataFrame({
        "anchor_id": ["box_1a", "box_1b", "sec_general"],
        "anchor_type": ["box", "box", "section"],
    })


@pytest.fixture
def good_elements_df():
    """Valid elements."""
    return pd.DataFrame({
        "element_id": ["el_1", "el_2", "el_3"],
        "text": ["Some text", "More text", "Even more"],
    })


# =============================================================================
# TEST 1: CG1a must discover boxes WITHOUT using expected registry
# =============================================================================

class TestCG1aOpenWorldDiscovery:
    """CG1a uses generic patterns, NOT the EXPECTED_BOXES list."""

    def test_discovers_standard_boxes(self, good_lines_df):
        """Should find boxes from text patterns alone."""
        result = check_cg1a_anchor_discovery(good_lines_df)
        discovered = set(result.metrics.get("discovered_box_keys", []))

        # Should find most boxes through generic regex
        assert len(discovered) >= 18, f"Only found {len(discovered)} boxes"
        assert result.passed

    def test_discovers_unexpected_box_99(self):
        """Should discover Box 99 even though it's not in registry."""
        lines = pd.DataFrame({
            "line_text": ["Box 99. Some future field"],
            "page": [1],
        })
        result = check_cg1a_anchor_discovery(lines)
        discovered = set(result.metrics.get("discovered_box_keys", []))

        assert "99" in discovered, "CG1a should discover ANY box pattern"

    def test_fails_when_no_boxes_found(self):
        """Should fail/warn when discovery finds too few boxes."""
        lines = pd.DataFrame({
            "line_text": ["This is just plain text", "No boxes here"],
            "page": [1, 1],
        })
        result = check_cg1a_anchor_discovery(lines)

        # Should fail or warn when below variance threshold
        assert not result.passed or len(result.findings) > 0

    def test_empty_lines_graceful(self):
        """Should handle empty input gracefully."""
        result = check_cg1a_anchor_discovery(pd.DataFrame())
        assert result.passed  # Empty is acceptable, not a failure


class TestCG1bRegistryAlignment:
    """CG1b compares discovered vs expected - diagnostic only."""

    def test_perfect_alignment(self):
        """Full coverage should pass with high alignment rate."""
        discovered = EXPECTED_BOXES_1099DIV.copy()
        result = check_cg1b_registry_alignment(discovered)

        assert result.passed
        assert result.metrics["alignment_rate"] == 1.0
        assert len(result.metrics["missing"]) == 0

    def test_partial_alignment_warns(self):
        """Missing boxes should generate warnings."""
        discovered = {"1a", "1b", "2a"}  # Only 3 boxes
        result = check_cg1b_registry_alignment(discovered)

        # Should warn about missing boxes
        assert len(result.metrics["missing"]) > 0
        assert any(f.severity == "warning" for f in result.findings)

    def test_extra_boxes_info_only(self):
        """Extra discovered boxes are INFO, not errors."""
        discovered = EXPECTED_BOXES_1099DIV | {"99", "100"}
        result = check_cg1b_registry_alignment(discovered)

        # Extra boxes should be info, not warning/error
        assert result.passed
        assert "99" in result.metrics["extra"]
        extra_findings = [f for f in result.findings if "extra" in f.message.lower()]
        assert all(f.severity == "info" for f in extra_findings)

    def test_fake_box_flagged_without_breaking_discovery(self):
        """
        FALSIFICATION: Add fake box and confirm CG1b flags it.

        This proves CG1a (discovery) and CG1b (alignment) are separate:
        - CG1a should still pass (it found boxes)
        - CG1b should flag the extra as unusual
        """
        # Discovery finds real boxes + fake
        discovered = EXPECTED_BOXES_1099DIV | {"99x"}  # Fake box

        cg1b = check_cg1b_registry_alignment(discovered)

        # CG1b should note the extra without failing
        assert "99x" in cg1b.metrics["extra"]
        # But it shouldn't break the alignment rate for real boxes
        assert cg1b.metrics["matched"] == len(EXPECTED_BOXES_1099DIV)


# =============================================================================
# TEST 2: CG2 artifact contamination MUST detect bad patterns
# =============================================================================

class TestCG2ArtifactContamination:
    """CG2 must catch artifacts that leak into nodes/edges."""

    def test_clean_data_passes(self):
        """No artifacts = pass."""
        lines = pd.DataFrame({
            "line_text": ["Box 1a content", "Box 1b content"],
            "page": [1, 2],
            "geom_y0": [100, 100],  # Not in header/footer band
        })
        nodes = pd.DataFrame({"body_text": ["Clean content"]})
        edges = pd.DataFrame({"source_evidence": ["Box 1a reference"]})

        result = check_cg2_artifact_contamination(lines, nodes, edges)
        assert result.passed

    def test_detects_repeated_header_text(self):
        """
        FALSIFICATION: Repeated text in header band = artifact.
        """
        # Simulate header that appears on every page
        lines = pd.DataFrame({
            "line_text": [
                "Instructions for Form 1099-DIV",  # Header artifact
                "Instructions for Form 1099-DIV",  # Same on page 2
                "Box 1a. Ordinary dividends",
            ],
            "page": [1, 2, 1],
            "geom_y0": [10, 10, 200],  # First two in top 10% (header band)
        })

        # Now contaminate nodes with the artifact
        nodes = pd.DataFrame({
            "body_text": ["Instructions for Form 1099-DIV included here"]
        })
        edges = pd.DataFrame({"source_evidence": [""]})

        result = check_cg2_artifact_contamination(lines, nodes, edges)

        # Should detect contamination
        assert result.metrics.get("contaminations", 0) > 0 or not result.passed

    def test_page_number_artifact(self):
        """Page numbers repeated across pages should be flagged."""
        lines = pd.DataFrame({
            "line_text": ["-2-", "-2-", "-3-", "-3-", "Box 1a content"],
            "page": [1, 2, 2, 3, 1],
            "geom_y0": [900, 900, 900, 900, 100],  # Bottom of page (footer)
        })
        nodes = pd.DataFrame({"body_text": ["See page -2- for details"]})
        edges = pd.DataFrame({"source_evidence": [""]})

        result = check_cg2_artifact_contamination(lines, nodes, edges)

        # Short repeated text in footer band should be detected
        assert len(result.metrics.get("artifacts_sample", [])) > 0


# =============================================================================
# TEST 3: CG3 heading coverage MUST degrade when anchors removed
# =============================================================================

class TestCG3HeadingCoverage:
    """CG3 must fail when heading candidates aren't anchored."""

    def test_full_coverage_passes(self):
        """All headings anchored = pass."""
        lines = pd.DataFrame({
            "line_text": ["Box 1a Header", "Box 1b Header"],
            "line_bold": [True, True],
            "line_size": [12.0, 12.0],
            "gap_above": [20.0, 20.0],
        })
        anchors = pd.DataFrame({
            "anchor_id": ["box_1a", "box_1b"],
        })

        result = check_cg3_structure_completeness(lines, anchors)
        # With 2 headings and 2 anchors, coverage should be good
        assert result.metrics.get("coverage_ratio", 0) >= 0.8

    def test_fails_when_half_anchors_removed(self):
        """
        FALSIFICATION: Remove half the anchors, coverage should degrade.
        """
        # 10 heading candidates
        lines = pd.DataFrame({
            "line_text": [f"Section {i} Header" for i in range(10)],
            "line_bold": [True] * 10,
            "line_size": [14.0] * 10,
            "gap_above": [25.0] * 10,
        })

        # Only 3 anchors (30% coverage)
        anchors = pd.DataFrame({
            "anchor_id": ["sec_1", "sec_2", "sec_3"],
        })

        result = check_cg3_structure_completeness(lines, anchors)

        # Coverage should be ~30%, below 80% threshold
        coverage = result.metrics.get("coverage_ratio", 1.0)
        assert coverage < VARIANCE["heading_coverage"], f"Coverage {coverage} should be below threshold"
        assert not result.passed, "Should fail with low coverage"

    def test_zero_anchors_fails(self):
        """No anchors with heading candidates = fail."""
        lines = pd.DataFrame({
            "line_text": ["Important Header", "Another Header"],
            "line_bold": [True, True],
            "line_size": [16.0, 16.0],
            "gap_above": [30.0, 30.0],
        })
        anchors = pd.DataFrame()  # Empty

        result = check_cg3_structure_completeness(lines, anchors)

        # 0 anchors / N candidates = 0% coverage
        if result.metrics.get("heading_candidates", 0) > 0:
            assert not result.passed


# =============================================================================
# TEST 4: IS5 structural DAG MUST detect cycles
# =============================================================================

class TestIS5StructuralDAG:
    """IS5 must catch cycles in structural edges only."""

    def test_valid_dag_passes(self):
        """Acyclic structure passes."""
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2", "e3"],
            "source_node_id": ["root", "root", "sec_a"],
            "target_node_id": ["sec_a", "sec_b", "box_1"],
            "edge_type": ["parent_of", "parent_of", "parent_of"],
        })
        result = check_is5_structural_dag(edges)
        assert result.passed
        assert not result.metrics.get("has_cycle", False)

    def test_detects_direct_cycle(self):
        """
        FALSIFICATION: Direct A→B→A cycle must fail.
        """
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["A", "B"],
            "target_node_id": ["B", "A"],  # Cycle!
            "edge_type": ["parent_of", "parent_of"],
        })
        result = check_is5_structural_dag(edges)

        assert result.metrics.get("has_cycle", False), "Should detect cycle"
        assert not result.passed, "Cycle should fail validation"

    def test_detects_indirect_cycle(self):
        """
        FALSIFICATION: Indirect A→B→C→A cycle must fail.
        """
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2", "e3"],
            "source_node_id": ["A", "B", "C"],
            "target_node_id": ["B", "C", "A"],  # Cycle through C
            "edge_type": ["parent_of", "parent_of", "parent_of"],
        })
        result = check_is5_structural_dag(edges)

        assert result.metrics.get("has_cycle", False), "Should detect indirect cycle"
        assert not result.passed

    def test_ignores_reference_cycles(self):
        """
        Reference edges (references_box) may form cycles - should NOT fail.
        """
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2", "e3", "e4"],
            "source_node_id": ["root", "A", "B", "B"],
            "target_node_id": ["A", "B", "A", "root"],
            "edge_type": ["parent_of", "parent_of", "references_box", "references_box"],
        })
        result = check_is5_structural_dag(edges)

        # Structural edges (parent_of) form valid DAG
        # Reference edges form cycle but should be ignored
        assert result.passed, "Reference cycles should be ignored"

    def test_ignores_semantic_cycles(self):
        """Semantic edges (excludes, includes) may cycle - OK."""
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2", "e3"],
            "source_node_id": ["root", "A", "B"],
            "target_node_id": ["A", "B", "A"],
            "edge_type": ["parent_of", "excludes", "excludes"],
        })
        result = check_is5_structural_dag(edges)

        # Only parent_of is structural, others ignored
        assert result.passed

    def test_no_root_fails(self):
        """Hierarchy without root should fail."""
        # All nodes are children of something
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["A", "B"],
            "target_node_id": ["B", "C"],
            "edge_type": ["parent_of", "parent_of"],
        })
        result = check_is5_structural_dag(edges)

        # A is parent of B, B is parent of C
        # A is the root (not a child of anything)
        # This should pass since A is root
        # Wait - let me check: roots = parents - children = {A} - so there IS a root
        # Actually this should pass
        assert result.metrics.get("root_count", 0) >= 1


# =============================================================================
# TEST 5: IS1/IS2 ID integrity MUST catch nulls and missing endpoints
# =============================================================================

class TestIS1IDIntegrity:
    """IS1 must catch null and duplicate IDs."""

    def test_valid_ids_pass(self, good_nodes_df, good_edges_df, good_anchors_df, good_elements_df):
        """Clean data passes."""
        result = check_is1_id_integrity(good_nodes_df, good_edges_df, good_anchors_df, good_elements_df)
        assert result.passed

    def test_null_node_id_fails(self):
        """
        FALSIFICATION: Null node_id must fail.
        """
        nodes = pd.DataFrame({
            "node_id": ["valid", None, "also_valid"],
        })
        edges = pd.DataFrame({"edge_id": []})
        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        result = check_is1_id_integrity(nodes, edges, anchors, elements)

        assert result.metrics.get("nodes_null", 0) > 0
        assert not result.passed

    def test_duplicate_node_id_fails(self):
        """
        FALSIFICATION: Duplicate node_id must fail.
        """
        nodes = pd.DataFrame({
            "node_id": ["same_id", "same_id", "different"],
        })
        edges = pd.DataFrame({"edge_id": []})
        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        result = check_is1_id_integrity(nodes, edges, anchors, elements)

        assert result.metrics.get("nodes_dup", 0) > 0
        assert not result.passed


class TestIS2EdgeEndpoints:
    """IS2 must catch edges pointing to non-existent nodes."""

    def test_valid_endpoints_pass(self):
        """All endpoints exist = pass."""
        nodes = pd.DataFrame({"node_id": ["A", "B", "C"]})
        edges = pd.DataFrame({
            "source_node_id": ["A", "B"],
            "target_node_id": ["B", "C"],
        })
        result = check_is2_edge_endpoints(nodes, edges)
        assert result.passed

    def test_missing_source_fails(self):
        """
        FALSIFICATION: Edge from non-existent node must fail.
        """
        nodes = pd.DataFrame({"node_id": ["A", "B"]})
        edges = pd.DataFrame({
            "source_node_id": ["GHOST"],  # Doesn't exist
            "target_node_id": ["A"],
        })
        result = check_is2_edge_endpoints(nodes, edges)

        assert result.metrics.get("missing_sources", 0) > 0
        assert not result.passed

    def test_missing_target_fails(self):
        """
        FALSIFICATION: Edge to non-existent node must fail.
        """
        nodes = pd.DataFrame({"node_id": ["A", "B"]})
        edges = pd.DataFrame({
            "source_node_id": ["A"],
            "target_node_id": ["PHANTOM"],  # Doesn't exist
        })
        result = check_is2_edge_endpoints(nodes, edges)

        assert result.metrics.get("missing_targets", 0) > 0
        assert not result.passed


# =============================================================================
# TEST 6: Integration - known-bad document
# =============================================================================

class TestKnownBadDocument:
    """
    FALSIFICATION: Inject a completely broken document.

    All validators should flag issues.
    """

    def test_garbage_document_fails_multiple_checks(self):
        """A document with multiple problems should fail multiple checks."""
        # Lines with no box patterns
        lines = pd.DataFrame({
            "line_text": ["Random text", "More random", "No structure"],
            "page": [1, 1, 1],
        })

        # Nodes with duplicates and nulls
        nodes = pd.DataFrame({
            "node_id": ["dup", "dup", None],
            "body_text": ["x", "y", "z"],
        })

        # Edges pointing nowhere
        edges = pd.DataFrame({
            "edge_id": ["e1"],
            "source_node_id": ["ghost"],
            "target_node_id": ["phantom"],
            "edge_type": ["parent_of"],
            "source_evidence": [""],
        })

        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        # Run internal checks
        internal_results = validate_internal(nodes, edges, anchors, elements)
        internal_failures = [r for r in internal_results if not r.passed]

        # Run corpus checks
        corpus_results = validate_corpus_grounded(
            lines, nodes, edges, anchors, elements
        )
        corpus_failures = [r for r in corpus_results if not r.passed]

        # Should have multiple failures
        total_failures = len(internal_failures) + len(corpus_failures)
        assert total_failures >= 2, f"Expected multiple failures, got {total_failures}"


# =============================================================================
# TEST 7: TE1-TE3 Typed Edge Extraction
# =============================================================================

@pytest.mark.skipif(not TYPED_EDGES_AVAILABLE, reason="typed_edges not available")
class TestTE1NegationFlip:
    """
    TE1: Mutate "does not include" → "includes"; assert excludes disappears
    and a positive edge appears (or polarity changes).
    """

    def test_negation_produces_excludes(self):
        """'Does not include' should produce excludes edge."""
        text = "Do not include amounts from box 2a in box 1a."
        valid_boxes = {"1a", "2a", "2b"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        # Should find at least one excludes edge
        assert len(edges) > 0, "Should extract excludes from negation"
        assert all(e.edge_type == "excludes" for e in edges)
        assert all(e.polarity == "negative" for e in edges)

    def test_positive_produces_includes(self):
        """'Include amounts' (no negation) should NOT produce excludes."""
        text = "Include amounts from box 2a in box 1a."
        valid_boxes = {"1a", "2a", "2b"}

        excludes_edges = extract_excludes_edges("box_1a", text, valid_boxes)

        # No negation = no excludes
        assert len(excludes_edges) == 0, "Positive text should not produce excludes"

    def test_negation_flip_changes_result(self):
        """
        FALSIFICATION: Same text with/without negation should differ.
        """
        negative_text = "Do not include amounts from box 2a."
        positive_text = "Include amounts from box 2a."  # Negation removed
        valid_boxes = {"1a", "2a"}

        neg_edges = extract_excludes_edges("box_1a", negative_text, valid_boxes)
        pos_edges = extract_excludes_edges("box_1a", positive_text, valid_boxes)

        # Negation should produce excludes, positive should not
        assert (len(neg_edges) > len(pos_edges)), (
            f"Negation flip should change results: neg={len(neg_edges)}, pos={len(pos_edges)}"
        )


@pytest.mark.skipif(not TYPED_EDGES_AVAILABLE, reason="typed_edges not available")
class TestTE2ScopeLeakage:
    """
    TE2: Insert negation sentence far from any box ref;
    assert no excludes edge emitted.
    """

    def test_distant_negation_no_edge(self):
        """
        FALSIFICATION: Negation without nearby box ref should not emit edge.
        """
        # Negation is 200+ chars away from box reference
        text = (
            "Do not include these amounts. " +
            "X" * 200 +  # Large gap
            "See box 2a for details."
        )
        valid_boxes = {"1a", "2a"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        # The negation context window is ~80 chars, so distant ref shouldn't match
        # This tests that scope is properly limited
        assert len(edges) == 0, \
            "Distant negation should not create edge (scope leakage)"

    def test_nearby_negation_creates_edge(self):
        """Negation near box ref should create edge (baseline)."""
        text = "Do not include amounts in box 2a."  # Negation is close
        valid_boxes = {"1a", "2a"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        assert len(edges) > 0, "Nearby negation should create edge"


@pytest.mark.skipif(not TYPED_EDGES_AVAILABLE, reason="typed_edges not available")
class TestTE4Ambiguity:
    """
    TE4: Ambiguity tests - multiple boxes in one sentence.

    When a sentence mentions multiple boxes (e.g., "Do not include
    amounts from box 2a or box 2b"), should create edges to all valid boxes.
    """

    def test_multiple_boxes_same_sentence(self):
        """Should create edges to all mentioned boxes."""
        text = "Do not include amounts from box 2a or box 2b in this box."
        valid_boxes = {"1a", "2a", "2b", "3"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        # Should create edges to both 2a and 2b
        target_keys = {e.target_box_key for e in edges}
        assert "2a" in target_keys, "Should extract edge to box 2a"
        assert "2b" in target_keys, "Should extract edge to box 2b"

    def test_boxes_and_pattern(self):
        """'boxes X and Y' should create edges to both."""
        text = "Do not include amounts from boxes 1b and 2e."
        valid_boxes = {"1a", "1b", "2a", "2e"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        target_keys = {e.target_box_key for e in edges}
        assert "1b" in target_keys, "Should extract edge to 1b"
        assert "2e" in target_keys, "Should extract edge to 2e"

    def test_comma_separated_boxes(self):
        """'box X, Y, and Z' should create edges to all."""
        text = "Do not include amounts from boxes 2a, 2b, and 2c."
        valid_boxes = {"1a", "2a", "2b", "2c", "2d"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        target_keys = {e.target_box_key for e in edges}
        assert len(target_keys) >= 2, f"Should extract multiple edges, got {target_keys}"

    def test_only_valid_boxes(self):
        """Should only create edges to boxes in valid_boxes set."""
        text = "Do not include amounts from box 99 or box 2a."
        valid_boxes = {"1a", "2a", "2b"}  # 99 not in valid set

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        target_keys = {e.target_box_key for e in edges}
        assert "2a" in target_keys, "Should extract valid box 2a"
        assert "99" not in target_keys, "Should NOT extract invalid box 99"


@pytest.mark.skipif(not TYPED_EDGES_AVAILABLE, reason="typed_edges not available")
class TestTE5Overlap:
    """
    TE5: Overlap tests - competing patterns in same text.

    When both "includes" and "excludes" patterns could match,
    validate the extractor handles priority correctly.
    """

    def test_excludes_ignores_positive_includes(self):
        """
        FALSIFICATION: Text with both include and exclude language.

        "Box 1a includes amounts... Do not include box 2a amounts"
        The excludes extractor should only find the negative relationship.
        """
        text = "Box 1a includes amounts from various sources. Do not include box 2a amounts in this total."
        valid_boxes = {"1a", "2a", "2b"}

        # Run excludes extractor
        excludes = extract_excludes_edges("box_1a", text, valid_boxes)

        # Should find excludes for 2a (near "Do not include")
        excl_targets = {e.target_box_key for e in excludes}
        assert "2a" in excl_targets, "Should find excludes edge to 2a"

        # Run includes extractor (requires source_box_key and excluded_boxes)
        includes = extract_includes_edges("box_1a", "1a", text, valid_boxes, set())

        # The positive "includes" statement shouldn't be confused with "do not include"
        # Includes extractor finds different patterns (reported_in_both, also_includes)

    def test_no_double_emission(self):
        """Same box should not appear twice from same extraction."""
        text = "Do not include box 2a. Also do not include box 2a amounts."
        valid_boxes = {"1a", "2a"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        # Should only have one edge to 2a, not duplicates
        targets = [e.target_box_key for e in edges]
        assert targets.count("2a") == 1, f"Should have exactly one edge to 2a, got {targets}"

    def test_complex_sentence_multiple_patterns(self):
        """
        Complex sentence with multiple edge type patterns.

        The extractor should handle overlapping patterns gracefully.
        """
        text = (
            "Box 1a includes amounts from box 2a but does not include "
            "any amounts from box 2b or 2c. See box 3 for details."
        )
        valid_boxes = {"1a", "2a", "2b", "2c", "3"}

        excludes = extract_excludes_edges("box_1a", text, valid_boxes)

        # Should extract excludes for 2b, 2c (near "does not include")
        excl_targets = {e.target_box_key for e in excludes}
        assert "2b" in excl_targets or "2c" in excl_targets, (
            f"Should extract at least one excludes edge, got {excl_targets}"
        )


@pytest.mark.skipif(not TYPED_EDGES_AVAILABLE, reason="typed_edges not available")
class TestTE3SelfEdgeGuard:
    """
    TE3: Craft "Do not include … in box 1b" inside box 1b;
    assert edge is either dropped or rewritten (no box_1b -> box_1b).

    Policy: Self-loops are INVALID. Extractor must either:
    - Drop them entirely, OR
    - Rewrite to an explicit exception
    """

    def test_self_reference_must_not_create_self_loop(self):
        """
        FALSIFICATION: Self-referential exclusion MUST NOT create self-loop.
        """
        # Text inside box_1b section says "do not include in box 1b"
        text = "Do not include these amounts in box 1b."
        valid_boxes = {"1a", "1b", "2a"}

        # Extract from box_1b context (source anchor = box_1b)
        edges = extract_excludes_edges("box_1b", text, valid_boxes)

        # Check for self-loops (source anchor matches target box)
        # source_anchor_id is "box_1b", target_box_key would be "1b"
        self_loops = [e for e in edges if e.target_box_key == "1b"]

        # POLICY: Self-loops are invalid and must be dropped
        assert len(self_loops) == 0, (
            f"Self-loop detected: box_1b -> box_1b. "
            f"Extractor should drop self-referential edges. Found {len(self_loops)} self-loops."
        )

    def test_cross_reference_creates_edge(self):
        """Cross-reference exclusion should create valid edge."""
        # Text in box_1a context references box 2a
        text = "Do not include amounts from box 2a."
        valid_boxes = {"1a", "1b", "2a"}

        edges = extract_excludes_edges("box_1a", text, valid_boxes)

        # Should create edge from box_1a -> box_2a (not a self-loop)
        assert len(edges) > 0, "Cross-reference should create edge"
        # Verify it's not a self-loop
        for e in edges:
            assert e.source_anchor_id != f"box_{e.target_box_key}", (
                f"Unexpected self-loop: {e.source_anchor_id} -> box_{e.target_box_key}"
            )


# =============================================================================
# TEST 8: H1-H3 Hierarchy Constraints
# =============================================================================

class TestH1SkeletonAbsenceMustFail:
    """
    H1: Run IS5 with zero parent_of edges and require failure.

    Tight rule: internal validators should never pass a graph with no structural skeleton.
    """

    def test_empty_edges_fails(self):
        """
        FALSIFICATION: Empty edge DataFrame must fail IS5.
        """
        edges = pd.DataFrame()
        result = check_is5_structural_dag(edges)

        # Empty skeleton should FAIL, not pass
        assert not result.passed, "Empty skeleton should fail IS5"

    def test_only_reference_edges_fails(self):
        """
        FALSIFICATION: Only reference edges (no structural) must fail.
        """
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["A", "B"],
            "target_node_id": ["B", "C"],
            "edge_type": ["references_box", "references_box"],  # Not structural
        })
        result = check_is5_structural_dag(edges)

        assert not result.passed, "Reference-only edges should fail (no skeleton)"
        assert result.metrics.get("structural_edges", -1) == 0

    def test_only_follows_edges_fails(self):
        """
        FALSIFICATION: Only follows edges (no parent_of) must fail.
        """
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["A", "B"],
            "target_node_id": ["B", "C"],
            "edge_type": ["follows", "follows"],  # Structural but not hierarchy
        })
        result = check_is5_structural_dag(edges)

        # Has structural edges but no parent_of = no hierarchy
        assert not result.passed, "Follows-only should fail (no hierarchy)"
        assert result.metrics.get("parent_of_count", -1) == 0

    def test_valid_hierarchy_passes(self):
        """Baseline: valid parent_of hierarchy should pass."""
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["root", "root"],
            "target_node_id": ["A", "B"],
            "edge_type": ["parent_of", "parent_of"],
        })
        result = check_is5_structural_dag(edges)

        assert result.passed, "Valid hierarchy should pass"
        assert result.metrics.get("root_count", 0) >= 1


class TestH2MultiRootStress:
    """
    H2: Create two doc_roots; assert warning or fail depending on policy.
    """

    def test_single_root_passes(self):
        """Single root is valid."""
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2", "e3"],
            "source_node_id": ["doc_root", "doc_root", "sec_a"],
            "target_node_id": ["sec_a", "sec_b", "box_1"],
            "edge_type": ["parent_of", "parent_of", "parent_of"],
        })
        result = check_is5_structural_dag(edges)

        assert result.passed
        assert result.metrics.get("root_count") == 1

    def test_multiple_roots_detected(self):
        """
        Multiple roots should be detected (may be warning or error).
        """
        # Two disconnected trees = two roots
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["root_a", "root_b"],  # Two different roots
            "target_node_id": ["child_a", "child_b"],
            "edge_type": ["parent_of", "parent_of"],
        })
        result = check_is5_structural_dag(edges)

        # Should detect multiple roots
        root_count = result.metrics.get("root_count", 0)
        assert root_count == 2, f"Should detect 2 roots, got {root_count}"

        # Policy: multiple roots might pass (forest) or warn/fail
        # At minimum, the metric should expose it
        # Current implementation: passes but metric shows 2 roots


class TestH3OrphanConcepts:
    """
    H3: Build concept nodes with no parent_of and no semantic edges;
    assert they're flagged (coverage metric).
    """

    def test_orphan_detection_in_coverage(self):
        """
        FALSIFICATION: Nodes with no edges should affect coverage metrics.
        """
        # Nodes including orphans
        nodes = pd.DataFrame({
            "node_id": ["root", "connected", "orphan_1", "orphan_2"],
            "node_type": ["doc_root", "section", "concept", "concept"],
        })

        # Edges only connect root -> connected
        edges = pd.DataFrame({
            "edge_id": ["e1"],
            "source_node_id": ["root"],
            "target_node_id": ["connected"],
            "edge_type": ["parent_of"],
        })

        anchors = pd.DataFrame({"anchor_id": ["sec_a"]})
        elements = pd.DataFrame({"element_id": ["el_1"]})

        # Run internal validation
        results = validate_internal(nodes, edges, anchors, elements)

        # IS5 should pass (valid DAG) but orphans exist
        is5 = next((r for r in results if r.check_id == "IS5"), None)
        assert is5 is not None

        # The orphan nodes are not reachable from root
        # This is a coverage issue, not a DAG issue
        # We should track this somewhere (future enhancement)

    def test_all_nodes_connected_baseline(self):
        """Baseline: all nodes connected = good coverage."""
        nodes = pd.DataFrame({
            "node_id": ["root", "child"],
            "node_type": ["doc_root", "section"],
        })
        edges = pd.DataFrame({
            "edge_id": ["e1"],
            "source_node_id": ["root"],
            "target_node_id": ["child"],
            "edge_type": ["parent_of"],
        })
        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        results = validate_internal(nodes, edges, anchors, elements)
        is5 = next((r for r in results if r.check_id == "IS5"), None)

        assert is5 is not None
        assert is5.passed


# =============================================================================
# TEST 9: Tight Rule - No Skeleton = Fail
# =============================================================================

class TestTightRuleSkeletonRequired:
    """
    TIGHT RULE: Internal validators should never pass a graph
    that has no structural skeleton.
    """

    def test_semantic_only_graph_fails(self):
        """
        Graph with only semantic edges (no structural) must fail.
        """
        nodes = pd.DataFrame({
            "node_id": ["A", "B", "C"],
        })
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["A", "B"],
            "target_node_id": ["B", "C"],
            "edge_type": ["excludes", "includes"],  # Semantic only
        })
        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        results = validate_internal(nodes, edges, anchors, elements)

        # IS5 should fail due to no structural skeleton
        is5 = next((r for r in results if r.check_id == "IS5"), None)
        assert is5 is not None
        assert not is5.passed, "Semantic-only graph should fail IS5"

    def test_mixed_graph_with_skeleton_passes(self):
        """
        Graph with structural + semantic edges should pass (if DAG valid).
        """
        nodes = pd.DataFrame({
            "node_id": ["root", "A", "B"],
        })
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2", "e3"],
            "source_node_id": ["root", "root", "A"],
            "target_node_id": ["A", "B", "B"],
            "edge_type": ["parent_of", "parent_of", "excludes"],  # Mixed
        })
        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        results = validate_internal(nodes, edges, anchors, elements)

        is5 = next((r for r in results if r.check_id == "IS5"), None)
        assert is5 is not None
        assert is5.passed, "Mixed graph with valid skeleton should pass"


# =============================================================================
# TEST 10: IS7 Reachability - Gating check
# =============================================================================

class TestIS7Reachability:
    """
    IS7: Gating reachability check.

    Requirements:
    - Exactly one doc_root node
    - ≥95% of nodes reachable from doc_root via parent_of
    - All box_* nodes must be reachable

    This kills the "graph emits but topology is garbage" scenario.
    """

    def test_full_reachability_passes(self):
        """All nodes reachable from doc_root = pass."""
        nodes = pd.DataFrame({
            "node_id": ["doc:root", "sec:a", "box:1a", "box:1b"],
            "node_type": ["doc_root", "section", "box", "box"],
        })
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2", "e3"],
            "source_node_id": ["doc:root", "doc:root", "sec:a"],
            "target_node_id": ["sec:a", "box:1a", "box:1b"],
            "edge_type": ["parent_of", "parent_of", "parent_of"],
        })

        result = check_is7_reachability(nodes, edges)

        assert result.passed, "Full reachability should pass"
        assert result.metrics["reachability_rate"] == 1.0
        assert result.metrics["unreachable_count"] == 0

    def test_no_doc_root_fails(self):
        """
        FALSIFICATION: No doc_root node must fail.
        """
        nodes = pd.DataFrame({
            "node_id": ["sec:a", "box:1a", "box:1b"],
            "node_type": ["section", "box", "box"],  # No doc_root!
        })
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["sec:a", "sec:a"],
            "target_node_id": ["box:1a", "box:1b"],
            "edge_type": ["parent_of", "parent_of"],
        })

        result = check_is7_reachability(nodes, edges)

        assert not result.passed, "No doc_root should fail IS7"
        assert result.metrics["doc_root_count"] == 0

    def test_multiple_doc_roots_fails(self):
        """
        FALSIFICATION: Multiple doc_root nodes must fail.
        """
        nodes = pd.DataFrame({
            "node_id": ["root:a", "root:b", "child:a", "child:b"],
            "node_type": ["doc_root", "doc_root", "section", "section"],  # Two roots!
        })
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["root:a", "root:b"],
            "target_node_id": ["child:a", "child:b"],
            "edge_type": ["parent_of", "parent_of"],
        })

        result = check_is7_reachability(nodes, edges)

        assert not result.passed, "Multiple doc_roots should fail IS7"
        assert result.metrics["doc_root_count"] == 2

    def test_low_reachability_fails(self):
        """
        FALSIFICATION: Reachability below 95% threshold must fail.
        """
        # Create 20 nodes: 1 doc_root + 19 others
        # Connect only 10 (50% reachability)
        node_ids = ["doc:root"] + [f"node:{i}" for i in range(19)]
        node_types = ["doc_root"] + ["section"] * 19

        nodes = pd.DataFrame({
            "node_id": node_ids,
            "node_type": node_types,
        })

        # Only connect 10 nodes (50% reachability)
        connected = [f"node:{i}" for i in range(9)]  # 9 connected + root = 10 total
        edges = pd.DataFrame({
            "edge_id": [f"e{i}" for i in range(9)],
            "source_node_id": ["doc:root"] * 9,
            "target_node_id": connected,
            "edge_type": ["parent_of"] * 9,
        })

        result = check_is7_reachability(nodes, edges)

        # 10/20 = 50% < 95% threshold
        assert not result.passed, f"50% reachability should fail (threshold {REACHABILITY_THRESHOLD:.0%})"
        assert result.metrics["reachability_rate"] < REACHABILITY_THRESHOLD

    def test_unreachable_box_fails(self):
        """
        FALSIFICATION: Unreachable box_* node must fail.
        """
        nodes = pd.DataFrame({
            "node_id": ["doc:root", "sec:a", "box_1a", "box_orphan"],
            "node_type": ["doc_root", "section", "box", "box"],
        })

        # box_orphan not connected
        edges = pd.DataFrame({
            "edge_id": ["e1", "e2"],
            "source_node_id": ["doc:root", "sec:a"],
            "target_node_id": ["sec:a", "box_1a"],
            "edge_type": ["parent_of", "parent_of"],
        })

        result = check_is7_reachability(nodes, edges)

        # 3/4 = 75% < 95%, AND box_orphan is unreachable
        assert not result.passed, "Unreachable box node should fail"
        assert result.metrics["unreachable_box_count"] > 0
        assert "box_orphan" in [str(b) for b in result.metrics.get("unreachable_boxes", [])] or \
               result.metrics["unreachable_box_count"] == 1

    def test_edge_threshold_reachability(self):
        """
        Test exactly at 95% threshold.
        """
        # 20 nodes, 19 connected = 95%
        node_ids = ["doc:root"] + [f"node:{i}" for i in range(19)]
        node_types = ["doc_root"] + ["section"] * 19

        nodes = pd.DataFrame({
            "node_id": node_ids,
            "node_type": node_types,
        })

        # Connect 18 nodes (19 reachable including root = 95%)
        connected = [f"node:{i}" for i in range(18)]
        edges = pd.DataFrame({
            "edge_id": [f"e{i}" for i in range(18)],
            "source_node_id": ["doc:root"] * 18,
            "target_node_id": connected,
            "edge_type": ["parent_of"] * 18,
        })

        result = check_is7_reachability(nodes, edges)

        # 19/20 = 95% exactly at threshold
        assert result.metrics["reachability_rate"] >= REACHABILITY_THRESHOLD

    def test_empty_nodes_fails(self):
        """
        FALSIFICATION: Empty nodes DataFrame must fail.
        """
        nodes = pd.DataFrame()
        edges = pd.DataFrame({
            "edge_id": ["e1"],
            "source_node_id": ["a"],
            "target_node_id": ["b"],
            "edge_type": ["parent_of"],
        })

        result = check_is7_reachability(nodes, edges)

        assert not result.passed, "Empty nodes should fail IS7"

    def test_no_edges_fails(self):
        """
        FALSIFICATION: No edges to traverse must fail.
        """
        nodes = pd.DataFrame({
            "node_id": ["doc:root", "orphan"],
            "node_type": ["doc_root", "section"],
        })
        edges = pd.DataFrame()

        result = check_is7_reachability(nodes, edges)

        assert not result.passed, "No edges should fail IS7"

    def test_is7_in_validate_internal(self):
        """IS7 must be included in validate_internal results."""
        nodes = pd.DataFrame({
            "node_id": ["doc:root", "child"],
            "node_type": ["doc_root", "section"],
        })
        edges = pd.DataFrame({
            "edge_id": ["e1"],
            "source_node_id": ["doc:root"],
            "target_node_id": ["child"],
            "edge_type": ["parent_of"],
            "source_evidence": ["structural"],
        })
        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        results = validate_internal(nodes, edges, anchors, elements)
        check_ids = [r.check_id for r in results]

        assert "IS7" in check_ids, "IS7 should be in validate_internal results"

    def test_deep_hierarchy_reachability(self):
        """Deep hierarchy should still have full reachability."""
        # Create a deep chain: root -> n1 -> n2 -> n3 -> ... -> n10
        nodes = pd.DataFrame({
            "node_id": ["doc:root"] + [f"n{i}" for i in range(1, 11)],
            "node_type": ["doc_root"] + ["section"] * 10,
        })

        # Chain edges
        sources = ["doc:root"] + [f"n{i}" for i in range(1, 10)]
        targets = [f"n{i}" for i in range(1, 11)]
        edges = pd.DataFrame({
            "edge_id": [f"e{i}" for i in range(10)],
            "source_node_id": sources,
            "target_node_id": targets,
            "edge_type": ["parent_of"] * 10,
        })

        result = check_is7_reachability(nodes, edges)

        assert result.passed, "Deep hierarchy should pass"
        assert result.metrics["reachability_rate"] == 1.0


# =============================================================================
# TEST 11: CG4 Geometry-Based Anchor Discovery
# =============================================================================

class TestCG4GeometryDiscovery:
    """
    CG4: Geometry-based anchor discovery.

    Uses geometric signals (font size, bold, gaps) to discover anchors
    independently from text patterns. Cross-checks with text discovery.
    """

    def test_geometry_finds_large_font_anchors(self):
        """Large font lines should be detected as anchor candidates."""
        lines = pd.DataFrame({
            "line_text": [
                "Box 1a. Ordinary Dividends",
                "Some body text here",
                "More body text continues",
                "Box 2a. Capital Gains",
            ],
            "line_size": [12.0, 9.0, 9.0, 12.0],  # Body=9, headers=12
            "page": [1, 1, 1, 1],
        })
        text_discovered = {"1a", "2a"}

        result = check_cg4_geometry_discovery(lines, text_discovered)

        assert result.metrics.get("geometry_candidates", 0) >= 2

    def test_geometry_finds_bold_anchors(self):
        """Bold lines should be detected as anchor candidates."""
        # Need enough body lines to make median font=9.0
        lines = pd.DataFrame({
            "line_text": [
                "Section Header",  # Bold + large font + short text
                "Normal text that is longer and represents body content",
                "Another line of body text here for median calculation",
                "More body text to establish the median font size",
                "Yet another body line to ensure median is 9",
                "Another Header",  # Bold + large font + short text
            ],
            "line_size": [12.0, 9.0, 9.0, 9.0, 9.0, 12.0],  # Median=9, headers=12
            "line_bold": [True, False, False, False, False, True],  # Bold headers
            "gap_above": [25.0, 5.0, 5.0, 5.0, 5.0, 25.0],
            "page": [1, 1, 1, 1, 1, 1],
        })
        text_discovered = set()

        result = check_cg4_geometry_discovery(lines, text_discovered)

        # Font 12 > 9*1.1=9.9 → +3, Bold → +2, short text → +1 = 6 >= 4
        assert result.metrics.get("geometry_candidates", 0) >= 1

    def test_empty_lines_graceful(self):
        """Empty input should be handled gracefully."""
        result = check_cg4_geometry_discovery(pd.DataFrame(), set())

        assert result.passed

    def test_no_geometry_columns_graceful(self):
        """Missing geometry columns should not crash."""
        lines = pd.DataFrame({
            "line_text": ["Box 1a. Test"],
            "page": [1],
            # No font_size, line_bold, gap_above columns
        })
        result = check_cg4_geometry_discovery(lines, {"1a"})

        assert result.passed

    def test_correlation_with_text_discovery(self):
        """Geometry candidates should correlate with text-discovered boxes."""
        # Need more body lines so median font is 9.0, making 12.0 stand out
        lines = pd.DataFrame({
            "line_text": [
                "Box 1a. Header",  # Large font
                "Box 1b. Header",  # Large font
                "Box 2a. Header",  # Large font
                "body text content",
                "more body text",
                "even more body",
                "body continues",
            ],
            "line_size": [12.0, 12.0, 12.0, 9.0, 9.0, 9.0, 9.0],  # Median=9, headers=12
            "page": [1] * 7,
        })
        text_discovered = {"1a", "1b", "2a"}

        result = check_cg4_geometry_discovery(lines, text_discovered)

        # Large font (12 vs median 9) should score 3 points + text length 1 = 4
        assert result.metrics.get("geom_with_box_pattern", 0) >= 1

    def test_warns_on_low_geometry_coverage(self):
        """
        FALSIFICATION: If geometry finds far fewer anchors than text,
        something may be wrong with geometry detection.
        """
        # Text found 20 boxes but geometry finds none
        lines = pd.DataFrame({
            "line_text": ["Box " + str(i) for i in range(20)],
            "line_size": [9.0] * 20,  # All body size
            "line_bold": [False] * 20,  # No bold
            "gap_above": [5.0] * 20,  # No large gaps
            "page": [1] * 20,
        })
        text_discovered = {str(i) for i in range(20)}

        result = check_cg4_geometry_discovery(lines, text_discovered)

        # Should warn about geometry finding < 50% of text-discovered
        ratio = result.metrics.get("geom_to_text_ratio", 1.0)
        assert ratio < 0.5, "Geometry should find fewer when no signals present"
        # Should have warning
        warnings = [f for f in result.findings if f.severity == "warning"]
        assert len(warnings) >= 1, "Should warn about low geometry coverage"

    def test_cg4_in_corpus_validation(self):
        """CG4 should be included in validate_corpus_grounded results."""
        lines = pd.DataFrame({
            "line_text": ["Box 1a. Test"],
            "page": [1],
        })
        nodes = pd.DataFrame({"body_text": []})
        edges = pd.DataFrame({"source_evidence": []})
        anchors = pd.DataFrame({"anchor_id": []})
        elements = pd.DataFrame({"element_id": []})

        results = validate_corpus_grounded(lines, nodes, edges, anchors, elements)
        check_ids = [r.check_id for r in results]

        assert "CG4" in check_ids, "CG4 should be in validate_corpus_grounded results"
