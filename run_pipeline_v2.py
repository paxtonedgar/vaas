#!/usr/bin/env python3
"""
Run the VaaS extraction pipeline on 1099-DIV.

This is the refactored orchestration layer that uses modular components
from the vaas package.

Usage:
    python run_pipeline_v2.py [--pdf PATH] [--output DIR] [--validate]
"""

import sys
sys.path.insert(0, 'src')

import argparse
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Set

# =============================================================================
# VAAS IMPORTS
# =============================================================================

from vaas.extraction import (
    # PDF extraction
    extract_spans_from_pdf,
    infer_body_font_size,
    # Line building
    build_line_dataframe,
    add_block_geometry,
    add_page_geometry,
    add_emphasis_flags,
    add_list_item_flags,
    add_text_properties,
    add_header_pattern_flags,
    add_page_marker_flags,
    add_structural_filters,
    # Column detection
    detect_columns_for_document,
    assign_line_columns,
    # Layout detection
    detect_subsection_candidates,
    assign_split_triggers,
    # Element extraction
    split_blocks_into_elements,
    classify_elements,
    # Anchor extraction
    extract_anchors,
    build_anchor_timeline,
    assign_elements_to_anchors,
    EXPECTED_BOXES_1099DIV,
    # Section materialization
    materialize_sections,
    # Merge
    merge_thin_subsections,
    # References
    extract_references,
    # Role constants
    ROLE_PAGE_ARTIFACT,
)

from vaas.semantic.concept_roles import classify_section_roles
from vaas.graph import build_nodes_legacy, build_edges_legacy


# =============================================================================
# PIPELINE CONFIG
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    pdf_path: str
    doc_id: str
    output_dir: str
    expected_boxes: Set[str]
    page_mid_x: float = 300.0
    validate: bool = False

    @classmethod
    def for_1099div(
        cls,
        pdf_path: str = "data/i1099div.pdf",
        output_dir: str = "output",
    ) -> "PipelineConfig":
        """Create config for 1099-DIV extraction."""
        return cls(
            pdf_path=pdf_path,
            doc_id="1099div_filer",
            output_dir=output_dir,
            expected_boxes=EXPECTED_BOXES_1099DIV,
        )


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def stage_1_extract_spans(config: PipelineConfig) -> pd.DataFrame:
    """Cell 1: Extract spans from PDF."""
    print("=" * 70)
    print("CELL 1: Extracting spans from PDF")
    print("=" * 70)

    result = extract_spans_from_pdf(config.pdf_path, config.doc_id)
    print(f"Pages: {result.page_count}")
    print(f"Spans extracted: {len(result.spans_df)}")

    return result.spans_df


def stage_2_infer_body_font(spans_df: pd.DataFrame) -> float:
    """Cell 2: Infer body font size."""
    print("\n" + "=" * 70)
    print("CELL 2: Inferring body font size")
    print("=" * 70)

    body_size = infer_body_font_size(spans_df)
    print(f"Body font size: {body_size}")

    return body_size


def stage_3_build_lines(
    spans_df: pd.DataFrame,
    body_size: float,
) -> tuple:
    """Cell 3: Build lines and detect structure."""
    print("\n" + "=" * 70)
    print("CELL 3: Building lines and detecting structure")
    print("=" * 70)

    # Build line DataFrame
    line_df = build_line_dataframe(spans_df, body_size)

    # Add geometry
    line_df = add_block_geometry(line_df)
    line_df = add_page_geometry(line_df)
    line_df = add_emphasis_flags(line_df, body_size)
    line_df = add_list_item_flags(line_df)

    # Detect columns
    col_info_df = detect_columns_for_document(line_df)
    line_df = assign_line_columns(line_df, col_info_df)

    # Clean up temporary columns
    line_df.drop(columns=["_is_bullet", "_is_enum", "_is_list_item"], inplace=True, errors="ignore")

    # Add text properties and header pattern flags (needed before layout detection)
    line_df = add_text_properties(line_df)
    line_df = add_page_marker_flags(line_df)
    line_df = add_header_pattern_flags(line_df)
    line_df = add_structural_filters(line_df)

    # Detect subsection candidates and assign split triggers
    line_df = detect_subsection_candidates(line_df)
    line_df = assign_split_triggers(line_df)

    print(f"Lines: {len(line_df)}")
    two_col = len(col_info_df[col_info_df["num_columns"] == 2]) if not col_info_df.empty else 0
    print(f"Column detection: {two_col}/{len(col_info_df)} pages have 2 columns")
    print(f"Split triggers: {line_df['split_trigger'].sum()}")

    return line_df, col_info_df


def stage_4_split_and_classify(
    line_df: pd.DataFrame,
    page_mid_x: float = 300.0,
) -> pd.DataFrame:
    """Cell 3-4: Split blocks into elements and classify."""
    print("\n" + "=" * 70)
    print("CELL 4: Splitting and classifying elements")
    print("=" * 70)

    elements_df = split_blocks_into_elements(line_df, page_mid_x)
    elements_df = classify_elements(elements_df)

    print(f"Elements: {len(elements_df)}")
    print("Role distribution:")
    print(elements_df["role"].value_counts().to_string())

    return elements_df


def stage_5_extract_anchors(
    elements_df: pd.DataFrame,
    expected_boxes: Set[str],
) -> tuple:
    """Cell 5-6: Extract anchors and build timeline."""
    print("\n" + "=" * 70)
    print("CELL 5-6: Anchor extraction and timeline")
    print("=" * 70)

    from vaas.extraction import (
        ROLE_BOX_HEADER, ROLE_SECTION_HEADER, ROLE_SUBSECTION_HEADER,
        validate_box_coverage
    )

    extraction = extract_anchors(
        elements_df,
        role_box_header=ROLE_BOX_HEADER,
        role_section_header=ROLE_SECTION_HEADER,
        role_subsection_header=ROLE_SUBSECTION_HEADER,
    )
    anchors_df = extraction.anchors_df
    print(f"Anchors extracted: {len(anchors_df)}")

    # Validate box coverage
    if expected_boxes and not anchors_df.empty:
        validation = validate_box_coverage(anchors_df, expected_boxes)
        coverage = len(validation.found) / len(validation.expected) if validation.expected else 0
        print(f"Box coverage: {coverage:.1%} ({len(validation.found)}/{len(validation.expected)})")
        if validation.missing:
            print(f"Missing boxes: {validation.missing}")

    # Build timeline and assign
    timeline = build_anchor_timeline(anchors_df, elements_df)
    elements_df = assign_elements_to_anchors(elements_df, timeline)

    return anchors_df, elements_df


def stage_6_materialize_sections(
    elements_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    col_info_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cell 7-8: Materialize sections and merge thin subsections."""
    print("\n" + "=" * 70)
    print("CELL 7-8: Section materialization and merge")
    print("=" * 70)

    from vaas.extraction import ROLE_BOX_HEADER, ROLE_SECTION_HEADER, ROLE_SUBSECTION_HEADER

    # Pass correct header_roles with proper case
    header_roles = {ROLE_BOX_HEADER, ROLE_SECTION_HEADER, ROLE_SUBSECTION_HEADER}
    result = materialize_sections(elements_df, anchors_df, header_roles=header_roles)
    sections_df = result.sections_df
    print(f"Sections materialized: {len(sections_df)}")

    # Merge thin subsections
    sections_df = merge_thin_subsections(sections_df, col_info_df)
    print(f"Sections after merge: {len(sections_df)}")

    # Classify concept roles
    sections_df = classify_section_roles(sections_df)

    # Report
    role_counts = sections_df[sections_df["anchor_type"] == "subsection"]["concept_role"].value_counts(dropna=False)
    print(f"\n--- Concept Roles ---")
    for role, count in role_counts.items():
        label = role if role else "NULL"
        print(f"  {label}: {count}")

    return sections_df


def stage_7_extract_references(
    elements_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cell 9: Extract references."""
    print("\n" + "=" * 70)
    print("CELL 9: Reference extraction")
    print("=" * 70)

    # Get valid box keys
    valid_box_keys = set()
    if not anchors_df.empty and "box_key" in anchors_df.columns:
        valid_box_keys = set(anchors_df["box_key"].dropna().str.lower())

    result = extract_references(elements_df, valid_box_keys)
    print(f"References extracted: {result.total}")
    print(f"  Box refs: {result.box_refs}")
    print(f"  Pub refs: {result.pub_refs}")
    print(f"  IRC refs: {result.irc_refs}")
    print(f"  Form refs: {result.form_refs}")

    return result.references_df


def stage_8_build_graph(
    sections_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    references_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    doc_id: str,
) -> tuple:
    """Cell 10: Build graph nodes and edges."""
    print("\n" + "=" * 70)
    print("CELL 10: Graph construction")
    print("=" * 70)

    graph_nodes, paragraph_nodes_df = build_nodes_legacy(
        sections_df=sections_df,
        elements_df=elements_df,
        doc_id=doc_id,
        doc_label="1099-DIV Filer Instructions",
        role_page_artifact=ROLE_PAGE_ARTIFACT,
    )

    graph_edges = build_edges_legacy(
        sections_df=sections_df,
        paragraph_nodes_df=paragraph_nodes_df,
        references_df=references_df,
        anchors_df=anchors_df,
        graph_nodes_df=graph_nodes,
        doc_id=doc_id,
    )

    return graph_nodes, graph_edges


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config: PipelineConfig) -> Dict:
    """
    Run the full extraction pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Dict with pipeline results and counts.
    """
    print("\n" + "=" * 70)
    print(f"VaaS Extraction Pipeline: {config.pdf_path}")
    print("=" * 70 + "\n")

    # Stage 1: Extract spans
    spans_df = stage_1_extract_spans(config)

    # Stage 2: Infer body font
    body_size = stage_2_infer_body_font(spans_df)

    # Stage 3: Build lines
    line_df, col_info_df = stage_3_build_lines(spans_df, body_size)

    # Stage 4: Split and classify
    elements_df = stage_4_split_and_classify(line_df, config.page_mid_x)

    # Stage 5: Extract anchors
    anchors_df, elements_df = stage_5_extract_anchors(elements_df, config.expected_boxes)

    # Stage 6: Materialize sections
    sections_df = stage_6_materialize_sections(elements_df, anchors_df, col_info_df)

    # Stage 7: Extract references
    references_df = stage_7_extract_references(elements_df, anchors_df)

    # Stage 8: Build graph
    graph_nodes, graph_edges = stage_8_build_graph(
        sections_df, elements_df, references_df, anchors_df, config.doc_id
    )

    # Save outputs
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    graph_nodes.to_parquet(output_dir / "graph_nodes.parquet", index=False)
    graph_edges.to_parquet(output_dir / "graph_edges.parquet", index=False)
    sections_df.to_parquet(output_dir / "sections.parquet", index=False)
    anchors_df.to_parquet(output_dir / "anchors.parquet", index=False)

    print("\n" + "=" * 70)
    print(f"Outputs saved to {output_dir}/")
    print("=" * 70)

    results = {
        "spans": len(spans_df),
        "lines": len(line_df),
        "elements": len(elements_df),
        "anchors": len(anchors_df),
        "sections": len(sections_df),
        "references": len(references_df),
        "nodes": len(graph_nodes),
        "edges": len(graph_edges),
    }

    print("\n--- Pipeline Summary ---")
    for key, value in results.items():
        print(f"  {key}: {value}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run VaaS extraction pipeline on 1099-DIV"
    )
    parser.add_argument(
        "--pdf",
        default="data/i1099div.pdf",
        help="Path to PDF file",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks",
    )

    args = parser.parse_args()

    config = PipelineConfig.for_1099div(
        pdf_path=args.pdf,
        output_dir=args.output,
    )
    config.validate = args.validate

    results = run_pipeline(config)
    print(f"\nPipeline complete: {results['nodes']} nodes, {results['edges']} edges")


if __name__ == "__main__":
    main()
