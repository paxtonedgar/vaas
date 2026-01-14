#!/usr/bin/env python3
"""
Run the VaaS extraction pipeline on 1099-DIV.

This is the refactored orchestration layer that uses modular components
from the vaas package.

Usage:
    python -m vaas.run_pipeline_v2 [--pdf PATH] [--output DIR] [--validate]
"""

import argparse
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from collections import Counter

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
from vaas.extraction.references import generate_ref_occurrence_id

from vaas.semantic.concept_roles import classify_section_roles
from vaas.semantic.regime_detection import (
    detect_regimes,
    build_regime_nodes,
    build_regime_edges,
)
from vaas.graph import build_all_nodes, build_all_edges
from vaas.graph.edges import Edge
from vaas.semantic.authority_materializer import materialize_authorities
from vaas.semantic.manifest import build_manifest, write_manifest
from vaas.semantic.claim_pipeline import emit_claim_artifacts
from vaas.semantic.sentence_index import build_sentence_index
from vaas.semantic.scope_resolver import build_struct_scope_overlay
from vaas.semantic.typed_edge_accounting import build_typed_edge_dataframe
from vaas.semantic.resolution import resolve_claims
from vaas.semantic.compiler import compile_constraints


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
    form_id: str = "form:unknown"
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
            form_id="form:1099-div",
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
    doc_id: str,
) -> pd.DataFrame:
    """Cell 9: Extract references."""
    print("\n" + "=" * 70)
    print("CELL 9: Reference extraction")
    print("=" * 70)

    # Get valid box keys
    valid_box_keys = set()
    if not anchors_df.empty and "box_key" in anchors_df.columns:
        valid_box_keys = set(anchors_df["box_key"].dropna().str.lower())

    result = extract_references(elements_df, doc_id=doc_id, valid_box_keys=valid_box_keys)
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
    form_id: str,
) -> tuple:
    """Cell 10: Build graph nodes and edges."""
    print("\n" + "=" * 70)
    print("CELL 10: Graph construction")
    print("=" * 70)

    reference_element_ids: Set[str] = set()
    if references_df is not None and not references_df.empty:
        reference_element_ids = set(
            references_df["source_element_id"].dropna().astype(str)
        )

    # Build nodes
    node_result = build_all_nodes(
        sections_df=sections_df,
        elements_df=elements_df,
        doc_id=doc_id,
        doc_label="1099-DIV Filer Instructions",
        skip_roles={ROLE_PAGE_ARTIFACT},
        force_element_ids=reference_element_ids,
    )
    print(f"Nodes: {node_result.total} (sections={node_result.section_node_count}, paragraphs={node_result.paragraph_node_count})")

    # Build edges
    edge_result = build_all_edges(
        sections_df=sections_df,
        paragraph_nodes_df=node_result.paragraph_nodes_df,
        references_df=references_df,
        anchors_df=anchors_df,
        graph_nodes_df=node_result.nodes_df,
        doc_id=doc_id,
        form_id=form_id,
    )
    print(f"Edges: {len(edge_result.edges_df)}")
    if edge_result.edges_filtered > 0:
        print(f"  (filtered {edge_result.edges_filtered} invalid edges)")
    print(f"Edge types: {edge_result.edge_counts}")

    return node_result.nodes_df, edge_result.edges_df, edge_result.typed_edges


def stage_9_detect_regimes(
    sections_df: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    graph_edges: pd.DataFrame,
    doc_id: str,
) -> tuple:
    """Detect regimes and add regime nodes/edges to graph."""
    print("\n" + "=" * 70)
    print("CELL 10b: Regime detection")
    print("=" * 70)

    # Detect regimes from sections
    regimes = detect_regimes(sections_df, doc_id)
    print(f"Regimes detected: {len(regimes)}")

    if regimes:
        for r in regimes:
            boxes_str = ", ".join(r.governed_boxes) if r.governed_boxes else "none"
            print(f"  {r.regime_id}: governs boxes [{boxes_str}]")

        # Build regime nodes
        regime_nodes = build_regime_nodes(regimes, sections_df, doc_id)
        print(f"Regime nodes: {len(regime_nodes)}")

        # Build regime edges (governs)
        regime_edges = build_regime_edges(regimes, sections_df, doc_id)
        print(f"Regime edges: {len(regime_edges)}")

        # Merge regime nodes into graph_nodes
        if not regime_nodes.empty:
            # Ensure columns match
            for col in graph_nodes.columns:
                if col not in regime_nodes.columns:
                    regime_nodes[col] = None
            for col in regime_nodes.columns:
                if col not in graph_nodes.columns:
                    graph_nodes[col] = None

            graph_nodes = pd.concat([graph_nodes, regime_nodes], ignore_index=True)

        # Merge regime edges into graph_edges
        if regime_edges:
            regime_edges_df = pd.DataFrame(regime_edges)
            # Ensure columns match
            for col in graph_edges.columns:
                if col not in regime_edges_df.columns:
                    regime_edges_df[col] = None
            for col in regime_edges_df.columns:
                if col not in graph_edges.columns:
                    graph_edges[col] = None

            graph_edges = pd.concat([graph_edges, regime_edges_df], ignore_index=True)

        print(f"Total nodes after regimes: {len(graph_nodes)}")
        print(f"Total edges after regimes: {len(graph_edges)}")

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

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

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

    sentence_index_df = build_sentence_index(elements_df, config.doc_id)
    sentence_index_df.to_parquet(output_dir / "element_sentences.parquet", index=False)

    # Stage 6: Materialize sections
    sections_df = stage_6_materialize_sections(elements_df, anchors_df, col_info_df)

    # Stage 7: Extract references
    references_df = stage_7_extract_references(elements_df, anchors_df, config.doc_id)
    references_df = annotate_reference_sentences(references_df, sentence_index_df)
    references_df.to_parquet(output_dir / "references.parquet", index=False)

    # Stage 8: Build graph
    graph_nodes, graph_edges, typed_semantic_edges = stage_8_build_graph(
        sections_df,
        elements_df,
        references_df,
        anchors_df,
        config.doc_id,
        config.form_id,
    )

    # Stage 9: Detect regimes and add to graph
    graph_nodes, graph_edges = stage_9_detect_regimes(
        sections_df, graph_nodes, graph_edges, config.doc_id
    )

    struct_overlay_df = build_struct_scope_overlay(graph_nodes, sentence_index_df)
    struct_overlay_df.to_parquet(output_dir / "struct_nodes.parquet", index=False)

    # Materialize authorities
    authorities_df, paragraph_authority_df, authority_mentions_df, authority_stats = materialize_authorities(
        config.doc_id,
        references_df,
        graph_nodes,
    )

    if not authorities_df.empty:
        authority_node_rows = []
        if graph_nodes.empty:
            node_columns = [
                "node_id",
                "doc_id",
                "node_type",
                "anchor_id",
                "box_key",
                "label",
                "text",
                "pages",
                "bbox",
                "element_id",
                "element_count",
                "char_count",
                "reading_order",
                "paragraph_kind",
                "anchor_type",
                "concept_role",
            ]
            graph_nodes = pd.DataFrame(columns=node_columns)
        else:
            node_columns = list(graph_nodes.columns)
        for _, auth in authorities_df.iterrows():
            row = {
                "node_id": auth["authority_id"],
                "doc_id": config.doc_id,
                "node_type": "authority",
                "anchor_id": None,
                "box_key": None,
                "label": auth.get("label"),
                "text": auth.get("raw_text"),
                "pages": [],
                "bbox": None,
                "element_id": None,
                "element_count": 0,
                "char_count": len(auth.get("raw_text", "") or ""),
                "reading_order": None,
                "paragraph_kind": None,
                "anchor_type": None,
                "concept_role": auth.get("authority_type"),
            }
            authority_node_rows.append(row)
        authority_nodes_df = pd.DataFrame(authority_node_rows, columns=node_columns)
        graph_nodes = pd.concat([graph_nodes, authority_nodes_df], ignore_index=True)

    typed_edges_df = build_typed_edge_dataframe(config.doc_id, typed_semantic_edges)
    claimable_edges_count = 0
    if not typed_edges_df.empty and "claimable" in typed_edges_df.columns:
        claimable_edges_count = int(typed_edges_df["claimable"].astype(bool).sum())
    typed_edges_df.to_parquet(output_dir / "typed_edges.parquet", index=False)

    if not paragraph_authority_df.empty:
        auth_edge_rows = []
        if graph_edges.empty:
            edge_columns = [
                "edge_id",
                "source_node_id",
                "target_node_id",
                "edge_type",
                "direction",
                "confidence",
                "source_evidence",
                "source_element_id",
                "created_by",
                "pattern_matched",
                "polarity",
                "evidence_sentence_idx",
                "evidence_char_start",
                "evidence_char_end",
                "rule_class",
                "precedence",
            ]
            graph_edges = pd.DataFrame(columns=edge_columns)
        else:
            edge_columns = list(graph_edges.columns)
        for _, link in paragraph_authority_df.iterrows():
            auth_edge_rows.append({
                "edge_id": link["edge_id"],
                "source_node_id": link["paragraph_node_id"],
                "target_node_id": link["authority_id"],
                "edge_type": "mentions_authority",
                "direction": "directed",
                "confidence": 1.0,
                "source_evidence": link.get("evidence_text"),
                "source_element_id": link.get("source_element_id"),
                "created_by": "semantic",
                "pattern_matched": None,
                "polarity": "positive",
                "evidence_sentence_idx": None,
                "evidence_char_start": None,
                "evidence_char_end": None,
                "rule_class": None,
                "precedence": None,
            })
        authority_edges_df = pd.DataFrame(auth_edge_rows, columns=edge_columns)
        graph_edges = pd.concat([graph_edges, authority_edges_df], ignore_index=True)

    graph_nodes.to_parquet(output_dir / "graph_nodes.parquet", index=False)
    graph_edges.to_parquet(output_dir / "graph_edges.parquet", index=False)
    sections_df.to_parquet(output_dir / "sections.parquet", index=False)
    anchors_df.to_parquet(output_dir / "anchors.parquet", index=False)
    authorities_df.to_parquet(output_dir / "authorities.parquet", index=False)
    paragraph_authority_df.to_parquet(output_dir / "paragraph_authority_edges.parquet", index=False)
    authority_mentions_df.to_parquet(output_dir / "authority_mentions.parquet", index=False)

    claim_stats = emit_claim_artifacts(
        doc_id=config.doc_id,
        form_id=config.form_id,
        typed_edges=typed_semantic_edges,
        nodes_df=graph_nodes,
        paragraph_authority_df=paragraph_authority_df,
        sentence_index_df=sentence_index_df,
        authority_mentions_df=authority_mentions_df,
        struct_overlay_df=struct_overlay_df,
        output_dir=output_dir,
    )

    claims_df = pd.read_parquet(output_dir / "claims.parquet")
    precedence_df = pd.read_parquet(output_dir / "claim_precedence.parquet")
    resolution_stats = resolve_claims(claims_df, precedence_df, output_dir)
    resolved_claims_df = pd.read_parquet(output_dir / "resolved_claims.parquet")
    compiler_stats = compile_constraints(resolved_claims_df, output_dir)

    manifest_tables = [
        "graph_nodes.parquet",
        "graph_edges.parquet",
        "struct_nodes.parquet",
        "sections.parquet",
        "anchors.parquet",
        "element_sentences.parquet",
        "references.parquet",
        "authorities.parquet",
        "paragraph_authority_edges.parquet",
        "authority_mentions.parquet",
        "typed_edges.parquet",
        "typed_edge_claims.parquet",
        "claims.parquet",
        "constraints.parquet",
        "claim_edges.parquet",
        "claim_rejections.parquet",
        "claim_authority_mentions.parquet",
        "claim_precedence.parquet",
        "resolved_claims.parquet",
        "resolution_groups.parquet",
        "compiled_directives.parquet",
        "constraints_resolved.parquet",
        "claim_constraints.parquet",
        "constraint_attributes.parquet",
    ]
    manifest = build_manifest(config.doc_id, manifest_tables)
    write_manifest(output_dir, manifest)

    print("\n" + "=" * 70)
    print(f"Outputs saved to {output_dir}/")
    print("=" * 70)

    typed_edge_claim_count = claim_stats.get("typed_edge_claims", 0)
    accounting_gap = claimable_edges_count - typed_edge_claim_count

    results = {
        "spans": len(spans_df),
        "lines": len(line_df),
        "elements": len(elements_df),
        "anchors": len(anchors_df),
        "sections": len(sections_df),
        "references": len(references_df),
        "authorities": len(authorities_df),
        "authority_links": len(paragraph_authority_df),
        "authority_mentions": len(authority_mentions_df),
        "sentence_records": len(sentence_index_df),
        "typed_edges": len(typed_edges_df),
        "claimable_edges": claimable_edges_count,
        "typed_edge_claims": typed_edge_claim_count,
        "claim_accounting_gap": accounting_gap,
        "claims": claim_stats.get("claims", 0),
        "claim_rejections": claim_stats.get("claim_rejections", 0),
        "claim_authority_mentions": claim_stats.get("claim_authority_mentions", 0),
        "claim_constraints": claim_stats.get("claim_constraints", 0),
        "constraints": claim_stats.get("constraints", 0),
        "constraint_attributes": claim_stats.get("constraint_attributes", 0),
        "resolved_claims": resolution_stats.get("resolved_claims", 0),
        "resolution_groups": resolution_stats.get("resolution_groups", 0),
        "compiled_directives": compiler_stats.get("compiled_directives", 0),
        "constraints_resolved": compiler_stats.get("constraints_resolved", 0),
        "nodes": len(graph_nodes),
        "edges": len(graph_edges),
        "typed_edge_type_counts": Counter(edge.edge_type for edge in typed_semantic_edges),
    }

    print("\n--- Pipeline Summary ---")
    for key, value in results.items():
        print(f"  {key}: {value}")

    typed_edge_type_counts = results["typed_edge_type_counts"]
    if typed_edge_type_counts:
        print(f"  typed_edge_types: {dict(typed_edge_type_counts)}")

    if authority_stats["canonicalized"]:
        print(
            f"  authority_join_rate: {authority_stats['joined']}/{authority_stats['canonicalized']} "
            f"({(authority_stats['joined'] / authority_stats['canonicalized']):.2%})"
        )
    rejection_reasons = claim_stats.get("rejection_reasons", {})
    if rejection_reasons:
        top_reasons = sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_pairs = ", ".join(f"{reason}:{count}" for reason, count in top_reasons)
        print(f"  top_claim_rejections: {top_pairs}")
    claim_predicate_counts = claim_stats.get("claim_predicate_counts", {})
    if claim_predicate_counts:
        print(f"  claim_predicates: {claim_predicate_counts}")

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


# =============================================================================
# HELPERS
# =============================================================================


def annotate_reference_sentences(
    references_df: pd.DataFrame,
    sentence_index_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach sentence metadata to references using the canonical index."""
    if references_df is None or references_df.empty:
        return references_df

    refs = references_df.copy()
    refs["element_char_start"] = refs["position"].fillna(-1).astype(int)
    refs["element_char_end"] = refs.apply(
        lambda row: (
            int(row.get("position", -1)) + len(str(row.get("ref_text", "")))
            if row.get("position") is not None else -1
        ),
        axis=1,
    )
    refs["sentence_idx"] = None
    refs["char_start"] = None
    refs["char_end"] = None
    refs["sentence_char_start_in_element"] = None
    refs["sentence_char_end_in_element"] = None

    if sentence_index_df is None or sentence_index_df.empty:
        return refs

    grouped: Dict[str, List[pd.Series]] = {}
    for _, sent in sentence_index_df.iterrows():
        element_id = str(sent.get("source_element_id"))
        grouped.setdefault(element_id, []).append(sent)

    for idx, row in refs.iterrows():
        element_id = str(row.get("source_element_id"))
        start_elem = int(row.get("element_char_start", -1))
        end_elem = int(row.get("element_char_end", -1))
        entries = grouped.get(element_id)
        if not entries or start_elem < 0:
            continue
        for entry in entries:
            try:
                sent_start = int(entry.get("sentence_char_start"))
                sent_end = int(entry.get("sentence_char_end"))
            except (TypeError, ValueError):
                continue
            if sent_start <= start_elem < sent_end:
                sentence_idx = int(entry.get("sentence_idx"))
                refs.at[idx, "sentence_idx"] = sentence_idx
                span_len = max(0, end_elem - start_elem)
                rel_start = max(0, start_elem - sent_start)
                rel_end = min(rel_start + span_len, sent_end - sent_start)
                refs.at[idx, "char_start"] = rel_start
                refs.at[idx, "char_end"] = rel_end
                refs.at[idx, "sentence_char_start_in_element"] = sent_start
                refs.at[idx, "sentence_char_end_in_element"] = sent_end
                break

    refs["ref_occurrence_id"] = refs.apply(
        lambda row: generate_ref_occurrence_id(
            doc_id=row.get("doc_id"),
            source_element_id=row.get("source_element_id"),
            sentence_idx=row.get("sentence_idx"),
            char_start=row.get("char_start"),
            char_end=row.get("char_end"),
            ref_text=row.get("ref_text"),
            ref_type=row.get("ref_type"),
            target_anchor_id=row.get("target_anchor_id"),
        ),
        axis=1,
    )
    return refs


if __name__ == "__main__":
    main()
