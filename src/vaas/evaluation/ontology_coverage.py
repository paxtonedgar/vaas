"""
Ontology Coverage Evaluator

Compares the parsed ontology (semantic target) against the current graph
to identify gaps at both semantic and code levels.

This produces actionable diagnostics:
- Which ontology elements are NOT represented in the graph
- Which predicates have no corresponding edge types
- Which node types are missing (Rules, Windows, Thresholds)
- Code-level recommendations for closing gaps
"""

import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from .ontology_parser import (
    ParsedOntology,
    OntologyElement,
    OntologyElementType,
    Relationship,
    parse_ontology,
    extract_predicate_vocabulary,
)


@dataclass
class PredicateMapping:
    """Maps an ontology predicate to current extraction capability."""
    ontology_predicate: str
    category: str                           # temporal/regulatory/conditional/etc
    current_edge_type: Optional[str]        # Our edge type if we have one
    coverage_status: str                    # "full", "partial", "none"
    gap_reason: Optional[str] = None        # Why we can't capture it
    code_location: Optional[str] = None     # Where to add support
    example_relationship: Optional[str] = None


@dataclass
class NodeTypeGap:
    """Identifies missing node types from the ontology."""
    ontology_type: OntologyElementType
    count_in_ontology: int
    count_in_graph: int
    examples: List[str]
    gap_reason: str
    code_location: str


@dataclass
class CoverageReport:
    """Complete coverage analysis."""
    ontology_summary: Dict
    graph_summary: Dict

    # Predicate coverage
    predicate_mappings: List[PredicateMapping]
    predicates_covered: int
    predicates_partial: int
    predicates_missing: int

    # Node type coverage
    node_type_gaps: List[NodeTypeGap]

    # Relationship coverage
    relationships_total: int
    relationships_matchable: int  # Could be matched if we had the predicates

    # Actionable gaps (prioritized)
    priority_gaps: List[Dict]

    def coverage_percentage(self) -> float:
        total = self.predicates_covered + self.predicates_partial + self.predicates_missing
        if total == 0:
            return 0.0
        return (self.predicates_covered + 0.5 * self.predicates_partial) / total * 100


# =============================================================================
# PREDICATE MAPPING LOGIC
# =============================================================================

# Map ontology predicates to our current edge types
PREDICATE_TO_EDGE_TYPE = {
    # Direct mappings (we have these)
    "includes": "includes",
    "excludes": "excludes",
    "portion_of": "portion_of",
    "requires": "requires",
    "applies_if": "applies_if",
    "defines": "defines",
    "qualifies": "qualifies",

    # Partial mappings (captured differently)
    "reported_in": "references_box",  # We capture box refs, not reporting semantics
    "reported_on": "references_box",
    "governed_by": None,              # We don't capture regulatory refs

    # Structural (we have these implicitly)
    "is_a": "parent_of",              # Hierarchy
    "contains": "parent_of",
}

# Predicates we cannot capture with current extraction
MISSING_PREDICATE_REASONS = {
    # Temporal predicates - need Window node type
    "length_days": ("No Window node type", "src/vaas/graph/nodes.py"),
    "starts_offset_from": ("No Window node type", "src/vaas/graph/nodes.py"),
    "start_offset_days": ("No Window node type", "src/vaas/graph/nodes.py"),
    "min_days_held": ("No Threshold node type", "src/vaas/graph/nodes.py"),
    "applies_within": ("No Window reference edges", "src/vaas/semantic/typed_edges.py"),

    # Rule semantics - need Rule node type
    "when": ("No Rule node type with conditions", "src/vaas/graph/nodes.py"),
    "effect": ("No Rule node type with effects", "src/vaas/graph/nodes.py"),
    "trigger_condition": ("No Rule node type", "src/vaas/graph/nodes.py"),

    # Regulatory references - need IRC/Pub extraction
    "governed_by": ("No IRC/Pub reference extraction", "src/vaas/extraction/references.py"),
    "defined_by": ("No IRC/Pub reference extraction", "src/vaas/extraction/references.py"),
    "subject_to": ("No IRC section edges", "src/vaas/semantic/typed_edges.py"),

    # Entity classification - need entity type nodes
    "classified_as": ("No entity classification edges", "src/vaas/semantic/typed_edges.py"),
    "is_a": ("Captured as parent_of but loses semantics", "src/vaas/graph/edges.py"),
    "may_be_treated_as": ("No conditional classification", "src/vaas/semantic/typed_edges.py"),

    # Reporting semantics - partially captured
    "required_when": ("No conditional requirement edges", "src/vaas/semantic/typed_edges.py"),
    "must_furnish": ("No statement requirement edges", "src/vaas/semantic/typed_edges.py"),
    "must_include_field": ("No field requirement edges", "src/vaas/semantic/typed_edges.py"),
}


def map_predicate(
    predicate: str,
    category: str,
    claims_df: Optional[pd.DataFrame] = None,
    example_rel: Optional[Relationship] = None,
) -> PredicateMapping:
    """Map a single ontology predicate to our current capability."""

    if claims_df is not None and not claims_df.empty:
        claim_count = (claims_df["predicate"] == predicate).sum()
        if claim_count > 0:
            return PredicateMapping(
                ontology_predicate=predicate,
                category=category,
                current_edge_type=predicate,
                coverage_status="full",
                example_relationship=f"{example_rel.source} —({predicate})—> {example_rel.target}" if example_rel else None,
            )

    # Check direct mapping
    if predicate in PREDICATE_TO_EDGE_TYPE:
        edge_type = PREDICATE_TO_EDGE_TYPE[predicate]
        if edge_type:
            return PredicateMapping(
                ontology_predicate=predicate,
                category=category,
                current_edge_type=edge_type,
                coverage_status="full" if predicate == edge_type else "partial",
                example_relationship=f"{example_rel.source} —({predicate})—> {example_rel.target}" if example_rel else None,
            )

    # Check if we know why it's missing
    if predicate in MISSING_PREDICATE_REASONS:
        reason, location = MISSING_PREDICATE_REASONS[predicate]
        return PredicateMapping(
            ontology_predicate=predicate,
            category=category,
            current_edge_type=None,
            coverage_status="none",
            gap_reason=reason,
            code_location=location,
            example_relationship=f"{example_rel.source} —({predicate})—> {example_rel.target}" if example_rel else None,
        )

    # Unknown predicate - needs investigation
    return PredicateMapping(
        ontology_predicate=predicate,
        category=category,
        current_edge_type=None,
        coverage_status="none",
        gap_reason="Not analyzed - needs pattern in typed_edges.py",
        code_location="src/vaas/semantic/typed_edges.py",
        example_relationship=f"{example_rel.source} —({predicate})—> {example_rel.target}" if example_rel else None,
    )


def analyze_node_type_gaps(
    ontology: ParsedOntology,
    nodes_df: pd.DataFrame
) -> List[NodeTypeGap]:
    """Identify missing node types."""
    gaps = []

    # Current node types
    current_types = set(nodes_df["node_type"].unique())

    # Check Rules
    if ontology.rules:
        rule_count = len(ontology.rules)
        graph_rules = 0  # We don't have rule nodes
        if "rule" not in current_types:
            gaps.append(NodeTypeGap(
                ontology_type=OntologyElementType.RULE,
                count_in_ontology=rule_count,
                count_in_graph=graph_rules,
                examples=[r.name for r in ontology.rules[:5]],
                gap_reason="No Rule node type - rules embedded in text, not extracted as nodes",
                code_location="src/vaas/graph/nodes.py + run_pipeline.py (add rule extraction stage)",
            ))

    # Check Windows
    if ontology.windows:
        window_count = len(ontology.windows)
        graph_windows = 0  # We don't have window nodes
        if "window" not in current_types:
            gaps.append(NodeTypeGap(
                ontology_type=OntologyElementType.WINDOW,
                count_in_ontology=window_count,
                count_in_graph=graph_windows,
                examples=[w.name for w in ontology.windows[:5]],
                gap_reason="No Window node type - temporal constructs not extracted",
                code_location="src/vaas/graph/nodes.py + new temporal_extraction.py",
            ))

    # Check Thresholds
    if ontology.thresholds:
        gaps.append(NodeTypeGap(
            ontology_type=OntologyElementType.THRESHOLD,
            count_in_ontology=len(ontology.thresholds),
            count_in_graph=0,
            examples=[t.name for t in ontology.thresholds[:5]],
            gap_reason="No Threshold node type",
            code_location="src/vaas/graph/nodes.py",
        ))

    return gaps


def prioritize_gaps(
    predicate_mappings: List[PredicateMapping],
    node_type_gaps: List[NodeTypeGap],
    ontology: ParsedOntology,
) -> List[Dict]:
    """Prioritize gaps by impact (relationship count)."""
    priorities = []

    # Count relationships per predicate
    pred_counts = defaultdict(int)
    for rel in ontology.all_relationships:
        pred_counts[rel.predicate] += 1

    # Add predicate gaps with counts
    for pm in predicate_mappings:
        if pm.coverage_status == "none":
            count = pred_counts.get(pm.ontology_predicate, 0)
            priorities.append({
                "type": "predicate",
                "name": pm.ontology_predicate,
                "impact": count,
                "category": pm.category,
                "reason": pm.gap_reason,
                "code_location": pm.code_location,
                "example": pm.example_relationship,
            })

    # Add node type gaps
    for ntg in node_type_gaps:
        priorities.append({
            "type": "node_type",
            "name": ntg.ontology_type.value,
            "impact": ntg.count_in_ontology * 5,  # Weight node types higher
            "category": "structural",
            "reason": ntg.gap_reason,
            "code_location": ntg.code_location,
            "example": ", ".join(ntg.examples[:3]),
        })

    # Sort by impact
    priorities.sort(key=lambda x: -x["impact"])

    return priorities


def run_coverage_analysis(
    ontology_path: str,
    graph_dir: str = "output",
) -> CoverageReport:
    """
    Run full coverage analysis comparing ontology to current graph.

    Args:
        ontology_path: Path to the ontology markdown file
        graph_dir: Directory containing graph parquet files

    Returns:
        CoverageReport with complete gap analysis
    """
    # Parse ontology
    ontology = parse_ontology(ontology_path)
    vocab = extract_predicate_vocabulary(ontology)

    # Load graph
    graph_path = Path(graph_dir)
    nodes_df = pd.read_parquet(graph_path / "graph_nodes.parquet")
    edges_df = pd.read_parquet(graph_path / "graph_edges.parquet")
    claims_path = graph_path / "claims.parquet"
    claims_df = pd.read_parquet(claims_path) if claims_path.exists() else pd.DataFrame()

    # Graph summary
    graph_summary = {
        "total_nodes": len(nodes_df),
        "node_types": nodes_df["node_type"].value_counts().to_dict(),
        "total_edges": len(edges_df),
        "edge_types": edges_df["edge_type"].value_counts().to_dict(),
    }

    # Map all predicates
    predicate_mappings = []
    pred_to_example = {}
    for rel in ontology.all_relationships:
        if rel.predicate not in pred_to_example:
            pred_to_example[rel.predicate] = rel

    for category, predicates in vocab.items():
        for pred in predicates:
            example = pred_to_example.get(pred)
            mapping = map_predicate(pred, category, claims_df, example)
            predicate_mappings.append(mapping)

    # Count coverage levels
    predicates_covered = sum(1 for pm in predicate_mappings if pm.coverage_status == "full")
    predicates_partial = sum(1 for pm in predicate_mappings if pm.coverage_status == "partial")
    predicates_missing = sum(1 for pm in predicate_mappings if pm.coverage_status == "none")

    # Analyze node type gaps
    node_type_gaps = analyze_node_type_gaps(ontology, nodes_df)

    # Prioritize gaps
    priority_gaps = prioritize_gaps(predicate_mappings, node_type_gaps, ontology)

    return CoverageReport(
        ontology_summary=ontology.summary(),
        graph_summary=graph_summary,
        predicate_mappings=predicate_mappings,
        predicates_covered=predicates_covered,
        predicates_partial=predicates_partial,
        predicates_missing=predicates_missing,
        node_type_gaps=node_type_gaps,
        relationships_total=len(ontology.all_relationships),
        relationships_matchable=sum(
            1 for r in ontology.all_relationships
            if r.predicate in PREDICATE_TO_EDGE_TYPE and PREDICATE_TO_EDGE_TYPE[r.predicate]
        ),
        priority_gaps=priority_gaps,
    )


def generate_coverage_report_md(report: CoverageReport) -> str:
    """Generate markdown report from coverage analysis."""
    lines = [
        "# Ontology Coverage Report",
        "",
        f"**Coverage Score:** {report.coverage_percentage():.1f}%",
        "",
        "---",
        "",
        "## 1. Summary",
        "",
        "### Ontology (Target)",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    for k, v in report.ontology_summary.items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "### Current Graph",
        "| Metric | Value |",
        "|--------|-------|",
    ])

    lines.append(f"| total_nodes | {report.graph_summary['total_nodes']} |")
    lines.append(f"| total_edges | {report.graph_summary['total_edges']} |")

    for nt, count in report.graph_summary.get("node_types", {}).items():
        lines.append(f"| node_type: {nt} | {count} |")

    for et, count in report.graph_summary.get("edge_types", {}).items():
        lines.append(f"| edge_type: {et} | {count} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Predicate Coverage",
        "",
        f"- **Fully covered:** {report.predicates_covered}",
        f"- **Partially covered:** {report.predicates_partial}",
        f"- **Missing:** {report.predicates_missing}",
        "",
        "### Missing Predicates (by category)",
        "",
    ])

    # Group missing by category
    missing_by_cat = defaultdict(list)
    for pm in report.predicate_mappings:
        if pm.coverage_status == "none":
            missing_by_cat[pm.category].append(pm)

    for cat, mappings in sorted(missing_by_cat.items()):
        lines.append(f"#### {cat.upper()}")
        lines.append("")
        lines.append("| Predicate | Reason | Code Location |")
        lines.append("|-----------|--------|---------------|")
        for pm in mappings:
            lines.append(f"| `{pm.ontology_predicate}` | {pm.gap_reason or 'Unknown'} | `{pm.code_location or 'TBD'}` |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Node Type Gaps",
        "",
    ])

    if report.node_type_gaps:
        lines.append("| Type | In Ontology | In Graph | Gap Reason |")
        lines.append("|------|-------------|----------|------------|")
        for ntg in report.node_type_gaps:
            lines.append(
                f"| {ntg.ontology_type.value} | {ntg.count_in_ontology} | {ntg.count_in_graph} | {ntg.gap_reason} |"
            )
            lines.append(f"|  | Examples: {', '.join(ntg.examples[:3])} | | Code: `{ntg.code_location}` |")
    else:
        lines.append("All node types covered!")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Priority Gaps (by Impact)",
        "",
        "These gaps, if closed, would have the highest impact on ontology coverage:",
        "",
    ])

    for i, gap in enumerate(report.priority_gaps[:15], 1):
        lines.append(f"### {i}. {gap['name']} ({gap['type']})")
        lines.append(f"- **Impact:** {gap['impact']} relationships affected")
        lines.append(f"- **Category:** {gap['category']}")
        lines.append(f"- **Reason:** {gap['reason']}")
        lines.append(f"- **Code:** `{gap['code_location']}`")
        if gap.get('example'):
            lines.append(f"- **Example:** `{gap['example']}`")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Recommendations",
        "",
        "### Immediate (High ROI)",
        "1. Add `governed_by` edge type for IRC/Pub references",
        "2. Add `when`/`effect` edges for rule extraction",
        "3. Introduce `Rule` node type for conditional logic",
        "",
        "### Medium Term",
        "1. Add `Window` node type for temporal constructs",
        "2. Extract holding period parameters (length_days, min_days_held)",
        "3. Add entity classification edges (classified_as, is_a)",
        "",
        "### Long Term",
        "1. Full rule extraction with condition/effect decomposition",
        "2. Cross-form entity resolution",
        "3. Temporal reasoning support",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    ontology_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/PaxtonEdgar/Downloads/Form_1099DIV_Ontology.md"
    graph_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    print(f"Analyzing coverage: {ontology_path} vs {graph_dir}")

    report = run_coverage_analysis(ontology_path, graph_dir)

    # Print summary
    print(f"\n=== Coverage Summary ===")
    print(f"Coverage Score: {report.coverage_percentage():.1f}%")
    print(f"Predicates: {report.predicates_covered} covered, {report.predicates_partial} partial, {report.predicates_missing} missing")
    print(f"Relationships: {report.relationships_matchable}/{report.relationships_total} matchable")

    print(f"\n=== Top 10 Priority Gaps ===")
    for i, gap in enumerate(report.priority_gaps[:10], 1):
        print(f"{i}. [{gap['type']}] {gap['name']} (impact: {gap['impact']})")
        print(f"   Reason: {gap['reason']}")
        print(f"   Code: {gap['code_location']}")

    # Generate markdown report
    md_report = generate_coverage_report_md(report)
    output_path = Path(graph_dir) / "ontology_coverage_report.md"
    output_path.write_text(md_report)
    print(f"\nFull report saved to: {output_path}")
