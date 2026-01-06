#!/usr/bin/env python3
"""
Graph Quality Report Generator

Implements deterministic audits (Phase A) and LLM-as-judge structure (Phase B)
for validating the knowledge graph extraction pipeline.

Outputs: Markdown report + JSON findings for CI integration
"""

import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

EXPECTED_BOXES_1099DIV = {
    "1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f",
    "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16"
}

# Thresholds
MAX_NODE_CHARS = 4000  # Monolith warning threshold
MAX_EVIDENCE_CHARS = 200  # Evidence snippet max length
MIN_CONTENT_CHARS = 20  # Minimum content for a valid section
ALLOWED_EDGE_TYPES = {
    "references_box", "same_group", "parent_of", "includes", "same_field",
    "excludes", "applies_if", "defines", "qualifies", "requires"
}
CONFIDENCE_BANDS = {
    "structural": (0.95, 1.0),
    "regex": (0.85, 1.0),
    "llm": (0.5, 1.0),
}

# Artifact patterns
PAGE_MARKER_ISOLATED_RX = re.compile(r'^\s*-\d{1,3}-\s*$', re.MULTILINE)
PAGE_MARKER_TOKEN_RX = re.compile(r'(?<!\d)-(\d{1,2})-(?!\d)')  # -2-, -3- but not -2003-
HEADER_FOOTER_RX = re.compile(
    r'(Instructions for Form \d+|Department of the Treasury|'
    r'Internal Revenue Service|www\.irs\.gov|Cat\.\s*No\.\s*\d+)',
    re.IGNORECASE
)

# =============================================================================
# Data Classes for Findings
# =============================================================================

@dataclass
class Finding:
    check_id: str
    severity: str  # "error", "warning", "info"
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    element_id: Optional[str] = None
    evidence: Optional[str] = None
    recommendation: Optional[str] = None

@dataclass
class CheckResult:
    check_id: str
    check_name: str
    passed: bool
    total_items: int
    passed_items: int
    failed_items: int
    findings: List[Finding] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed_items / self.total_items if self.total_items > 0 else 1.0

@dataclass
class QualityReport:
    timestamp: str
    doc_id: str
    total_nodes: int
    total_edges: int
    checks: List[CheckResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_check(self, result: CheckResult):
        self.checks.append(result)

    def compute_summary(self):
        passed = sum(1 for c in self.checks if c.passed)
        failed = len(self.checks) - passed

        # Count by severity
        errors = sum(1 for c in self.checks for f in c.findings if f.severity == "error")
        warnings = sum(1 for c in self.checks for f in c.findings if f.severity == "warning")

        self.summary = {
            "checks_passed": passed,
            "checks_failed": failed,
            "total_errors": errors,
            "total_warnings": warnings,
            "overall_status": "PASS" if failed == 0 else "FAIL",
        }

# =============================================================================
# Phase A: Deterministic Audits
# =============================================================================

def audit_a1_anchor_coverage(nodes_df: pd.DataFrame, sections_df: pd.DataFrame) -> CheckResult:
    """A1: Anchor coverage + uniqueness"""
    findings = []

    # Check expected boxes present
    box_nodes = nodes_df[nodes_df["node_type"] == "box_section"]
    found_boxes = set(box_nodes["box_key"].dropna().tolist())

    missing = EXPECTED_BOXES_1099DIV - found_boxes
    extras = found_boxes - EXPECTED_BOXES_1099DIV

    for box in missing:
        findings.append(Finding(
            check_id="A1",
            severity="error",
            message=f"Missing expected box: {box}",
            recommendation="Check anchor extraction regex patterns"
        ))

    for box in extras:
        findings.append(Finding(
            check_id="A1",
            severity="warning",
            message=f"Unexpected extra box: {box}",
            recommendation="Verify this is a valid box for this form"
        ))

    # Check each anchor has content
    for _, row in sections_df.iterrows():
        if row["char_count"] < MIN_CONTENT_CHARS and row["anchor_type"] == "box":
            findings.append(Finding(
                check_id="A1",
                severity="warning",
                message=f"Anchor '{row['anchor_id']}' has minimal content ({row['char_count']} chars)",
                node_id=row["anchor_id"],
                recommendation="Check content assignment logic"
            ))

    # Check for duplicate content assignment (same element_id in multiple sections)
    element_assignments = defaultdict(list)
    for _, row in sections_df.iterrows():
        element_ids = row.get("element_ids")
        if element_ids is not None and isinstance(element_ids, (list, tuple)) and len(element_ids) > 0:
            for eid in element_ids:
                element_assignments[eid].append(row["anchor_id"])

    # Note: grouped anchors are expected to share elements
    duplicates_found = 0
    for eid, anchors in element_assignments.items():
        if len(anchors) > 1:
            # Check if they're in the same group (expected)
            unique_anchors = set(anchors)
            if len(unique_anchors) > 1:
                # Check if these are grouped boxes (14, 15, 16)
                grouped_boxes = {"box_14", "box_15", "box_16"}
                if not unique_anchors.issubset(grouped_boxes):
                    duplicates_found += 1
                    if duplicates_found <= 5:  # Limit findings
                        findings.append(Finding(
                            check_id="A1",
                            severity="warning",
                            message=f"Element '{eid}' assigned to multiple anchors: {anchors}",
                            element_id=eid,
                            recommendation="Check anchor timeline overlap"
                        ))

    passed = len(missing) == 0
    return CheckResult(
        check_id="A1",
        check_name="Anchor Coverage + Uniqueness",
        passed=passed,
        total_items=len(EXPECTED_BOXES_1099DIV),
        passed_items=len(EXPECTED_BOXES_1099DIV) - len(missing),
        failed_items=len(missing),
        findings=findings
    )


def audit_a2_artifact_contamination(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> CheckResult:
    """A2: Artifact contamination tests"""
    findings = []
    contaminated_nodes = 0
    contaminated_edges = 0

    for _, row in nodes_df.iterrows():
        text = row.get("text", "") or ""
        node_id = row["node_id"]

        # Check for isolated page markers (the bad kind)
        # Pattern: standalone -N- that's NOT part of a year reference like -2003-
        lines = text.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if PAGE_MARKER_ISOLATED_RX.match(stripped):
                contaminated_nodes += 1
                findings.append(Finding(
                    check_id="A2",
                    severity="error",
                    message=f"Isolated page marker found in node",
                    node_id=node_id,
                    evidence=f"Line {i+1}: '{stripped}'",
                    recommendation="Page markers should be classified as PageArtifact and excluded"
                ))

        # Check for header/footer text that should have been excluded
        if row.get("node_type") == "box_section":
            if HEADER_FOOTER_RX.search(text):
                match = HEADER_FOOTER_RX.search(text)
                # Only flag if it's at the start of a line (likely a header/footer artifact)
                for line in lines:
                    if HEADER_FOOTER_RX.match(line.strip()):
                        contaminated_nodes += 1
                        findings.append(Finding(
                            check_id="A2",
                            severity="warning",
                            message=f"Header/footer text found in box section",
                            node_id=node_id,
                            evidence=f"'{match.group()[:50]}...'",
                            recommendation="Improve PageArtifact classification"
                        ))
                        break

    # Check edge evidence
    for _, row in edges_df.iterrows():
        evidence = row.get("source_evidence", "") or ""
        edge_id = row.get("edge_id", "")

        # Check for newlines in evidence (bad snippet extraction)
        if '\n' in evidence:
            contaminated_edges += 1
            findings.append(Finding(
                check_id="A2",
                severity="warning",
                message=f"Edge evidence contains newline",
                edge_id=edge_id,
                evidence=f"'{evidence[:60]}...'",
                recommendation="Tighten evidence extraction to stop at newlines"
            ))

        # Check for page markers in evidence
        if PAGE_MARKER_TOKEN_RX.search(evidence):
            match = PAGE_MARKER_TOKEN_RX.search(evidence)
            # Exclude legitimate patterns like "two-notices-in-3-years"
            if not re.search(r'in-\d+-', evidence):
                contaminated_edges += 1
                findings.append(Finding(
                    check_id="A2",
                    severity="error",
                    message=f"Page marker in edge evidence",
                    edge_id=edge_id,
                    evidence=f"'{evidence[:60]}...'",
                    recommendation="Exclude PageArtifact elements from reference extraction"
                ))

        # Check evidence length
        if len(evidence) > MAX_EVIDENCE_CHARS:
            findings.append(Finding(
                check_id="A2",
                severity="info",
                message=f"Evidence exceeds max length ({len(evidence)} chars)",
                edge_id=edge_id,
                evidence=f"'{evidence[:50]}...'",
                recommendation="Consider truncating evidence snippets"
            ))

    total_items = len(nodes_df) + len(edges_df)
    failed_items = contaminated_nodes + contaminated_edges

    return CheckResult(
        check_id="A2",
        check_name="Artifact Contamination",
        passed=failed_items == 0,
        total_items=total_items,
        passed_items=total_items - failed_items,
        failed_items=failed_items,
        findings=findings
    )


def audit_a3_preamble_monolith(nodes_df: pd.DataFrame, sections_df: pd.DataFrame) -> CheckResult:
    """A3: Structure completeness test

    Fails ONLY on structural criteria:
    1. Any preamble/section node exceeds 4000 chars

    Note: Internal heading detection requires element-level layout data (split_kind, role).
    Text-based regex detection is unreliable and not used for pass/fail decisions.
    To enable full structure completeness checking, export elements_df during pipeline run.
    """
    findings = []
    monolith_count = 0

    # Types of nodes that should have structure broken out
    structure_nodes = nodes_df[nodes_df["node_type"].isin(["preamble", "section"])]

    for _, row in structure_nodes.iterrows():
        text = row.get("text", "") or ""
        char_count = len(text)
        node_id = row["node_id"]
        node_type = row.get("node_type", "")

        # STRUCTURAL CHECK: Size threshold (deterministic, reliable)
        if char_count > MAX_NODE_CHARS:
            monolith_count += 1
            findings.append(Finding(
                check_id="A3",
                severity="error",
                message=f"Monolith node: {char_count} chars exceeds threshold ({MAX_NODE_CHARS})",
                node_id=node_id,
                evidence=f"Node type: {node_type}, chars: {char_count}",
                recommendation="Split into subsection anchors using layout heuristics in run_pipeline.py"
            ))

    # Also check box sections that might have grown too large
    box_nodes = nodes_df[nodes_df["node_type"] == "box_section"]
    for _, row in box_nodes.iterrows():
        text = row.get("text", "") or ""
        char_count = len(text)
        if char_count > MAX_NODE_CHARS * 1.5:  # Higher threshold for boxes
            findings.append(Finding(
                check_id="A3",
                severity="warning",
                message=f"Box section unusually large: {char_count} chars",
                node_id=row["node_id"],
                recommendation="Check if multiple boxes were merged incorrectly"
            ))

    # Summary info
    if len(structure_nodes) > 0:
        total_chars = sum(len(row.get("text", "") or "") for _, row in structure_nodes.iterrows())
        avg_chars = total_chars / len(structure_nodes)
        findings.append(Finding(
            check_id="A3",
            severity="info",
            message=f"Structure nodes: {len(structure_nodes)}, avg chars: {avg_chars:.0f}",
            recommendation=None
        ))

    return CheckResult(
        check_id="A3",
        check_name="Structure Completeness",
        passed=monolith_count == 0,
        total_items=len(structure_nodes),
        passed_items=len(structure_nodes) - monolith_count,
        failed_items=monolith_count,
        findings=findings
    )


def audit_a4_edge_integrity(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> CheckResult:
    """A4: Edge integrity tests"""
    findings = []
    failed_edges = 0

    node_ids = set(nodes_df["node_id"].tolist())
    node_texts = dict(zip(nodes_df["node_id"], nodes_df["text"].fillna("")))

    for _, row in edges_df.iterrows():
        edge_id = row.get("edge_id", "")
        edge_type = row.get("edge_type", "")
        confidence = row.get("confidence", 0)
        created_by = row.get("created_by", "")
        source_id = row.get("source_node_id", "")
        target_id = row.get("target_node_id", "")
        evidence = row.get("source_evidence", "") or ""

        edge_valid = True

        # Check edge_type is allowed
        if edge_type not in ALLOWED_EDGE_TYPES:
            edge_valid = False
            findings.append(Finding(
                check_id="A4",
                severity="error",
                message=f"Unknown edge type: '{edge_type}'",
                edge_id=edge_id,
                recommendation=f"Add to ALLOWED_EDGE_TYPES or fix edge creation"
            ))

        # Check confidence is within expected band for created_by
        if created_by in CONFIDENCE_BANDS:
            lo, hi = CONFIDENCE_BANDS[created_by]
            if not (lo <= confidence <= hi):
                findings.append(Finding(
                    check_id="A4",
                    severity="warning",
                    message=f"Confidence {confidence} outside expected band [{lo}, {hi}] for {created_by}",
                    edge_id=edge_id,
                    recommendation="Review confidence assignment logic"
                ))

        # Check source and target nodes exist
        if source_id not in node_ids:
            edge_valid = False
            findings.append(Finding(
                check_id="A4",
                severity="error",
                message=f"Source node not found: '{source_id}'",
                edge_id=edge_id
            ))

        if target_id not in node_ids:
            edge_valid = False
            findings.append(Finding(
                check_id="A4",
                severity="error",
                message=f"Target node not found: '{target_id}'",
                edge_id=edge_id
            ))

        # Check evidence is non-empty for reference edges
        if edge_type == "references_box" and not evidence.strip():
            findings.append(Finding(
                check_id="A4",
                severity="warning",
                message=f"Empty evidence for reference edge",
                edge_id=edge_id,
                recommendation="Reference edges should have source_evidence"
            ))

        # Check evidence is findable in source node text (for references)
        if edge_type == "references_box" and evidence.strip() and source_id in node_texts:
            source_text = node_texts[source_id].lower()
            # Extract the core reference from evidence (e.g., "box 1a" from "...in box 1a that...")
            core_ref = re.search(r'box(?:es)?\s+\d+[a-z]?', evidence.lower())
            if core_ref and core_ref.group() not in source_text:
                findings.append(Finding(
                    check_id="A4",
                    severity="warning",
                    message=f"Evidence reference not found in source node text",
                    edge_id=edge_id,
                    evidence=f"Looking for '{core_ref.group()}' in source"
                ))

        if not edge_valid:
            failed_edges += 1

    return CheckResult(
        check_id="A4",
        check_name="Edge Integrity",
        passed=failed_edges == 0,
        total_items=len(edges_df),
        passed_items=len(edges_df) - failed_edges,
        failed_items=failed_edges,
        findings=findings
    )


def audit_a5_provenance_completeness(nodes_df: pd.DataFrame, sections_df: pd.DataFrame) -> CheckResult:
    """A5: Provenance pointer completeness"""
    findings = []
    missing_provenance = 0

    required_fields = ["doc_id", "pages", "anchor_id"]
    recommended_fields = ["element_count", "char_count"]

    def _is_empty(val):
        """Check if a value is empty (handles scalars and arrays)"""
        if val is None:
            return True
        if isinstance(val, (list, tuple)):
            return len(val) == 0
        try:
            return pd.isna(val) or val == ""
        except ValueError:
            # Array comparison - check if any values
            return False

    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]

        # Check required fields
        for field in required_fields:
            if _is_empty(row.get(field)):
                missing_provenance += 1
                findings.append(Finding(
                    check_id="A5",
                    severity="error",
                    message=f"Missing required provenance field: {field}",
                    node_id=node_id,
                    recommendation="Ensure all nodes have complete provenance"
                ))

        # Check recommended fields
        for field in recommended_fields:
            if _is_empty(row.get(field)):
                findings.append(Finding(
                    check_id="A5",
                    severity="info",
                    message=f"Missing recommended field: {field}",
                    node_id=node_id
                ))

    # Check sections have element_ids for pointer resolution
    for _, row in sections_df.iterrows():
        element_ids = row.get("element_ids")
        has_elements = element_ids is not None and isinstance(element_ids, (list, tuple)) and len(element_ids) > 0
        if not has_elements:
            if row.get("anchor_type") == "box":
                findings.append(Finding(
                    check_id="A5",
                    severity="warning",
                    message=f"Section missing element_ids for pointer resolution",
                    node_id=row["anchor_id"],
                    recommendation="Track source elements for highlighting capability"
                ))

    total_checks = len(nodes_df) * len(required_fields)

    return CheckResult(
        check_id="A5",
        check_name="Provenance Completeness",
        passed=missing_provenance == 0,
        total_items=total_checks,
        passed_items=total_checks - missing_provenance,
        failed_items=missing_provenance,
        findings=findings
    )


# =============================================================================
# Phase B: LLM-as-Judge Structure (placeholder for actual LLM calls)
# =============================================================================

def sample_boundary_cases(sections_df: pd.DataFrame, elements_df: pd.DataFrame, k: int = 2) -> List[Dict]:
    """Sample elements near anchor boundaries for B1 judge evaluation"""
    boundary_samples = []

    # This would sample elements at reading_order boundaries
    # For now, return structure for LLM evaluation

    return boundary_samples


def judge_b1_anchor_assignment(samples: List[Dict]) -> CheckResult:
    """B1: Judge anchor assignment on boundary cases (LLM placeholder)"""
    findings = []

    # Placeholder: In production, this would call Claude with structured output
    # For each sample:
    #   - Present element text + nearby anchor headers
    #   - Ask: "Does this element belong to Box X or Box Y?"
    #   - Require evidence pointers with quotes

    findings.append(Finding(
        check_id="B1",
        severity="info",
        message="LLM boundary judgment not yet implemented",
        recommendation="Integrate Claude API for boundary case evaluation"
    ))

    return CheckResult(
        check_id="B1",
        check_name="Anchor Assignment (LLM Judge)",
        passed=True,  # Placeholder pass
        total_items=0,
        passed_items=0,
        failed_items=0,
        findings=findings
    )


def judge_b2_edge_correctness(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> CheckResult:
    """B2: Judge edge correctness for expansion-critical edges (LLM placeholder)"""
    findings = []

    # Filter to expansion-critical edge types
    critical_types = {"includes", "references_box", "references_section", "defines"}
    critical_edges = edges_df[edges_df["edge_type"].isin(critical_types)]

    # Placeholder: Sample and have LLM judge:
    #   - Is the edge supported by the quoted evidence?
    #   - Is the direction correct?
    #   - Is the edge type correct?
    #   - Should confidence be adjusted?

    findings.append(Finding(
        check_id="B2",
        severity="info",
        message=f"LLM edge judgment not yet implemented ({len(critical_edges)} critical edges to sample)",
        recommendation="Integrate Claude API for edge semantic validation"
    ))

    return CheckResult(
        check_id="B2",
        check_name="Edge Correctness (LLM Judge)",
        passed=True,  # Placeholder pass
        total_items=len(critical_edges),
        passed_items=0,
        failed_items=0,
        findings=findings
    )


def judge_b3_pair_suitability() -> CheckResult:
    """B3: Judge pair suitability (LLM placeholder - for after pair generation)"""
    findings = []

    findings.append(Finding(
        check_id="B3",
        severity="info",
        message="Pair suitability judgment requires training pairs (not yet generated)",
        recommendation="Run after pair generation phase"
    ))

    return CheckResult(
        check_id="B3",
        check_name="Pair Suitability (LLM Judge)",
        passed=True,  # N/A until pairs exist
        total_items=0,
        passed_items=0,
        failed_items=0,
        findings=findings
    )


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(report: QualityReport) -> str:
    """Generate markdown report from QualityReport"""

    lines = [
        f"# Graph Quality Report",
        f"",
        f"**Generated:** {report.timestamp}",
        f"**Document:** {report.doc_id}",
        f"**Nodes:** {report.total_nodes} | **Edges:** {report.total_edges}",
        f"",
        f"---",
        f"",
        f"## 1. Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall Status | **{report.summary['overall_status']}** |",
        f"| Checks Passed | {report.summary['checks_passed']} |",
        f"| Checks Failed | {report.summary['checks_failed']} |",
        f"| Total Errors | {report.summary['total_errors']} |",
        f"| Total Warnings | {report.summary['total_warnings']} |",
        f"",
    ]

    # Check summary table
    lines.extend([
        f"### Check Results",
        f"",
        f"| Check | Status | Pass Rate | Items |",
        f"|-------|--------|-----------|-------|",
    ])

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        status_icon = "✅" if check.passed else "❌"
        lines.append(
            f"| {check.check_id}: {check.check_name} | {status_icon} {status} | "
            f"{check.pass_rate:.1%} | {check.passed_items}/{check.total_items} |"
        )

    lines.extend(["", "---", ""])

    # Phase A Findings
    lines.extend([
        f"## 2. Deterministic Findings (Phase A)",
        f"",
    ])

    phase_a_checks = [c for c in report.checks if c.check_id.startswith("A")]
    for check in phase_a_checks:
        error_findings = [f for f in check.findings if f.severity == "error"]
        warning_findings = [f for f in check.findings if f.severity == "warning"]

        lines.extend([
            f"### {check.check_id}: {check.check_name}",
            f"",
            f"**Status:** {'PASS' if check.passed else 'FAIL'} "
            f"({check.passed_items}/{check.total_items})",
            f"",
        ])

        if error_findings:
            lines.append(f"**Errors ({len(error_findings)}):**")
            for f in error_findings[:10]:  # Limit displayed
                lines.append(f"- {f.message}")
                if f.evidence:
                    lines.append(f"  - Evidence: `{f.evidence[:80]}`")
                if f.recommendation:
                    lines.append(f"  - Fix: {f.recommendation}")
            if len(error_findings) > 10:
                lines.append(f"- ... and {len(error_findings) - 10} more errors")
            lines.append("")

        if warning_findings:
            lines.append(f"**Warnings ({len(warning_findings)}):**")
            for f in warning_findings[:5]:  # Limit displayed
                lines.append(f"- {f.message}")
                if f.evidence:
                    lines.append(f"  - Evidence: `{f.evidence[:80]}`")
            if len(warning_findings) > 5:
                lines.append(f"- ... and {len(warning_findings) - 5} more warnings")
            lines.append("")

        if not error_findings and not warning_findings:
            lines.append("No issues found.")
            lines.append("")

    lines.extend(["---", ""])

    # Phase B Findings
    lines.extend([
        f"## 3. LLM Judge Findings (Phase B)",
        f"",
    ])

    phase_b_checks = [c for c in report.checks if c.check_id.startswith("B")]
    for check in phase_b_checks:
        lines.extend([
            f"### {check.check_id}: {check.check_name}",
            f"",
        ])

        for f in check.findings:
            lines.append(f"- {f.message}")
            if f.recommendation:
                lines.append(f"  - {f.recommendation}")
        lines.append("")

    lines.extend(["---", ""])

    # Fix Plan
    lines.extend([
        f"## 4. Recommended Fix Plan (Ranked by ROI)",
        f"",
    ])

    # Aggregate recommendations by frequency/impact
    recommendations = defaultdict(int)
    for check in report.checks:
        for f in check.findings:
            if f.recommendation and f.severity in ["error", "warning"]:
                recommendations[f.recommendation] += 1

    sorted_recs = sorted(recommendations.items(), key=lambda x: -x[1])
    for i, (rec, count) in enumerate(sorted_recs[:10], 1):
        lines.append(f"{i}. **{rec}** ({count} occurrences)")

    if not sorted_recs:
        lines.append("No fixes needed - all checks passed!")

    lines.extend(["", "---", ""])

    # Appendix: Raw findings for CI
    lines.extend([
        f"## Appendix: Machine-Readable Summary",
        f"",
        f"```json",
        json.dumps(report.summary, indent=2),
        f"```",
    ])

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_validation(output_dir: str = "output") -> QualityReport:
    """Run all validation checks and generate report"""

    output_path = Path(output_dir)

    # Load data
    nodes_df = pd.read_parquet(output_path / "graph_nodes.parquet")
    edges_df = pd.read_parquet(output_path / "graph_edges.parquet")
    sections_df = pd.read_parquet(output_path / "sections.parquet")

    # Initialize report
    doc_id = nodes_df["doc_id"].iloc[0] if not nodes_df.empty else "unknown"
    report = QualityReport(
        timestamp=datetime.now().isoformat(),
        doc_id=doc_id,
        total_nodes=len(nodes_df),
        total_edges=len(edges_df),
    )

    print("Running Graph Quality Validation...")
    print("=" * 60)

    # Phase A: Deterministic Audits
    print("\n[Phase A] Deterministic Audits")

    print("  A1: Anchor Coverage...")
    report.add_check(audit_a1_anchor_coverage(nodes_df, sections_df))

    print("  A2: Artifact Contamination...")
    report.add_check(audit_a2_artifact_contamination(nodes_df, edges_df))

    print("  A3: Preamble Monolith Risk...")
    report.add_check(audit_a3_preamble_monolith(nodes_df, sections_df))

    print("  A4: Edge Integrity...")
    report.add_check(audit_a4_edge_integrity(edges_df, nodes_df))

    print("  A5: Provenance Completeness...")
    report.add_check(audit_a5_provenance_completeness(nodes_df, sections_df))

    # Phase B: LLM-as-Judge (placeholders)
    print("\n[Phase B] LLM-as-Judge (Placeholders)")

    print("  B1: Anchor Assignment...")
    report.add_check(judge_b1_anchor_assignment([]))

    print("  B2: Edge Correctness...")
    report.add_check(judge_b2_edge_correctness(edges_df, nodes_df))

    print("  B3: Pair Suitability...")
    report.add_check(judge_b3_pair_suitability())

    # Compute summary
    report.compute_summary()

    # Generate markdown report
    md_report = generate_markdown_report(report)

    # Save report
    report_path = output_path / "graph_quality_report.md"
    with open(report_path, "w") as f:
        f.write(md_report)

    # Save JSON for CI
    json_path = output_path / "graph_quality_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": report.summary,
            "checks": [
                {
                    "check_id": c.check_id,
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "pass_rate": c.pass_rate,
                    "findings_count": len(c.findings),
                }
                for c in report.checks
            ]
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Report saved to: {report_path}")
    print(f"JSON summary: {json_path}")
    print(f"\nOverall Status: {report.summary['overall_status']}")
    print(f"Checks: {report.summary['checks_passed']}/{len(report.checks)} passed")

    return report


if __name__ == "__main__":
    report = run_validation()

    # Print summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    for check in report.checks:
        status = "✅" if check.passed else "❌"
        errors = len([f for f in check.findings if f.severity == "error"])
        warnings = len([f for f in check.findings if f.severity == "warning"])
        print(f"{status} {check.check_id}: {check.check_name} - {errors} errors, {warnings} warnings")
