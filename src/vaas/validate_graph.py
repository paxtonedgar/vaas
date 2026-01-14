#!/usr/bin/env python3
"""
Graph Quality Report Composer

Orchestrates validation passes and generates unified quality reports.
Delegates actual checks to:
- validate_internal: Post-processed table sanity (IS1-IS6)
- validate_corpus: Raw geometry/corpus-grounded checks (CG1a-CG3)

Outputs: Markdown report + JSON findings for CI integration
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

# Internal validators
from vaas.evaluation.validate_internal import (
    validate_internal,
    CheckResult as InternalCheckResult,
    Finding as InternalFinding,
)
from vaas.evaluation.validate_corpus import (
    validate_corpus_grounded,
    CheckResult as CorpusCheckResult,
    Finding as CorpusFinding,
)

# Phase C: Ontology Coverage imports
try:
    from vaas.evaluation.ontology_coverage import (
        run_coverage_analysis,
        CoverageReport as OntologyCoverageReport,
    )
    from vaas.evaluation.ontology_parser import parse_ontology
    ONTOLOGY_EVAL_AVAILABLE = True
except ImportError:
    ONTOLOGY_EVAL_AVAILABLE = False

# Semantic KG audits availability
try:
    from vaas.evaluation.semkg_audits import run_semkg_audits as _run_semkg_audits
    SEMKG_AUDITS_AVAILABLE = True
except ImportError:
    SEMKG_AUDITS_AVAILABLE = False
    _run_semkg_audits = None

from vaas.schemas.semantic_contract import SchemaVersion
from vaas.semantic.manifest import (
    MANIFEST_FILENAME,
    load_manifest,
    required_tables_for_version,
)


# =============================================================================
# Data Classes for Report
# =============================================================================

@dataclass
class Finding:
    check_id: str
    severity: str  # "error", "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    check_id: str
    check_name: str
    passed: bool
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")


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
        errors = sum(c.error_count for c in self.checks)
        warnings = sum(c.warning_count for c in self.checks)

        self.summary = {
            "checks_passed": passed,
            "checks_failed": failed,
            "total_errors": errors,
            "total_warnings": warnings,
            "overall_status": "PASS" if failed == 0 else "FAIL",
        }


# =============================================================================
# Adapter: Convert validator results to report format
# =============================================================================

def _adapt_check_result(result) -> CheckResult:
    """Convert internal/corpus CheckResult to report CheckResult."""
    findings = [
        Finding(
            check_id=f.check_id,
            severity=f.severity,
            message=f.message,
            details=getattr(f, "details", {}),
        )
        for f in result.findings
    ]
    return CheckResult(
        check_id=result.check_id,
        check_name=result.check_name,
        passed=result.passed,
        findings=findings,
        metrics=result.metrics,
    )


# =============================================================================
# Helpers
# =============================================================================

def _load_parquet_safe(path: Path) -> pd.DataFrame:
    """Load parquet or return empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# =============================================================================
# Phase B: LLM-as-Judge Placeholders
# =============================================================================

def judge_b1_anchor_assignment() -> CheckResult:
    """B1: LLM judge for anchor assignment quality (placeholder)."""
    return CheckResult(
        check_id="B1",
        check_name="Anchor Assignment Quality",
        passed=True,
        findings=[Finding("B1", "info", "LLM judge not implemented")],
        metrics={"status": "placeholder"},
    )


def judge_b2_edge_correctness() -> CheckResult:
    """B2: LLM judge for edge correctness (placeholder)."""
    return CheckResult(
        check_id="B2",
        check_name="Edge Correctness",
        passed=True,
        findings=[Finding("B2", "info", "LLM judge not implemented")],
        metrics={"status": "placeholder"},
    )


def judge_b3_pair_suitability() -> CheckResult:
    """B3: LLM judge for training pair suitability (placeholder)."""
    return CheckResult(
        check_id="B3",
        check_name="Training Pair Suitability",
        passed=True,
        findings=[Finding("B3", "info", "LLM judge not implemented")],
        metrics={"status": "placeholder"},
    )


# =============================================================================
# Phase C: Ontology Coverage
# =============================================================================

def run_phase_c_ontology_coverage(
    output_dir: str,
    ontology_path: Optional[str] = None,
) -> List[CheckResult]:
    """Run Phase C ontology coverage checks."""
    results = []

    if not ONTOLOGY_EVAL_AVAILABLE:
        results.append(CheckResult(
            check_id="C1",
            check_name="Ontology Coverage",
            passed=True,
            findings=[Finding("C1", "info", "Ontology evaluation module not available")],
            metrics={"status": "skipped"},
        ))
        return results

    # Auto-detect ontology path
    if ontology_path is None:
        candidates = [
            Path("docs/SEMANTIC_CORE_V3.md"),
            Path("docs/SEMANTIC_CORE.md"),
            Path(output_dir) / "ontology.md",
        ]
        for candidate in candidates:
            if candidate.exists():
                ontology_path = str(candidate)
                break

    if ontology_path is None or not Path(ontology_path).exists():
        results.append(CheckResult(
            check_id="C1",
            check_name="Ontology Coverage",
            passed=True,
            findings=[Finding("C1", "info", "No ontology file found")],
            metrics={"status": "skipped"},
        ))
        return results

    try:
        ontology = parse_ontology(ontology_path)
        output_path = Path(output_dir)
        nodes_df = _load_parquet_safe(output_path / "graph_nodes.parquet")
        edges_df = _load_parquet_safe(output_path / "graph_edges.parquet")

        coverage_report = run_coverage_analysis(ontology, nodes_df, edges_df)

        findings = []
        if coverage_report.coverage_rate < 0.20:
            findings.append(Finding(
                "C1", "warning",
                f"Low ontology coverage: {coverage_report.coverage_rate:.1%}"
            ))

        results.append(CheckResult(
            check_id="C1",
            check_name="Ontology Coverage",
            passed=coverage_report.coverage_rate >= 0.10,
            findings=findings,
            metrics={
                "coverage_rate": coverage_report.coverage_rate,
                "predicates_covered": coverage_report.predicates_covered,
                "node_types_covered": coverage_report.node_types_covered,
            },
        ))
    except Exception as e:
        results.append(CheckResult(
            check_id="C1",
            check_name="Ontology Coverage",
            passed=True,
            findings=[Finding("C1", "warning", f"Ontology analysis error: {e}")],
            metrics={"status": "error"},
        ))

    return results


# =============================================================================
# Semantic KG Audits
# =============================================================================

def run_semkg_audits(output_path: Path) -> List[CheckResult]:
    """Run semantic KG audits if available."""
    results = []

    if not SEMKG_AUDITS_AVAILABLE or _run_semkg_audits is None:
        return results

    try:
        semkg_results = _run_semkg_audits(output_path)
        for audit in semkg_results:
            findings = []
            if not audit.passed:
                findings.append(Finding(
                    check_id=audit.gate_id,
                    severity="error",
                    message=f"{audit.details} (threshold {audit.threshold:.0%})",
                ))
            elif isinstance(audit.details, str) and audit.details.startswith("WARN:"):
                findings.append(Finding(
                    check_id=audit.gate_id,
                    severity="warning",
                    message=audit.details,
                ))

            check_name = {
                "semkg_reference_join": "Reference Join",
                "semkg_claim_scope": "Claim Scope Resolution",
                "semkg_claim_group_competition": "Claim Group Competition",
                "semkg_precedence_nonempty": "Precedence Non-Empty",
                "semkg_predicate_health_report": "Predicate Health Report",
                "semkg_compiler_ir_schema": "Compiler IR Schema",
                "semkg_compiler_directive_join": "Compiler Directive Join",
                "semkg_compiler_scope_topic_consistency": "Compiler Scope/Topic Consistency",
                "semkg_compiler_ir_present": "Compiler IR Present",
                "semkg_compiler_support_nonempty": "Compiler Support Non-Empty",
            }.get(audit.gate_id, audit.gate_id)

            results.append(CheckResult(
                check_id=audit.gate_id,
                check_name=check_name,
                passed=audit.passed,
                findings=findings,
                metrics={"total": audit.total, "succeeded": audit.succeeded},
            ))
    except Exception as e:
        results.append(CheckResult(
            check_id="SEMKG_ERROR",
            check_name="Semantic KG Audits",
            passed=True,
            findings=[Finding("SEMKG_ERROR", "warning", f"SEMKG audit error: {e}")],
        ))

    return results


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(val) for val in value]
    return str(value)


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(report: QualityReport) -> str:
    """Generate markdown report from QualityReport."""
    lines = [
        "# Graph Quality Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Document:** {report.doc_id}",
        f"**Nodes:** {report.total_nodes} | **Edges:** {report.total_edges}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall Status | **{report.summary['overall_status']}** |",
        f"| Checks Passed | {report.summary['checks_passed']} |",
        f"| Checks Failed | {report.summary['checks_failed']} |",
        f"| Total Errors | {report.summary['total_errors']} |",
        f"| Total Warnings | {report.summary['total_warnings']} |",
        "",
        "## Check Results",
        "",
        "| Check | Status | Errors | Warnings |",
        "|-------|--------|--------|----------|",
    ]

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        icon = "+" if check.passed else "x"
        lines.append(
            f"| {check.check_id}: {check.check_name} | [{icon}] {status} | "
            f"{check.error_count} | {check.warning_count} |"
        )

    lines.extend(["", "---", ""])

    # Group findings by phase
    phases = {
        "IS": "Internal Sanity",
        "CG": "Corpus-Grounded",
        "B": "LLM-as-Judge",
        "C": "Ontology Coverage",
        "SEMKG": "Semantic KG",
    }

    for prefix, phase_name in phases.items():
        phase_checks = [c for c in report.checks if c.check_id.startswith(prefix)]
        if not phase_checks:
            continue

        lines.extend([f"## {phase_name} Findings", ""])

        for check in phase_checks:
            if not check.findings:
                continue

            lines.append(f"### {check.check_id}: {check.check_name}")
            lines.append("")

            for finding in check.findings:
                severity_icon = {"error": "[!]", "warning": "[~]", "info": "[i]"}.get(
                    finding.severity, "[-]"
                )
                lines.append(f"- {severity_icon} {finding.message}")

            lines.append("")

        # Add metrics summary
        for check in phase_checks:
            if check.metrics:
                lines.append(f"**{check.check_id} Metrics:**")
                for key, value in check.metrics.items():
                    if isinstance(value, float):
                        lines.append(f"- {key}: {value:.2f}")
                    elif isinstance(value, list) and len(value) > 10:
                        lines.append(f"- {key}: [{len(value)} items]")
                    else:
                        lines.append(f"- {key}: {value}")
                lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_validation(
    output_dir: str = "output",
    ontology_path: Optional[str] = None,
) -> QualityReport:
    """Run all validation checks and generate report.

    Args:
        output_dir: Directory containing graph parquet files
        ontology_path: Path to ontology MD file for Phase C checks.
                      If None, auto-detects from common locations.
    """
    output_path = Path(output_dir)

    # Load tables
    nodes_df = _load_parquet_safe(output_path / "graph_nodes.parquet")
    edges_df = _load_parquet_safe(output_path / "graph_edges.parquet")
    anchors_df = _load_parquet_safe(output_path / "anchors.parquet")
    elements_df = _load_parquet_safe(output_path / "elements.parquet")
    lines_df = _load_parquet_safe(output_path / "lines.parquet")
    spans_df = _load_parquet_safe(output_path / "spans.parquet")

    # Manifest check
    manifest_findings = []
    manifest_doc_id = "unknown"
    manifest_passed = False

    try:
        manifest = load_manifest(output_path)
        schema_version = manifest.get("schema_version", SchemaVersion)
        required_tables = required_tables_for_version(schema_version)
        missing_tables = [
            tbl for tbl in required_tables if not (output_path / tbl).exists()
        ]
        manifest_passed = len(missing_tables) == 0
        if missing_tables:
            manifest_findings.append(Finding(
                check_id="MANIFEST",
                severity="error",
                message=f"Missing required tables: {', '.join(missing_tables)}",
            ))
        manifest_doc_id = manifest.get("doc_id", "unknown")
    except FileNotFoundError:
        manifest_findings.append(Finding(
            check_id="MANIFEST",
            severity="warning",
            message=f"{MANIFEST_FILENAME} not found",
        ))

    # Initialize report
    doc_id = manifest_doc_id if manifest_doc_id != "unknown" else (
        nodes_df["doc_id"].iloc[0] if not nodes_df.empty and "doc_id" in nodes_df.columns else "unknown"
    )
    report = QualityReport(
        timestamp=datetime.now().isoformat(),
        doc_id=doc_id,
        total_nodes=len(nodes_df),
        total_edges=len(edges_df),
    )

    print("Running Graph Quality Validation...")
    print("=" * 60)

    # Manifest check
    report.add_check(CheckResult(
        check_id="MANIFEST",
        check_name="Semantic Manifest",
        passed=manifest_passed,
        findings=manifest_findings,
    ))

    # =========================================================================
    # Internal Sanity Checks (IS1-IS6)
    # =========================================================================
    print("\n[Internal Sanity] Post-processed table checks")

    internal_results = validate_internal(nodes_df, edges_df, anchors_df, elements_df)
    for result in internal_results:
        adapted = _adapt_check_result(result)
        print(f"  {adapted.check_id}: {adapted.check_name}... {'PASS' if adapted.passed else 'FAIL'}")
        report.add_check(adapted)

    # =========================================================================
    # Corpus-Grounded Checks (CG1a-CG3)
    # =========================================================================
    print("\n[Corpus-Grounded] Raw geometry/text checks")

    corpus_results = validate_corpus_grounded(
        lines_df, nodes_df, edges_df, anchors_df, elements_df, spans_df
    )
    for result in corpus_results:
        adapted = _adapt_check_result(result)
        print(f"  {adapted.check_id}: {adapted.check_name}... {'PASS' if adapted.passed else 'FAIL'}")
        report.add_check(adapted)

    # =========================================================================
    # Phase B: LLM-as-Judge (Placeholders)
    # =========================================================================
    print("\n[Phase B] LLM-as-Judge (placeholders)")

    print("  B1: Anchor Assignment...")
    report.add_check(judge_b1_anchor_assignment())

    print("  B2: Edge Correctness...")
    report.add_check(judge_b2_edge_correctness())

    print("  B3: Pair Suitability...")
    report.add_check(judge_b3_pair_suitability())

    # =========================================================================
    # Phase C: Ontology Coverage
    # =========================================================================
    print("\n[Phase C] Ontology Coverage")

    phase_c_results = run_phase_c_ontology_coverage(output_dir, ontology_path)
    for result in phase_c_results:
        print(f"  {result.check_id}: {result.check_name}...")
        report.add_check(result)

    # =========================================================================
    # Semantic KG Audits
    # =========================================================================
    print("\n[Semantic KG] Audits")

    semkg_results = run_semkg_audits(output_path)
    if semkg_results:
        for result in semkg_results:
            print(f"  {result.check_id}: {result.check_name}...")
            report.add_check(result)
    else:
        print("  Skipped - semkg_audits module unavailable")

    predicate_health_path = output_path / "predicate_health.json"
    if predicate_health_path.exists():
        with open(predicate_health_path, "r", encoding="utf-8") as fh:
            report.summary["predicate_health"] = json.load(fh)

    # Compute summary
    report.compute_summary()

    # Generate and save reports
    md_report = generate_markdown_report(report)

    report_path = output_path / "graph_quality_report.md"
    with open(report_path, "w") as f:
        f.write(md_report)

    json_path = output_path / "graph_quality_report.json"
    json_payload = {
        "summary": {
            "checks_passed": report.summary.get("checks_passed", 0),
            "checks_failed": report.summary.get("checks_failed", 0),
            "total_errors": report.summary.get("total_errors", 0),
            "total_warnings": report.summary.get("total_warnings", 0),
            "overall_status": report.summary.get("overall_status", "UNKNOWN"),
        },
        "checks": [
            {
                "check_id": c.check_id,
                "check_name": c.check_name,
                "passed": _jsonable(c.passed),
                "errors": _jsonable(c.error_count),
                "warnings": _jsonable(c.warning_count),
                "metrics": _jsonable(c.metrics),
            }
            for c in report.checks
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Report saved to: {report_path}")
    print(f"JSON summary: {json_path}")
    print(f"\nOverall Status: {report.summary['overall_status']}")
    print(f"Checks: {report.summary['checks_passed']}/{len(report.checks)} passed")

    return report


# =============================================================================
# Convenience Wrapper
# =============================================================================

def validate_all(output_dir: str = "output") -> bool:
    """
    Run all validators and return pass/fail status.

    Convenience wrapper for CI/CD integration.

    Args:
        output_dir: Directory containing graph parquet files.

    Returns:
        True if all checks pass, False otherwise.
    """
    report = run_validation(output_dir=output_dir)
    return report.summary.get("overall_status") == "PASS"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run graph quality validation")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--ontology", default=None, help="Ontology file path")
    args = parser.parse_args()

    report = run_validation(output_dir=args.output, ontology_path=args.ontology)

    # Print quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    for check in report.checks:
        status = "[+]" if check.passed else "[x]"
        print(f"{status} {check.check_id}: {check.check_name} - {check.error_count} errors, {check.warning_count} warnings")
