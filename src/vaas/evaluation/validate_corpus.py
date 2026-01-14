"""
Corpus-Grounded Validator - Raw geometry/text checks.

Uses raw line/span data, NOT post-processed node text.
Does NOT reuse pipeline regex/constants.

Checks:
- CG1a: Open-world anchor discovery (generic patterns + geometry)
- CG1b: Registry alignment (discovered vs expected, as alignment metric)
- CG2: Artifact contamination (geometry-based detection)
- CG3: Structure completeness (unanchored heading detection)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


# =============================================================================
# CONFIG / VARIANCE (soft expectations, not assertions)
# =============================================================================

VARIANCE = {
    "box_count": (18, 30),          # min, max acceptable discovered boxes
    "heading_coverage": 0.80,       # min ratio of heading candidates anchored
}

# Registry is used ONLY for alignment reporting, not discovery
EXPECTED_BOXES_1099DIV = {
    "1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f",
    "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Finding:
    check_id: str
    severity: str  # error | warning | info
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    check_id: str
    check_name: str
    passed: bool
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CG1a: OPEN-WORLD ANCHOR DISCOVERY
# =============================================================================

_BOX_SINGLE = re.compile(r'^Box(?:es)?\s+(\d+[a-z]?)\b', re.IGNORECASE)
_BOX_RANGE = re.compile(r'^Box(?:es)?\s+(\d+)\s*[-–]\s*(\d+)', re.IGNORECASE)
_BOX_THROUGH = re.compile(r'^Box(?:es)?\s+(\d+)\s+through\s+(\d+)', re.IGNORECASE)


def _extract_box_keys(text: str) -> Set[str]:
    """Extract box keys using generic, open-world patterns."""
    text = (text or "").strip()
    keys: Set[str] = set()

    m = _BOX_RANGE.match(text)
    if m:
        try:
            lo, hi = int(m.group(1)), int(m.group(2))
            keys.update(str(k) for k in range(min(lo, hi), max(lo, hi) + 1))
            return keys
        except ValueError:
            pass

    m = _BOX_THROUGH.match(text)
    if m:
        try:
            lo, hi = int(m.group(1)), int(m.group(2))
            keys.update(str(k) for k in range(min(lo, hi), max(lo, hi) + 1))
            return keys
        except ValueError:
            pass

    m = _BOX_SINGLE.match(text)
    if m:
        keys.add(m.group(1).lower())

    return keys


def check_cg1a_anchor_discovery(lines_df: pd.DataFrame) -> CheckResult:
    """CG1a: Discover anchors without using expected registry."""
    findings = []
    metrics = {}

    if lines_df.empty:
        return CheckResult("CG1a", "Anchor Discovery", True, metrics={"discovered_box_keys": []})

    text_col = next((c for c in ["line_text", "text"] if c in lines_df.columns), None)
    if not text_col:
        findings.append(Finding("CG1a", "warning", "No text column available"))
        return CheckResult("CG1a", "Anchor Discovery", True, findings, metrics)

    discovered: Set[str] = set()
    for txt in lines_df[text_col].fillna(""):
        discovered.update(_extract_box_keys(str(txt)))

    metrics["lines_examined"] = len(lines_df)
    metrics["discovered_box_keys"] = sorted(discovered)
    metrics["discovered_count"] = len(discovered)

    lo, hi = VARIANCE["box_count"]
    if len(discovered) < lo:
        findings.append(Finding("CG1a", "warning", f"Only {len(discovered)} boxes discovered"))
    elif len(discovered) > hi:
        findings.append(Finding("CG1a", "info", f"{len(discovered)} boxes discovered (above typical)"))

    return CheckResult(
        "CG1a",
        "Anchor Discovery",
        passed=len(discovered) >= lo,
        findings=findings,
        metrics=metrics,
    )


# =============================================================================
# CG1b: REGISTRY ALIGNMENT (DIAGNOSTIC ONLY)
# =============================================================================

def check_cg1b_registry_alignment(
    discovered: Set[str],
    expected: Set[str] = EXPECTED_BOXES_1099DIV,
) -> CheckResult:
    """CG1b: Alignment reporting against registry (non-fatal)."""
    missing = expected - discovered
    extra = discovered - expected
    matched = expected & discovered

    metrics = {
        "expected": len(expected),
        "discovered": len(discovered),
        "matched": len(matched),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "alignment_rate": len(matched) / len(expected) if expected else 1.0,
    }

    findings = []
    if missing:
        findings.append(Finding("CG1b", "warning", f"{len(missing)} expected boxes missing"))
    if extra:
        findings.append(Finding("CG1b", "info", f"{len(extra)} extra boxes discovered"))

    return CheckResult(
        "CG1b",
        "Registry Alignment",
        passed=metrics["alignment_rate"] >= 0.80,
        findings=findings,
        metrics=metrics,
    )


# =============================================================================
# CG2: ARTIFACT CONTAMINATION (GEOMETRY-BASED)
# =============================================================================

def _detect_artifacts(
    lines_df: pd.DataFrame,
    header_pct: float = 0.10,
    footer_pct: float = 0.10,
) -> Tuple[Set[str], Dict[str, Any]]:
    artifacts: Set[str] = set()
    metrics: Dict[str, Any] = {}

    if lines_df.empty:
        return artifacts, metrics

    text_col = next((c for c in ["line_text", "text"] if c in lines_df.columns), None)
    if not text_col:
        return artifacts, {"error": "no_text_col"}

    df = lines_df.copy()

    if {"geom_y0", "page"}.issubset(df.columns):
        bounds = df.groupby("page")["geom_y0"].agg(["min", "max"])
        bounds["height"] = bounds["max"] - bounds["min"]
        bounds["header"] = bounds["min"] + bounds["height"] * header_pct
        bounds["footer"] = bounds["max"] - bounds["height"] * footer_pct

        df = df.merge(bounds[["header", "footer"]], left_on="page", right_index=True)
        df["_in_band"] = (df["geom_y0"] <= df["header"]) | (df["geom_y0"] >= df["footer"])
    else:
        df["_in_band"] = False

    df["_norm"] = df[text_col].fillna("").str.strip().str.lower()
    page_counts = df.groupby("_norm")["page"].nunique()
    repeated = page_counts[page_counts >= 2].index

    for txt in repeated:
        if 0 < len(txt) < 60:
            rows = df[df["_norm"] == txt]
            if rows["_in_band"].any():
                artifacts.add(txt)

    metrics["artifact_count"] = len(artifacts)
    return artifacts, metrics


def check_cg2_artifact_contamination(
    lines_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> CheckResult:
    findings = []

    artifacts, detect_metrics = _detect_artifacts(lines_df)
    metrics = dict(detect_metrics)
    metrics["artifacts_sample"] = sorted(artifacts)[:10]

    if not artifacts:
        return CheckResult("CG2", "Artifact Contamination", True,
                           [Finding("CG2", "info", "No artifacts detected")],
                           metrics)

    contaminations = 0
    if not nodes_df.empty and "body_text" in nodes_df.columns:
        text = nodes_df["body_text"].fillna("").str.lower()
        for a in artifacts:
            contaminations += text.str.contains(re.escape(a)).sum()

    if not edges_df.empty and "source_evidence" in edges_df.columns:
        ev = edges_df["source_evidence"].fillna("").str.lower()
        for a in artifacts:
            contaminations += ev.str.contains(re.escape(a)).sum()

    metrics["contaminations"] = contaminations

    if contaminations:
        findings.append(Finding("CG2", "warning", f"{contaminations} artifact contaminations found"))

    return CheckResult(
        "CG2",
        "Artifact Contamination",
        passed=contaminations == 0,
        findings=findings,
        metrics=metrics,
    )


# =============================================================================
# CG3: STRUCTURE COMPLETENESS (RAW HEADING DETECTION)
# =============================================================================

def _detect_headings(lines_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    if lines_df.empty:
        return pd.DataFrame(), metrics

    df = lines_df.copy()
    df["_score"] = 0

    if "line_bold" in df.columns:
        df["_score"] += df["line_bold"].fillna(False).astype(int) * 2

    if "line_size" in df.columns:
        median = df["line_size"].median()
        df["_score"] += (df["line_size"] > median * 1.1).astype(int) * 2

    if "gap_above" in df.columns:
        med_gap = df["gap_above"].median()
        if pd.notna(med_gap):
            df["_score"] += (df["gap_above"] > med_gap * 2).astype(int)

    text_col = next((c for c in ["line_text", "text"] if c in df.columns), None)
    if text_col:
        l = df[text_col].fillna("").str.len()
        df["_score"] += ((l > 5) & (l < 80)).astype(int)

    candidates = df[df["_score"] >= 3]
    metrics["lines_examined"] = len(df)
    metrics["heading_candidates"] = len(candidates)
    return candidates, metrics


def check_cg3_structure_completeness(
    lines_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
) -> CheckResult:
    findings = []

    candidates, detect_metrics = _detect_headings(lines_df)
    metrics = dict(detect_metrics)

    if candidates.empty:
        return CheckResult("CG3", "Structure Completeness", True,
                           [Finding("CG3", "info", "No heading candidates detected")],
                           metrics)

    anchor_count = len(anchors_df) if not anchors_df.empty else 0
    coverage = anchor_count / len(candidates)
    metrics["anchor_count"] = anchor_count
    metrics["coverage_ratio"] = coverage

    if coverage < VARIANCE["heading_coverage"]:
        findings.append(Finding(
            "CG3", "warning",
            f"Coverage {coverage:.0%} below threshold"
        ))

    return CheckResult(
        "CG3",
        "Structure Completeness",
        passed=coverage >= VARIANCE["heading_coverage"],
        findings=findings,
        metrics=metrics,
    )


# =============================================================================
# CG4: GEOMETRY-BASED ANCHOR DISCOVERY
# =============================================================================

def _score_geometry_anchor(row: pd.Series, body_font: float, median_gap: float) -> int:
    """Score a line based on geometric anchor signals."""
    score = 0

    # Font size signal: larger than body = potential anchor
    font_size = row.get("font_size") or row.get("line_size") or 0
    if font_size > body_font * 1.1:  # 10% larger than body
        score += 3
    elif font_size > body_font * 1.05:  # 5% larger
        score += 1

    # Bold signal
    if row.get("line_bold") or row.get("is_bold"):
        score += 2

    # Vertical gap signal: large gap above = section start
    gap = row.get("gap_above") or 0
    if median_gap > 0 and gap > median_gap * 2:
        score += 2
    elif median_gap > 0 and gap > median_gap * 1.5:
        score += 1

    # Text length signal: short text more likely header
    text = str(row.get("line_text") or row.get("text") or "")
    if 5 < len(text) < 60:
        score += 1

    return score


def _detect_geometry_anchors(
    lines_df: pd.DataFrame,
    score_threshold: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect anchors using purely geometric signals.

    Args:
        lines_df: DataFrame with lines and geometry columns.
        score_threshold: Minimum score to be considered an anchor.

    Returns:
        Tuple of (candidates DataFrame, metrics dict).
    """
    metrics: Dict[str, Any] = {}

    if lines_df.empty:
        return pd.DataFrame(), metrics

    df = lines_df.copy()

    # Determine body font size (most common or median)
    font_col = next(
        (c for c in ["font_size", "line_size"] if c in df.columns), None
    )
    if font_col:
        body_font = df[font_col].median()
        metrics["body_font_detected"] = float(body_font)
    else:
        body_font = 9.0  # Default for IRS docs
        metrics["body_font_detected"] = None

    # Determine median gap
    if "gap_above" in df.columns:
        median_gap = df["gap_above"].median()
        if pd.isna(median_gap):
            median_gap = 0
    else:
        median_gap = 0
    metrics["median_gap"] = float(median_gap) if median_gap else 0

    # Score each line
    df["_geom_score"] = df.apply(
        lambda r: _score_geometry_anchor(r, body_font, median_gap), axis=1
    )

    candidates = df[df["_geom_score"] >= score_threshold]
    metrics["lines_examined"] = len(df)
    metrics["geometry_candidates"] = len(candidates)

    return candidates, metrics


def check_cg4_geometry_discovery(
    lines_df: pd.DataFrame,
    text_discovered: Set[str],
) -> CheckResult:
    """
    CG4: Discover anchors from geometry, cross-check with text discovery.

    This provides independent validation: if geometry says "this looks like
    an anchor header" but text didn't find a box pattern, that's a potential
    issue (either text regex is missing patterns, or geometry is wrong).

    Args:
        lines_df: DataFrame with lines and geometry columns.
        text_discovered: Set of box keys from CG1a text discovery.

    Returns:
        CheckResult with correlation metrics.
    """
    findings = []
    metrics = {}

    geom_candidates, detect_metrics = _detect_geometry_anchors(lines_df)
    metrics.update(detect_metrics)

    if geom_candidates.empty:
        # Still compute ratio for monitoring
        if len(text_discovered) > 0:
            metrics["geom_to_text_ratio"] = 0.0
            findings.append(Finding(
                "CG4", "warning",
                f"Geometry found 0 candidates but text found {len(text_discovered)} boxes",
            ))
        return CheckResult(
            "CG4", "Geometry Discovery", True,
            findings or [Finding("CG4", "info", "No geometry-based anchors detected")],
            metrics,
        )

    # Cross-check: which geometry candidates have box patterns?
    text_col = next((c for c in ["line_text", "text"] if c in geom_candidates.columns), None)
    if text_col:
        geom_with_box = 0
        geom_without_box = 0
        undetected_samples = []

        for _, row in geom_candidates.iterrows():
            text = str(row.get(text_col, ""))
            found_keys = _extract_box_keys(text)
            if found_keys:
                geom_with_box += 1
            else:
                geom_without_box += 1
                if len(undetected_samples) < 5:
                    undetected_samples.append(text[:60])

        metrics["geom_with_box_pattern"] = geom_with_box
        metrics["geom_without_box_pattern"] = geom_without_box
        metrics["undetected_samples"] = undetected_samples

        # Correlation: what % of geometry anchors have box patterns?
        total_geom = geom_with_box + geom_without_box
        if total_geom > 0:
            correlation = geom_with_box / total_geom
            metrics["box_correlation"] = correlation

            # Warning if many geometry anchors don't have box patterns
            # (some will be sections, subsections - that's expected)
            if geom_without_box > geom_with_box:
                findings.append(Finding(
                    "CG4", "info",
                    f"{geom_without_box} geometry anchors without box patterns (may be sections)",
                ))

    # Check for geometry-text agreement on count
    # Geometry should find at least as many candidates as text found boxes
    if len(text_discovered) > 0:
        geom_coverage = metrics.get("geometry_candidates", 0) / len(text_discovered)
        metrics["geom_to_text_ratio"] = geom_coverage

        if geom_coverage < 0.5:
            findings.append(Finding(
                "CG4", "warning",
                f"Geometry found {metrics.get('geometry_candidates', 0)} candidates "
                f"but text found {len(text_discovered)} boxes",
            ))

    return CheckResult(
        "CG4", "Geometry Discovery", True,
        findings=findings,
        metrics=metrics,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def validate_corpus_grounded(
    lines_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    spans_df: pd.DataFrame = None,
) -> List[CheckResult]:
    results: List[CheckResult] = []

    cg1a = check_cg1a_anchor_discovery(lines_df)
    results.append(cg1a)

    discovered = set(cg1a.metrics.get("discovered_box_keys", []))
    results.append(check_cg1b_registry_alignment(discovered))

    results.append(check_cg2_artifact_contamination(lines_df, nodes_df, edges_df))
    results.append(check_cg3_structure_completeness(lines_df, anchors_df))
    results.append(check_cg4_geometry_discovery(lines_df, discovered))

    return results
