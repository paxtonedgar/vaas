"""
Typed edge extraction for knowledge graph.

Extracts semantic edge types from text using template-first matching.
Edges point from rule to thing being ruled on.

Edge types (Phase 1b + 2a):
- excludes: Negation/exception relationship (concept → box)
- applies_if: Conditional applicability (concept → box)
- defines: Semantic meaning (concept → box)
- qualifies: Scope/constraint (concept → box)
- requires: Computational dependency (box → box)
"""

import re
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class TypedEdgeCandidate:
    """A candidate typed edge extracted from text."""
    edge_type: str
    source_anchor_id: str
    target_box_key: str
    confidence: float
    evidence_text: str
    pattern_matched: str
    polarity: str  # "positive" or "negative"


# =============================================================================
# PATTERN TABLES (Phase 1b + 2a)
# =============================================================================
# Format: (pattern_name, compiled_regex, confidence)
# Patterns are evaluated in priority order within each type.

# --- EXCLUDES (Phase 1b) - Negation patterns ---
EXCLUDES_PATTERNS = [
    ("does_not_include", re.compile(r"(?i)does\s+not\s+include"), 0.95),
    ("do_not_include", re.compile(r"(?i)do\s+not\s+include"), 0.95),
    ("do_not_report", re.compile(r"(?i)do\s+not\s+report"), 0.95),
    ("not_reported", re.compile(r"(?i)(?:is|are)\s+not\s+reported"), 0.90),
    ("except", re.compile(r"(?i)\bexcept\s+(?:for\s+)?(?:the\s+)?(?:amounts?|dividends?|distributions?)?"), 0.85),
    ("excluding", re.compile(r"(?i)\bexcluding\b"), 0.90),
    ("other_than", re.compile(r"(?i)\bother\s+than\b"), 0.80),
    ("not_qualified", re.compile(r"(?i)(?:is|are)\s+not\s+qualified"), 0.90),
    ("not_eligible", re.compile(r"(?i)(?:is|are)\s+not\s+eligible"), 0.85),
]

# --- APPLIES_IF (Phase 2a) - Conditional applicability ---
# Direction: concept → box
APPLIES_IF_PATTERNS = [
    ("report_if", re.compile(r"(?i)report\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)\s+if\b"), 0.90),
    ("include_when", re.compile(r"(?i)include\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)\s+when\b"), 0.90),
    ("if_report", re.compile(r"(?i)if\s+.{1,60}?,\s*report\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)"), 0.85),
    ("only_if", re.compile(r"(?i)only\s+if\s+.{1,60}?,\s*include\s+(?:.+?\s+)?in\s+box\s+(\d+[a-z]?)"), 0.85),
]

# --- DEFINES (Phase 2a) - Semantic meaning ---
# Direction: concept → box
DEFINES_PATTERNS = [
    ("box_is", re.compile(r"(?i)box\s+(\d+[a-z]?)\s+(?:is|means)\b"), 0.95),
    ("term_refers", re.compile(r"(?i)([A-Z][A-Za-z\s]{3,40})\s+refers\s+to\b"), 0.85),
    ("noun_phrase_are", re.compile(r"(?i)^([A-Z][A-Za-z\s]{3,40})\s+are\s+dividends\s+that\b"), 0.85),
    ("includes_definition", re.compile(r"(?i)([A-Z][A-Za-z\s]{3,40})\s+includes?\s+dividends\s+that\b"), 0.80),
]

# --- QUALIFIES (Phase 2a) - Scope/constraint ---
# Direction: concept → box
QUALIFIES_PATTERNS = [
    ("includes_only", re.compile(r"(?i)box\s+(\d+[a-z]?)\s+includes\s+only\b"), 0.95),
    ("applies_to_box", re.compile(r"(?i)for\s+box\s+(\d+[a-z]?),\s+this\s+applies\s+to\b"), 0.90),
    ("only_portion", re.compile(r"(?i)only\s+the\s+.+?\s+portion\s+.{0,40}?\s+reported\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
    ("limited_to", re.compile(r"(?i)limited\s+to\s+amounts\s+reported\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
]

# --- REQUIRES (Phase 2a) - Computational dependency ---
# Direction: box → box (source = current box anchor, target = captured boxes)
REQUIRES_PATTERNS = [
    ("include_from", re.compile(r"(?i)include\s+amounts?\s+from\s+box(?:es)?\s+((?:\d+[a-z]?(?:,?\s*(?:and\s+)?)?)+)"), 0.95),
    ("also_report", re.compile(r"(?i)also\s+report\s+.{0,40}?\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
    ("see_box", re.compile(r"(?i)see\s+box\s+(\d+[a-z]?)"), 0.80),
    ("combine_with", re.compile(r"(?i)combine\s+with\s+amounts?\s+in\s+box\s+(\d+[a-z]?)"), 0.85),
]

# Box reference pattern for general use
BOX_REF_PATTERN = re.compile(
    r"[Bb]ox(?:es)?\s+(\d+[a-z]?(?:\s*(?:,|and|or)\s*\d+[a-z]?)*)",
    re.IGNORECASE
)


def _clean_evidence(text: str, ref_pos: int, match_len: int) -> str:
    """Extract and clean evidence snippet around a match."""
    start = max(0, ref_pos - 40)
    end = min(len(text), ref_pos + match_len + 40)
    evidence = text[start:end].strip()
    # Clean newlines (prevents validation warnings)
    evidence = " ".join(evidence.split())
    if start > 0:
        evidence = "..." + evidence
    if end < len(text):
        evidence = evidence + "..."
    return evidence


def _extract_box_keys(ref_text: str) -> List[str]:
    """Extract individual box keys from a reference string."""
    individual = re.findall(r"\b(\d+[a-z]?)\b", ref_text, re.IGNORECASE)
    return [k.lower() for k in individual]


def _has_negation_context(text: str, pos: int, window: int = 80) -> bool:
    """Check if position has negation context (excludes should win)."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    context = text[start:end]

    for _, pattern, _ in EXCLUDES_PATTERNS:
        if pattern.search(context):
            return True
    return False


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_excludes_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract excludes edges from text.
    Direction: concept → box
    """
    if not text:
        return []

    edges = []
    seen = set()  # (box_key) - one edge per box per anchor

    for match in BOX_REF_PATTERN.finditer(text):
        ref_text = match.group(1)
        ref_pos = match.start()

        # Check for negation context
        start = max(0, ref_pos - 80)
        end = min(len(text), ref_pos + 80)
        context = text[start:end]

        negation_match = None
        for pattern_name, pattern, confidence in EXCLUDES_PATTERNS:
            m = pattern.search(context)
            if m:
                negation_match = (pattern_name, confidence)
                break

        if not negation_match:
            continue

        pattern_name, confidence = negation_match
        box_keys = _extract_box_keys(ref_text)

        for box_key in box_keys:
            if box_key not in valid_box_keys:
                continue
            if box_key in seen:
                continue
            seen.add(box_key)

            evidence = _clean_evidence(text, ref_pos, len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="excludes",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="negative",
            ))

    return edges


def extract_applies_if_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract applies_if edges from text.
    Direction: concept → box
    """
    if not text:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in APPLIES_IF_PATTERNS:
        for match in pattern.finditer(text):
            box_key = match.group(1).lower()

            if box_key not in valid_box_keys:
                continue
            if box_key in excluded_boxes:
                continue  # excludes wins
            if box_key in seen:
                continue

            # Check for negation context (excludes wins)
            if _has_negation_context(text, match.start()):
                continue

            seen.add(box_key)
            evidence = _clean_evidence(text, match.start(), len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="applies_if",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="positive",
            ))

    return edges


def extract_defines_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract defines edges from text.
    Direction: concept → box

    Note: Some patterns capture terms, not boxes. Those need box resolution
    via nearby references. For now, only emit when box is directly captured.
    """
    if not text:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in DEFINES_PATTERNS:
        for match in pattern.finditer(text):
            captured = match.group(1)

            # Check if captured is a box key or a term
            if re.match(r"^\d+[a-z]?$", captured, re.IGNORECASE):
                # Direct box reference
                box_key = captured.lower()
            else:
                # Term captured - need to find nearby box reference
                # For now, skip (Phase 2b can add term→box resolution)
                continue

            if box_key not in valid_box_keys:
                continue
            if box_key in excluded_boxes:
                continue
            if box_key in seen:
                continue

            if _has_negation_context(text, match.start()):
                continue

            seen.add(box_key)
            evidence = _clean_evidence(text, match.start(), len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="defines",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="positive",
            ))

    return edges


def extract_qualifies_edges(
    anchor_id: str,
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract qualifies edges from text.
    Direction: concept → box
    """
    if not text:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in QUALIFIES_PATTERNS:
        for match in pattern.finditer(text):
            box_key = match.group(1).lower()

            if box_key not in valid_box_keys:
                continue
            if box_key in excluded_boxes:
                continue
            if box_key in seen:
                continue

            if _has_negation_context(text, match.start()):
                continue

            seen.add(box_key)
            evidence = _clean_evidence(text, match.start(), len(match.group(0)))

            edges.append(TypedEdgeCandidate(
                edge_type="qualifies",
                source_anchor_id=anchor_id,
                target_box_key=box_key,
                confidence=confidence,
                evidence_text=evidence,
                pattern_matched=pattern_name,
                polarity="positive",
            ))

    return edges


def extract_requires_edges(
    anchor_id: str,
    source_box_key: Optional[str],
    text: str,
    valid_box_keys: Set[str],
    excluded_boxes: Set[str],
) -> List[TypedEdgeCandidate]:
    """
    Extract requires edges from text.
    Direction: box → box (source = current section's box, target = referenced boxes)

    Only emits edges when source is a box anchor.
    """
    if not text or not source_box_key:
        return []

    edges = []
    seen = set()

    for pattern_name, pattern, confidence in REQUIRES_PATTERNS:
        for match in pattern.finditer(text):
            ref_text = match.group(1)
            target_keys = _extract_box_keys(ref_text)

            for box_key in target_keys:
                if box_key not in valid_box_keys:
                    continue
                if box_key in excluded_boxes:
                    continue
                if box_key == source_box_key:
                    continue  # No self-edges for requires
                if box_key in seen:
                    continue

                if _has_negation_context(text, match.start()):
                    continue

                seen.add(box_key)
                evidence = _clean_evidence(text, match.start(), len(match.group(0)))

                edges.append(TypedEdgeCandidate(
                    edge_type="requires",
                    source_anchor_id=anchor_id,
                    target_box_key=box_key,
                    confidence=confidence,
                    evidence_text=evidence,
                    pattern_matched=pattern_name,
                    polarity="positive",
                ))

    return edges


def extract_typed_edges_from_section(
    anchor_id: str,
    body_text: str,
    valid_box_keys: Set[str],
    source_box_key: Optional[str] = None,
) -> List[TypedEdgeCandidate]:
    """
    Extract all typed edges from a section's body text.

    Priority: excludes wins - if a box has an excludes edge, don't emit
    other edge types for that same box from this section.

    Args:
        anchor_id: The source anchor ID
        body_text: Body text of the section
        valid_box_keys: Set of valid box keys
        source_box_key: If section is a box, its key (for requires edges)

    Returns:
        List of TypedEdgeCandidate objects
    """
    all_edges = []

    # Phase 1b: Extract excludes edges FIRST (highest priority)
    excludes_edges = extract_excludes_edges(anchor_id, body_text, valid_box_keys)
    all_edges.extend(excludes_edges)

    # Track boxes with excludes (they don't get other edge types)
    excluded_boxes = {e.target_box_key for e in excludes_edges}

    # Phase 2a: Extract other typed edges
    all_edges.extend(extract_applies_if_edges(
        anchor_id, body_text, valid_box_keys, excluded_boxes
    ))
    all_edges.extend(extract_defines_edges(
        anchor_id, body_text, valid_box_keys, excluded_boxes
    ))
    all_edges.extend(extract_qualifies_edges(
        anchor_id, body_text, valid_box_keys, excluded_boxes
    ))
    all_edges.extend(extract_requires_edges(
        anchor_id, source_box_key, body_text, valid_box_keys, excluded_boxes
    ))

    return all_edges
