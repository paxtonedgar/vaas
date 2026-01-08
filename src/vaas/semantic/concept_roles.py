"""
Concept role classification for subsection/concept nodes.

Classifies subsection anchors into semantic roles using regex-first heuristics
with position-based tiebreakers. Returns NULL when confidence is low.

Roles:
- definition: Defines a term or concept (noun phrase headers)
- qualification: Constrains applicability or adds conditions
- condition: Introduces conditional logic (if/when/only if)
- exception: Excludes or negates applicability
- procedure: Describes actions to take (imperative verbs)
"""

import re
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


# Valid concept roles (closed set)
CONCEPT_ROLES = frozenset({
    "definition",
    "qualification",
    "condition",
    "exception",
    "procedure",
})


@dataclass
class RoleClassification:
    """Result of concept role classification."""
    role: Optional[str]
    confidence: float
    method: str  # "regex", "position", or "null"
    pattern_matched: Optional[str] = None


# =============================================================================
# REGEX PATTERNS (ordered by priority - first match wins)
# =============================================================================

# Pattern table: (name, compiled_regex, role, confidence, applies_to)
# applies_to: "header" = match against header_text, "body" = match against body_text

_ROLE_PATTERNS = [
    # Exception patterns (highest priority - explicit negation)
    (
        "negation_phrase",
        re.compile(
            r"(?i)^.{0,80}(?:does\s+not\s+include|do\s+not\s+|except\s+|excluding\s+|not\s+reported|other\s+than\s+)"
        ),
        "exception",
        0.9,
        "body",
    ),

    # Condition patterns (explicit conditional logic)
    (
        "early_conditional",
        re.compile(
            r"(?i)^.{0,60}(?:if\s+(?:you|the|a|any|this)|when\s+(?:you|the|a)|only\s+if\s+)"
        ),
        "condition",
        0.9,
        "body",
    ),

    # Procedure patterns (imperative verbs at start)
    (
        "imperative_verb",
        re.compile(
            r"(?i)^(?:Enter|Report|File|Include|Use|Complete|Check|Attach|See)\s+"
        ),
        "procedure",
        0.85,
        "body",
    ),

    # Qualification patterns
    (
        "qualifier_phrase",
        re.compile(
            r"(?i)^.{0,40}(?:the\s+following|these\s+(?:dividends|amounts|distributions)|this\s+applies|applies\s+to)"
        ),
        "qualification",
        0.75,
        "body",
    ),

    # Definition patterns (noun phrase headers - checked last as fallback)
    (
        "noun_phrase_header",
        re.compile(
            r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+\([^)]+\))?$"
        ),
        "definition",
        0.8,
        "header",
    ),

    # Noun phrase with numbers (e.g., "Section 404(k) Dividends")
    (
        "noun_phrase_with_section",
        re.compile(
            r"^(?:Section\s+)?\d+[a-zA-Z]?\s*(?:\([a-z]\))?\s+[A-Z][a-z]+"
        ),
        "definition",
        0.75,
        "header",
    ),
]


def classify_concept_role(
    header_text: Optional[str],
    body_text: Optional[str],
    position_info: Optional[Dict[str, Any]] = None,
) -> RoleClassification:
    """
    Classify a subsection/concept node into a semantic role.

    Uses regex-first heuristics with position-based tiebreakers.
    Returns role=None when confidence is low.

    Args:
        header_text: The header/label text of the subsection
        body_text: The body content of the subsection
        position_info: Optional dict with position signals:
            - is_before_first_box: bool
            - is_after_box_header: bool
            - body_char_count: int

    Returns:
        RoleClassification with role, confidence, and method
    """
    header = (header_text or "").strip()
    body = (body_text or "").strip()

    # Try regex patterns in priority order
    for pattern_name, pattern, role, confidence, applies_to in _ROLE_PATTERNS:
        text_to_match = header if applies_to == "header" else body

        if not text_to_match:
            continue

        if pattern.search(text_to_match):
            return RoleClassification(
                role=role,
                confidence=confidence,
                method="regex",
                pattern_matched=pattern_name,
            )

    # Position-based tiebreakers (only when regex fails)
    if position_info:
        # Before first box → likely definition
        if position_info.get("is_before_first_box", False):
            return RoleClassification(
                role="definition",
                confidence=0.6,
                method="position",
                pattern_matched="before_first_box",
            )

        # After box header with short body → likely qualification
        if (
            position_info.get("is_after_box_header", False)
            and position_info.get("body_char_count", 0) < 200
        ):
            return RoleClassification(
                role="qualification",
                confidence=0.5,
                method="position",
                pattern_matched="after_box_short_body",
            )

    # No confident classification - return NULL
    return RoleClassification(
        role=None,
        confidence=0.0,
        method="null",
        pattern_matched=None,
    )


def classify_concept_roles_batch(
    anchors: list,
    header_col: str = "label",
    body_col: str = "body_text",
) -> list:
    """
    Classify concept roles for a batch of anchor rows.

    Args:
        anchors: List of anchor dicts or DataFrame rows
        header_col: Column name for header text
        body_col: Column name for body text

    Returns:
        List of RoleClassification objects (same order as input)
    """
    results = []
    for anchor in anchors:
        if hasattr(anchor, "get"):
            # Dict-like
            header = anchor.get(header_col, "")
            body = anchor.get(body_col, "")
        else:
            # Named tuple or object
            header = getattr(anchor, header_col, "")
            body = getattr(anchor, body_col, "")

        result = classify_concept_role(header, body)
        results.append(result)

    return results


def classify_section_roles(sections_df) -> "pd.DataFrame":
    """
    Apply concept role classification to subsection anchors in a sections DataFrame.

    Adds columns:
    - concept_role: Classified role (or None for low confidence)
    - concept_role_confidence: Confidence score (0.0-1.0)
    - concept_role_method: Classification method ("regex", "position", "null")

    Args:
        sections_df: Sections DataFrame with anchor_type, label, body_text columns.

    Returns:
        Sections DataFrame with concept role columns added.
    """
    import pandas as pd

    sdf = sections_df.copy()
    sdf["concept_role"] = None
    sdf["concept_role_confidence"] = 0.0
    sdf["concept_role_method"] = "null"

    subsection_mask = sdf["anchor_type"] == "subsection"

    for idx in sdf[subsection_mask].index:
        header = sdf.loc[idx, "label"] or ""
        body = sdf.loc[idx, "body_text"] or ""

        result = classify_concept_role(header, body)

        sdf.loc[idx, "concept_role"] = result.role
        sdf.loc[idx, "concept_role_confidence"] = result.confidence
        sdf.loc[idx, "concept_role_method"] = result.method

    return sdf
