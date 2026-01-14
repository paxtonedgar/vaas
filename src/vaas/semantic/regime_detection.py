"""
Regime detection for knowledge graph.

Regimes are scoping containers that modify rules within them.
Examples:
- "For U.S. owners of PFICs, the holding period requirements differ"
- "If the recipient is a domestic corporation..."
- "For section 897 purposes..."

Regimes create two types of edges:
- governs: Regime → Concept (the regime applies to this concept)
- overrides: Regime → Rule (the regime overrides the default rule)

Usage:
    from vaas.semantic.regime_detection import detect_regimes, build_regime_edges
    regimes = detect_regimes(sections_df)
    regime_edges = build_regime_edges(regimes, nodes_df, doc_id)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class DetectedRegime:
    """A detected regime scope in the document."""
    regime_id: str
    label: str
    condition: str  # The triggering condition (e.g., "U.S. owners of PFICs")
    source_anchor_id: str  # The anchor where regime was detected
    source_element_id: Optional[str]
    evidence_text: str
    confidence: float
    char_start: int
    char_end: int
    # Scope information
    scope_type: str  # "entity", "section", "condition", "purpose"
    scope_depth: int = 0  # Nesting level (0 = top-level)


@dataclass
class RegimeEdge:
    """An edge from regime to governed/overridden node."""
    regime_id: str
    target_node_id: str
    edge_type: str  # "governs" or "overrides"
    confidence: float
    evidence_text: str


# =============================================================================
# REGIME PATTERNS
# =============================================================================

# Named regime patterns - explicit entity-type containers
# These create specific named regime containers
NAMED_REGIME_PATTERNS = [
    # RIC/REIT regime - "Only RICs and REITs should complete boxes 2e and 2f"
    ("RIC_REIT_Regime", re.compile(
        r"(?i)only\s+RICs?\s+and\s+REITs?\s+should\s+complete\s+box(?:es)?\s+"
        r"((?:\d+[a-z]?\s*(?:,?\s*(?:and|or)\s*)?)+)"
    ), "entity", 0.95),
    # RIC-specific regime - "RICs—Special reporting instructions"
    ("RIC_Regime", re.compile(
        r"(?i)RICs?[—–\-]\s*(?:Special\s+)?(?:reporting\s+)?instructions?"
    ), "entity", 0.95),
    # REIT-specific regime - headers mentioning REIT context
    ("REIT_Regime", re.compile(
        r"(?i)REITs?[—–\-]\s*(?:Special\s+)?(?:reporting\s+)?instructions?"
    ), "entity", 0.95),
    # Section 199A regime - "Section 199A dividends"
    ("Section199A_Regime", re.compile(
        r"(?i)(?:section\s+)?199A\s+dividends?"
    ), "section", 0.95),
    # Section 897 regime - for FIRPTA/foreign investment
    ("Section897_Regime", re.compile(
        r"(?i)section\s+897\s+(?:ordinary\s+dividends?|capital\s+gain|gain)"
    ), "section", 0.95),
]

# Entity-scoped regimes ("For [entity type]...")
ENTITY_REGIME_PATTERNS = [
    # "For U.S. owners of PFICs..."
    (re.compile(
        r"(?i)for\s+(?:U\.?S\.?\s+)?(?:owners?|holders?|shareholders?|recipients?)\s+of\s+"
        r"([A-Z][A-Za-z\s\-]+?)(?:,|\.|$)"
    ), "entity", 0.90),
    # "If the recipient is a [entity]..."
    (re.compile(
        r"(?i)if\s+(?:the\s+)?(?:recipient|payee|holder|owner)\s+is\s+(?:a\s+)?"
        r"([A-Za-z\s\-]+?)(?:,|\.|$)"
    ), "condition", 0.85),
    # "For domestic corporations..."
    (re.compile(
        r"(?i)for\s+(?:domestic|foreign)\s+(?:corporations?|entities|persons?)(?:,|\.|$)"
    ), "entity", 0.85),
]

# IRC section-scoped regimes ("For section 897 purposes...")
SECTION_REGIME_PATTERNS = [
    # "For section XXX purposes..."
    (re.compile(
        r"(?i)for\s+(?:section|sec\.?|IRC\s+§?)\s*(\d+[a-z]?(?:\([a-z]\))?)\s+purposes?"
    ), "purpose", 0.90),
    # "Under section XXX..."
    (re.compile(
        r"(?i)under\s+(?:section|sec\.?|IRC\s+§?)\s*(\d+[a-z]?(?:\([a-z]\))?)"
    ), "section", 0.85),
    # "Subject to section XXX..."
    (re.compile(
        r"(?i)subject\s+to\s+(?:section|sec\.?|IRC\s+§?)\s*(\d+[a-z]?(?:\([a-z]\))?)"
    ), "section", 0.85),
]

# Conditional regimes ("If [condition]...")
CONDITIONAL_REGIME_PATTERNS = [
    # "If the holding period requirement is not met..."
    (re.compile(
        r"(?i)if\s+(?:the\s+)?([a-z\s]+?requirement)\s+(?:is|has)\s+(?:not\s+)?(?:been\s+)?met"
    ), "condition", 0.85),
    # "If you report on an accrual basis..."
    (re.compile(
        r"(?i)if\s+you\s+(?:report|file|use)\s+(?:on\s+)?(?:an?\s+)?([a-z\s]+?basis)"
    ), "condition", 0.80),
    # "When the dividend is from a REIT..."
    (re.compile(
        r"(?i)when\s+(?:the\s+)?(?:dividend|distribution|payment)\s+is\s+from\s+(?:a\s+)?([A-Z]+)"
    ), "condition", 0.85),
]

# Box reference pattern for extracting governed boxes
BOX_REF_PATTERN = re.compile(
    r"[Bb]ox(?:es)?\s+(\d+[a-z]?(?:\s*(?:,|and|or)\s*\d+[a-z]?)*)"
)


@dataclass
class DetectedRegimeWithBoxes(DetectedRegime):
    """A detected regime with its governed boxes."""
    governed_boxes: List[str] = field(default_factory=list)


def _extract_box_keys(box_ref: str) -> List[str]:
    """Extract individual box keys from a box reference string."""
    # Split on comma, 'and', 'or' and clean
    parts = re.split(r'\s*(?:,|and|or)\s*', box_ref.strip())
    keys = []
    for part in parts:
        part = part.strip().lower()
        # Extract just the number/letter (e.g., "2e" from "box 2e")
        m = re.search(r'(\d+[a-z]?)', part)
        if m:
            keys.append(m.group(1))
    return keys


def detect_regimes(
    sections_df: pd.DataFrame,
    doc_id: str,
) -> List[DetectedRegimeWithBoxes]:
    """
    Detect regime scopes in sections.

    Args:
        sections_df: Sections DataFrame with full_text, anchor_id
        doc_id: Document identifier

    Returns:
        List of detected regimes with governed boxes
    """
    regimes: List[DetectedRegimeWithBoxes] = []
    seen_regimes: Dict[str, DetectedRegimeWithBoxes] = {}  # Track by regime_id to merge boxes

    if sections_df is None or sections_df.empty:
        return regimes

    for _, section in sections_df.iterrows():
        anchor_id = section.get("anchor_id", "")
        full_text = section.get("full_text", "") or ""
        box_key = (section.get("box_key", "") or "").lower()

        if not full_text.strip():
            continue

        # -------------------------------------------------------------------------
        # PASS 1: Named regime patterns (specific containers like RIC_REIT_Regime)
        # -------------------------------------------------------------------------
        for regime_name, pattern, scope_type, confidence in NAMED_REGIME_PATTERNS:
            for match in pattern.finditer(full_text):
                # Extract governed boxes from the match or nearby text
                governed_boxes = []

                # If pattern captures box refs, extract them
                if match.lastindex and match.lastindex >= 1:
                    box_ref = match.group(1)
                    governed_boxes = _extract_box_keys(box_ref)

                # Also look for box refs in the surrounding text (within 200 chars)
                context_start = max(0, match.start() - 50)
                context_end = min(len(full_text), match.end() + 200)
                context = full_text[context_start:context_end]

                for box_match in BOX_REF_PATTERN.finditer(context):
                    for bk in _extract_box_keys(box_match.group(1)):
                        if bk not in governed_boxes:
                            governed_boxes.append(bk)

                # If this section IS a box, the regime governs that box
                if box_key and box_key not in governed_boxes:
                    governed_boxes.append(box_key)

                # Merge with existing regime of same name or create new
                if regime_name in seen_regimes:
                    # Add new governed boxes
                    for bk in governed_boxes:
                        if bk not in seen_regimes[regime_name].governed_boxes:
                            seen_regimes[regime_name].governed_boxes.append(bk)
                else:
                    regime = DetectedRegimeWithBoxes(
                        regime_id=regime_name,
                        label=regime_name.replace("_", " "),
                        condition=match.group(0).strip()[:100],
                        source_anchor_id=anchor_id,
                        source_element_id=None,
                        evidence_text=match.group(0),
                        confidence=confidence,
                        char_start=match.start(),
                        char_end=match.end(),
                        scope_type=scope_type,
                        governed_boxes=governed_boxes,
                    )
                    seen_regimes[regime_name] = regime

        # -------------------------------------------------------------------------
        # PASS 2: Generic regime patterns (for additional context)
        # -------------------------------------------------------------------------
        all_generic_patterns = (
            ENTITY_REGIME_PATTERNS +
            SECTION_REGIME_PATTERNS +
            CONDITIONAL_REGIME_PATTERNS
        )

        for pattern, scope_type, confidence in all_generic_patterns:
            for match in pattern.finditer(full_text):
                # Extract condition from capture group or full match
                if match.lastindex and match.lastindex >= 1:
                    condition = match.group(1).strip()
                else:
                    condition = match.group(0).strip()

                # Generate unique ID based on condition
                regime_id = f"regime_{condition[:20].replace(' ', '_').lower()}"

                # Skip if we already have a named regime covering this
                if any(r.source_anchor_id == anchor_id for r in seen_regimes.values()):
                    continue

                governed_boxes = []
                if box_key:
                    governed_boxes.append(box_key)

                if regime_id not in seen_regimes:
                    regime = DetectedRegimeWithBoxes(
                        regime_id=regime_id,
                        label=f"{scope_type.title()}: {condition[:50]}",
                        condition=condition,
                        source_anchor_id=anchor_id,
                        source_element_id=None,
                        evidence_text=match.group(0),
                        confidence=confidence,
                        char_start=match.start(),
                        char_end=match.end(),
                        scope_type=scope_type,
                        governed_boxes=governed_boxes,
                    )
                    seen_regimes[regime_id] = regime

    return list(seen_regimes.values())


def build_regime_nodes(
    regimes: List[DetectedRegimeWithBoxes],
    sections_df: pd.DataFrame,
    doc_id: str,
) -> pd.DataFrame:
    """
    Build regime nodes DataFrame.

    Args:
        regimes: List of detected regimes with governed boxes
        sections_df: Sections DataFrame for metadata
        doc_id: Document identifier

    Returns:
        DataFrame with regime nodes
    """
    if not regimes:
        return pd.DataFrame()

    # Build anchor_id → section metadata lookup
    section_meta = {}
    for _, row in sections_df.iterrows():
        aid = row.get("anchor_id")
        if aid:
            section_meta[aid] = {
                "pages": row.get("pages", []),
                "bbox": row.get("bbox"),
            }

    nodes = []
    for regime in regimes:
        meta = section_meta.get(regime.source_anchor_id, {})

        nodes.append({
            "node_id": f"{doc_id}:{regime.regime_id}",
            "doc_id": doc_id,
            "node_type": "regime",
            "anchor_id": regime.regime_id,
            "label": regime.label,
            "text": regime.evidence_text,
            "pages": meta.get("pages", []),
            "bbox": meta.get("bbox"),
            "element_count": 0,
            "char_count": len(regime.evidence_text),
            "regime_condition": regime.condition,
            "regime_scope_type": regime.scope_type,
            "source_anchor_id": regime.source_anchor_id,
            "governed_boxes": regime.governed_boxes,  # List of governed box keys
        })

    return pd.DataFrame(nodes)


def build_regime_edges(
    regimes: List[DetectedRegimeWithBoxes],
    sections_df: pd.DataFrame,
    doc_id: str,
    valid_box_keys: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Build governs edges from regimes to boxes and concepts.

    Logic:
    - Regime governs boxes explicitly listed in governed_boxes
    - Regime governs concepts in sections that reference the regime's boxes

    Args:
        regimes: List of detected regimes with governed boxes
        sections_df: Sections DataFrame
        doc_id: Document identifier
        valid_box_keys: Set of valid box keys (for validation)

    Returns:
        List of edge dictionaries
    """
    edges = []

    if not regimes:
        return edges

    # Build set of valid boxes for validation
    if valid_box_keys is None:
        valid_box_keys = set()
        for _, row in sections_df.iterrows():
            bk = (row.get("box_key", "") or "").lower()
            if bk:
                valid_box_keys.add(bk)

    # -------------------------------------------------------------------------
    # PASS 0: Create parent_of edges from doc_root → regime (structural connectivity)
    # This ensures regime nodes are connected to the graph hierarchy
    # -------------------------------------------------------------------------
    doc_root_id = f"{doc_id}:doc_root"
    for regime in regimes:
        regime_node_id = f"{doc_id}:{regime.regime_id}"
        edges.append({
            "edge_id": f"e:{doc_root_id}->parent_of->{regime_node_id}",
            "source_node_id": doc_root_id,
            "target_node_id": regime_node_id,
            "edge_type": "parent_of",
            "confidence": 1.0,
            "source_evidence": f"Regime '{regime.label}' is child of document root",
            "created_by": "structural",
        })

    # -------------------------------------------------------------------------
    # PASS 1: Create governs edges from regime → box (direct governance)
    # -------------------------------------------------------------------------
    for regime in regimes:
        for box_key in regime.governed_boxes:
            if valid_box_keys and box_key not in valid_box_keys:
                continue

            edges.append({
                "source_node_id": f"{doc_id}:{regime.regime_id}",
                "target_node_id": f"{doc_id}:box_{box_key}",
                "edge_type": "governs",
                "confidence": regime.confidence,
                "source_evidence": f"Regime '{regime.label}' governs Box {box_key}",
                "created_by": "regex",
                "pattern_matched": "named_regime",
            })

    # -------------------------------------------------------------------------
    # PASS 2: Create governs edges from regime → subsections in governed boxes
    # -------------------------------------------------------------------------
    # Map box_key → subsections under that box
    box_subsections: Dict[str, List[str]] = {}
    for _, section in sections_df.iterrows():
        anchor_id = section.get("anchor_id", "")
        anchor_type = section.get("anchor_type", "")
        box_key = (section.get("box_key", "") or "").lower()

        if anchor_type == "subsection" and box_key:
            if box_key not in box_subsections:
                box_subsections[box_key] = []
            box_subsections[box_key].append(anchor_id)

    # For each regime, also govern subsections under its governed boxes
    for regime in regimes:
        for box_key in regime.governed_boxes:
            if box_key in box_subsections:
                for subsection_id in box_subsections[box_key]:
                    edges.append({
                        "source_node_id": f"{doc_id}:{regime.regime_id}",
                        "target_node_id": f"{doc_id}:{subsection_id}",
                        "edge_type": "governs",
                        "confidence": regime.confidence * 0.9,  # Slightly lower for inferred
                        "source_evidence": f"Regime '{regime.label}' governs subsection under Box {box_key}",
                        "created_by": "regex",
                        "pattern_matched": "regime_subsection",
                    })

    return edges


def _get_parent_anchors(sections_df: pd.DataFrame, anchor_id: str) -> List[str]:
    """Get parent anchor IDs for a given anchor."""
    # Simplified: return sections that appear before this one
    # Full implementation would use parent_of edges
    parents = []

    anchor_row = sections_df[sections_df["anchor_id"] == anchor_id]
    if anchor_row.empty:
        return parents

    # Get all sections (simplified containment)
    for _, row in sections_df.iterrows():
        aid = row.get("anchor_id", "")
        atype = row.get("anchor_type", "")

        # Sections and boxes can contain regimes
        if atype in ("section", "box") and aid != anchor_id:
            parents.append(aid)

    return parents


def generate_regime_summary(regimes: List[DetectedRegime]) -> str:
    """Generate human-readable summary of detected regimes."""
    if not regimes:
        return "No regimes detected.\n"

    lines = [f"# Regime Detection Summary\n", f"**Total regimes:** {len(regimes)}\n"]

    by_type: Dict[str, List[DetectedRegime]] = {}
    for r in regimes:
        if r.scope_type not in by_type:
            by_type[r.scope_type] = []
        by_type[r.scope_type].append(r)

    for scope_type, type_regimes in sorted(by_type.items()):
        lines.append(f"\n## {scope_type.title()} Regimes ({len(type_regimes)})\n")
        for r in type_regimes:
            lines.append(f"- **{r.regime_id}**: {r.condition}")
            lines.append(f"  - Source: {r.source_anchor_id}")
            lines.append(f"  - Confidence: {r.confidence:.2f}")

    return "\n".join(lines)
