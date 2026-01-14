"""
Graph edge construction for knowledge graph.

This module builds edges between nodes in the knowledge graph. Edge types include:
- Structural edges (parent_of, follows, in_section) from document layout
- Reference edges (references_box, same_group) from cross-references
- Typed edges (excludes, includes, applies_if, etc.) from semantic patterns

Edge Construction Strategy:
1. Build section hierarchy first (parent_of: doc_root → section → box → concept)
2. Build reading order edges (follows between consecutive anchors)
3. Build anchor → paragraph edges (fine-grained containment)
4. Build paragraph follows edges (reading order within anchor)
5. Build in_section containment edges (denormalized for fast queries)
6. Build reference edges from extracted cross-references
7. Build same_group edges for grouped boxes (e.g., Boxes 14-16)
8. Build typed semantic edges (excludes, includes, etc.)
9. Filter edges to only reference active nodes
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from vaas.extraction.geometry import reading_order_sort_key
from vaas.utils.text import stable_hash
from vaas.utils.term_bindings import exception_term_from_context


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

# =============================================================================
# RULE CLASS CONSTANTS (for semantic precedence)
# =============================================================================
# Lower = higher priority. Rule class determines base precedence.
# Tie-breaker within class uses reading order.

RULE_CLASS_GATING = "GATING"          # "Only X should complete..." - base 100
RULE_CLASS_PROHIBITION = "PROHIBITION"  # "Do not include..." - base 200
RULE_CLASS_FALLBACK = "FALLBACK"      # "If impractical..." - base 300
RULE_CLASS_POPULATION = "POPULATION"  # "Enter/Include X in box Y" - base 1000
RULE_CLASS_AGGREGATION = "AGGREGATION"  # "Box A includes box B" - base 1100

RULE_CLASS_BASE_PRECEDENCE = {
    RULE_CLASS_GATING: 1,       # "Only X should complete..." - highest priority
    RULE_CLASS_PROHIBITION: 2,  # "Do not include..."
    RULE_CLASS_FALLBACK: 3,     # "If impractical..."
    RULE_CLASS_POPULATION: 10,  # "Enter/Include X in box Y"
    RULE_CLASS_AGGREGATION: 11, # "Box A includes box B"
    None: 20,                   # Unclassified edges
}

# Scale factor to ensure rule_class completely dominates position
RULE_CLASS_SCALE = 1_000_000_000


def compute_precedence(rule_class: Optional[str], reading_order: int, sentence_idx: int) -> int:
    """
    Compute precedence from rule class and position.

    Formula: base(rule_class) * 1_000_000_000 + reading_order * 1000 + sentence_idx
    Rule class completely dominates; position is only a tie-breaker within class.

    Args:
        rule_class: Rule classification (GATING, PROHIBITION, etc.)
        reading_order: Document reading order (0-based)
        sentence_idx: Sentence index within section (0-based)

    Returns:
        Precedence value (lower = higher priority)
    """
    base = RULE_CLASS_BASE_PRECEDENCE.get(rule_class, 20)
    # Position is pure tie-breaker within same rule class
    return base * RULE_CLASS_SCALE + reading_order * 1000 + sentence_idx


@dataclass
class Edge:
    """
    A single graph edge.

    Attributes:
        edge_id: Unique identifier for this edge.
        source_node_id: Source node ID (fully qualified: "doc_id:anchor_id").
        target_node_id: Target node ID (fully qualified).
        edge_type: Type of edge (parent_of, follows, references_box, etc.).
        direction: "directed" or "bidirectional".
        confidence: Confidence score (0.0-1.0).
        source_evidence: Evidence text supporting this edge.
        source_element_id: Element ID containing evidence (for provenance).
        source_element_ids: List of element IDs for multi-source edges (provenance).
        source_pages: Page numbers where evidence appears (provenance).
        source_bbox: Bounding box of evidence [x0, y0, x1, y1] (provenance).
        created_by: How edge was created ("structural", "regex", "llm").
        pattern_matched: For typed edges, the pattern that matched.
        polarity: For typed edges, "positive" or "negative".
        evidence_sentence_idx: Sentence index within section (for sentence-gated edges).
        evidence_char_start: Character offset start (relative to section full_text).
        evidence_char_end: Character offset end (relative to section full_text).
        rule_class: Semantic rule classification (GATING, PROHIBITION, FALLBACK, etc.).
            Determines base precedence. None for structural/reference edges.
        precedence: Rule priority (lower = higher priority). First-match-wins semantics.
            Computed as: base(rule_class) + tie_breaker(reading_order, sentence_idx)
            None means no precedence context (structural edges).
    """

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    direction: str = "directed"
    confidence: float = 1.0
    source_evidence: Optional[str] = None
    source_element_id: Optional[str] = None
    source_element_ids: List[str] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    source_bbox: Optional[List[float]] = None
    created_by: str = "structural"
    pattern_matched: Optional[str] = None
    polarity: Optional[str] = None
    # Sentence-level provenance for semantic edges
    evidence_sentence_idx: Optional[int] = None
    evidence_char_start: Optional[int] = None
    evidence_char_end: Optional[int] = None
    # Rule classification and precedence (semantic priority)
    rule_class: Optional[str] = None
    precedence: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type,
            "direction": self.direction,
            "confidence": self.confidence,
            "source_evidence": self.source_evidence,
            "source_element_id": self.source_element_id,
            "source_element_ids": self.source_element_ids,
            "source_pages": self.source_pages,
            "source_bbox": self.source_bbox,
            "created_by": self.created_by,
            "pattern_matched": self.pattern_matched,
            "polarity": self.polarity,
            "evidence_sentence_idx": self.evidence_sentence_idx,
            "evidence_char_start": self.evidence_char_start,
            "evidence_char_end": self.evidence_char_end,
            "rule_class": self.rule_class,
            "precedence": self.precedence,
        }


@dataclass
class EdgeBuildResult:
    """
    Result of edge building.

    Attributes:
        edges_df: DataFrame with all edges.
        edge_counts: Dictionary of edge type to count.
        edges_filtered: Number of edges filtered (referencing non-existent nodes).
        typed_edges: List of typed semantic Edge objects (pre-dedup).
    """

    edges_df: pd.DataFrame
    edge_counts: Dict[str, int] = field(default_factory=dict)
    edges_filtered: int = 0
    typed_edges: List[Edge] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of edges."""
        return sum(self.edge_counts.values())

    def __repr__(self) -> str:
        return f"EdgeBuildResult(total={self.total}, types={len(self.edge_counts)})"


# =============================================================================
# EDGE ID GENERATION
# =============================================================================

def generate_edge_id(
    edge_type: str,
    source_id: str,
    target_id: str,
    suffix: Optional[str] = None,
) -> str:
    """
    Generate stable edge ID from components.

    Args:
        edge_type: Type of edge.
        source_id: Source node ID.
        target_id: Target node ID.
        suffix: Optional suffix for uniqueness.

    Returns:
        Stable edge ID.
    """
    parts = [edge_type, source_id, target_id]
    if suffix:
        parts.append(suffix)
    hash_suffix = stable_hash(parts, length=8)
    return f"e_{edge_type[:3]}_{hash_suffix}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sort_key_for_section(row: pd.Series) -> Tuple[int, int, float, float]:
    """
    Get sort key for ordering sections by reading order.

    Delegates to geometry.reading_order_sort_key() for column-aware ordering.

    Args:
        row: Section row with pages and bbox.

    Returns:
        Tuple of (page, column, y0, x0) for sorting.
        Column is 0 for left column, 1 for right column (based on page midpoint).
    """
    return reading_order_sort_key(row)


def extract_page_from_row(row: pd.Series) -> int:
    """Extract page number from row."""
    pages = row.get("pages", [])
    if isinstance(pages, (list, tuple, np.ndarray)) and len(pages) > 0:
        return int(pages[0])
    elif isinstance(pages, (int, float, np.integer)):
        return int(pages)
    return 0


# =============================================================================
# STRUCTURAL EDGE BUILDERS
# =============================================================================

def build_section_hierarchy_edges(
    sections_df: pd.DataFrame,
    doc_id: str,
) -> List[Edge]:
    """
    Build parent_of edges for section hierarchy.

    Uses "nearest preceding" containment:
    - preamble → doc_root
    - section → doc_root (resets current_box)
    - box → current_section OR doc_root
    - concept (subsection) → current_box OR current_section OR doc_root

    Args:
        sections_df: Sections DataFrame with anchor_id, anchor_type, pages, bbox.
        doc_id: Document ID prefix.

    Returns:
        List of parent_of Edge objects.
    """
    if sections_df.empty:
        return []

    edges: List[Edge] = []
    doc_root_id = f"{doc_id}:doc_root"

    # Sort sections by reading order
    ordered = sections_df.copy()
    ordered["_sort_key"] = ordered.apply(get_sort_key_for_section, axis=1)
    ordered = ordered.sort_values("_sort_key").reset_index(drop=True)

    # Track hierarchy state
    current_section = None
    current_box = None

    for _, row in ordered.iterrows():
        anchor_id = row["anchor_id"]
        anchor_type = row.get("anchor_type", "")
        node_id = f"{doc_id}:{anchor_id}"

        # Determine parent using "nearest preceding" containment
        parent_id = None

        if anchor_type == "preamble":
            parent_id = doc_root_id

        elif anchor_type == "section":
            parent_id = doc_root_id
            current_section = anchor_id
            current_box = None  # Reset box when entering new section

        elif anchor_type == "box":
            if current_section:
                parent_id = f"{doc_id}:{current_section}"
            else:
                parent_id = doc_root_id
            current_box = anchor_id

        elif anchor_type == "subsection":  # concept
            if current_box:
                parent_id = f"{doc_id}:{current_box}"
            elif current_section:
                parent_id = f"{doc_id}:{current_section}"
            else:
                parent_id = doc_root_id

        else:
            # Unknown type - attach to doc_root
            parent_id = doc_root_id

        # Add parent edge
        if parent_id and parent_id != node_id:
            edges.append(Edge(
                edge_id=generate_edge_id("parent_of", parent_id, node_id),
                source_node_id=parent_id,
                target_node_id=node_id,
                edge_type="parent_of",
                direction="directed",
                confidence=1.0,
                source_evidence=f"Structural: {anchor_type} under {parent_id.split(':')[-1]}",
                created_by="structural",
            ))

    logger.debug(f"Built {len(edges)} section hierarchy edges")
    return edges


def build_section_follows_edges(
    sections_df: pd.DataFrame,
    doc_id: str,
) -> List[Edge]:
    """
    Build follows edges between sections in reading order.

    Args:
        sections_df: Sections DataFrame.
        doc_id: Document ID prefix.

    Returns:
        List of follows Edge objects.
    """
    if sections_df.empty or len(sections_df) < 2:
        return []

    edges: List[Edge] = []

    # Sort sections by reading order
    ordered = sections_df.copy()
    ordered["_sort_key"] = ordered.apply(get_sort_key_for_section, axis=1)
    ordered = ordered.sort_values("_sort_key").reset_index(drop=True)

    prev_anchor_id = None
    prev_page = None

    for _, row in ordered.iterrows():
        anchor_id = row["anchor_id"]
        page = extract_page_from_row(row)
        node_id = f"{doc_id}:{anchor_id}"

        if prev_anchor_id and prev_anchor_id != anchor_id:
            edges.append(Edge(
                edge_id=generate_edge_id("follows", f"{doc_id}:{prev_anchor_id}", node_id),
                source_node_id=f"{doc_id}:{prev_anchor_id}",
                target_node_id=node_id,
                edge_type="follows",
                direction="directed",
                confidence=1.0,
                source_evidence=f"Reading order: page {page}",
                created_by="structural",
            ))

        prev_anchor_id = anchor_id
        prev_page = page

    logger.debug(f"Built {len(edges)} section follows edges")
    return edges


def build_anchor_paragraph_edges(
    paragraph_nodes_df: pd.DataFrame,
    doc_id: str,
    valid_anchor_node_ids: Set[str],
) -> List[Edge]:
    """
    Build parent_of edges from anchors to paragraphs.

    Guarantees: Every paragraph gets a parent_of edge.
    - If anchor_id is valid, use it
    - If anchor_id is missing or invalid, fall back to doc_root

    Args:
        paragraph_nodes_df: Paragraph nodes DataFrame.
        doc_id: Document ID prefix.
        valid_anchor_node_ids: Set of valid anchor node IDs.

    Returns:
        List of parent_of Edge objects.
    """
    if paragraph_nodes_df.empty:
        return []

    edges: List[Edge] = []
    doc_root_id = f"{doc_id}:doc_root"
    orphan_count = 0

    for _, para in paragraph_nodes_df.iterrows():
        para_node_id = para["node_id"]
        anchor_id = para.get("anchor_id")
        element_id = para.get("element_id")

        # Determine parent - fall back to doc_root if anchor invalid
        if anchor_id:
            anchor_node_id = f"{doc_id}:{anchor_id}"
            if anchor_node_id in valid_anchor_node_ids:
                parent_node_id = anchor_node_id
                evidence = f"Paragraph under {anchor_id}"
            else:
                # Anchor doesn't exist - orphan recovery
                parent_node_id = doc_root_id
                evidence = f"Orphan paragraph (missing anchor: {anchor_id})"
                orphan_count += 1
                logger.warning(f"Orphan paragraph {para_node_id}: anchor {anchor_id} not found")
        else:
            # No anchor_id at all - attach to doc_root
            parent_node_id = doc_root_id
            evidence = "Paragraph with no anchor_id"
            orphan_count += 1

        edges.append(Edge(
            edge_id=generate_edge_id("parent_of", parent_node_id, para_node_id),
            source_node_id=parent_node_id,
            target_node_id=para_node_id,
            edge_type="parent_of",
            direction="directed",
            confidence=1.0 if parent_node_id != doc_root_id else 0.5,  # Lower confidence for orphan recovery
            source_evidence=evidence,
            source_element_id=str(element_id) if element_id else None,
            created_by="structural",
        ))

    if orphan_count > 0:
        logger.info(f"Recovered {orphan_count} orphan paragraphs (attached to doc_root)")

    logger.debug(f"Built {len(edges)} anchor→paragraph edges")
    return edges


def build_paragraph_follows_edges(
    paragraph_nodes_df: pd.DataFrame,
    doc_id: str,
) -> List[Edge]:
    """
    Build follows edges between paragraphs within same anchor.

    Args:
        paragraph_nodes_df: Paragraph nodes DataFrame with anchor_id, reading_order.
        doc_id: Document ID prefix.

    Returns:
        List of follows Edge objects.
    """
    if paragraph_nodes_df.empty:
        return []

    edges: List[Edge] = []

    # Group by anchor and sort by reading order
    for anchor_id, group in paragraph_nodes_df.groupby("anchor_id"):
        sorted_paras = group.sort_values("reading_order").reset_index(drop=True)

        for i in range(len(sorted_paras) - 1):
            p1 = sorted_paras.iloc[i]
            p2 = sorted_paras.iloc[i + 1]

            edges.append(Edge(
                edge_id=generate_edge_id("follows", p1["node_id"], p2["node_id"]),
                source_node_id=p1["node_id"],
                target_node_id=p2["node_id"],
                edge_type="follows",
                direction="directed",
                confidence=1.0,
                source_evidence=f"Reading order within {anchor_id}",
                source_element_id=str(p1.get("element_id")) if p1.get("element_id") else None,
                created_by="structural",
            ))

    logger.debug(f"Built {len(edges)} paragraph follows edges")
    return edges


def build_in_section_edges(
    sections_df: pd.DataFrame,
    doc_id: str,
) -> List[Edge]:
    """
    Build in_section containment edges.

    Denormalized edges for fast "all boxes in section X" queries.
    Only emitted for subsection (concept) → section relationships.

    Args:
        sections_df: Sections DataFrame.
        doc_id: Document ID prefix.

    Returns:
        List of in_section Edge objects.
    """
    if sections_df.empty:
        return []

    edges: List[Edge] = []

    # Sort sections by reading order
    ordered = sections_df.copy()
    ordered["_sort_key"] = ordered.apply(get_sort_key_for_section, axis=1)
    ordered = ordered.sort_values("_sort_key").reset_index(drop=True)

    # Build anchor → section map
    anchor_to_section: Dict[str, str] = {}
    current_section = None

    for _, row in ordered.iterrows():
        anchor_id = row["anchor_id"]
        anchor_type = row.get("anchor_type", "")

        if anchor_type == "section":
            current_section = anchor_id
        elif anchor_type in ("box", "subsection") and current_section:
            anchor_to_section[anchor_id] = current_section

    # Emit in_section edges only for subsections (concepts)
    for anchor_id, section_id in anchor_to_section.items():
        # Get anchor type
        anchor_row = sections_df[sections_df["anchor_id"] == anchor_id]
        if anchor_row.empty:
            continue

        anchor_type = anchor_row.iloc[0].get("anchor_type", "")
        if anchor_type != "subsection":
            continue  # Only emit for concepts

        anchor_node_id = f"{doc_id}:{anchor_id}"
        section_node_id = f"{doc_id}:{section_id}"

        edges.append(Edge(
            edge_id=generate_edge_id("in_section", anchor_node_id, section_node_id),
            source_node_id=anchor_node_id,
            target_node_id=section_node_id,
            edge_type="in_section",
            direction="directed",
            confidence=1.0,
            source_evidence=f"Concept {anchor_id} in section {section_id}",
            created_by="structural",
        ))

    logger.debug(f"Built {len(edges)} in_section edges")
    return edges


# =============================================================================
# REFERENCE EDGE BUILDERS
# =============================================================================

def build_box_reference_edges(
    references_df: pd.DataFrame,
    valid_node_ids: Set[str],
    doc_id: str,
) -> List[Edge]:
    """
    Build references_box edges from extracted references.

    Deduplicates by (source_node_id, target_node_id) - only one edge per pair.

    Args:
        references_df: References DataFrame with source_anchor_id, target_anchor_id.
        valid_node_ids: Set of valid node IDs to filter against.
        doc_id: Document ID prefix.

    Returns:
        List of references_box Edge objects (deduplicated).
    """
    if references_df.empty:
        return []

    edges: List[Edge] = []
    seen_pairs: Set[Tuple[str, str]] = set()  # (source, target) pairs

    for _, ref in references_df.iterrows():
        # Only process internal box references
        if ref.get("ref_type") != "box_reference":
            continue
        if not ref.get("target_exists", False):
            continue

        source_anchor = ref.get("source_anchor_id")
        target_anchor = ref.get("target_anchor_id")

        if not source_anchor or not target_anchor:
            continue

        source_node_id = f"{doc_id}:{source_anchor}"
        target_node_id = f"{doc_id}:{target_anchor}"

        # Validate both nodes exist
        if source_node_id not in valid_node_ids:
            continue
        if target_node_id not in valid_node_ids:
            continue

        # Dedupe by (source, target) pair
        pair = (source_node_id, target_node_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Extract provenance fields
        source_page = ref.get("page")
        source_pages = [int(source_page)] if source_page is not None and not pd.isna(source_page) else []

        source_bbox = None
        if "geom_x0" in ref.index:
            try:
                source_bbox = [
                    float(ref["geom_x0"]),
                    float(ref["geom_y0"]),
                    float(ref["geom_x1"]),
                    float(ref["geom_y1"]),
                ]
            except (ValueError, TypeError, KeyError):
                pass

        edges.append(Edge(
            edge_id=generate_edge_id("references_box", source_node_id, target_node_id),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type="references_box",
            direction="directed",
            confidence=float(ref.get("confidence", 0.9)),
            source_evidence=ref.get("evidence_text") or ref.get("ref_text", ""),
            source_element_id=str(ref.get("source_element_id")) if ref.get("source_element_id") else None,
            source_pages=source_pages,
            source_bbox=source_bbox,
            created_by=ref.get("created_by", "regex"),
        ))

    logger.debug(f"Built {len(edges)} box reference edges (deduped from {len(references_df)} refs)")
    return edges


def build_same_group_edges(
    anchors_df: pd.DataFrame,
    doc_id: str,
) -> List[Edge]:
    """
    Build same_group edges for grouped boxes (e.g., Boxes 14-16).

    Creates bidirectional edges between all members of each group.

    Args:
        anchors_df: Anchors DataFrame with anchor_id, group_id, is_grouped.
        doc_id: Document ID prefix.

    Returns:
        List of same_group Edge objects.
    """
    if anchors_df.empty:
        return []

    if "group_id" not in anchors_df.columns or "is_grouped" not in anchors_df.columns:
        return []

    edges: List[Edge] = []

    grouped = anchors_df[anchors_df["is_grouped"] == True]
    if grouped.empty:
        return []

    for group_id in grouped["group_id"].dropna().unique():
        group_members = grouped[grouped["group_id"] == group_id]["anchor_id"].tolist()

        # Get source_element_id from first member
        group_source_elem = None
        if "source_element_id" in grouped.columns:
            first_member = grouped[grouped["group_id"] == group_id].iloc[0]
            group_source_elem = first_member.get("source_element_id")

        # Create edges between all pairs
        for i, a1 in enumerate(group_members):
            for a2 in group_members[i + 1:]:
                node1 = f"{doc_id}:{a1}"
                node2 = f"{doc_id}:{a2}"

                edges.append(Edge(
                    edge_id=generate_edge_id("same_group", node1, node2),
                    source_node_id=node1,
                    target_node_id=node2,
                    edge_type="same_group",
                    direction="bidirectional",
                    confidence=1.0,
                    source_evidence=f"Grouped header: {group_id}",
                    source_element_id=str(group_source_elem) if group_source_elem else None,
                    created_by="structural",
                ))

    logger.debug(f"Built {len(edges)} same_group edges")
    return edges


# =============================================================================
# TYPED EDGE BUILDERS
# =============================================================================

def _build_paragraph_semantic_edges(
    paragraph_nodes_df: pd.DataFrame,
    valid_box_keys: Set[str],
    doc_id: str,
    form_id: Optional[str],
    anchor_context: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[List[Edge], Dict[str, int]]:
    """
    Build concept→box semantic edges from paragraphs.

    The paragraph is the rule-holder.
    Edges: excludes, applies_if, defines, qualifies, portion_of

    Returns:
        Tuple of (edges list, stats dict)
    """
    from vaas.semantic.typed_edges import extract_concept_to_box_edges

    edges: List[Edge] = []
    stats = {
        "paragraphs_scanned": 0,
        "paragraphs_with_edges": 0,
        "edges_by_type": {},
    }

    if paragraph_nodes_df is None or paragraph_nodes_df.empty:
        return edges, stats

    for _, para in paragraph_nodes_df.iterrows():
        stats["paragraphs_scanned"] += 1

        para_node_id = para.get("node_id", "")
        para_text = para.get("text", "") or ""
        parent_box_key = (para.get("box_key", "") or "").lower() or None
        anchor_id = para.get("anchor_id")
        anchor_key = str(anchor_id) if anchor_id else ""
        context = anchor_context.get(anchor_key, {}) if anchor_context else {}
        exception_term = (
            exception_term_from_context(context.get("label"))
            or exception_term_from_context(context.get("full_text"))
        )
        paragraph_kind = str(para.get("paragraph_kind") or "")
        if paragraph_kind != "list":
            exception_term = None

        if not para_text.strip() or not para_node_id:
            continue

        # Get reading_order for precedence computation
        reading_order = para.get("reading_order")
        if reading_order is None or pd.isna(reading_order):
            reading_order = 0
        else:
            reading_order = int(reading_order)

        # Extract using clean public API (full node_id, no reconstruction)
        candidates = extract_concept_to_box_edges(
            source_node_id=para_node_id,
            text=para_text,
            valid_box_keys=valid_box_keys,
            parent_box_key=parent_box_key,
            form_id=form_id,
            exception_term=exception_term,
        )

        if candidates:
            stats["paragraphs_with_edges"] += 1

        for te in candidates:
            target_node_id = f"{doc_id}:box_{te.target_box_key}"

            # Count by type
            stats["edges_by_type"][te.edge_type] = stats["edges_by_type"].get(te.edge_type, 0) + 1

            # Compute precedence using rule_class + position tie-breaker
            # Rule class completely dominates; position only breaks ties within class
            sentence_idx = te.sentence_idx if te.sentence_idx is not None else 0
            precedence = compute_precedence(te.rule_class, reading_order, sentence_idx)

            edges.append(Edge(
                edge_id=generate_edge_id(te.edge_type, para_node_id, target_node_id),
                source_node_id=para_node_id,
                target_node_id=target_node_id,
                edge_type=te.edge_type,
                direction="directed",
                confidence=te.confidence,
                source_evidence=te.evidence_text,
                source_element_id=str(para.get("element_id")) if para.get("element_id") else None,
                created_by="regex",
                pattern_matched=te.pattern_matched,
                polarity=te.polarity,
                evidence_sentence_idx=te.sentence_idx,
                evidence_char_start=te.sentence_char_start,
                evidence_char_end=te.sentence_char_end,
                rule_class=te.rule_class,
                precedence=precedence,
            ))

    return edges, stats


def _build_box_dependency_edges(
    sections_df: pd.DataFrame,
    valid_box_keys: Set[str],
    doc_id: str,
) -> Tuple[List[Edge], Dict[str, int]]:
    """
    Build box→box dependency edges from box sections.

    These describe relationships between boxes.
    Edges: aggregates, requires, includes

    Returns:
        Tuple of (edges list, stats dict)
    """
    from vaas.semantic.typed_edges import extract_box_to_box_edges

    edges: List[Edge] = []
    stats = {
        "box_sections_scanned": 0,
        "edges_by_type": {},
    }

    if sections_df is None or sections_df.empty:
        return edges, stats

    # Build section order for precedence (based on reading order)
    box_sections = sections_df[sections_df["anchor_type"] == "box"].copy()
    if not box_sections.empty:
        box_sections["_sort_key"] = box_sections.apply(get_sort_key_for_section, axis=1)
        box_sections = box_sections.sort_values("_sort_key").reset_index(drop=True)
        section_order = {row["anchor_id"]: idx for idx, row in box_sections.iterrows()}
    else:
        section_order = {}

    for _, section in sections_df.iterrows():
        anchor_type = section.get("anchor_type", "")
        if anchor_type != "box":
            continue

        stats["box_sections_scanned"] += 1

        anchor_id = section["anchor_id"]
        source_box_key = (section.get("box_key", "") or "").lower() or None
        full_text = section.get("full_text", "") or ""

        if not source_box_key or not full_text.strip():
            continue

        source_node_id = f"{doc_id}:{anchor_id}"

        # Extract using clean public API
        candidates = extract_box_to_box_edges(
            source_node_id=source_node_id,
            source_box_key=source_box_key,
            text=full_text,
            valid_box_keys=valid_box_keys,
        )

        for te in candidates:
            target_node_id = f"{doc_id}:box_{te.target_box_key}"

            # Skip self-edges (box→same box)
            if source_node_id == target_node_id:
                continue

            stats["edges_by_type"][te.edge_type] = stats["edges_by_type"].get(te.edge_type, 0) + 1

            # Compute precedence using rule_class + section order as tie-breaker
            sentence_idx = te.sentence_idx if te.sentence_idx is not None else 0
            section_idx = section_order.get(anchor_id, 0)
            precedence = compute_precedence(te.rule_class, section_idx, sentence_idx)

            edges.append(Edge(
                edge_id=generate_edge_id(te.edge_type, source_node_id, target_node_id),
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=te.edge_type,
                direction="directed",
                confidence=te.confidence,
                source_evidence=te.evidence_text,
                source_element_id=None,  # Section-level, no specific element
                created_by="regex",
                pattern_matched=te.pattern_matched,
                polarity=te.polarity,
                evidence_sentence_idx=te.sentence_idx,
                evidence_char_start=te.sentence_char_start,
                evidence_char_end=te.sentence_char_end,
                rule_class=te.rule_class,
                precedence=precedence,
            ))

    return edges, stats


def _build_anchor_semantic_edges(
    sections_df: pd.DataFrame,
    valid_box_keys: Set[str],
    doc_id: str,
    form_id: Optional[str],
) -> Tuple[List[Edge], Dict[str, int]]:
    """
    Build concept→box semantic edges from anchor sections with single-source text.

    This captures definition-style anchors that have a single backing element,
    allowing sentence offsets to align with element-level evidence.
    """
    from vaas.semantic.typed_edges import extract_concept_to_box_edges

    edges: List[Edge] = []
    stats = {
        "anchors_scanned": 0,
        "anchors_with_edges": 0,
        "edges_by_type": {},
    }

    if sections_df is None or sections_df.empty:
        return edges, stats

    sections_sorted = sections_df.copy()
    sections_sorted["_sort_key"] = sections_sorted.apply(get_sort_key_for_section, axis=1)
    sections_sorted = sections_sorted.sort_values("_sort_key").reset_index(drop=True)
    section_order = {row["anchor_id"]: idx for idx, row in sections_sorted.iterrows()}

    for _, section in sections_df.iterrows():
        anchor_type = section.get("anchor_type", "")
        if anchor_type == "box":
            continue
        if (section.get("concept_role") or "") != "definition":
            continue

        element_ids = section.get("element_ids") or []
        if not isinstance(element_ids, (list, tuple)) or len(element_ids) != 1:
            continue

        full_text = section.get("full_text", "") or ""
        if not full_text.strip():
            continue

        stats["anchors_scanned"] += 1
        anchor_id = section.get("anchor_id")
        source_node_id = f"{doc_id}:{anchor_id}"
        candidates = extract_concept_to_box_edges(
            source_node_id=source_node_id,
            text=full_text,
            valid_box_keys=valid_box_keys,
            form_id=form_id,
        )

        if candidates:
            stats["anchors_with_edges"] += 1

        for te in candidates:
            target_node_id = f"{doc_id}:box_{te.target_box_key}"
            stats["edges_by_type"][te.edge_type] = stats["edges_by_type"].get(te.edge_type, 0) + 1
            sentence_idx = te.sentence_idx if te.sentence_idx is not None else 0
            section_idx = section_order.get(anchor_id, 0)
            precedence = compute_precedence(te.rule_class, section_idx, sentence_idx)

            edges.append(Edge(
                edge_id=generate_edge_id(te.edge_type, source_node_id, target_node_id),
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=te.edge_type,
                direction="directed",
                confidence=te.confidence,
                source_evidence=te.evidence_text,
                source_element_id=str(element_ids[0]),
                created_by="regex",
                pattern_matched=te.pattern_matched,
                polarity=te.polarity,
                evidence_sentence_idx=te.sentence_idx,
                evidence_char_start=te.sentence_char_start,
                evidence_char_end=te.sentence_char_end,
                rule_class=te.rule_class,
                precedence=precedence,
            ))

    return edges, stats


def _dedupe_edges(edges: List[Edge]) -> List[Edge]:
    """
    Dedupe edges by (edge_type, source_node_id, target_node_id, sentence, span).

    Keeps the highest-confidence edge for each unique key.
    """
    if not edges:
        return []

    # Group by key, keep highest confidence
    best: Dict[Tuple[str, str, str, Optional[int], Optional[int], Optional[int]], Edge] = {}
    for e in edges:
        key = (
            e.edge_type,
            e.source_node_id,
            e.target_node_id,
            e.evidence_sentence_idx,
            e.evidence_char_start,
            e.evidence_char_end,
        )
        if key not in best or e.confidence > best[key].confidence:
            best[key] = e

    return list(best.values())


def build_typed_edges(
    sections_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    doc_id: str,
    paragraph_nodes_df: Optional[pd.DataFrame] = None,
    form_id: Optional[str] = None,
) -> Tuple[List[Edge], Dict[str, int]]:
    """
    Build typed semantic edges (excludes, includes, applies_if, etc.).

    Phase B architecture:
    - Concept→box edges (excludes, applies_if, defines, qualifies, portion_of)
      are extracted from PARAGRAPHS - the paragraph is the rule-holder.
    - Box→box edges (requires, includes) are extracted from SECTIONS -
      these describe relationships between boxes, not rules.

    Deduplication: keeps highest-confidence edge per (type, source, target).

    Args:
        sections_df: Sections DataFrame with anchor_id, anchor_type, box_key, full_text.
        anchors_df: Anchors DataFrame for valid box keys.
        doc_id: Document ID prefix.
        paragraph_nodes_df: Optional paragraph nodes DataFrame for paragraph-scoped extraction.

    Returns:
        List of typed Edge objects (deduplicated).
    """
    # Build valid box keys
    valid_box_keys: Set[str] = set()
    if not anchors_df.empty and "anchor_type" in anchors_df.columns:
        box_anchors = anchors_df[anchors_df["anchor_type"] == "box"]
        if "box_key" in box_anchors.columns:
            valid_box_keys = set(box_anchors["box_key"].str.lower().dropna())

    anchor_context: Dict[str, Dict[str, str]] = {}
    if not sections_df.empty:
        for _, row in sections_df.iterrows():
            anchor_id = row.get("anchor_id")
            if not anchor_id:
                continue
            anchor_context[str(anchor_id)] = {
                "label": str(row.get("label") or ""),
                "full_text": str(row.get("full_text") or ""),
            }

    # Build edges from sources
    para_edges, para_stats = _build_paragraph_semantic_edges(
        paragraph_nodes_df,
        valid_box_keys,
        doc_id,
        form_id,
        anchor_context=anchor_context,
    )
    anchor_edges, anchor_stats = _build_anchor_semantic_edges(
        sections_df,
        valid_box_keys,
        doc_id,
        form_id,
    )
    box_edges, box_stats = _build_box_dependency_edges(
        sections_df, valid_box_keys, doc_id
    )

    # Combine and dedupe
    typed_edges_raw = para_edges + anchor_edges + box_edges
    all_edges = typed_edges_raw
    before_dedupe = len(all_edges)
    all_edges = _dedupe_edges(all_edges)
    deduped_count = before_dedupe - len(all_edges)

    # Logging with useful metrics
    logger.info(
        f"Typed edges: paragraphs scanned={para_stats['paragraphs_scanned']}, "
        f"with_edges={para_stats['paragraphs_with_edges']}, "
        f"anchors scanned={anchor_stats['anchors_scanned']}, "
        f"anchors_with_edges={anchor_stats['anchors_with_edges']}, "
        f"box_sections={box_stats['box_sections_scanned']}"
    )

    type_counts = {}
    for e in all_edges:
        type_counts[e.edge_type] = type_counts.get(e.edge_type, 0) + 1

    if type_counts:
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
        logger.info(f"Typed edge counts: {counts_str}")

    if deduped_count > 0:
        logger.info(f"Deduped {deduped_count} duplicate edges")

    logger.debug(f"Built {len(all_edges)} typed edges total")
    stats_summary = {
        "paragraphs_scanned": para_stats["paragraphs_scanned"],
        "paragraphs_with_edges": para_stats["paragraphs_with_edges"],
        "anchors_scanned": anchor_stats["anchors_scanned"],
        "anchors_with_edges": anchor_stats["anchors_with_edges"],
        "box_sections_scanned": box_stats["box_sections_scanned"],
    }
    return all_edges, stats_summary


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def build_all_edges(
    sections_df: pd.DataFrame,
    paragraph_nodes_df: pd.DataFrame,
    references_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    doc_id: str,
    graph_nodes_df: pd.DataFrame,
    include_typed_edges: bool = True,
    form_id: Optional[str] = None,
) -> EdgeBuildResult:
    """
    Build all edge types and combine.

    Args:
        sections_df: Sections DataFrame.
        paragraph_nodes_df: Paragraph nodes DataFrame.
        references_df: References DataFrame.
        anchors_df: Anchors DataFrame.
        doc_id: Document ID.
        graph_nodes_df: Graph nodes DataFrame (for filtering).
        include_typed_edges: Whether to build typed semantic edges.

    Returns:
        EdgeBuildResult with combined edges and counts.
    """
    all_edges: List[Edge] = []
    edge_counts: Dict[str, int] = {}

    # Get valid node IDs for filtering
    valid_node_ids = set(graph_nodes_df["node_id"].astype(str)) if not graph_nodes_df.empty else set()

    # Get valid anchor node IDs specifically
    valid_anchor_node_ids = set()
    if not sections_df.empty:
        for aid in sections_df["anchor_id"]:
            valid_anchor_node_ids.add(f"{doc_id}:{aid}")
    valid_anchor_node_ids.add(f"{doc_id}:doc_root")

    # 1. Section hierarchy (parent_of)
    hierarchy_edges = build_section_hierarchy_edges(sections_df, doc_id)
    all_edges.extend(hierarchy_edges)
    edge_counts["parent_of_section"] = len(hierarchy_edges)

    # 2. Section follows (reading order)
    section_follows = build_section_follows_edges(sections_df, doc_id)
    all_edges.extend(section_follows)
    edge_counts["follows_section"] = len(section_follows)

    # 3. Anchor → paragraph edges
    anchor_para_edges = build_anchor_paragraph_edges(
        paragraph_nodes_df, doc_id, valid_anchor_node_ids
    )
    all_edges.extend(anchor_para_edges)
    edge_counts["parent_of_para"] = len(anchor_para_edges)

    # 4. Paragraph follows edges
    para_follows = build_paragraph_follows_edges(paragraph_nodes_df, doc_id)
    all_edges.extend(para_follows)
    edge_counts["follows_para"] = len(para_follows)

    # 5. In-section containment
    in_section_edges = build_in_section_edges(sections_df, doc_id)
    all_edges.extend(in_section_edges)
    edge_counts["in_section"] = len(in_section_edges)

    # 6. Box reference edges
    box_ref_edges = build_box_reference_edges(references_df, valid_node_ids, doc_id)
    all_edges.extend(box_ref_edges)
    edge_counts["references_box"] = len(box_ref_edges)

    # 7. Same group edges
    same_group_edges = build_same_group_edges(anchors_df, doc_id)
    all_edges.extend(same_group_edges)
    edge_counts["same_group"] = len(same_group_edges)

    # 8. Typed semantic edges (Phase B: paragraph-scoped)
    typed_edges: List[Edge] = []
    if include_typed_edges:
        typed_edges, typed_stats = build_typed_edges(
            sections_df=sections_df,
            anchors_df=anchors_df,
            doc_id=doc_id,
            paragraph_nodes_df=paragraph_nodes_df,
            form_id=form_id,
        )
        all_edges.extend(typed_edges)

        # Count by type
        for edge in typed_edges:
            key = f"typed_{edge.edge_type}"
            edge_counts[key] = edge_counts.get(key, 0) + 1

    # Convert to DataFrame
    if all_edges:
        edges_df = pd.DataFrame([e.to_dict() for e in all_edges])
    else:
        edges_df = pd.DataFrame()

    # Filter edges by active nodes
    edges_filtered = 0
    if not edges_df.empty and valid_node_ids:
        before = len(edges_df)
        edges_df = edges_df[
            edges_df["source_node_id"].isin(valid_node_ids) &
            edges_df["target_node_id"].isin(valid_node_ids)
        ].reset_index(drop=True)
        edges_filtered = before - len(edges_df)
        if edges_filtered > 0:
            logger.info(f"Filtered {edges_filtered} edges referencing non-existent nodes")

    # Compute final counts by edge_type
    final_counts: Dict[str, int] = {}
    if not edges_df.empty:
        final_counts = edges_df["edge_type"].value_counts().to_dict()

    logger.info(f"Built {len(edges_df)} total edges")

    return EdgeBuildResult(
        edges_df=edges_df,
        edge_counts=final_counts,
        edges_filtered=edges_filtered,
        typed_edges=typed_edges,
    )
