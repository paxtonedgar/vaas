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


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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
        created_by: How edge was created ("structural", "regex", "llm").
        pattern_matched: For typed edges, the pattern that matched.
        polarity: For typed edges, "positive" or "negative".
        evidence_sentence_idx: Sentence index within section (for sentence-gated edges).
        evidence_char_start: Character offset start (relative to section full_text).
        evidence_char_end: Character offset end (relative to section full_text).
    """

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    direction: str = "directed"
    confidence: float = 1.0
    source_evidence: Optional[str] = None
    source_element_id: Optional[str] = None
    created_by: str = "structural"
    pattern_matched: Optional[str] = None
    polarity: Optional[str] = None
    # Sentence-level provenance for semantic edges
    evidence_sentence_idx: Optional[int] = None
    evidence_char_start: Optional[int] = None
    evidence_char_end: Optional[int] = None

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
            "created_by": self.created_by,
            "pattern_matched": self.pattern_matched,
            "polarity": self.polarity,
            "evidence_sentence_idx": self.evidence_sentence_idx,
            "evidence_char_start": self.evidence_char_start,
            "evidence_char_end": self.evidence_char_end,
        }


@dataclass
class EdgeBuildResult:
    """
    Result of edge building.

    Attributes:
        edges_df: DataFrame with all edges.
        edge_counts: Dictionary of edge type to count.
        edges_filtered: Number of edges filtered (referencing non-existent nodes).
    """

    edges_df: pd.DataFrame
    edge_counts: Dict[str, int] = field(default_factory=dict)
    edges_filtered: int = 0

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

    for _, para in paragraph_nodes_df.iterrows():
        para_node_id = para["node_id"]
        anchor_id = para.get("anchor_id")
        element_id = para.get("element_id")

        if not anchor_id:
            continue

        anchor_node_id = f"{doc_id}:{anchor_id}"

        # Verify anchor exists
        if anchor_node_id not in valid_anchor_node_ids:
            continue

        edges.append(Edge(
            edge_id=generate_edge_id("parent_of", anchor_node_id, para_node_id),
            source_node_id=anchor_node_id,
            target_node_id=para_node_id,
            edge_type="parent_of",
            direction="directed",
            confidence=1.0,
            source_evidence=f"Paragraph under {anchor_id}",
            source_element_id=str(element_id) if element_id else None,
            created_by="structural",
        ))

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

    Args:
        references_df: References DataFrame with source_anchor_id, target_anchor_id.
        valid_node_ids: Set of valid node IDs to filter against.
        doc_id: Document ID prefix.

    Returns:
        List of references_box Edge objects.
    """
    if references_df.empty:
        return []

    edges: List[Edge] = []

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

        edges.append(Edge(
            edge_id=generate_edge_id("references_box", source_node_id, target_node_id),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type="references_box",
            direction="directed",
            confidence=float(ref.get("confidence", 0.9)),
            source_evidence=ref.get("evidence_text") or ref.get("ref_text", ""),
            source_element_id=str(ref.get("source_element_id")) if ref.get("source_element_id") else None,
            created_by=ref.get("created_by", "regex"),
        ))

    logger.debug(f"Built {len(edges)} box reference edges")
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

        if not para_text.strip() or not para_node_id:
            continue

        # Extract using clean public API (full node_id, no reconstruction)
        candidates = extract_concept_to_box_edges(
            source_node_id=para_node_id,
            text=para_text,
            valid_box_keys=valid_box_keys,
            parent_box_key=parent_box_key,
        )

        if candidates:
            stats["paragraphs_with_edges"] += 1

        for te in candidates:
            target_node_id = f"{doc_id}:box_{te.target_box_key}"

            # Count by type
            stats["edges_by_type"][te.edge_type] = stats["edges_by_type"].get(te.edge_type, 0) + 1

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
    Edges: requires, includes

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
            ))

    return edges, stats


def _dedupe_edges(edges: List[Edge]) -> List[Edge]:
    """
    Dedupe edges by (edge_type, source_node_id, target_node_id).

    Keeps the highest-confidence edge for each unique triple.
    """
    if not edges:
        return []

    # Group by key, keep highest confidence
    best: Dict[Tuple[str, str, str], Edge] = {}
    for e in edges:
        key = (e.edge_type, e.source_node_id, e.target_node_id)
        if key not in best or e.confidence > best[key].confidence:
            best[key] = e

    return list(best.values())


def build_typed_edges(
    sections_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    doc_id: str,
    paragraph_nodes_df: Optional[pd.DataFrame] = None,
) -> List[Edge]:
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

    # Build edges from two sources
    para_edges, para_stats = _build_paragraph_semantic_edges(
        paragraph_nodes_df, valid_box_keys, doc_id
    )
    box_edges, box_stats = _build_box_dependency_edges(
        sections_df, valid_box_keys, doc_id
    )

    # Combine and dedupe
    all_edges = para_edges + box_edges
    before_dedupe = len(all_edges)
    all_edges = _dedupe_edges(all_edges)
    deduped_count = before_dedupe - len(all_edges)

    # Logging with useful metrics
    logger.info(
        f"Typed edges: paragraphs scanned={para_stats['paragraphs_scanned']}, "
        f"with_edges={para_stats['paragraphs_with_edges']}, "
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
    return all_edges


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
    if include_typed_edges:
        typed_edges = build_typed_edges(
            sections_df=sections_df,
            anchors_df=anchors_df,
            doc_id=doc_id,
            paragraph_nodes_df=paragraph_nodes_df,
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
    )


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def build_edges_legacy(
    sections_df: pd.DataFrame,
    paragraph_nodes_df: pd.DataFrame,
    references_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    graph_nodes_df: pd.DataFrame,
    doc_id: str = "1099div_filer",
) -> pd.DataFrame:
    """
    Legacy-compatible wrapper for build_all_edges.

    Matches original run_pipeline.py behavior for drop-in replacement.

    Args:
        sections_df: Sections DataFrame.
        paragraph_nodes_df: Paragraph nodes DataFrame.
        references_df: References DataFrame.
        anchors_df: Anchors DataFrame.
        graph_nodes_df: Graph nodes DataFrame.
        doc_id: Document ID.

    Returns:
        Edges DataFrame.
    """
    result = build_all_edges(
        sections_df=sections_df,
        paragraph_nodes_df=paragraph_nodes_df,
        references_df=references_df,
        anchors_df=anchors_df,
        doc_id=doc_id,
        graph_nodes_df=graph_nodes_df,
        include_typed_edges=True,
    )

    # Print summary (matching original behavior)
    print(f"\nGraph edges: {len(result.edges_df)}")
    if result.edges_filtered > 0:
        print(f"Dropped {result.edges_filtered} edges referencing pruned/missing nodes")

    if not result.edges_df.empty:
        print(f"\n--- Edge Types ---")
        print(result.edges_df["edge_type"].value_counts().to_string())

    return result.edges_df
