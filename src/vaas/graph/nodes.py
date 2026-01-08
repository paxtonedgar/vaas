"""
Graph node construction for knowledge graph.

This module builds nodes for the knowledge graph. Node types include:
- doc_root: Document root node (one per document)
- section_nodes: Anchor-level nodes (box, section, subsection/concept)
- paragraph_nodes: Element-level nodes (fine-grained content)

Node Construction Strategy:
1. Create doc_root node as graph entry point
2. Create section nodes from sections_df (one per anchor)
3. Create paragraph nodes from elements_df (one per non-header element)
4. Combine into single graph_nodes DataFrame
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from vaas.constants import (
    ROLE_BOX_HEADER,
    ROLE_SECTION_HEADER,
    ROLE_SUBSECTION_HEADER,
    ROLE_LIST_BLOCK,
    ROLE_PAGE_ARTIFACT,
    ROLE_BODY_TEXT,
    DEFAULT_SKIP_ROLES,
    HEADER_ROLES,
    ANCHOR_TYPE_TO_NODE_TYPE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Node:
    """
    A single graph node.

    Attributes:
        node_id: Unique identifier (format: "doc_id:anchor_id" or "doc_id:el_{element_id}").
        doc_id: Document identifier.
        node_type: Type of node ("doc_root", "box_section", "section", "concept", "paragraph").
        anchor_id: Anchor ID for section nodes, parent anchor ID for paragraph nodes.
        box_key: Box key for box nodes (e.g., "1a", "2e").
        label: Short display label.
        text: Full text content.
        pages: List of page numbers.
        bbox: Bounding box [x0, y0, x1, y1].
        element_id: Element ID for paragraph nodes.
        element_count: Number of elements (1 for paragraphs).
        char_count: Character count of text.
        reading_order: Reading order within anchor (for paragraphs).
        paragraph_kind: "body" or "list" for paragraph nodes.
        anchor_type: Anchor type of parent (for paragraphs).
        concept_role: Role for concept/subsection nodes.
    """

    node_id: str
    doc_id: str
    node_type: str
    anchor_id: Optional[str] = None
    box_key: Optional[str] = None
    label: str = ""
    text: str = ""
    pages: List[int] = field(default_factory=list)
    bbox: Optional[List[float]] = None
    element_id: Optional[str] = None
    element_count: int = 0
    char_count: int = 0
    reading_order: Optional[int] = None
    paragraph_kind: Optional[str] = None
    anchor_type: Optional[str] = None
    concept_role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "node_id": self.node_id,
            "doc_id": self.doc_id,
            "node_type": self.node_type,
            "anchor_id": self.anchor_id,
            "box_key": self.box_key,
            "label": self.label,
            "text": self.text,
            "pages": self.pages,
            "bbox": self.bbox,
            "element_id": self.element_id,
            "element_count": self.element_count,
            "char_count": self.char_count,
            "reading_order": self.reading_order,
            "paragraph_kind": self.paragraph_kind,
            "anchor_type": self.anchor_type,
            "concept_role": self.concept_role,
        }


@dataclass
class NodeBuildResult:
    """
    Result of node building.

    Attributes:
        nodes_df: DataFrame with all nodes.
        paragraph_nodes_df: DataFrame with only paragraph nodes (for edge building).
        section_node_count: Number of section-level nodes.
        paragraph_node_count: Number of paragraph-level nodes.
    """

    nodes_df: pd.DataFrame
    paragraph_nodes_df: pd.DataFrame
    section_node_count: int = 0
    paragraph_node_count: int = 0

    @property
    def total(self) -> int:
        """Total number of nodes (including doc_root)."""
        return self.section_node_count + self.paragraph_node_count + 1

    def __repr__(self) -> str:
        return (
            f"NodeBuildResult(total={self.total}, "
            f"sections={self.section_node_count}, paragraphs={self.paragraph_node_count})"
        )


# =============================================================================
# NODE ID GENERATION
# =============================================================================

def generate_node_id(doc_id: str, anchor_id: str) -> str:
    """
    Generate node ID for section-level nodes.

    Args:
        doc_id: Document identifier.
        anchor_id: Anchor identifier.

    Returns:
        Node ID in format "doc_id:anchor_id".
    """
    return f"{doc_id}:{anchor_id}"


def generate_paragraph_node_id(doc_id: str, element_id: str) -> str:
    """
    Generate node ID for paragraph-level nodes.

    Args:
        doc_id: Document identifier.
        element_id: Element identifier (may or may not have doc_id prefix).

    Returns:
        Node ID in format "doc_id:el_{element_id}".
    """
    # If element_id already has doc_id prefix, strip it to avoid duplication
    # e.g., "1099div_filer:1:2:seg0" -> "1:2:seg0"
    if element_id.startswith(f"{doc_id}:"):
        element_id = element_id[len(doc_id) + 1:]
    return f"{doc_id}:el_{element_id}"


# =============================================================================
# NODE TYPE DETERMINATION
# =============================================================================

def get_node_type_for_section(anchor_id: str, anchor_type: str) -> str:
    """
    Determine node_type for a section based on anchor_id and anchor_type.

    Args:
        anchor_id: Anchor identifier.
        anchor_type: Type of anchor ("box", "section", "subsection").

    Returns:
        Node type string.
    """
    # Special cases
    if anchor_id == "preamble":
        return "preamble"
    if anchor_id == "unassigned":
        return "unassigned"

    # Map anchor_type to node_type
    return ANCHOR_TYPE_TO_NODE_TYPE.get(anchor_type, "section")


# =============================================================================
# NODE BUILDERS
# =============================================================================

def build_doc_root_node(
    doc_id: str,
    label: str = "Document Root",
) -> Node:
    """
    Create the document root node.

    Args:
        doc_id: Document identifier.
        label: Label for the root node.

    Returns:
        Document root Node.
    """
    return Node(
        node_id=f"{doc_id}:doc_root",
        doc_id=doc_id,
        node_type="doc_root",
        anchor_id="doc_root",
        box_key="",
        label=label,
        text="",
        pages=[],
        bbox=None,
        element_count=0,
        char_count=0,
    )


def build_section_nodes(
    sections_df: pd.DataFrame,
    doc_id: str,
) -> List[Node]:
    """
    Create nodes for all sections (box, section, subsection).

    Args:
        sections_df: Sections DataFrame with anchor_id, anchor_type, etc.
        doc_id: Document identifier.

    Returns:
        List of section Node objects.
    """
    if sections_df.empty:
        return []

    nodes: List[Node] = []

    for _, section in sections_df.iterrows():
        anchor_id = section["anchor_id"]
        anchor_type = section.get("anchor_type", "")

        # Determine node type
        node_type = get_node_type_for_section(anchor_id, anchor_type)

        # Get concept_role for subsection/concept nodes
        concept_role = None
        if anchor_type == "subsection":
            concept_role = section.get("concept_role")

        # Normalize pages to list
        pages = section.get("pages", [])
        if isinstance(pages, (int, float)):
            pages = [int(pages)]
        elif not isinstance(pages, list):
            pages = list(pages) if pages is not None else []

        # Normalize bbox
        bbox = section.get("bbox")
        if bbox is not None and not isinstance(bbox, list):
            bbox = list(bbox) if hasattr(bbox, '__iter__') else None

        nodes.append(Node(
            node_id=generate_node_id(doc_id, anchor_id),
            doc_id=doc_id,
            node_type=node_type,
            anchor_id=anchor_id,
            box_key=section.get("box_key") or "",
            label=section.get("label", "") or "",
            text=section.get("full_text", "") or "",
            pages=pages,
            bbox=bbox,
            element_count=int(section.get("element_count", 0) or 0),
            char_count=int(section.get("char_count", 0) or 0),
            concept_role=concept_role,
            anchor_type=anchor_type,
        ))

    logger.debug(f"Built {len(nodes)} section nodes")
    return nodes


def build_paragraph_nodes(
    elements_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    doc_id: str,
    skip_roles: Optional[Set[str]] = None,
) -> List[Node]:
    """
    Create paragraph-level nodes from elements.

    Skips:
    - Elements with roles in skip_roles (default: PageArtifact)
    - Header elements (BoxHeader, SectionHeader, SubsectionHeader)
    - Empty text elements
    - Unassigned or invalid anchor_id

    Args:
        elements_df: Elements DataFrame with element_id, anchor_id, text, role, etc.
        sections_df: Sections DataFrame for anchor metadata lookup.
        doc_id: Document identifier.
        skip_roles: Set of roles to skip (default: {PageArtifact}).

    Returns:
        List of paragraph Node objects.
    """
    if elements_df.empty:
        return []

    if skip_roles is None:
        skip_roles = DEFAULT_SKIP_ROLES

    # Build anchor_id -> metadata lookup from sections_df
    anchor_meta: Dict[str, Dict[str, str]] = {}
    if not sections_df.empty:
        for _, sec in sections_df.iterrows():
            aid = sec.get("anchor_id")
            if aid:
                anchor_meta[aid] = {
                    "anchor_type": sec.get("anchor_type", "") or "",
                    "box_key": sec.get("box_key", "") or "",
                }

    nodes: List[Node] = []

    for _, elem in elements_df.iterrows():
        role = elem.get("role", "")

        # Skip artifacts
        if role in skip_roles:
            continue

        # Skip header elements - they're already represented as anchor nodes
        if role in HEADER_ROLES:
            continue

        # Skip empty elements
        text = elem.get("text", "")
        if not text or not str(text).strip():
            continue

        # Skip unassigned/invalid anchor_id
        aid = elem.get("anchor_id")
        if not isinstance(aid, str) or not aid or aid == "unassigned":
            continue

        element_id = str(elem["element_id"])

        # Determine paragraph_kind from role
        if role == ROLE_LIST_BLOCK:
            para_kind = "list"
        else:
            para_kind = "body"

        # Get metadata from parent anchor
        meta = anchor_meta.get(aid, {})
        box_key = meta.get("box_key", "")
        anchor_type = meta.get("anchor_type", "")

        # Build bbox from geom fields if available
        bbox: Optional[List[float]] = None
        if "geom_x0" in elements_df.columns:
            try:
                bbox = [
                    float(elem["geom_x0"]),
                    float(elem["geom_y0"]),
                    float(elem["geom_x1"]),
                    float(elem["geom_y1"]),
                ]
            except (ValueError, TypeError, KeyError):
                bbox = None
        elif "bbox" in elements_df.columns:
            raw_bbox = elem.get("bbox")
            if raw_bbox is not None:
                if isinstance(raw_bbox, list):
                    bbox = raw_bbox
                elif hasattr(raw_bbox, '__iter__'):
                    bbox = list(raw_bbox)

        # Safe reading_order extraction
        ro = elem.get("reading_order", 0)
        if pd.isna(ro):
            ro = 0
        else:
            ro = int(ro)

        # Safe page extraction
        page = elem.get("page", 0)
        if pd.isna(page):
            page = 0
        else:
            page = int(page)

        # Create label from first 60 chars
        label = str(text)[:60].replace("\n", " ")

        nodes.append(Node(
            node_id=generate_paragraph_node_id(doc_id, element_id),
            doc_id=doc_id,
            node_type="paragraph",
            anchor_id=aid,
            anchor_type=anchor_type,
            element_id=element_id,
            box_key=box_key,
            label=label,
            text=str(text),
            pages=[page],
            bbox=bbox,
            reading_order=ro,
            paragraph_kind=para_kind,
            element_count=1,
            char_count=len(str(text)),
            concept_role=None,
        ))

    logger.debug(f"Built {len(nodes)} paragraph nodes")
    return nodes


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def build_all_nodes(
    sections_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    doc_id: str,
    doc_label: str = "Document Root",
    skip_roles: Optional[Set[str]] = None,
) -> NodeBuildResult:
    """
    Build all node types and combine into DataFrame.

    Args:
        sections_df: Sections DataFrame.
        elements_df: Elements DataFrame.
        doc_id: Document identifier.
        doc_label: Label for the document root node.
        skip_roles: Set of roles to skip for paragraph nodes.

    Returns:
        NodeBuildResult with combined nodes and counts.
    """
    all_nodes: List[Node] = []

    # 1. Doc root node
    doc_root = build_doc_root_node(doc_id, doc_label)
    all_nodes.append(doc_root)

    # 2. Section nodes
    section_nodes = build_section_nodes(sections_df, doc_id)
    all_nodes.extend(section_nodes)

    # 3. Paragraph nodes
    paragraph_nodes = build_paragraph_nodes(
        elements_df, sections_df, doc_id, skip_roles
    )
    all_nodes.extend(paragraph_nodes)

    # Convert to DataFrame
    nodes_df = pd.DataFrame([n.to_dict() for n in all_nodes])

    # Create separate paragraph_nodes_df for edge building
    if paragraph_nodes:
        paragraph_nodes_df = pd.DataFrame([n.to_dict() for n in paragraph_nodes])
    else:
        paragraph_nodes_df = pd.DataFrame()

    logger.info(
        f"Built {len(nodes_df)} total nodes: "
        f"1 doc_root, {len(section_nodes)} sections, {len(paragraph_nodes)} paragraphs"
    )

    return NodeBuildResult(
        nodes_df=nodes_df,
        paragraph_nodes_df=paragraph_nodes_df,
        section_node_count=len(section_nodes),
        paragraph_node_count=len(paragraph_nodes),
    )


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def build_nodes_legacy(
    sections_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    doc_id: str = "1099div_filer",
    doc_label: str = "1099-DIV Filer Instructions",
    role_page_artifact: str = "PageArtifact",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy-compatible wrapper for build_all_nodes.

    Matches original run_pipeline.py behavior for drop-in replacement.

    Args:
        sections_df: Sections DataFrame.
        elements_df: Elements DataFrame.
        doc_id: Document identifier.
        doc_label: Label for document root.
        role_page_artifact: Role to skip (for backwards compatibility).

    Returns:
        Tuple of (graph_nodes_df, paragraph_nodes_df).
    """
    skip_roles = {role_page_artifact}

    result = build_all_nodes(
        sections_df=sections_df,
        elements_df=elements_df,
        doc_id=doc_id,
        doc_label=doc_label,
        skip_roles=skip_roles,
    )

    # Print summary (matching original behavior)
    print(f"\n--- Building Graph Nodes ---")
    print(f"  Section nodes: {result.section_node_count}")
    print(f"  Paragraph nodes: {result.paragraph_node_count}")
    print(f"  Total nodes: {len(result.nodes_df)}")

    if not result.paragraph_nodes_df.empty and "paragraph_kind" in result.paragraph_nodes_df.columns:
        print(f"  Paragraphs by kind: {result.paragraph_nodes_df['paragraph_kind'].value_counts().to_dict()}")

    return result.nodes_df, result.paragraph_nodes_df
