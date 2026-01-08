"""
Section materialization from anchors and elements.

This module converts extracted anchors and their assigned elements into
materialized sections with aggregated text, metadata, and bounding boxes.

Process Overview:
1. Group elements by their anchor_id assignment
2. Separate header elements from body elements by role
3. Concatenate text with appropriate separators
4. Compute bounding box union across all elements
5. Track element provenance and page references
6. Apply text repairs (hyphenation fix)

Sections are the primary retrieval units for RAG. Each section contains:
- Full text for embedding and retrieval
- Metadata for filtering and display
- Element IDs for provenance tracking
- Bounding boxes for document highlighting
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from vaas.utils.text import repair_hyphenation


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Section:
    """
    A materialized document section.

    Sections are the primary units of content for RAG retrieval.
    Each section corresponds to one anchor (box, section, or subsection).

    Attributes:
        anchor_id: Unique identifier (e.g., "box_1a", "sec_general_instructions").
        anchor_type: Type of anchor (box, section, subsection, preamble).
        box_key: Box key if anchor_type is "box" (e.g., "1a").
        label: Human-readable label/title.
        header_text: Header portion of content (titles, headings).
        body_text: Body portion of content.
        full_text: Combined header + body text.
        char_count: Total character count.
        element_count: Number of elements in this section.
        element_ids: List of source element IDs.
        pages: List of page numbers where section appears.
        bbox: Bounding box union [x0, y0, x1, y1].
        is_grouped: Whether this anchor is part of a grouped header.
        group_id: Group identifier for grouped anchors.
    """

    anchor_id: str
    anchor_type: str
    box_key: Optional[str] = None
    label: str = ""
    header_text: str = ""
    body_text: str = ""
    full_text: str = ""
    char_count: int = 0
    element_count: int = 0
    element_ids: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    bbox: List[float] = field(default_factory=list)
    is_grouped: bool = False
    group_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "anchor_id": self.anchor_id,
            "box_key": self.box_key or "",
            "anchor_type": self.anchor_type,
            "label": self.label,
            "header_text": self.header_text,
            "body_text": self.body_text,
            "full_text": self.full_text,
            "pages": self.pages,
            "bbox": self.bbox,
            "element_count": self.element_count,
            "element_ids": self.element_ids,
            "char_count": self.char_count,
            "is_grouped": self.is_grouped,
            "group_id": self.group_id,
        }


@dataclass
class SectionMaterializationResult:
    """
    Result of section materialization.

    Attributes:
        sections_df: DataFrame with all materialized sections.
        total_sections: Total number of sections created.
        box_sections: Number of box sections.
        main_sections: Number of main (document) sections.
        subsections: Number of subsections.
        preamble_sections: Number of preamble sections.
        avg_char_count: Average character count per section.
        empty_sections: List of anchor_ids with no content.
    """

    sections_df: pd.DataFrame
    total_sections: int = 0
    box_sections: int = 0
    main_sections: int = 0
    subsections: int = 0
    preamble_sections: int = 0
    avg_char_count: float = 0.0
    empty_sections: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SectionMaterializationResult(total={self.total_sections}, "
            f"box={self.box_sections}, section={self.main_sections}, "
            f"subsection={self.subsections}, empty={len(self.empty_sections)})"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_bbox_union(bboxes: List[Any]) -> List[float]:
    """
    Compute bounding box union from multiple bboxes.

    Takes minimum x0/y0 and maximum x1/y1 to create a box
    that encompasses all input boxes.

    Args:
        bboxes: List of bboxes, each as [x0, y0, x1, y1].
                Handles None, empty, and invalid bboxes gracefully.

    Returns:
        Union bbox as [x0, y0, x1, y1], or [0, 0, 0, 0] if no valid boxes.

    Example:
        >>> compute_bbox_union([[72, 100, 200, 150], [100, 140, 300, 200]])
        [72, 100, 300, 200]
    """
    valid_bboxes = []
    for b in bboxes:
        if b is None:
            continue
        if isinstance(b, np.ndarray):
            b = b.tolist()
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            try:
                # Validate all values are numeric
                coords = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                if not any(np.isnan(c) for c in coords):
                    valid_bboxes.append(coords)
            except (TypeError, ValueError, IndexError):
                continue

    if not valid_bboxes:
        return [0.0, 0.0, 0.0, 0.0]

    x0 = min(b[0] for b in valid_bboxes)
    y0 = min(b[1] for b in valid_bboxes)
    x1 = max(b[2] for b in valid_bboxes)
    y1 = max(b[3] for b in valid_bboxes)

    return [float(x0), float(y0), float(x1), float(y1)]


def split_header_body(
    header_elements_text: List[str],
    body_elements_text: List[str],
    repair_hyphens: bool = True,
) -> Tuple[str, str, str]:
    """
    Combine header and body element text into section text.

    Header elements are joined with single newlines.
    Body elements are joined with double newlines (paragraph separation).
    Optionally repairs PDF hyphenation artifacts.

    Args:
        header_elements_text: List of header element text strings.
        body_elements_text: List of body element text strings.
        repair_hyphens: Whether to fix hyphenation artifacts.

    Returns:
        Tuple of (header_text, body_text, full_text).

    Example:
        >>> split_header_body(["Box 1a. Title"], ["Body paragraph 1.", "Body paragraph 2."])
        ('Box 1a. Title', 'Body paragraph 1.\\n\\nBody paragraph 2.', 'Box 1a. Title\\n\\nBody...')
    """
    header_text = "\n".join(header_elements_text).strip()
    body_text = "\n\n".join(body_elements_text).strip()

    if header_text:
        full_text = f"{header_text}\n\n{body_text}".strip()
    else:
        full_text = body_text

    if repair_hyphens:
        header_text = repair_hyphenation(header_text)
        body_text = repair_hyphenation(body_text)
        full_text = repair_hyphenation(full_text)

    return header_text, body_text, full_text


def get_section_sort_key(row: pd.Series) -> Tuple[int, int, str]:
    """
    Generate sort key for section ordering.

    Order:
    1. Preamble first (priority 0)
    2. Regular sections/boxes by box_key (priority 1)
    3. Unassigned last (priority 2)

    Box keys are sorted numerically then alphabetically (1, 1a, 1b, 2, 2a, ...).

    Args:
        row: Section row with anchor_id and box_key.

    Returns:
        Sort key tuple (priority, numeric_key, alpha_key).
    """
    anchor_id = row.get("anchor_id", "")

    if anchor_id == "preamble":
        return (0, 0, "")
    elif anchor_id == "unassigned":
        return (2, 0, "")
    else:
        key = row.get("box_key", "") or ""
        match = re.match(r"(\d+)([a-z]?)", str(key))
        if match:
            num = int(match.group(1))
            letter = match.group(2) or ""
            return (1, num, letter)
        return (1, 999, str(key))


def build_anchor_to_elements_map(
    elements_df: pd.DataFrame,
    anchor_id_col: str = "anchor_id",
    anchor_ids_col: str = "anchor_ids",
) -> Dict[str, List[int]]:
    """
    Build mapping from anchor_id to element indices.

    Handles both primary anchor assignments and grouped anchor sharing.

    Args:
        elements_df: DataFrame with elements.
        anchor_id_col: Column name for primary anchor assignment.
        anchor_ids_col: Column name for grouped anchor list.

    Returns:
        Dict mapping anchor_id to list of element DataFrame indices.
    """
    # Collect all unique anchor IDs
    all_anchor_ids: Set[str] = set(elements_df[anchor_id_col].unique())

    # Add grouped anchor members
    if anchor_ids_col in elements_df.columns:
        for anchor_ids_list in elements_df[anchor_ids_col].dropna():
            if isinstance(anchor_ids_list, list):
                all_anchor_ids.update(anchor_ids_list)

    # Build reverse map
    anchor_to_elements: Dict[str, List[int]] = {aid: [] for aid in all_anchor_ids}

    for idx, row in elements_df.iterrows():
        primary = row[anchor_id_col]
        grouped = row.get(anchor_ids_col)

        if grouped and isinstance(grouped, list):
            # Element belongs to ALL anchors in the group
            for aid in grouped:
                if aid in anchor_to_elements:
                    anchor_to_elements[aid].append(idx)
        elif primary in anchor_to_elements:
            anchor_to_elements[primary].append(idx)

    return anchor_to_elements


# =============================================================================
# SECTION MATERIALIZATION
# =============================================================================

def materialize_section(
    anchor_id: str,
    elements_df: pd.DataFrame,
    element_indices: List[int],
    anchor_meta: Optional[pd.Series],
    header_roles: Set[str],
    exclude_roles: Set[str],
    repair_hyphens: bool = True,
) -> Section:
    """
    Materialize a single section from its elements.

    Args:
        anchor_id: The anchor ID for this section.
        elements_df: Full elements DataFrame.
        element_indices: Indices of elements belonging to this anchor.
        anchor_meta: Anchor metadata row (from anchors_df).
        header_roles: Roles to treat as header elements.
        exclude_roles: Roles to exclude from body (headers + artifacts).
        repair_hyphens: Whether to repair hyphenation artifacts.

    Returns:
        Materialized Section object.
    """
    # Get anchor metadata
    if anchor_meta is not None:
        box_key = anchor_meta.get("box_key", "")
        label = anchor_meta.get("label", "")
        anchor_type = anchor_meta.get("anchor_type", "box")
        is_grouped = anchor_meta.get("is_grouped", False)
        group_id = anchor_meta.get("group_id", None)
    else:
        box_key = ""
        label = anchor_id
        anchor_type = "preamble" if anchor_id == "preamble" else "other"
        is_grouped = False
        group_id = None

    # Handle empty element list
    if not element_indices:
        return Section(
            anchor_id=anchor_id,
            anchor_type=anchor_type,
            box_key=box_key if box_key else None,
            label=label,
            is_grouped=is_grouped,
            group_id=group_id,
        )

    # Get elements and sort by reading order
    anchor_elements = elements_df.loc[element_indices].copy()
    anchor_elements = anchor_elements.sort_values(["page", "reading_order"])

    # Separate header and body elements
    header_mask = anchor_elements["role"].isin(header_roles)
    body_mask = ~anchor_elements["role"].isin(exclude_roles)

    header_elements = anchor_elements[header_mask]
    body_elements = anchor_elements[body_mask]

    # Build text
    header_text_list = header_elements["text"].tolist()
    body_text_list = body_elements["text"].tolist()

    header_text, body_text, full_text = split_header_body(
        header_text_list, body_text_list, repair_hyphens
    )

    # Compute bbox union
    all_bboxes = anchor_elements["bbox"].tolist()
    bbox = compute_bbox_union(all_bboxes)

    # Collect pages and element IDs
    pages = sorted(anchor_elements["page"].unique().tolist())
    element_ids = anchor_elements["element_id"].tolist()

    return Section(
        anchor_id=anchor_id,
        anchor_type=anchor_type,
        box_key=box_key if box_key else None,
        label=label,
        header_text=header_text,
        body_text=body_text,
        full_text=full_text,
        char_count=len(full_text),
        element_count=len(anchor_elements),
        element_ids=element_ids,
        pages=pages,
        bbox=bbox,
        is_grouped=is_grouped,
        group_id=group_id,
    )


def materialize_sections(
    elements_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    anchor_timeline_df: Optional[pd.DataFrame] = None,
    header_roles: Optional[Set[str]] = None,
    artifact_roles: Optional[Set[str]] = None,
    repair_hyphens: bool = True,
    sort_sections: bool = True,
    element_id_col: str = "element_id",
    anchor_id_col: str = "anchor_id",
    anchor_ids_col: str = "anchor_ids",
    role_col: str = "role",
) -> SectionMaterializationResult:
    """
    Materialize all sections from elements and anchors.

    Each anchor becomes a section. Elements assigned to that anchor
    are aggregated into the section's content with separated header
    and body text.

    Args:
        elements_df: DataFrame with elements (must have anchor_id assigned).
        anchors_df: DataFrame with anchor metadata.
        anchor_timeline_df: Optional timeline DataFrame (used if anchors_df missing metadata).
        header_roles: Roles to treat as headers (default: box_header, section_header, subsection_header).
        artifact_roles: Roles to exclude from body (default: PageArtifact, page_artifact).
        repair_hyphens: Whether to repair hyphenation artifacts.
        sort_sections: Whether to sort sections by box key.
        element_id_col: Column name for element ID.
        anchor_id_col: Column name for anchor ID.
        anchor_ids_col: Column name for grouped anchor IDs.
        role_col: Column name for element role.

    Returns:
        SectionMaterializationResult with sections DataFrame and statistics.

    Example:
        >>> result = materialize_sections(elements_df, anchors_df)
        >>> print(f"Created {result.total_sections} sections")
        >>> sections_df = result.sections_df
    """
    # Default role sets
    if header_roles is None:
        # Use actual role constants (PascalCase) to match classify_elements output
        header_roles = {"BoxHeader", "SectionHeader", "SubsectionHeader"}
    if artifact_roles is None:
        artifact_roles = {"PageArtifact", "page_artifact"}

    # Roles to exclude from body (headers + artifacts)
    exclude_roles = header_roles | artifact_roles

    # Use timeline if provided and anchors_df is minimal
    anchor_meta_df = anchor_timeline_df if anchor_timeline_df is not None else anchors_df

    # Build anchor -> elements map
    anchor_to_elements = build_anchor_to_elements_map(
        elements_df, anchor_id_col, anchor_ids_col
    )

    # Materialize each section
    sections: List[Section] = []
    empty_sections: List[str] = []

    for anchor_id, element_indices in anchor_to_elements.items():
        # Get anchor metadata
        anchor_meta = None
        if not anchor_meta_df.empty and "anchor_id" in anchor_meta_df.columns:
            meta_rows = anchor_meta_df[anchor_meta_df["anchor_id"] == anchor_id]
            if not meta_rows.empty:
                anchor_meta = meta_rows.iloc[0]

        # Materialize section
        section = materialize_section(
            anchor_id=anchor_id,
            elements_df=elements_df,
            element_indices=element_indices,
            anchor_meta=anchor_meta,
            header_roles=header_roles,
            exclude_roles=exclude_roles,
            repair_hyphens=repair_hyphens,
        )

        sections.append(section)

        if section.element_count == 0:
            empty_sections.append(anchor_id)

    # Convert to DataFrame
    if sections:
        sections_df = pd.DataFrame([s.to_dict() for s in sections])
    else:
        sections_df = pd.DataFrame()

    # Sort sections
    if sort_sections and not sections_df.empty:
        sections_df["_sort"] = sections_df.apply(get_section_sort_key, axis=1)
        sections_df = sections_df.sort_values("_sort").drop(columns=["_sort"])
        sections_df = sections_df.reset_index(drop=True)

    # Compute statistics
    total_sections = len(sections_df)
    box_sections = 0
    main_sections = 0
    subsections = 0
    preamble_sections = 0

    if not sections_df.empty and "anchor_type" in sections_df.columns:
        type_counts = sections_df["anchor_type"].value_counts()
        box_sections = int(type_counts.get("box", 0))
        main_sections = int(type_counts.get("section", 0))
        subsections = int(type_counts.get("subsection", 0))
        preamble_sections = int(type_counts.get("preamble", 0))

    avg_char_count = 0.0
    if not sections_df.empty and "char_count" in sections_df.columns:
        avg_char_count = float(sections_df["char_count"].mean())

    logger.info(
        f"Materialized {total_sections} sections "
        f"(box={box_sections}, section={main_sections}, subsection={subsections})"
    )

    if empty_sections:
        logger.warning(f"{len(empty_sections)} empty sections: {empty_sections[:5]}...")

    return SectionMaterializationResult(
        sections_df=sections_df,
        total_sections=total_sections,
        box_sections=box_sections,
        main_sections=main_sections,
        subsections=subsections,
        preamble_sections=preamble_sections,
        avg_char_count=avg_char_count,
        empty_sections=empty_sections,
    )


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def materialize_sections_legacy(
    elements_df: pd.DataFrame,
    anchor_timeline: pd.DataFrame,
    role_box_header: str = "box_header",
    role_section_header: str = "section_header",
    role_subsection_header: str = "subsection_header",
    role_page_artifact: str = "PageArtifact",
) -> pd.DataFrame:
    """
    Legacy-compatible wrapper for materialize_sections.

    Matches original run_pipeline.py behavior for drop-in replacement.

    Args:
        elements_df: Elements DataFrame with anchor_id assigned.
        anchor_timeline: Anchor timeline DataFrame.
        role_box_header: Role value for box headers.
        role_section_header: Role value for section headers.
        role_subsection_header: Role value for subsection headers.
        role_page_artifact: Role value for page artifacts.

    Returns:
        Sections DataFrame in original format.
    """
    header_roles = {role_box_header, role_section_header, role_subsection_header}
    artifact_roles = {role_page_artifact, "page_artifact"}

    result = materialize_sections(
        elements_df=elements_df,
        anchors_df=anchor_timeline,
        anchor_timeline_df=anchor_timeline,
        header_roles=header_roles,
        artifact_roles=artifact_roles,
    )

    # Print summary (matching original behavior)
    print(f"\nSections created: {result.total_sections}")

    if not result.sections_df.empty:
        print("\n--- Section Summary ---")
        for _, r in result.sections_df.iterrows():
            label = r.get("label", "")[:30]
            print(f"  {r['anchor_id']}: {r['element_count']} elements, {r['char_count']} chars - {label}...")

    return result.sections_df


# =============================================================================
# UTILITIES
# =============================================================================

def filter_sections_by_type(
    sections_df: pd.DataFrame,
    anchor_type: str,
) -> pd.DataFrame:
    """
    Filter sections to a specific anchor type.

    Args:
        sections_df: Sections DataFrame.
        anchor_type: Type to filter (box, section, subsection, preamble).

    Returns:
        Filtered DataFrame.
    """
    if sections_df.empty:
        return sections_df
    return sections_df[sections_df["anchor_type"] == anchor_type].copy()


def get_section_by_anchor_id(
    sections_df: pd.DataFrame,
    anchor_id: str,
) -> Optional[pd.Series]:
    """
    Get a single section by anchor_id.

    Args:
        sections_df: Sections DataFrame.
        anchor_id: Anchor ID to find.

    Returns:
        Section row or None if not found.
    """
    if sections_df.empty:
        return None
    matches = sections_df[sections_df["anchor_id"] == anchor_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def get_sections_summary(sections_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for sections.

    Args:
        sections_df: Sections DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    if sections_df.empty:
        return {
            "total": 0,
            "by_type": {},
            "total_chars": 0,
            "avg_chars": 0.0,
            "total_elements": 0,
        }

    type_counts = sections_df["anchor_type"].value_counts().to_dict()
    total_chars = int(sections_df["char_count"].sum())
    avg_chars = float(sections_df["char_count"].mean())
    total_elements = int(sections_df["element_count"].sum())

    return {
        "total": len(sections_df),
        "by_type": type_counts,
        "total_chars": total_chars,
        "avg_chars": avg_chars,
        "total_elements": total_elements,
    }
