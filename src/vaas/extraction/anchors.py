"""
Anchor detection and timeline building for PDF extraction.

This module handles:
1. Parsing box headers (Box 1a, Boxes 14-16, etc.) into structured box keys
2. Extracting anchors (boxes, sections, subsections) from classified elements
3. Building the anchor timeline with reading order ranges
4. Assigning content elements to their respective anchors

Anchors are the structural skeleton of the document. Each anchor represents
a semantic unit (box, section, or subsection) that will be populated with
content elements based on reading order proximity.

Algorithm Overview:
1. Parse box headers using regex patterns to extract box keys
2. Create anchor records from box, section, and subsection headers
3. Sort anchors by (page, reading_order) to establish timeline
4. Compute reading order ranges [start, end) for each anchor
5. Assign content elements to anchors based on range containment
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from vaas.utils.text import stable_hash, slug_title
from vaas.constants import ROLE_BOX_HEADER, ROLE_SECTION_HEADER, ROLE_SUBSECTION_HEADER


logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Toggle vectorized anchor assignment (10-50x faster than iterrows)
# Set to False to use legacy implementation for parity testing
USE_VECTORIZED_ASSIGNMENT = True


# =============================================================================
# BOX KEY PARSING
# =============================================================================

# Regex patterns for box header parsing
# Order matters: check more specific patterns (range, through, double) before single
BOX_SINGLE_PARSE = re.compile(r"^Box(?:es)?\s+(\d+[a-z]?)\b", re.IGNORECASE)
BOX_RANGE_PARSE = re.compile(r"^Box(?:es)?\s+(\d+[a-z]?)\s*[-–]\s*(\d+[a-z]?)\b", re.IGNORECASE)
BOX_DOUBLE_PARSE = re.compile(r"^Box(?:es)?\s+(\d+[a-z]?)\s+and\s+(\d+[a-z]?)\b", re.IGNORECASE)
BOX_THROUGH_PARSE = re.compile(r"^Box(?:es)?\s+(\d+)\s+through\s+(\d+)\b", re.IGNORECASE)


@dataclass
class BoxParseResult:
    """
    Result of parsing a box header.

    Attributes:
        kind: Parse type (single, range, double, through, unknown).
        keys: List of box keys extracted (e.g., ["1a"], ["14", "15", "16"]).
        label: Remaining text after box reference (the description).
        raw_text: Original input text.
    """

    kind: str
    keys: List[str]
    label: str
    raw_text: str = ""

    @property
    def is_valid(self) -> bool:
        """True if parsing found at least one box key."""
        return self.kind != "unknown" and len(self.keys) > 0

    @property
    def is_grouped(self) -> bool:
        """True if multiple box keys were parsed (range/double/through)."""
        return len(self.keys) > 1


def _expand_numeric_range(lo: str, hi: str) -> List[str]:
    """
    Expand a numeric range like ("14", "16") to ["14", "15", "16"].

    Handles both pure numeric and alphanumeric (strips letters for range).
    Falls back to [lo, hi] if expansion fails.
    """
    try:
        lo_num = int(re.sub(r"[a-z]", "", lo))
        hi_num = int(re.sub(r"[a-z]", "", hi))
        return [str(k) for k in range(min(lo_num, hi_num), max(lo_num, hi_num) + 1)]
    except (ValueError, TypeError):
        return [lo, hi]


def parse_box_keys(text: str) -> BoxParseResult:
    """
    Parse a box header text into structured box keys.

    Handles multiple formats:
    - Single: "Box 1a" → ["1a"]
    - Range: "Box 14-16" or "Boxes 14–16" → ["14", "15", "16"]
    - Double: "Boxes 2a and 2b" → ["2a", "2b"]
    - Through: "Boxes 14 through 16" → ["14", "15", "16"]

    Args:
        text: Raw box header text from PDF extraction.

    Returns:
        BoxParseResult with parsed keys and remaining label.

    Examples:
        >>> parse_box_keys("Box 1a. Ordinary Dividends")
        BoxParseResult(kind='single', keys=['1a'], label='Ordinary Dividends')

        >>> parse_box_keys("Boxes 14 through 16. State information")
        BoxParseResult(kind='through', keys=['14', '15', '16'], label='State information')

        >>> parse_box_keys("Boxes 2a and 2b. Capital Gains")
        BoxParseResult(kind='double', keys=['2a', '2b'], label='Capital Gains')
    """
    t = (text or "").strip()

    # Try range pattern first (14-16, 14–16)
    m = BOX_RANGE_PARSE.match(t)
    if m:
        lo, hi = m.group(1).lower(), m.group(2).lower()
        keys = _expand_numeric_range(lo, hi)
        label = BOX_RANGE_PARSE.sub("", t).strip().lstrip(".-–: ")
        return BoxParseResult(kind="range", keys=keys, label=label, raw_text=text)

    # Try "through" pattern (14 through 16)
    m = BOX_THROUGH_PARSE.match(t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        keys = [str(k) for k in range(min(lo, hi), max(lo, hi) + 1)]
        label = BOX_THROUGH_PARSE.sub("", t).strip().lstrip(".-–: ")
        return BoxParseResult(kind="through", keys=keys, label=label, raw_text=text)

    # Try "and" pattern (2a and 2b)
    m = BOX_DOUBLE_PARSE.match(t)
    if m:
        keys = [m.group(1).lower(), m.group(2).lower()]
        label = BOX_DOUBLE_PARSE.sub("", t).strip().lstrip(".-–: ")
        return BoxParseResult(kind="double", keys=keys, label=label, raw_text=text)

    # Try single box pattern
    m = BOX_SINGLE_PARSE.match(t)
    if m:
        keys = [m.group(1).lower()]
        label = BOX_SINGLE_PARSE.sub("", t).strip().lstrip(".-–: ")
        return BoxParseResult(kind="single", keys=keys, label=label, raw_text=text)

    # No match found
    return BoxParseResult(kind="unknown", keys=[], label=t, raw_text=text)


# =============================================================================
# ANCHOR DATA STRUCTURES
# =============================================================================

@dataclass
class AnchorRecord:
    """
    A single anchor record representing a structural unit.

    Attributes:
        anchor_id: Unique identifier (e.g., "box_1a", "sec_general_instructions").
        anchor_type: Type of anchor (box, section, subsection).
        label: Human-readable label.
        box_key: Box key if anchor_type is "box", empty otherwise.
        source_element_id: Element ID that created this anchor.
        source_text: Original header text.
        parse_kind: How the anchor was parsed (single, range, section, etc.).
        is_grouped: True if part of a grouped header (e.g., Boxes 14-16).
        group_id: Group identifier for grouped anchors.
        page: Page number where anchor appears.
        reading_order: Reading order of source element.
        geom_y0: Y-coordinate of anchor for sorting.
        geom_x0: X-coordinate of anchor for column detection.
    """

    anchor_id: str
    anchor_type: str  # "box", "section", "subsection"
    label: str
    box_key: str = ""
    source_element_id: str = ""
    source_text: str = ""
    parse_kind: str = ""
    is_grouped: bool = False
    group_id: Optional[str] = None
    page: int = 0
    reading_order: int = 0
    geom_y0: float = 0.0
    geom_x0: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "anchor_id": self.anchor_id,
            "box_key": self.box_key,
            "anchor_type": self.anchor_type,
            "label": self.label,
            "source_element_id": self.source_element_id,
            "source_text": self.source_text,
            "parse_kind": self.parse_kind,
            "is_grouped": self.is_grouped,
            "group_id": self.group_id,
            "page": self.page,
            "reading_order": self.reading_order,
            "geom_y0": self.geom_y0,
            "geom_x0": self.geom_x0,
        }


@dataclass
class AnchorExtractionResult:
    """
    Result of anchor extraction from elements.

    Attributes:
        anchors_df: DataFrame with all extracted anchors.
        box_anchors: Count of box anchors.
        section_anchors: Count of section anchors.
        subsection_anchors: Count of subsection anchors.
        parse_warnings: List of unparseable headers.
        duplicates_dropped: Count of duplicate anchors removed.
    """

    anchors_df: pd.DataFrame
    box_anchors: int = 0
    section_anchors: int = 0
    subsection_anchors: int = 0
    parse_warnings: List[str] = field(default_factory=list)
    duplicates_dropped: int = 0

    def __repr__(self) -> str:
        return (
            f"AnchorExtractionResult(boxes={self.box_anchors}, "
            f"sections={self.section_anchors}, "
            f"subsections={self.subsection_anchors})"
        )


# =============================================================================
# SECTION ID MAPPING
# =============================================================================

# Default mapping from section header text to canonical section IDs
# These are common section headers in IRS instruction documents
DEFAULT_SECTION_ID_MAP: Dict[str, str] = {
    "future developments": "sec_future_developments",
    "reminders": "sec_reminders",
    "general instructions": "sec_general_instructions",
    "specific instructions": "sec_specific_instructions",
    "what's new": "sec_whats_new",
    "definitions": "sec_definitions",
    "how to": "sec_how_to",
    "where to": "sec_where_to",
    "paperwork reduction act notice": "sec_paperwork_reduction",
    "additional information": "sec_additional_info",
}


def get_section_id(
    text: str,
    section_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    Get canonical section ID for a section header.

    Args:
        text: Section header text.
        section_map: Custom mapping of text prefixes to IDs.

    Returns:
        Tuple of (section_id, label).
    """
    section_map = section_map or DEFAULT_SECTION_ID_MAP
    section_text = text.strip()
    section_label = section_text

    # Check for known section patterns
    for pattern, sid in section_map.items():
        if section_text.lower().startswith(pattern):
            return sid, section_label

    # Fallback: create ID from text
    section_id = "sec_" + re.sub(r"[^a-z0-9]+", "_", section_text.lower()[:30]).strip("_")
    return section_id, section_label


def get_subsection_id(text: str, source_element_id: str) -> Tuple[str, str]:
    """
    Generate stable subsection ID from content and position.

    Uses a combination of title slug (human-readable) and position hash
    (guarantees uniqueness) to create IDs.

    Args:
        text: Subsection header text.
        source_element_id: Element ID for position-based hash.

    Returns:
        Tuple of (subsection_id, label).
    """
    subsection_text = text.strip()
    subsection_label = subsection_text.split("\n")[0][:60]  # First line, truncated

    # Generate stable ID: human-readable slug + position hash
    title_slug = slug_title(subsection_text, max_len=30)
    position_hash = stable_hash([source_element_id], length=8)
    subsection_id = f"sub_{title_slug}_{position_hash}"

    return subsection_id, subsection_label


# =============================================================================
# ANCHOR EXTRACTION
# =============================================================================

def extract_box_anchors(
    box_headers: pd.DataFrame,
) -> Tuple[List[AnchorRecord], List[str]]:
    """
    Extract box anchors from box header elements.

    Args:
        box_headers: DataFrame with box header elements.
            Required columns: element_id, text, page, reading_order, geom_y0, geom_x0

    Returns:
        Tuple of (anchor_records, warnings).
    """
    anchors: List[AnchorRecord] = []
    warnings: List[str] = []

    for _, row in box_headers.iterrows():
        parsed = parse_box_keys(row["text"])

        if not parsed.is_valid:
            warnings.append(f"Could not parse box header: {row['text'][:60]}")
            continue

        group_id = f"group_{row['element_id']}" if parsed.is_grouped else None

        for box_key in parsed.keys:
            anchors.append(
                AnchorRecord(
                    anchor_id=f"box_{box_key}",
                    anchor_type="box",
                    box_key=box_key,
                    label=parsed.label,
                    source_element_id=row["element_id"],
                    source_text=row["text"],
                    parse_kind=parsed.kind,
                    is_grouped=parsed.is_grouped,
                    group_id=group_id,
                    page=int(row["page"]),
                    reading_order=int(row.get("reading_order", 0)),
                    geom_y0=float(row.get("geom_y0", 0)),
                    geom_x0=float(row.get("geom_x0", 0)),
                )
            )

    return anchors, warnings


def extract_section_anchors(
    section_headers: pd.DataFrame,
    section_map: Optional[Dict[str, str]] = None,
) -> List[AnchorRecord]:
    """
    Extract section anchors from section header elements.

    Args:
        section_headers: DataFrame with section header elements.
        section_map: Custom mapping of text to section IDs.

    Returns:
        List of section anchor records.
    """
    anchors: List[AnchorRecord] = []

    for _, row in section_headers.iterrows():
        section_id, section_label = get_section_id(row["text"], section_map)

        anchors.append(
            AnchorRecord(
                anchor_id=section_id,
                anchor_type="section",
                box_key="",
                label=section_label,
                source_element_id=row["element_id"],
                source_text=row["text"],
                parse_kind="section",
                is_grouped=False,
                group_id=None,
                page=int(row["page"]),
                reading_order=int(row.get("reading_order", 0)),
                geom_y0=float(row.get("geom_y0", 0)),
                geom_x0=float(row.get("geom_x0", 0)),
            )
        )

    return anchors


def extract_subsection_anchors(
    subsection_headers: pd.DataFrame,
) -> List[AnchorRecord]:
    """
    Extract subsection anchors from subsection header elements.

    Args:
        subsection_headers: DataFrame with subsection header elements.

    Returns:
        List of subsection anchor records.
    """
    anchors: List[AnchorRecord] = []

    for _, row in subsection_headers.iterrows():
        subsection_id, subsection_label = get_subsection_id(
            row["text"], row["element_id"]
        )

        anchors.append(
            AnchorRecord(
                anchor_id=subsection_id,
                anchor_type="subsection",
                box_key="",
                label=subsection_label,
                source_element_id=row["element_id"],
                source_text=row["text"],
                parse_kind="subsection",
                is_grouped=False,
                group_id=None,
                page=int(row["page"]),
                reading_order=int(row.get("reading_order", 0)),
                geom_y0=float(row.get("geom_y0", 0)),
                geom_x0=float(row.get("geom_x0", 0)),
            )
        )

    return anchors


def deduplicate_anchors(
    anchors_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate anchors (same page + similar label).

    Args:
        anchors_df: DataFrame with anchor records.

    Returns:
        Tuple of (deduplicated DataFrame, count dropped).
    """
    if anchors_df.empty:
        return anchors_df, 0

    dropped_count = 0

    # Check for near-duplicate subsections (same page + same normalized label)
    subsections = anchors_df[anchors_df["anchor_type"] == "subsection"].copy()
    if len(subsections) > 0:
        subsections["_norm_label"] = subsections["label"].str.lower().str.strip()
        dup_mask = subsections.duplicated(subset=["page", "_norm_label"], keep="first")

        if dup_mask.any():
            dropped_count = dup_mask.sum()
            dup_ids = subsections[dup_mask]["anchor_id"].tolist()
            anchors_df = anchors_df[~anchors_df["anchor_id"].isin(dup_ids)]
            logger.warning(f"Dropped {dropped_count} near-duplicate subsection anchors")

    # Drop exact duplicates on anchor_id
    before = len(anchors_df)
    anchors_df = anchors_df.drop_duplicates(subset=["anchor_id"], keep="first")
    exact_dups = before - len(anchors_df)
    if exact_dups > 0:
        dropped_count += exact_dups
        logger.warning(f"Dropped {exact_dups} exact-duplicate anchor IDs")

    return anchors_df.reset_index(drop=True), dropped_count


def extract_anchors(
    elements_df: pd.DataFrame,
    role_box_header: str = ROLE_BOX_HEADER,
    role_section_header: str = ROLE_SECTION_HEADER,
    role_subsection_header: str = ROLE_SUBSECTION_HEADER,
    section_map: Optional[Dict[str, str]] = None,
) -> AnchorExtractionResult:
    """
    Extract all anchors from classified elements.

    This is the main entry point for anchor extraction. It processes
    box headers, section headers, and subsection headers to create
    a unified anchors DataFrame.

    Args:
        elements_df: DataFrame with classified elements.
            Required columns: element_id, text, role, page, reading_order, geom_y0, geom_x0
        role_box_header: Role value for box headers.
        role_section_header: Role value for section headers.
        role_subsection_header: Role value for subsection headers.
        section_map: Custom mapping of section text to IDs.

    Returns:
        AnchorExtractionResult with anchors DataFrame and statistics.

    Example:
        >>> result = extract_anchors(elements_df)
        >>> print(f"Found {result.box_anchors} box anchors")
        >>> anchors_df = result.anchors_df
    """
    all_anchors: List[AnchorRecord] = []
    all_warnings: List[str] = []

    # Extract box anchors
    box_headers = elements_df[elements_df["role"] == role_box_header]
    box_anchors, box_warnings = extract_box_anchors(box_headers)
    all_anchors.extend(box_anchors)
    all_warnings.extend(box_warnings)
    logger.info(f"Extracted {len(box_anchors)} box anchors from {len(box_headers)} headers")

    # Extract section anchors
    section_headers = elements_df[elements_df["role"] == role_section_header]
    section_anchors = extract_section_anchors(section_headers, section_map)
    all_anchors.extend(section_anchors)
    logger.info(f"Extracted {len(section_anchors)} section anchors")

    # Extract subsection anchors
    subsection_headers = elements_df[elements_df["role"] == role_subsection_header]
    subsection_anchors = extract_subsection_anchors(subsection_headers)
    all_anchors.extend(subsection_anchors)
    logger.info(f"Extracted {len(subsection_anchors)} subsection anchors")

    # Convert to DataFrame
    if all_anchors:
        anchors_df = pd.DataFrame([a.to_dict() for a in all_anchors])
        anchors_df = anchors_df.sort_values(["page", "reading_order"])
    else:
        anchors_df = pd.DataFrame()

    # Deduplicate
    anchors_df, dups_dropped = deduplicate_anchors(anchors_df)

    return AnchorExtractionResult(
        anchors_df=anchors_df,
        box_anchors=len(box_anchors),
        section_anchors=len(section_anchors),
        subsection_anchors=len(subsection_anchors),
        parse_warnings=all_warnings,
        duplicates_dropped=dups_dropped,
    )


# =============================================================================
# BOX VALIDATION
# =============================================================================

@dataclass
class BoxValidationResult:
    """
    Result of validating extracted boxes against expected set.

    Attributes:
        expected: Set of expected box keys.
        found: Set of actually found box keys.
        missing: Set of expected but not found box keys.
        extras: Set of found but not expected box keys.
        passed: True if all expected boxes were found.
    """

    expected: Set[str]
    found: Set[str]
    missing: Set[str]
    extras: Set[str]

    @property
    def passed(self) -> bool:
        return len(self.missing) == 0

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"BoxValidation({status}: "
            f"expected={len(self.expected)}, "
            f"found={len(self.found)}, "
            f"missing={len(self.missing)})"
        )


# Expected boxes for common IRS forms
EXPECTED_BOXES_1099DIV: Set[str] = {
    "1a", "1b", "2a", "2b", "2c", "2d", "2e", "2f",
    "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16",
}

EXPECTED_BOXES_1099INT: Set[str] = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16", "17",
}


def validate_box_coverage(
    anchors_df: pd.DataFrame,
    expected_boxes: Set[str],
) -> BoxValidationResult:
    """
    Validate that all expected boxes were extracted.

    Args:
        anchors_df: DataFrame with anchor records.
        expected_boxes: Set of expected box keys.

    Returns:
        BoxValidationResult with comparison details.

    Example:
        >>> result = validate_box_coverage(anchors_df, EXPECTED_BOXES_1099DIV)
        >>> if not result.passed:
        ...     print(f"Missing boxes: {result.missing}")
    """
    box_anchors = anchors_df[anchors_df["anchor_type"] == "box"] if not anchors_df.empty else pd.DataFrame()
    found = set(box_anchors["box_key"].tolist()) if not box_anchors.empty else set()

    return BoxValidationResult(
        expected=expected_boxes,
        found=found,
        missing=expected_boxes - found,
        extras=found - expected_boxes,
    )


# =============================================================================
# ANCHOR TIMELINE
# =============================================================================

@dataclass
class AnchorTimeline:
    """
    Anchor timeline with reading order ranges.

    The timeline assigns each anchor a range of reading orders
    [start_reading_order, end_reading_order] that defines which
    content elements belong to that anchor.

    Attributes:
        timeline_df: DataFrame with anchors and their reading order ranges.
        grouped_anchor_map: Map from source_element_id to list of anchor_ids
                           for grouped headers (e.g., Boxes 14-16).
    """

    timeline_df: pd.DataFrame
    grouped_anchor_map: Dict[str, List[str]] = field(default_factory=dict)

    def get_anchor_at(self, page: int, reading_order: int) -> Optional[str]:
        """Get anchor_id for a given page and reading order."""
        page_anchors = self.timeline_df[self.timeline_df["page"] == page]
        for _, anchor in page_anchors.iterrows():
            start = anchor["start_reading_order"]
            end = anchor["end_reading_order"]
            if start is not None and end is not None:
                if start <= reading_order <= end:
                    return anchor["anchor_id"]
        return None


def build_anchor_timeline(
    anchors_df: pd.DataFrame,
    elements_df: pd.DataFrame,
) -> AnchorTimeline:
    """
    Build anchor timeline with reading order ranges.

    For each anchor, computes:
    - start_reading_order: Reading order of the anchor's source element
    - end_reading_order: Reading order just before the next anchor starts
                        (or max reading order on page for last anchor)

    Args:
        anchors_df: DataFrame with anchor records.
        elements_df: DataFrame with all elements (for reading order lookup).

    Returns:
        AnchorTimeline with ranges and grouped anchor map.

    Example:
        >>> timeline = build_anchor_timeline(anchors_df, elements_df)
        >>> print(timeline.timeline_df[['anchor_id', 'start_reading_order', 'end_reading_order']])
    """
    if anchors_df.empty:
        return AnchorTimeline(timeline_df=pd.DataFrame())

    # Merge with elements for reading order
    timeline = anchors_df.merge(
        elements_df[["element_id", "reading_order", "page"]].rename(
            columns={"element_id": "source_element_id"}
        ),
        on="source_element_id",
        how="left",
        suffixes=("", "_elem"),
    )

    # Use element reading order if available, fall back to anchor reading order
    timeline["start_reading_order"] = timeline["reading_order_elem"].fillna(
        timeline["reading_order"]
    ).astype(float)  # Ensure numeric dtype for IntervalIndex
    timeline = timeline.sort_values(["page", "start_reading_order"]).reset_index(drop=True)

    # Compute end_reading_order for each anchor
    # Initialize with np.nan (float) to ensure numeric dtype for IntervalIndex
    timeline["end_reading_order"] = np.nan

    for page in timeline["page"].unique():
        page_mask = timeline["page"] == page
        page_indices = timeline[page_mask].index.tolist()

        for i, idx in enumerate(page_indices):
            if i + 1 < len(page_indices):
                # End at next anchor's start - 1
                next_idx = page_indices[i + 1]
                next_start = timeline.loc[next_idx, "start_reading_order"]
                timeline.loc[idx, "end_reading_order"] = next_start - 1
            else:
                # Last anchor on page: end at max reading order
                max_ro = elements_df[elements_df["page"] == page]["reading_order"].max()
                timeline.loc[idx, "end_reading_order"] = max_ro

    # Build grouped anchor map
    grouped_anchor_map: Dict[str, List[str]] = {}
    if "source_element_id" in timeline.columns:
        for src_elem, grp in timeline.groupby("source_element_id"):
            if len(grp) > 1:  # Grouped header
                grouped_anchor_map[src_elem] = grp["anchor_id"].tolist()

    logger.info(f"Built timeline with {len(timeline)} anchors, {len(grouped_anchor_map)} groups")

    return AnchorTimeline(
        timeline_df=timeline,
        grouped_anchor_map=grouped_anchor_map,
    )


# =============================================================================
# CONTENT ASSIGNMENT
# =============================================================================

def _assign_page_elements_vectorized(
    elements_df: pd.DataFrame,
    page_mask: np.ndarray,
    page_anchors: pd.DataFrame,
    grouped_map: Dict[str, List[str]],
) -> Tuple[int, int, int]:
    """
    Vectorized anchor assignment for a single page using IntervalIndex.

    This replaces the O(n*m) nested iterrows with O(n+m) vectorized operations.

    Args:
        elements_df: Full elements DataFrame (modified in place).
        page_mask: Boolean mask for elements on this page.
        page_anchors: Anchors for this page, sorted by start_reading_order.
        grouped_map: Map from source_element_id to list of anchor_ids.

    Returns:
        Tuple of (assigned_count, preamble_count, unassigned_count) for logging.
    """
    page_indices = elements_df.index[page_mask].values
    page_ro = elements_df.loc[page_mask, "reading_order"].values

    # Filter valid anchors (non-null start/end, start <= end)
    valid_mask = (
        page_anchors["start_reading_order"].notna() &
        page_anchors["end_reading_order"].notna() &
        (page_anchors["start_reading_order"] <= page_anchors["end_reading_order"])
    )
    valid_anchors = page_anchors[valid_mask]

    if valid_anchors.empty:
        return 0, 0, len(page_indices)

    # Build IntervalIndex from anchor ranges (closed="both" for start <= ro <= end)
    intervals = pd.IntervalIndex.from_arrays(
        valid_anchors["start_reading_order"].values,
        valid_anchors["end_reading_order"].values,
        closed="both",
    )
    anchor_ids = valid_anchors["anchor_id"].values
    source_elem_ids = valid_anchors["source_element_id"].values if "source_element_id" in valid_anchors.columns else None

    # Vectorized lookup: which interval contains each reading_order?
    # Returns -1 for no match (element falls outside all anchor ranges)
    idx_matches = intervals.get_indexer(page_ro)

    # Build assignment array
    matched_mask = idx_matches >= 0
    assigned_anchor_ids = np.where(
        matched_mask,
        anchor_ids[idx_matches],
        elements_df.loc[page_mask, "anchor_id"].values,  # Keep existing (preamble or unassigned)
    )

    # Single vectorized assignment for anchor_id
    elements_df.loc[page_mask, "anchor_id"] = assigned_anchor_ids

    # Handle grouped anchor_ids (for boxes 14-16 etc.)
    # Only iterate over matched elements that have grouped mappings
    if grouped_map and source_elem_ids is not None:
        matched_indices = np.where(matched_mask)[0]
        for i in matched_indices:
            src_elem = source_elem_ids[idx_matches[i]]
            if src_elem in grouped_map:
                elements_df.at[page_indices[i], "anchor_ids"] = grouped_map[src_elem]

    # Return counts for logging
    assigned_count = matched_mask.sum()
    preamble_count = (assigned_anchor_ids == "preamble").sum()
    unassigned_count = (assigned_anchor_ids == "unassigned").sum()

    return int(assigned_count), int(preamble_count), int(unassigned_count)


def _assign_page_elements_iterrows(
    elements_df: pd.DataFrame,
    page_mask: np.ndarray,
    page_anchors: pd.DataFrame,
    grouped_map: Dict[str, List[str]],
) -> Tuple[int, int, int]:
    """
    Legacy iterrows implementation for parity testing.

    This is the original O(n*m) nested loop implementation.
    Keep for comparison during rollout, then delete.
    """
    page_indices = elements_df.index[page_mask]
    assigned_count = 0

    for idx in page_indices:
        ro = elements_df.at[idx, "reading_order"]

        for _, anchor in page_anchors.iterrows():
            start = anchor["start_reading_order"]
            end = anchor["end_reading_order"]

            if start is not None and end is not None:
                if start <= ro <= end:
                    anchor_id = anchor["anchor_id"]
                    elements_df.loc[idx, "anchor_id"] = anchor_id
                    assigned_count += 1

                    # Check for grouped assignment
                    src_elem = anchor.get("source_element_id")
                    if src_elem in grouped_map:
                        elements_df.at[idx, "anchor_ids"] = grouped_map[src_elem]
                    break

    preamble_count = (elements_df.loc[page_mask, "anchor_id"] == "preamble").sum()
    unassigned_count = (elements_df.loc[page_mask, "anchor_id"] == "unassigned").sum()

    return assigned_count, int(preamble_count), int(unassigned_count)


def assign_elements_to_anchors(
    elements_df: pd.DataFrame,
    timeline: AnchorTimeline,
    preamble_anchor_id: str = "preamble",
) -> pd.DataFrame:
    """
    Assign content elements to anchors based on reading order ranges.

    Each element is assigned to the anchor whose reading order range
    contains the element's reading order. Elements before the first
    anchor are assigned to "preamble".

    For grouped anchors (e.g., Boxes 14-16), elements are assigned to
    ALL anchors in the group via the anchor_ids column.

    Args:
        elements_df: DataFrame with elements to assign.
        timeline: AnchorTimeline with reading order ranges.
        preamble_anchor_id: Anchor ID for pre-first-anchor content.

    Returns:
        DataFrame with added columns:
        - anchor_id: Primary anchor assignment
        - anchor_ids: List of all anchor assignments (for grouped)

    Example:
        >>> elements_df = assign_elements_to_anchors(elements_df, timeline)
        >>> box_1a_content = elements_df[elements_df['anchor_id'] == 'box_1a']
    """
    elements_df = elements_df.copy()
    elements_df["anchor_id"] = "unassigned"
    elements_df["anchor_ids"] = None

    anchor_timeline = timeline.timeline_df
    grouped_map = timeline.grouped_anchor_map

    if anchor_timeline.empty:
        elements_df["anchor_id"] = preamble_anchor_id
        return elements_df

    # Select assignment implementation based on feature flag
    assign_fn = (
        _assign_page_elements_vectorized
        if USE_VECTORIZED_ASSIGNMENT
        else _assign_page_elements_iterrows
    )

    # Track per-page stats for debugging
    total_assigned = 0
    total_preamble = 0
    total_unassigned = 0

    for page in sorted(elements_df["page"].unique()):
        page_mask = (elements_df["page"] == page).values
        page_anchors = anchor_timeline[anchor_timeline["page"] == page].copy()

        if page_anchors.empty:
            # No anchors on this page - all elements go to preamble
            elements_df.loc[page_mask, "anchor_id"] = preamble_anchor_id
            total_preamble += page_mask.sum()
            continue

        # Sort anchors by start_reading_order for deterministic precedence
        page_anchors = page_anchors.sort_values("start_reading_order")
        first_anchor_start = page_anchors["start_reading_order"].min()

        # Elements before first anchor go to preamble (vectorized)
        preamble_mask = page_mask & (
            elements_df["reading_order"].values < first_anchor_start
        )
        elements_df.loc[preamble_mask, "anchor_id"] = preamble_anchor_id

        # Assign remaining elements to anchors
        assigned, preamble, unassigned = assign_fn(
            elements_df, page_mask, page_anchors, grouped_map
        )
        total_assigned += assigned
        total_preamble += preamble
        total_unassigned += unassigned

        logger.debug(
            f"Page {page}: assigned={assigned}, preamble={preamble}, unassigned={unassigned}"
        )

    # Final sanity check: no elements should remain unassigned unless expected
    final_unassigned = (elements_df["anchor_id"] == "unassigned").sum()
    if final_unassigned > 0:
        logger.warning(
            f"{final_unassigned} elements remain unassigned - check anchor ranges"
        )

    total = len(elements_df)
    logger.info(
        f"Assigned {total - final_unassigned}/{total} elements to anchors "
        f"(vectorized={USE_VECTORIZED_ASSIGNMENT})"
    )

    return elements_df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_and_assign_anchors(
    elements_df: pd.DataFrame,
    role_box_header: str = ROLE_BOX_HEADER,
    role_section_header: str = ROLE_SECTION_HEADER,
    role_subsection_header: str = ROLE_SUBSECTION_HEADER,
    section_map: Optional[Dict[str, str]] = None,
    expected_boxes: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, AnchorTimeline, Optional[BoxValidationResult]]:
    """
    Complete anchor extraction and assignment pipeline.

    This convenience function runs the full workflow:
    1. Extract anchors from classified elements
    2. Build anchor timeline
    3. Assign elements to anchors
    4. Validate box coverage (optional)

    Args:
        elements_df: DataFrame with classified elements.
        role_box_header: Role value for box headers.
        role_section_header: Role value for section headers.
        role_subsection_header: Role value for subsection headers.
        section_map: Custom section ID mapping.
        expected_boxes: Set of expected box keys for validation.

    Returns:
        Tuple of (anchors_df, elements_df_with_assignments, timeline, validation_result)

    Example:
        >>> anchors_df, elements_df, timeline, validation = extract_and_assign_anchors(
        ...     elements_df,
        ...     expected_boxes=EXPECTED_BOXES_1099DIV
        ... )
        >>> if not validation.passed:
        ...     print(f"Missing: {validation.missing}")
    """
    # Extract anchors
    extraction = extract_anchors(
        elements_df,
        role_box_header=role_box_header,
        role_section_header=role_section_header,
        role_subsection_header=role_subsection_header,
        section_map=section_map,
    )

    # Build timeline
    timeline = build_anchor_timeline(extraction.anchors_df, elements_df)

    # Assign elements
    elements_df = assign_elements_to_anchors(elements_df, timeline)

    # Validate if expected boxes provided
    validation = None
    if expected_boxes:
        validation = validate_box_coverage(extraction.anchors_df, expected_boxes)

    return extraction.anchors_df, elements_df, timeline, validation


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def verify_vectorized_parity(
    elements_df: pd.DataFrame,
    timeline: AnchorTimeline,
    preamble_anchor_id: str = "preamble",
) -> Dict[str, Any]:
    """
    Verify vectorized implementation matches iterrows implementation.

    Runs both implementations on the same input and compares results.
    Use this during rollout to ensure correctness before removing legacy code.

    Args:
        elements_df: DataFrame with elements to assign.
        timeline: AnchorTimeline with reading order ranges.
        preamble_anchor_id: Anchor ID for pre-first-anchor content.

    Returns:
        Dict with comparison results:
        - matched: True if both implementations produce identical results
        - vectorized_time_ms: Time for vectorized implementation
        - iterrows_time_ms: Time for iterrows implementation
        - speedup: vectorized speedup factor
        - mismatched_count: Number of differing anchor_id assignments
        - mismatched_indices: List of indices with different assignments (first 10)

    Example:
        >>> result = verify_vectorized_parity(elements_df, timeline)
        >>> assert result["matched"], f"Mismatch: {result['mismatched_count']} rows differ"
        >>> print(f"Speedup: {result['speedup']:.1f}x")
    """
    import time

    grouped_map = timeline.grouped_anchor_map

    # Run vectorized implementation
    df_vec = elements_df.copy()
    df_vec["anchor_id"] = "unassigned"
    df_vec["anchor_ids"] = None

    t0 = time.perf_counter()
    for page in sorted(df_vec["page"].unique()):
        page_mask = (df_vec["page"] == page).values
        page_anchors = timeline.timeline_df[timeline.timeline_df["page"] == page].copy()
        if page_anchors.empty:
            df_vec.loc[page_mask, "anchor_id"] = preamble_anchor_id
            continue
        page_anchors = page_anchors.sort_values("start_reading_order")
        first_anchor_start = page_anchors["start_reading_order"].min()
        preamble_mask = page_mask & (df_vec["reading_order"].values < first_anchor_start)
        df_vec.loc[preamble_mask, "anchor_id"] = preamble_anchor_id
        _assign_page_elements_vectorized(df_vec, page_mask, page_anchors, grouped_map)
    vec_time = (time.perf_counter() - t0) * 1000

    # Run iterrows implementation
    df_iter = elements_df.copy()
    df_iter["anchor_id"] = "unassigned"
    df_iter["anchor_ids"] = None

    t0 = time.perf_counter()
    for page in sorted(df_iter["page"].unique()):
        page_mask = (df_iter["page"] == page).values
        page_anchors = timeline.timeline_df[timeline.timeline_df["page"] == page].copy()
        if page_anchors.empty:
            df_iter.loc[page_mask, "anchor_id"] = preamble_anchor_id
            continue
        page_anchors = page_anchors.sort_values("start_reading_order")
        first_anchor_start = page_anchors["start_reading_order"].min()
        preamble_mask = page_mask & (df_iter["reading_order"].values < first_anchor_start)
        df_iter.loc[preamble_mask, "anchor_id"] = preamble_anchor_id
        _assign_page_elements_iterrows(df_iter, page_mask, page_anchors, grouped_map)
    iter_time = (time.perf_counter() - t0) * 1000

    # Compare results
    mismatched = df_vec["anchor_id"] != df_iter["anchor_id"]
    mismatched_indices = df_vec.index[mismatched].tolist()

    return {
        "matched": len(mismatched_indices) == 0,
        "vectorized_time_ms": round(vec_time, 2),
        "iterrows_time_ms": round(iter_time, 2),
        "speedup": round(iter_time / vec_time, 1) if vec_time > 0 else float("inf"),
        "mismatched_count": len(mismatched_indices),
        "mismatched_indices": mismatched_indices[:10],
        "total_elements": len(elements_df),
    }
