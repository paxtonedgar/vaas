"""
Element extraction and classification.

This module handles:
- Splitting blocks into elements based on split triggers
- Classifying elements into roles (BoxHeader, SectionHeader, etc.)
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from vaas.constants import (
    ROLE_BOX_HEADER,
    ROLE_SECTION_HEADER,
    ROLE_SUBSECTION_HEADER,
    ROLE_LIST_BLOCK,
    ROLE_PAGE_ARTIFACT,
    ROLE_BODY_TEXT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Page artifact pattern
PAGE_ARTIFACT_RX = re.compile(
    r"(?:^Page\s+\d+|Instructions for Form|Department of the Treasury|"
    r"Internal Revenue Service|\(Rev\.\s*\w*\s*\d{4}\)|www\.irs\.gov|"
    r"^-\d+-$|Cat\.\s*No\.\s*\d+)",
    re.IGNORECASE
)

# Box pattern for classification
BOX_WEAK_RX = re.compile(
    r"^Box(?:es)?\s+\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?[.:]",
    re.IGNORECASE
)

# List patterns
BULLET_RX = re.compile(r"^\s*[•\-\*]\s*")
ENUM_RX = re.compile(r"^\s*\(?\d{1,2}\s*[.)]\s+[A-Za-z]")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ElementSplitResult:
    """Result of element splitting."""
    elements_df: pd.DataFrame
    element_count: int
    split_trigger_count: int


@dataclass
class ElementClassificationResult:
    """Result of element classification."""
    elements_df: pd.DataFrame
    role_distribution: Dict[str, int]


# =============================================================================
# BLOCK SPLITTING
# =============================================================================

def split_blocks_into_elements(
    line_df: pd.DataFrame,
    page_mid_x: float = 300.0,
) -> pd.DataFrame:
    """
    Split blocks into elements based on split triggers.

    Uses split_trigger and split_kind flags from line DataFrame to determine
    segment boundaries within blocks.

    Args:
        line_df: Line DataFrame with split_trigger, split_kind columns.
        page_mid_x: X coordinate for column split (left vs right).

    Returns:
        Elements DataFrame with one row per element.
    """
    if line_df.empty:
        return pd.DataFrame()

    if "split_trigger" not in line_df.columns:
        logger.warning("No split_trigger column, treating entire blocks as elements")
        line_df = line_df.copy()
        line_df["split_trigger"] = False
        line_df["split_kind"] = "body"

    rows = []

    for (doc_id, page, block_id), group in line_df.groupby(
        ["doc_id", "page", "block_id"], sort=False
    ):
        g = group.sort_values(
            ["geom_y0", "geom_x0", "line_id"], kind="mergesort"
        ).reset_index(drop=True)

        # Find split points
        starts = [0]
        for i in range(len(g)):
            if g.loc[i, "split_trigger"] and i not in starts:
                starts.append(i)
                if i + 1 < len(g):
                    starts.append(i + 1)

        starts = sorted(set(starts))

        # Build segment bounds
        seg_bounds = []
        for si, st in enumerate(starts):
            en = starts[si + 1] if si + 1 < len(starts) else len(g)
            if st < en:
                seg_bounds.append((st, en))

        # Create elements
        seg_idx = 0
        for st, en in seg_bounds:
            seg = g.iloc[st:en].copy()
            first_is_trigger = bool(seg.iloc[0]["split_trigger"])
            kind = seg.iloc[0]["split_kind"] if first_is_trigger else "body"

            # Aggregate text
            seg_text = "\n".join([
                str(x).strip()
                for x in seg["line_text"].tolist()
                if str(x).strip()
            ]).strip()

            # Compute bounding box
            x0 = float(seg["geom_x0"].min())
            y0 = float(seg["geom_y0"].min())
            x1 = float(seg["geom_x1"].max())
            y1 = float(seg["geom_y1"].max())
            bbox = [x0, y0, x1, y1]

            # Collect source IDs
            source_line_ids = list(map(str, seg["line_id"].tolist()))
            span_ids = []
            for sids in seg["span_ids"].tolist():
                span_ids.extend(list(map(str, sids)))
            seen = set()
            span_ids = [s for s in span_ids if not (s in seen or seen.add(s))]

            element_id = f"{doc_id}:{int(page)}:{block_id}:seg{seg_idx}"
            seg_idx += 1

            rows.append({
                "doc_id": str(doc_id),
                "page": int(page),
                "block_id": str(block_id),
                "element_id": element_id,
                "text": seg_text,
                "bbox": bbox,
                "geom_x0": x0,
                "geom_y0": y0,
                "geom_x1": x1,
                "geom_y1": y1,
                "source_line_ids": source_line_ids,
                "source_span_ids": span_ids,
                "split_kind": kind,
            })

    elements_df = pd.DataFrame(rows)

    if not elements_df.empty:
        # Assign column based on x position
        elements_df["x_column"] = (elements_df["geom_x0"] >= page_mid_x).astype(int)

        # Sort by reading order (page, column, y0, x0)
        # This is the canonical document order
        elements_df = elements_df.sort_values(
            ["doc_id", "page", "x_column", "geom_y0", "geom_x0", "element_id"],
            kind="mergesort"
        )

        # doc_reading_order: document-global, strictly increasing 0..N-1
        # This is the canonical order used for cross-page timeline logic
        elements_df["doc_reading_order"] = elements_df.groupby("doc_id").cumcount()

        # page_reading_order: page-local, for debugging/page-specific logic only
        # DO NOT use this for any cross-page comparison or interval assignment
        elements_df["page_reading_order"] = elements_df.groupby(["doc_id", "page"]).cumcount()

        # Keep 'reading_order' as alias for doc_reading_order for backward compatibility
        # All downstream code should treat reading_order as document-global
        elements_df["reading_order"] = elements_df["doc_reading_order"]

        elements_df = elements_df.reset_index(drop=True)

    logger.info(f"Split into {len(elements_df)} elements")
    return elements_df


def split_elements_with_result(
    line_df: pd.DataFrame,
    page_mid_x: float = 300.0,
) -> ElementSplitResult:
    """
    Split blocks into elements and return result with metadata.

    Args:
        line_df: Line DataFrame with split flags.
        page_mid_x: X coordinate for column split.

    Returns:
        ElementSplitResult with elements and counts.
    """
    split_trigger_count = int(line_df["split_trigger"].sum()) if "split_trigger" in line_df.columns else 0

    elements_df = split_blocks_into_elements(line_df, page_mid_x)

    return ElementSplitResult(
        elements_df=elements_df,
        element_count=len(elements_df),
        split_trigger_count=split_trigger_count,
    )


# =============================================================================
# ELEMENT CLASSIFICATION
# =============================================================================

def classify_elements(
    elements_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify elements into roles based on split_kind and text patterns.

    Roles:
    - BoxHeader: Box header elements (from split_kind="box")
    - SectionHeader: Main section headers
    - SubsectionHeader: Subsection headers
    - ListBlock: Bullet or enumerated list items
    - PageArtifact: Page markers, footers, headers
    - BodyTextBlock: Regular text content (default)

    Args:
        elements_df: Elements DataFrame with split_kind column.

    Returns:
        Elements DataFrame with role and role_conf columns.
    """
    if elements_df.empty:
        return elements_df

    df = elements_df.copy()

    # Initialize with default
    df["role"] = ROLE_BODY_TEXT
    df["role_conf"] = 0.7

    text = df["text"].fillna("").astype(str).str.strip()

    # Box headers
    is_box_split = df["split_kind"] == "box"
    box_pattern_match = text.str.match(BOX_WEAK_RX, na=False)
    df.loc[is_box_split & box_pattern_match, "role"] = ROLE_BOX_HEADER
    df.loc[is_box_split & box_pattern_match, "role_conf"] = 0.95
    df.loc[is_box_split & ~box_pattern_match, "role"] = ROLE_BOX_HEADER
    df.loc[is_box_split & ~box_pattern_match, "role_conf"] = 0.75

    # Section headers
    is_section_split = df["split_kind"] == "section"
    df.loc[is_section_split, "role"] = ROLE_SECTION_HEADER
    df.loc[is_section_split, "role_conf"] = 0.90

    # Subsection headers
    is_subsection_split = df["split_kind"] == "subsection"
    df.loc[is_subsection_split, "role"] = ROLE_SUBSECTION_HEADER
    df.loc[is_subsection_split, "role_conf"] = 0.85

    # List items (only if not already classified)
    is_list = text.str.match(BULLET_RX, na=False) | text.str.match(ENUM_RX, na=False)
    is_list = is_list & (df["role"] == ROLE_BODY_TEXT)
    df.loc[is_list, "role"] = ROLE_LIST_BLOCK
    df.loc[is_list, "role_conf"] = 0.85

    # Page artifacts from split_kind
    is_split_artifact = df["split_kind"] == "page_artifact"
    df.loc[is_split_artifact, "role"] = ROLE_PAGE_ARTIFACT
    df.loc[is_split_artifact, "role_conf"] = 0.99

    # Other artifacts by content pattern
    is_artifact = text.str.contains(PAGE_ARTIFACT_RX, na=False, regex=True)
    is_short = text.str.len() < 100
    is_artifact = is_artifact & is_short & (df["role"] == ROLE_BODY_TEXT)
    df.loc[is_artifact, "role"] = ROLE_PAGE_ARTIFACT
    df.loc[is_artifact, "role_conf"] = 0.90

    logger.debug(f"Classified {len(df)} elements")
    return df


def classify_elements_with_result(
    elements_df: pd.DataFrame,
) -> ElementClassificationResult:
    """
    Classify elements and return result with distribution.

    Args:
        elements_df: Elements DataFrame.

    Returns:
        ElementClassificationResult with classified elements and distribution.
    """
    classified = classify_elements(elements_df)

    role_distribution = {}
    if not classified.empty and "role" in classified.columns:
        role_distribution = classified["role"].value_counts().to_dict()

    return ElementClassificationResult(
        elements_df=classified,
        role_distribution=role_distribution,
    )


