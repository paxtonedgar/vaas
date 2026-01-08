"""
Line-level extraction and geometry computation.

This module builds line DataFrames from spans, computes geometry features,
and prepares lines for structural analysis.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .geometry import safe_bbox, is_bold, margin_tolerance

logger = logging.getLogger(__name__)


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Box header patterns
BOX_STRONG_RX = re.compile(
    r"^Box(?:es)?\s+\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?"
    r"(?:\.\s+[A-Z]|:\s+[A-Z]|\s+[-–]\s+[A-Z])",
    re.IGNORECASE
)
BOX_WEAK_RX = re.compile(
    r"^Box(?:es)?\s+\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?[.:]",
    re.IGNORECASE
)

# Section header patterns
SECTION_HDR_RX = re.compile(
    r"^(Specific Instructions|General Instructions|Definitions|"
    r"Future Developments|Reminders|What's New|How To|Where To|"
    r"Paperwork Reduction Act Notice|Additional Information)\b",
    re.IGNORECASE
)

# Page marker pattern
PAGE_MARKER_RX = re.compile(r"^\s*-\s*\d{1,3}\s*-\s*$")

# List item patterns
BULLET_RX = re.compile(r"^\s*[•\-\*]\s*")
ENUM_RX = re.compile(r"^\s*\(?\d{1,2}\s*[.)]\s+[A-Za-z]")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LineBuildResult:
    """Result of line DataFrame building."""
    line_df: pd.DataFrame
    line_count: int
    col_info_df: pd.DataFrame


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def build_line_dataframe(
    spans_df: pd.DataFrame,
    body_size: float,
) -> pd.DataFrame:
    """
    Build line-level DataFrame from spans.

    Aggregates spans into lines, computing:
    - Line text (concatenated spans)
    - Line geometry (bounding box union)
    - Font properties (size, bold)

    Args:
        spans_df: Spans DataFrame from PDF extraction.
        body_size: Body font size for emphasis detection.

    Returns:
        Line DataFrame with geometry and properties.
    """
    if spans_df.empty:
        return pd.DataFrame()

    tmp = spans_df.copy()

    # Extract bbox components
    tmp[["x0", "y0", "x1", "y1"]] = pd.DataFrame(
        tmp["bbox"].apply(safe_bbox).tolist(), index=tmp.index
    )

    # Compute bold flag
    tmp["bold"] = tmp.apply(
        lambda r: is_bold(r.get("font", ""), r.get("flags", 0)), axis=1
    )
    tmp["text"] = tmp["text"].fillna("").astype(str)

    # Group by line
    line_keys = ["doc_id", "page", "block_id", "line_id"]
    tmp_sorted = tmp.sort_values(line_keys + ["x0", "span_id"], kind="mergesort")

    line_df = tmp_sorted.groupby(line_keys, as_index=False).agg(
        line_text=("text", lambda x: "".join(x).strip()),
        geom_x0=("x0", "min"),
        geom_y0=("y0", "min"),
        geom_x1=("x1", "max"),
        geom_y1=("y1", "max"),
        line_size=("size", "median"),
        line_bold=("bold", lambda x: bool(np.any(x))),
        span_ids=("span_id", lambda x: list(map(str, x))),
    )

    # Sort by reading order
    line_df = line_df.sort_values(
        ["doc_id", "page", "block_id", "geom_y0", "geom_x0", "line_id"],
        kind="mergesort"
    )

    # Line index within block
    line_df["line_idx_in_block"] = line_df.groupby(
        ["doc_id", "page", "block_id"]
    ).cumcount()

    logger.debug(f"Built {len(line_df)} lines from {len(spans_df)} spans")
    return line_df


def add_block_geometry(line_df: pd.DataFrame) -> pd.DataFrame:
    """Add block-level geometry columns."""
    if line_df.empty:
        return line_df

    df = line_df.copy()

    block_geom = df.groupby(["doc_id", "page", "block_id"], as_index=False).agg(
        block_x0=("geom_x0", "min"),
        block_x1=("geom_x1", "max")
    )
    df = df.merge(block_geom, on=["doc_id", "page", "block_id"], how="left")
    df["block_width"] = df["block_x1"] - df["block_x0"]
    df["margin_tol"] = df["block_width"].apply(margin_tolerance)
    df["left_aligned"] = (df["geom_x0"] - df["block_x0"]).abs() <= df["margin_tol"]

    return df


def add_page_geometry(line_df: pd.DataFrame) -> pd.DataFrame:
    """Add page-level geometry columns."""
    if line_df.empty:
        return line_df

    df = line_df.copy()

    page_geom = df.groupby(["doc_id", "page"], as_index=False).agg(
        page_x0=("geom_x0", "min"),
        page_x1=("geom_x1", "max")
    )
    page_geom["page_mid_x"] = (page_geom["page_x0"] + page_geom["page_x1"]) / 2
    page_geom["page_width"] = page_geom["page_x1"] - page_geom["page_x0"]
    df = df.merge(
        page_geom[["doc_id", "page", "page_mid_x", "page_width"]],
        on=["doc_id", "page"],
        how="left"
    )

    return df


def add_emphasis_flags(
    line_df: pd.DataFrame,
    body_size: float,
) -> pd.DataFrame:
    """Add emphasis detection flags."""
    if line_df.empty:
        return line_df

    df = line_df.copy()
    df["has_emphasis"] = (
        (df["line_size"] > float(body_size) + 0.5) |
        (df["line_bold"] == True)
    )
    return df


def add_list_item_flags(line_df: pd.DataFrame) -> pd.DataFrame:
    """Add list item detection flags."""
    if line_df.empty:
        return line_df

    df = line_df.copy()
    t = df["line_text"].fillna("").astype(str).str.strip()
    df["_is_bullet"] = t.str.match(BULLET_RX, na=False)
    df["_is_enum"] = t.str.match(ENUM_RX, na=False)
    df["_is_list_item"] = df["_is_bullet"] | df["_is_enum"]
    return df


def add_text_properties(line_df: pd.DataFrame) -> pd.DataFrame:
    """Add text property columns (char count, heading detection)."""
    if line_df.empty:
        return line_df

    df = line_df.copy()
    t = df["line_text"].fillna("").astype(str).str.strip()

    # Basic text properties
    df["line_char_count"] = t.str.len()
    df["is_heading_length"] = df["line_char_count"].between(12, 60)
    df["ends_with_period"] = t.str.endswith(".")
    df["has_multiple_words"] = t.str.count(r"\s+") >= 1

    # Title-case detection
    df["is_title_case_ish"] = t.str.match(r"^[A-Z][A-Za-z0-9\s\-\(\)\']+$", na=False)

    # Date detection
    df["looks_like_date"] = t.str.match(r"^[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}$", na=False)

    return df


def add_header_pattern_flags(line_df: pd.DataFrame) -> pd.DataFrame:
    """Add header pattern detection flags."""
    if line_df.empty:
        return line_df

    df = line_df.copy()
    t = df["line_text"].fillna("").astype(str).str.strip()

    df["is_box_strong"] = t.str.match(BOX_STRONG_RX, na=False)
    df["is_box_weak"] = t.str.match(BOX_WEAK_RX, na=False)
    df["is_section"] = t.str.match(SECTION_HDR_RX, na=False)

    return df


def add_page_marker_flags(line_df: pd.DataFrame) -> pd.DataFrame:
    """Add page marker detection flags."""
    if line_df.empty:
        return line_df

    df = line_df.copy()
    t = df["line_text"].fillna("").astype(str).str.strip()

    df["line_center_x"] = (df["geom_x0"] + df["geom_x1"]) / 2
    df["line_width"] = df["geom_x1"] - df["geom_x0"]
    df["is_page_marker_text"] = t.str.match(PAGE_MARKER_RX, na=False)
    df["is_centered"] = (df["line_center_x"] - df["page_mid_x"]).abs() <= df["page_width"] * 0.15
    df["is_narrow"] = df["line_width"] <= df["page_width"] * 0.15
    df["is_page_marker"] = df["is_page_marker_text"] & df["is_centered"] & df["is_narrow"]

    return df


def add_structural_filters(line_df: pd.DataFrame) -> pd.DataFrame:
    """Add structural filter columns (header/footer bands, repeated lines, gaps)."""
    if line_df.empty:
        return line_df

    df = line_df.copy()
    t = df["line_text"].fillna("").astype(str).str.strip()

    # Basic text properties
    df["line_char_count"] = t.str.len()
    df["is_heading_length"] = df["line_char_count"].between(12, 60)
    df["ends_with_period"] = t.str.endswith(".")
    df["has_multiple_words"] = t.str.count(r"\s+") >= 1
    df["is_title_case_ish"] = t.str.match(r"^[A-Z][A-Za-z0-9\s\-\(\)\']+$", na=False)
    df["looks_like_date"] = t.str.match(r"^[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}$", na=False)

    # Early-in-doc check
    df["min_page_doc"] = df.groupby("doc_id")["page"].transform("min")
    df["is_early_in_doc"] = (df["page"] == df["min_page_doc"]) & (df["line_idx_in_block"] < 8)

    # Header/footer band exclusion
    page_y_bounds = df.groupby(["doc_id", "page"], as_index=False).agg(
        page_y_min=("geom_y0", "min"),
        page_y_max=("geom_y1", "max")
    )
    page_y_bounds["page_h"] = page_y_bounds["page_y_max"] - page_y_bounds["page_y_min"]
    df = df.merge(
        page_y_bounds[["doc_id", "page", "page_y_min", "page_y_max", "page_h"]],
        on=["doc_id", "page"],
        how="left"
    )
    df["in_header_band"] = df["geom_y0"] <= df["page_y_min"] + 0.10 * df["page_h"]
    df["in_footer_band"] = df["geom_y1"] >= df["page_y_max"] - 0.10 * df["page_h"]

    # Repeated-line artifact detection
    df["_norm_text"] = t.str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True).str.strip()
    page_count_per_text = df.groupby(["doc_id", "_norm_text"])["page"].transform("nunique")
    df["is_repeated_across_pages"] = page_count_per_text >= 2
    df["is_artifact_repeat"] = (
        df["is_repeated_across_pages"] &
        (df["in_header_band"] | df["in_footer_band"])
    )

    # Gap above computation
    df = df.sort_values(["doc_id", "page", "block_id", "geom_y0"]).reset_index(drop=True)
    df["prev_y1"] = df.groupby(["doc_id", "page", "block_id"])["geom_y1"].shift(1)
    df["gap_above"] = df["geom_y0"] - df["prev_y1"].fillna(df["geom_y0"])
    median_gap = df.groupby(["doc_id", "page", "block_id"])["gap_above"].transform("median")
    df["has_large_gap_above"] = (df["gap_above"] >= median_gap * 1.3) | (df["line_idx_in_block"] <= 1)

    return df


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def build_lines_with_features(
    spans_df: pd.DataFrame,
    body_size: float,
    col_info_df: Optional[pd.DataFrame] = None,
) -> LineBuildResult:
    """
    Build complete line DataFrame with all features.

    Args:
        spans_df: Spans DataFrame from PDF extraction.
        body_size: Body font size for emphasis detection.
        col_info_df: Optional pre-computed column info.

    Returns:
        LineBuildResult with line DataFrame and metadata.
    """
    # Build basic line DataFrame
    line_df = build_line_dataframe(spans_df, body_size)

    if line_df.empty:
        return LineBuildResult(
            line_df=line_df,
            line_count=0,
            col_info_df=pd.DataFrame(),
        )

    # Add geometry
    line_df = add_block_geometry(line_df)
    line_df = add_page_geometry(line_df)
    line_df = add_emphasis_flags(line_df, body_size)
    line_df = add_list_item_flags(line_df)

    # Import column detection (circular import avoidance)
    from .columns import detect_columns_for_document, assign_line_columns

    # Detect columns if not provided
    if col_info_df is None:
        col_info_df = detect_columns_for_document(line_df)

    # Assign columns
    line_df = assign_line_columns(line_df, col_info_df)

    # Clean up temporary columns
    line_df.drop(columns=["_is_bullet", "_is_enum", "_is_list_item"], inplace=True, errors="ignore")

    # Add header patterns and structural filters
    line_df = add_page_marker_flags(line_df)
    line_df = add_header_pattern_flags(line_df)
    line_df = add_structural_filters(line_df)

    logger.info(f"Built {len(line_df)} lines with features")

    return LineBuildResult(
        line_df=line_df,
        line_count=len(line_df),
        col_info_df=col_info_df,
    )


# Legacy wrapper
def build_lines_legacy(
    spans_df: pd.DataFrame,
    body_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy wrapper returning (line_df, col_info_df).

    For drop-in replacement in run_pipeline.py.
    """
    result = build_lines_with_features(spans_df, body_size)

    print(f"Lines: {result.line_count}")
    two_col_pages = result.col_info_df[result.col_info_df["num_columns"] == 2] if not result.col_info_df.empty else pd.DataFrame()
    print(f"Column detection: {len(two_col_pages)}/{len(result.col_info_df)} pages have 2 columns")

    return result.line_df, result.col_info_df
