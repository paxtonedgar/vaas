"""
Column detection for multi-column PDF layouts.

This module detects whether pages have one or two columns and assigns
lines to their respective columns. This is critical for correct reading
order in two-column IRS instruction PDFs.

Algorithm Overview:
1. Collect x0 (left edge) coordinates of all non-list-item lines
2. Round x0 values to cluster nearby values (4px buckets)
3. Count frequency of each x0 bucket to find peaks
4. If two peaks exist with sufficient count ratio AND distance, it's two-column
5. Split point is midway between the two column margins

Key Parameters:
- min_peak_ratio: Second peak must have at least this ratio of first peak's count
- min_distance_pct: Columns must be at least this % of page width apart
- x0_bucket_size: Rounding bucket for x0 clustering
- min_peak_occurrences: Minimum occurrences to be considered a peak
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnInfo:
    """
    Column detection result for a single page.

    Attributes:
        num_columns: Number of columns detected (1 or 2).
        col_0_x0: X-coordinate of left column's left margin.
        col_1_x0: X-coordinate of right column's left margin (if 2 columns).
        split_x: X-coordinate that divides the two columns (if 2 columns).
    """

    num_columns: int
    col_0_x0: float
    col_1_x0: Optional[float] = None
    split_x: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "num_columns": self.num_columns,
            "col_0_x0": self.col_0_x0,
            "col_1_x0": self.col_1_x0 if self.col_1_x0 is not None else np.nan,
            "col_split_x": self.split_x if self.split_x is not None else np.nan,
        }


# Default regex patterns for list item detection
# These are excluded from column detection to avoid locking onto indent levels
DEFAULT_BULLET_PATTERN = re.compile(r"^\s*[•\-\*]\s*")
DEFAULT_ENUM_PATTERN = re.compile(r"^\s*\(?\d{1,2}\s*[.)]\s+[A-Za-z]")


def detect_columns_for_page(
    page_df: pd.DataFrame,
    min_peak_ratio: float = 0.25,
    min_distance_pct: float = 0.25,
    x0_bucket_size: float = 4.0,
    min_peak_occurrences: int = 3,
    min_text_length: int = 20,
    page_width_col: str = "page_width",
    x0_col: str = "geom_x0",
    text_col: str = "line_text",
    list_item_col: Optional[str] = "_is_list_item",
) -> ColumnInfo:
    """
    Detect 1 or 2 columns for a single page based on x0 peak analysis.

    Uses rounded x0 values and frequency counts to find dominant column margins.
    Excludes bullets/enumerations to avoid locking onto indent levels.

    Algorithm:
    1. Filter to non-list-item lines with sufficient text length
    2. Round x0 values to buckets (default 4px) to cluster nearby values
    3. Count frequency of each bucket
    4. Find peaks (buckets with >= min_peak_occurrences)
    5. Check if two peaks satisfy both conditions:
       - Second peak count >= min_peak_ratio * first peak count
       - Distance between peaks >= min_distance_pct * page_width
    6. If both conditions met, it's two-column; otherwise one-column

    Args:
        page_df: DataFrame containing lines for a single page.
        min_peak_ratio: Minimum ratio of second peak count to first peak count.
        min_distance_pct: Minimum distance between columns as fraction of page width.
        x0_bucket_size: Size of x0 rounding bucket in pixels.
        min_peak_occurrences: Minimum count for a bucket to be considered a peak.
        min_text_length: Minimum text length for lines used in detection.
        page_width_col: Column name for page width.
        x0_col: Column name for x0 coordinate.
        text_col: Column name for line text.
        list_item_col: Column name for list item flag (None to skip filtering).

    Returns:
        ColumnInfo with detection results.

    Example:
        >>> info = detect_columns_for_page(page_df)
        >>> if info.num_columns == 2:
        ...     print(f"Two columns: left at {info.col_0_x0}, right at {info.col_1_x0}")
    """
    # Get page width (default to typical letter size if not available)
    if page_width_col in page_df.columns and len(page_df) > 0:
        page_width = float(page_df[page_width_col].iloc[0])
    else:
        page_width = 612.0  # Standard letter width in points

    # Build filter mask
    if x0_col not in page_df.columns:
        return ColumnInfo(num_columns=1, col_0_x0=0.0)

    # Compute text length if needed
    if text_col in page_df.columns:
        text_len = page_df[text_col].fillna("").astype(str).str.len()
        text_mask = text_len >= min_text_length
    else:
        text_mask = pd.Series(True, index=page_df.index)

    # Filter out list items if column exists
    if list_item_col and list_item_col in page_df.columns:
        list_mask = ~page_df[list_item_col].fillna(False)
    else:
        list_mask = pd.Series(True, index=page_df.index)

    # Apply filters
    mask = text_mask & list_mask
    filtered = page_df.loc[mask, x0_col].dropna()

    # Fall back to all lines if not enough filtered data
    if len(filtered) < 5:
        filtered = page_df[x0_col].dropna()

    # Not enough data for detection
    if len(filtered) < 2:
        min_x0 = float(filtered.min()) if len(filtered) > 0 else 0.0
        return ColumnInfo(num_columns=1, col_0_x0=min_x0)

    # Round x0 to buckets to cluster nearby values
    x0_rounded = (filtered / x0_bucket_size).round() * x0_bucket_size

    # Count frequencies of rounded x0
    x0_counts = x0_rounded.value_counts().sort_index()

    # Get significant peaks (at least min_peak_occurrences)
    significant = x0_counts[x0_counts >= min_peak_occurrences].sort_values(ascending=False)

    if len(significant) == 0:
        return ColumnInfo(num_columns=1, col_0_x0=float(x0_rounded.min()))

    peak1_x0 = float(significant.index[0])
    peak1_count = int(significant.iloc[0])

    # Look for second column: needs sufficient count AND distance from peak1
    if len(significant) >= 2:
        min_distance = page_width * min_distance_pct

        for i in range(1, len(significant)):
            peak2_x0 = float(significant.index[i])
            peak2_count = int(significant.iloc[i])
            distance = abs(peak2_x0 - peak1_x0)

            # Check both conditions
            if peak2_count >= peak1_count * min_peak_ratio and distance >= min_distance:
                left_x0 = min(peak1_x0, peak2_x0)
                right_x0 = max(peak1_x0, peak2_x0)
                split_x = (left_x0 + right_x0) / 2

                return ColumnInfo(
                    num_columns=2,
                    col_0_x0=left_x0,
                    col_1_x0=right_x0,
                    split_x=split_x,
                )

    return ColumnInfo(num_columns=1, col_0_x0=peak1_x0)


def mark_list_items(
    df: pd.DataFrame,
    text_col: str = "line_text",
    bullet_pattern: re.Pattern = DEFAULT_BULLET_PATTERN,
    enum_pattern: re.Pattern = DEFAULT_ENUM_PATTERN,
) -> pd.DataFrame:
    """
    Mark lines that are bullet points or enumerated list items.

    These lines are excluded from column detection because their indentation
    doesn't represent column margins.

    Args:
        df: DataFrame with line data.
        text_col: Column name for line text.
        bullet_pattern: Regex pattern for bullet points.
        enum_pattern: Regex pattern for enumerated items.

    Returns:
        DataFrame with added columns: _is_bullet, _is_enum, _is_list_item
    """
    df = df.copy()

    text = df[text_col].fillna("").astype(str).str.strip()
    df["_is_bullet"] = text.str.match(bullet_pattern, na=False)
    df["_is_enum"] = text.str.match(enum_pattern, na=False)
    df["_is_list_item"] = df["_is_bullet"] | df["_is_enum"]

    return df


def detect_columns_for_document(
    df: pd.DataFrame,
    min_peak_ratio: float = 0.25,
    min_distance_pct: float = 0.25,
    doc_id_col: str = "doc_id",
    page_col: str = "page",
    **kwargs,
) -> pd.DataFrame:
    """
    Detect columns for all pages in a document.

    Iterates over each page, runs column detection, and returns a DataFrame
    with column info per page.

    Args:
        df: DataFrame with line data for all pages.
        min_peak_ratio: Minimum ratio for second peak (passed to detect_columns_for_page).
        min_distance_pct: Minimum distance between columns (passed to detect_columns_for_page).
        doc_id_col: Column name for document ID.
        page_col: Column name for page number.
        **kwargs: Additional arguments passed to detect_columns_for_page.

    Returns:
        DataFrame with columns: doc_id, page, num_columns, col_0_x0, col_1_x0, col_split_x

    Example:
        >>> col_info_df = detect_columns_for_document(line_df)
        >>> two_col_pages = col_info_df[col_info_df['num_columns'] == 2]
        >>> print(f"{len(two_col_pages)} pages have 2 columns")
    """
    col_info_rows = []

    group_cols = [doc_id_col, page_col] if doc_id_col in df.columns else [page_col]

    for group_key, page_df in df.groupby(group_cols):
        col_info = detect_columns_for_page(
            page_df,
            min_peak_ratio=min_peak_ratio,
            min_distance_pct=min_distance_pct,
            **kwargs,
        )

        row = col_info.to_dict()

        # Add grouping columns
        if isinstance(group_key, tuple):
            row[doc_id_col] = group_key[0]
            row[page_col] = group_key[1]
        else:
            row[page_col] = group_key
            if doc_id_col in df.columns:
                row[doc_id_col] = df[doc_id_col].iloc[0]

        col_info_rows.append(row)

    return pd.DataFrame(col_info_rows)


def assign_line_columns(
    df: pd.DataFrame,
    col_info_df: pd.DataFrame,
    doc_id_col: str = "doc_id",
    page_col: str = "page",
    x0_col: str = "geom_x0",
    page_width_col: str = "page_width",
    col_margin_tol_min: float = 6.0,
    col_margin_tol_pct: float = 0.02,
) -> pd.DataFrame:
    """
    Assign lines to columns and compute column-aware alignment.

    Merges column info into line DataFrame and adds:
    - line_column: 0 for left column, 1 for right column
    - col_left_x0: Left margin of the line's column
    - col_margin_tol: Tolerance for alignment detection
    - col_left_aligned: Whether line is aligned to column margin

    Args:
        df: DataFrame with line data.
        col_info_df: DataFrame from detect_columns_for_document.
        doc_id_col: Column name for document ID.
        page_col: Column name for page number.
        x0_col: Column name for line x0 coordinate.
        page_width_col: Column name for page width.
        col_margin_tol_min: Minimum margin tolerance in points.
        col_margin_tol_pct: Margin tolerance as fraction of page width.

    Returns:
        DataFrame with added columns: line_column, col_left_x0, col_margin_tol, col_left_aligned

    Example:
        >>> df = assign_line_columns(line_df, col_info_df)
        >>> right_col_lines = df[df['line_column'] == 1]
    """
    # Determine merge columns
    merge_cols = [page_col]
    if doc_id_col in df.columns and doc_id_col in col_info_df.columns:
        merge_cols = [doc_id_col, page_col]

    # Merge column info
    df = df.merge(col_info_df, on=merge_cols, how="left")

    # Initialize line_column to 0 (left column)
    df["line_column"] = 0

    # For two-column pages, assign to right column if x0 >= split_x
    is_two_col = df["num_columns"] == 2
    if "col_split_x" in df.columns:
        df.loc[is_two_col & (df[x0_col] >= df["col_split_x"]), "line_column"] = 1

    # Set column left margin based on assigned column
    df["col_left_x0"] = df["col_0_x0"]
    if "col_1_x0" in df.columns:
        df.loc[df["line_column"] == 1, "col_left_x0"] = df["col_1_x0"]

    # Compute adaptive tolerance: percentage of page width with minimum floor
    if page_width_col in df.columns:
        df["col_margin_tol"] = np.maximum(
            col_margin_tol_min,
            df[page_width_col] * col_margin_tol_pct
        )
    else:
        df["col_margin_tol"] = col_margin_tol_min

    # Column-aware left alignment check
    df["col_left_aligned"] = (df[x0_col] - df["col_left_x0"]).abs() <= df["col_margin_tol"]

    return df


def cleanup_list_item_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove temporary list item columns added during column detection.

    Args:
        df: DataFrame with _is_bullet, _is_enum, _is_list_item columns.

    Returns:
        DataFrame with temporary columns removed.
    """
    cols_to_drop = ["_is_bullet", "_is_enum", "_is_list_item"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
