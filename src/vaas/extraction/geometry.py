"""
Geometry utilities for PDF extraction.

This module provides functions for working with bounding boxes, font detection,
and spatial sorting in PDF documents. These are low-level utilities used by
the extraction pipeline for layout analysis.

Key concepts:
- Bounding box (bbox): [x0, y0, x1, y1] where (x0, y0) is top-left corner
- Reading order: left-to-right, top-to-bottom within columns
- Column detection: based on x-coordinate relative to page midpoint
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Type aliases
BBox = Tuple[float, float, float, float]
BBoxLike = Union[list, tuple, np.ndarray, None]


def safe_bbox(bbox: BBoxLike) -> Tuple[float, float, float, float]:
    """
    Safely extract bounding box coordinates, handling invalid inputs.

    PDF extraction can produce malformed bbox data (None, wrong length,
    non-numeric values). This function normalizes all inputs to a consistent
    4-tuple of floats, using NaN for invalid data.

    Args:
        bbox: Bounding box as [x0, y0, x1, y1], or invalid/None value.

    Returns:
        Tuple of (x0, y0, x1, y1) as floats. Returns (NaN, NaN, NaN, NaN)
        if input is invalid.

    Example:
        >>> safe_bbox([72.0, 100, 540, 120])
        (72.0, 100.0, 540.0, 120.0)
        >>> safe_bbox(None)
        (nan, nan, nan, nan)
        >>> safe_bbox([1, 2])  # Wrong length
        (nan, nan, nan, nan)
    """
    try:
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return np.nan, np.nan, np.nan, np.nan
    except (TypeError, ValueError, IndexError):
        return np.nan, np.nan, np.nan, np.nan


def is_bold(font: Optional[str], flags: Optional[int]) -> bool:
    """
    Detect if text is bold from font name or PDF flags.

    Bold detection uses two methods:
    1. Font name contains "bold" (case-insensitive)
    2. PDF flags bit 4 (0x10 = 16) is set (standard PDF bold flag)

    Args:
        font: Font name string (e.g., "Helvetica-Bold", "Arial").
        flags: PDF text flags bitmask.

    Returns:
        True if text appears to be bold.

    Example:
        >>> is_bold("Helvetica-Bold", 0)
        True
        >>> is_bold("Arial", 16)  # Bit 4 set
        True
        >>> is_bold("Times", 0)
        False
    """
    # Check font name for "bold"
    font_str = str(font or "").lower()
    if "bold" in font_str:
        return True

    # Check PDF flags bit 4 (bold flag)
    # Flag value 16 = 0x10 = bit 4 set
    return bool(int(flags or 0) & 16)


def margin_tolerance(block_width: Optional[float], min_tol: float = 2.0, pct: float = 0.02) -> float:
    """
    Calculate margin tolerance for alignment detection.

    When determining if a line is "left-aligned", we need tolerance for
    minor variations in x-coordinates. This function calculates a tolerance
    as a percentage of block width, with a minimum floor.

    Args:
        block_width: Width of the containing block in points.
        min_tol: Minimum tolerance in points (default 2.0).
        pct: Percentage of block width to use (default 0.02 = 2%).

    Returns:
        Tolerance value in points.

    Example:
        >>> margin_tolerance(500.0)
        10.0  # 2% of 500
        >>> margin_tolerance(50.0)
        2.0   # Minimum floor
        >>> margin_tolerance(None)
        2.0   # Fallback for invalid input
    """
    if block_width is None or pd.isna(block_width):
        return min_tol
    return float(max(min_tol, pct * float(block_width)))


def bbox_y0(bbox: BBoxLike, default: float = 0.0) -> float:
    """
    Extract y0 (top) coordinate from a bounding box.

    This is a common operation when sorting elements by vertical position.
    Handles various input formats gracefully.

    Args:
        bbox: Bounding box as [x0, y0, x1, y1] or similar.
        default: Value to return if extraction fails.

    Returns:
        The y0 coordinate, or default if invalid.

    Example:
        >>> bbox_y0([72, 100, 540, 120])
        100.0
        >>> bbox_y0(None)
        0.0
    """
    if bbox is None:
        return default

    try:
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 2:
            val = float(bbox[1])
            return val if not np.isnan(val) else default
        return default
    except (TypeError, ValueError, IndexError):
        return default


def bbox_x0(bbox: BBoxLike, default: float = 0.0) -> float:
    """
    Extract x0 (left) coordinate from a bounding box.

    Args:
        bbox: Bounding box as [x0, y0, x1, y1] or similar.
        default: Value to return if extraction fails.

    Returns:
        The x0 coordinate, or default if invalid.

    Example:
        >>> bbox_x0([72, 100, 540, 120])
        72.0
    """
    if bbox is None:
        return default

    try:
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 1:
            val = float(bbox[0])
            return val if not np.isnan(val) else default
        return default
    except (TypeError, ValueError, IndexError):
        return default


def reading_order_sort_key(
    row: Dict[str, Any],
    page_mid_x: float = 306.0,
) -> Tuple[int, int, float, float]:
    """
    Generate a sort key for reading order within a document.

    Reading order in two-column PDFs follows this pattern:
    1. First by page number
    2. Then by column (left column before right)
    3. Then by y-position (top to bottom)
    4. Then by x-position (left to right, for same-line elements)

    Args:
        row: Dictionary or Series with 'pages'/'page' and 'bbox' keys.
        page_mid_x: X-coordinate that divides left/right columns.
                    Elements with x0 < page_mid_x are in left column (0),
                    others are in right column (1).

    Returns:
        Tuple of (page, column, y0, x0) for sorting.

    Example:
        >>> reading_order_sort_key({'pages': [3], 'bbox': [72, 100, 300, 120]})
        (3, 0, 100.0, 72.0)  # Page 3, left column
        >>> reading_order_sort_key({'pages': [3], 'bbox': [350, 100, 540, 120]})
        (3, 1, 100.0, 350.0)  # Page 3, right column
    """
    # Extract page number safely
    pages = row.get("pages") if row.get("pages") is not None else row.get("page")

    if isinstance(pages, (list, tuple, np.ndarray)) and len(pages) > 0:
        page = int(pages[0])
    elif isinstance(pages, (int, float, np.integer, np.floating)):
        page = int(pages) if not np.isnan(pages) else 0
    else:
        page = 0

    # Extract bbox coordinates safely
    bbox = row.get("bbox", [0, 0, 0, 0])
    x0 = bbox_x0(bbox, default=0.0)
    y0 = bbox_y0(bbox, default=0.0)

    # Determine column: 0 = left, 1 = right
    col = 0 if x0 < page_mid_x else 1

    return (page, col, y0, x0)


def compute_line_geometry(
    geom_x0: float,
    geom_y0: float,
    geom_x1: float,
    geom_y1: float,
) -> Dict[str, float]:
    """
    Compute derived geometry values for a line.

    Args:
        geom_x0: Left x-coordinate.
        geom_y0: Top y-coordinate.
        geom_x1: Right x-coordinate.
        geom_y1: Bottom y-coordinate.

    Returns:
        Dictionary with computed values:
        - center_x: Horizontal center
        - center_y: Vertical center
        - width: Line width
        - height: Line height

    Example:
        >>> compute_line_geometry(72, 100, 540, 120)
        {'center_x': 306.0, 'center_y': 110.0, 'width': 468.0, 'height': 20.0}
    """
    return {
        "center_x": (geom_x0 + geom_x1) / 2,
        "center_y": (geom_y0 + geom_y1) / 2,
        "width": geom_x1 - geom_x0,
        "height": geom_y1 - geom_y0,
    }


def is_centered(
    line_center_x: float,
    page_mid_x: float,
    page_width: float,
    tolerance_pct: float = 0.15,
) -> bool:
    """
    Check if a line is horizontally centered on the page.

    Args:
        line_center_x: X-coordinate of the line's center.
        page_mid_x: X-coordinate of the page's center.
        page_width: Total page width.
        tolerance_pct: Percentage of page width for tolerance (default 15%).

    Returns:
        True if the line center is within tolerance of page center.

    Example:
        >>> is_centered(306, 306, 612, tolerance_pct=0.15)
        True
        >>> is_centered(100, 306, 612, tolerance_pct=0.15)
        False
    """
    tolerance = page_width * tolerance_pct
    return abs(line_center_x - page_mid_x) <= tolerance
