"""
Serialization utilities for VaaS extraction pipeline.

This module provides normalization functions to prepare data for parquet
serialization. Parquet has strict type requirements, so numpy arrays and
mixed types need to be converted to plain Python types.
"""

from typing import Any, List, Union

import numpy as np


def normalize_cell_list(x: Any) -> List:
    """
    Normalize list-like values to plain Python lists for parquet serialization.

    Handles various input types that may appear in DataFrame cells:
    - None -> empty list
    - numpy.ndarray -> list
    - tuple -> list
    - list -> list (pass-through)
    - scalar (int/float) -> single-element list

    Args:
        x: Value to normalize (may be None, array, list, tuple, or scalar).

    Returns:
        Plain Python list suitable for parquet serialization.

    Example:
        >>> normalize_cell_list(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> normalize_cell_list(None)
        []
        >>> normalize_cell_list(42)
        [42]
    """
    if x is None:
        return []

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (list, tuple)):
        return list(x)

    if isinstance(x, (int, float, np.integer, np.floating)):
        return [int(x) if isinstance(x, (int, np.integer)) else float(x)]

    # Fallback for unexpected types
    return []


def normalize_bbox(bbox: Any) -> List[float]:
    """
    Normalize bounding box to a list of 4 floats, or empty list if invalid.

    Bounding boxes should be [x0, y0, x1, y1] representing the rectangle
    coordinates. This function ensures consistent format for serialization.

    Args:
        bbox: Bounding box value (may be list, tuple, array, or invalid).

    Returns:
        List of exactly 4 floats, or empty list if input is invalid.

    Example:
        >>> normalize_bbox([72.0, 100, 540.0, 120])
        [72.0, 100.0, 540.0, 120.0]
        >>> normalize_bbox(np.array([72, 100, 540, 120]))
        [72.0, 100.0, 540.0, 120.0]
        >>> normalize_bbox(None)
        []
        >>> normalize_bbox([1, 2])  # Invalid - not 4 elements
        []
    """
    # First normalize to list
    bbox_list = normalize_cell_list(bbox)

    # Validate: must have exactly 4 elements
    if len(bbox_list) != 4:
        return []

    # Convert all elements to float
    try:
        return [float(x) for x in bbox_list]
    except (ValueError, TypeError):
        return []


def normalize_pages(pages: Any) -> List[int]:
    """
    Normalize page numbers to a list of integers.

    Page numbers may come in as scalars, lists, or arrays. This ensures
    a consistent list of integers for serialization.

    Args:
        pages: Page value(s) - may be int, list, array, or None.

    Returns:
        List of integer page numbers.

    Example:
        >>> normalize_pages(3)
        [3]
        >>> normalize_pages([1, 2, 3])
        [1, 2, 3]
        >>> normalize_pages(np.array([1.0, 2.0]))
        [1, 2]
    """
    pages_list = normalize_cell_list(pages)

    try:
        return [int(p) for p in pages_list]
    except (ValueError, TypeError):
        return []


def safe_extract_page(pages: Any, default: int = 0) -> int:
    """
    Safely extract a single page number from various input formats.

    This handles the common pattern of extracting the first page from
    a pages field that may be a list, scalar, or array.

    Args:
        pages: Page value(s) - may be int, list, array, or None.
        default: Value to return if extraction fails.

    Returns:
        First page number as int, or default if invalid.

    Example:
        >>> safe_extract_page([3, 4, 5])
        3
        >>> safe_extract_page(7)
        7
        >>> safe_extract_page(None)
        0
    """
    if pages is None:
        return default

    if isinstance(pages, (int, np.integer)):
        return int(pages)

    if isinstance(pages, (float, np.floating)):
        return int(pages) if not np.isnan(pages) else default

    if isinstance(pages, (list, tuple, np.ndarray)) and len(pages) > 0:
        first = pages[0]
        if isinstance(first, (int, float, np.integer, np.floating)):
            return int(first) if not (isinstance(first, float) and np.isnan(first)) else default

    if isinstance(pages, str):
        # Handle string like "3" or "[3]"
        digits = "".join(ch for ch in pages if ch.isdigit())
        return int(digits) if digits else default

    return default
