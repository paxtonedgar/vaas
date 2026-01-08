"""
Utility modules for VaaS extraction pipeline.

Submodules:
    text: Text processing utilities (hashing, hyphenation repair, slugification)
    serialize: Serialization utilities for parquet-safe data normalization
"""

from vaas.utils.text import stable_hash, repair_hyphenation, slug_title
from vaas.utils.serialize import (
    normalize_cell_list,
    normalize_bbox,
    normalize_pages,
    safe_extract_page,
)

__all__ = [
    # Text utilities
    "stable_hash",
    "repair_hyphenation",
    "slug_title",
    # Serialization utilities
    "normalize_cell_list",
    "normalize_bbox",
    "normalize_pages",
    "safe_extract_page",
]
