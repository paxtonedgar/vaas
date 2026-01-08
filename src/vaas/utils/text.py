"""
Text processing utilities for VaaS extraction pipeline.

This module provides text manipulation functions used throughout the pipeline:
- Stable hashing for ID generation
- Hyphenation repair for PDF text extraction
- Slug generation for human-readable IDs
"""

import hashlib
import re
from typing import List


def stable_hash(parts: List[str], length: int = 16) -> str:
    """
    Generate a stable hash from a list of string parts.

    Creates a deterministic hash that can be used for generating unique IDs
    from content. The same inputs will always produce the same hash.

    Args:
        parts: List of strings to hash together.
        length: Number of hex characters to return (max 64 for SHA256).

    Returns:
        Hex string of specified length.

    Example:
        >>> stable_hash(["box_1a", "page_3"], length=8)
        'a1b2c3d4'
    """
    combined = "|".join(parts)
    full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return full_hash[:length]


def repair_hyphenation(text: str) -> str:
    """
    Fix hyphenated line breaks in extracted PDF text.

    PDF text extraction often preserves line-break hyphens that split words
    across lines (e.g., "fur-\\nnishing" should become "furnishing").

    This function:
    - Joins hyphenated words split across lines
    - Only joins if the continuation starts with lowercase (avoids proper nouns)
    - Handles both immediate newlines and newlines followed by whitespace

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Text with hyphenated line breaks repaired.

    Example:
        >>> repair_hyphenation("The fur-\\nnishing costs were high.")
        'The furnishing costs were high.'
        >>> repair_hyphenation("See Pub-\\n  lication 15.")
        'See Publication 15.'
    """
    if not text:
        return text

    # Pattern: word ending with hyphen, newline, then lowercase continuation
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)

    # Also handle with space after newline
    text = re.sub(r"(\w)-\n\s+([a-z])", r"\1\2", text)

    return text


def slug_title(text: str, max_len: int = 30) -> str:
    """
    Convert title text to a URL-safe slug.

    Creates a human-readable identifier from title text by:
    - Taking the first line only
    - Converting to lowercase
    - Replacing non-alphanumeric characters with underscores
    - Stripping leading/trailing underscores
    - Truncating to max_len

    Args:
        text: Title text to slugify.
        max_len: Maximum length of the resulting slug.

    Returns:
        URL-safe slug string.

    Example:
        >>> slug_title("Qualified Dividends and Capital Gains", max_len=20)
        'qualified_dividends_'
        >>> slug_title("Box 1a.\\nTotal Ordinary Dividends")
        'box_1a'
    """
    if not text:
        return ""

    # Take first line only
    first_line = text.strip().split("\n")[0]

    # Normalize: lowercase, replace non-alphanum with underscore
    slug = re.sub(r"[^a-z0-9]+", "_", first_line.lower())

    # Strip leading/trailing underscores, truncate
    return slug.strip("_")[:max_len]
