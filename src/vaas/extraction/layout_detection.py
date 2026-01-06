"""
Layout-driven detection for document structure elements.

Provides functions for detecting subsections, headers, and other
structural elements using layout heuristics (font, position, emphasis).
"""

import pandas as pd
from typing import Tuple


def detect_subsection_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect subsection candidates using layout-driven heuristics.

    This function adds subsection detection columns to a line-level DataFrame.
    It uses a multi-criteria approach:
    1. Base candidate criteria (bold, length, position, etc.)
    2. Structural confirmations (gap above, followed by body, not in header/footer)
    3. Final filter requiring base criteria + 2 structural confirmations

    Required input columns:
        - line_bold: bool - whether line is bold
        - is_heading_length: bool - char count in heading range (12-60)
        - has_multiple_words: bool - ≥2 words
        - ends_with_period: bool - ends with period
        - col_left_aligned: bool - column-aware left alignment
        - looks_like_date: bool - date pattern match
        - is_box_strong, is_box_weak: bool - box header detection
        - is_section: bool - section header detection
        - is_page_marker: bool - page marker detection
        - in_header_band, in_footer_band: bool - header/footer zones
        - has_large_gap_above: bool - vertical gap indicator
        - is_artifact_repeat: bool - repeated artifact text
        - doc_id, page, block_id: grouping columns
        - line_char_count: int - character count

    Added columns:
        - is_base_candidate: bool - passes primary candidate criteria
        - followed_by_body: bool - next line looks like body text
        - confirm_not_in_band: bool - not in header/footer
        - confirm_gap_or_early: bool - has gap above
        - confirm_followed_by_body: bool - followed by body text
        - structural_confirms: int - count of confirmations (0-3)
        - is_subsection_candidate: bool - final subsection candidate flag

    Args:
        df: Line-level DataFrame with required columns

    Returns:
        DataFrame with added subsection detection columns
    """
    df = df.copy()

    # STRUCTURAL FILTER: Followed-by-body confirmation
    # Next line should be: not bold OR longer than this line OR body-sized font
    # FIX: Don't penalize end-of-block headings - treat "no next line" as passing
    df["_next_bold_raw"] = df.groupby(["doc_id", "page", "block_id"])["line_bold"].shift(-1)
    df["_next_char_count_raw"] = df.groupby(["doc_id", "page", "block_id"])["line_char_count"].shift(-1)
    df["next_line_exists"] = df["_next_bold_raw"].notna()

    # Use nullable dtypes to avoid FutureWarning on fillna downcasting
    df["next_bold"] = df["_next_bold_raw"].astype("boolean").fillna(False).astype(bool)
    df["next_char_count"] = df["_next_char_count_raw"].astype("Int64").fillna(0).astype(int)

    df["followed_by_body"] = (
        (~df["next_line_exists"]) |  # No next line = OK (don't penalize end-of-block)
        (~df["next_bold"]) |
        (df["next_char_count"] > df["line_char_count"] * 1.2)  # Next line is longer
    )

    # PRIMARY CANDIDATE CRITERIA (all required):
    # - bold, 12-60 chars, ≥2 words, not ending with period, COLUMN-ALIGNED
    # - not already classified as box/section/page_marker
    # - not a date pattern
    # NOTE: Using col_left_aligned (column-aware) instead of left_aligned (block-relative)
    df["is_base_candidate"] = (
        df["line_bold"] &
        df["is_heading_length"] &
        df["has_multiple_words"] &
        ~df["ends_with_period"] &
        df["col_left_aligned"] &
        ~df["looks_like_date"] &
        ~df["is_box_strong"] &
        ~df["is_box_weak"] &
        ~df["is_section"] &
        ~df["is_page_marker"]
    )

    # STRUCTURAL CONFIRMATIONS (require 2 of 3):
    df["confirm_not_in_band"] = ~(df["in_header_band"] | df["in_footer_band"])
    df["confirm_gap_or_early"] = df["has_large_gap_above"]
    df["confirm_followed_by_body"] = df["followed_by_body"]

    # Count structural confirmations
    df["structural_confirms"] = (
        df["confirm_not_in_band"].astype(int) +
        df["confirm_gap_or_early"].astype(int) +
        df["confirm_followed_by_body"].astype(int)
    )

    # Final subsection candidate: base criteria + at least 2 structural confirmations
    # FIX: Only exclude artifact repeats (repeated AND in header/footer band)
    df["is_subsection_candidate"] = (
        df["is_base_candidate"] &
        (df["structural_confirms"] >= 2) &
        ~df["is_artifact_repeat"]
    )

    # Clean up temporary columns
    df = df.drop(columns=["_next_bold_raw", "_next_char_count_raw"], errors="ignore")

    return df


def assign_split_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign split trigger flags based on detected structural elements.

    This determines where to split text blocks into separate elements
    based on detected boxes, sections, subsections, and page markers.

    Required input columns:
        - is_page_marker, is_box_strong, is_box_weak: bool
        - is_section, is_subsection_candidate: bool
        - has_emphasis, left_aligned, col_left_aligned: bool
        - line_idx_in_block: int

    Added columns:
        - split_trigger: bool - whether this line triggers a split
        - split_kind: str - type of split (box, section, subsection, page_artifact, none)

    Args:
        df: Line-level DataFrame with detection columns

    Returns:
        DataFrame with split trigger columns added
    """
    df = df.copy()

    df["split_kind"] = "none"
    df["split_trigger"] = False

    early = df["line_idx_in_block"] <= 1

    # Page markers are ALWAYS split triggers (isolate them into own elements)
    df.loc[df["is_page_marker"], "split_trigger"] = True
    df.loc[df["is_page_marker"], "split_kind"] = "page_artifact"

    # Strong box headers
    strong_box = df["is_box_strong"] & ~df["is_page_marker"]
    df.loc[strong_box, "split_trigger"] = True
    df.loc[strong_box, "split_kind"] = "box"

    # Weak box headers (with additional gates)
    weak_box = df["is_box_weak"] & ~df["is_box_strong"] & ~df["is_page_marker"]
    weak_box_trigger = weak_box & (early | df["left_aligned"] | df["has_emphasis"])
    df.loc[weak_box_trigger, "split_trigger"] = True
    df.loc[weak_box_trigger, "split_kind"] = "box"

    # Main section headers (require emphasis)
    sec_trigger = df["is_section"] & df["has_emphasis"] & ~df["is_page_marker"]
    sec_trigger = sec_trigger & ~df["split_trigger"]
    df.loc[sec_trigger, "split_trigger"] = True
    df.loc[sec_trigger, "split_kind"] = "section"

    # LAYOUT-DRIVEN subsection detection
    # Gate: must be early in block OR column-left-aligned
    subsec_trigger = df["is_subsection_candidate"] & ~df["split_trigger"]
    subsec_trigger = subsec_trigger & (df["col_left_aligned"] | early)
    df.loc[subsec_trigger, "split_trigger"] = True
    df.loc[subsec_trigger, "split_kind"] = "subsection"

    return df


def detect_and_assign_structure(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Combined function to detect subsections and assign split triggers.

    This is a convenience function that calls detect_subsection_candidates
    followed by assign_split_triggers.

    Args:
        df: Line-level DataFrame with required columns

    Returns:
        Tuple of (processed DataFrame, count of subsection candidates)
    """
    df = detect_subsection_candidates(df)
    df = assign_split_triggers(df)

    subsec_count = (df["split_kind"] == "subsection").sum()

    return df, subsec_count
